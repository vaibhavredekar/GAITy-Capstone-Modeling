"""
predict_binary_stgcn.py

Production-grade script for loading a pre-trained binary ST-GCN model and making predictions.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('binary_stgcn_predictions.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Model Architecture (from your prediction script) ---
def clean_state_dict(state_dict):
    """Remove 'module.' prefix if present."""
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    return new_dict

class BinarySTGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for binary classification."""
    def __init__(self, in_channels=3, num_joints=14, out_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
        self.fc = nn.Linear(256 * num_joints, out_classes)
        
    def forward(self, x):
        # x: (N, C, T, V, M) -> (N, C, T, V)
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

# --- Predictor Class ---
class BinarySTGCPredictor:
    """Handles loading the binary ST-GCN model and making predictions."""
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.metadata_path = self.models_dir / "stgcn_metadata.json"
        self.model = None
        self.metadata = None
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> bool:
        """Load the binary model and its metadata."""
        try:
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
            
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)

            # Load binary model
            binary_model_path = self.models_dir / self.metadata["binary_model"]["path"]
            self.model = BinarySTGCN(
                num_joints=self.metadata["binary_model"]["num_joints"], 
                out_classes=1
            ).to(self.device)
            self.model.load_state_dict(clean_state_dict(torch.load(str(binary_model_path), map_location='cpu')))
            self.model.eval()

            self.is_loaded = True
            logger.info(f"Binary ST-GCN model loaded successfully from {binary_model_path.name}")
            return True
        except Exception as e:
            logger.error(f"Error loading binary model: {e}", exc_info=True)
            return False

    def _csv_to_stgcn_tensor(self, df):
        """
        Convert a DataFrame of landmarks to the (N, C, T, V) ST-GCN tensor.
        
        NOTE: This is a simplified version. It assumes one person per CSV and that landmark IDs
        are contiguous from 0 to 13 for the 14 GAIT_JOINTS. A real-world implementation might need
        more robust handling of multiple subjects or non-standard landmark IDs.
        """
        num_frames = df['frame'].nunique()
        num_joints = 14 # From metadata["binary_model"]["num_joints"]
        num_channels = 3
        X = np.zeros((1, num_channels, num_frames, num_joints), dtype=np.float32)
        
        # Create a mapping from landmark_id to a 0-indexed joint index
        # This assumes landmark IDs for the 14 GAIT_JOINTS are 0-13.
        # If your data has different IDs, this mapping will need to be adjusted.
        landmark_to_joint_idx = {i: i for i in range(num_joints)}

        for frame_idx, frame in enumerate(sorted(df['frame'].unique())):
            frame_data = df[df['frame'] == frame_idx]
            for _, row in frame_data.iterrows():
                landmark_id = int(row['landmark_id'])
                if landmark_id in landmark_to_joint_idx:
                    joint_idx = landmark_to_joint_idx[landmark_id]
                    X[0, 0, frame_idx, joint_idx, 0] = [row['x_norm'], row['y_norm'], row['z_norm']]
        
        return torch.from_numpy(X)

    def predict(self, input_df):
        """Make predictions on a DataFrame of landmarks."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert DataFrame to tensor
        X_tensor = self._csv_to_stgcn_tensor(input_df).to(self.device)
        
        with torch.no_grad():
            # Get binary prediction
            logits = self.model(X_tensor)
            probability = torch.sigmoid(logits).item()
            prediction = "abnormal" if probability >= self.metadata["binary_model"]["threshold"] else "normal"
            
            return {
                "prediction": prediction,
                "probability": float(probability),
                "model_type": "binary_stgcn"
            }

def main():
    parser = argparse.ArgumentParser(description="Predict gait using the binary ST-GCN model.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to the input CSV file with landmarks.")
    parser.add_argument("--models-dir", "-m", type=Path, default="models", help="Directory containing models.")
    args = parser.parse_args()

    predictor = BinarySTGCPredictor(models_dir=args.models_dir)
    if not predictor.load_model():
        sys.exit(1)

    try:
        df = pd.read_csv(args.input)
        results = predictor.predict(df)
        
        print(f"\nPrediction for: {args.input.name}")
        print(f"Overall gait: {results['prediction']} (probability={results['probability']:.3f})")

    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()