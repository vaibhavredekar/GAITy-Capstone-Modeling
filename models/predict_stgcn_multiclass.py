"""
predict_multilabel_stgcn.py

Production-grade script for loading pre-trained ST-GCN models and making predictions.
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
    handlers=[logging.FileHandler('multilabel_stgcn_predictions.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Model Architectures ---
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
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

class MultiSTGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for multi-label classification."""
    def __init__(self, in_channels=3, num_joints=14, out_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
        self.fc = nn.Linear(256 * num_joints, out_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

# --- Predictor Class ---
class MultiLabelSTGCPredictor:
    """Handles loading both binary and multi-label ST-GCN models for hierarchical prediction."""
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.metadata_path = self.models_dir / "stgcn_metadata.json"
        self.binary_model = None
        self.multi_model = None
        self.metadata = None
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self) -> bool:
        """Load both models and their metadata."""
        try:
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
            
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)

            # Load binary model
            binary_model_path = self.models_dir / self.metadata["binary_model"]["path"]
            self.binary_model = BinarySTGCN(
                num_joints=self.metadata["binary_model"]["num_joints"], 
                out_classes=1
            ).to(self.device)
            self.binary_model.load_state_dict(clean_state_dict(torch.load(str(binary_model_path), map_location='cpu')))
            self.binary_model.eval()

            # Load multi-label model
            multi_model_path = self.models_dir / self.metadata["multilabel_model"]["path"]
            self.multi_model = MultiSTGCN(
                num_joints=self.metadata["multilabel_model"]["num_joints"],
                out_classes=self.metadata["multilabel_model"]["num_labels"]
            ).to(self.device)
            self.multi_model.load_state_dict(clean_state_dict(torch.load(str(multi_model_path), map_location='cpu')))
            self.multi_model.eval()

            self.is_loaded = True
            logger.info("Binary and Multi-label ST-GCN models loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            return False

    def _csv_to_stgcn_tensor(self, df):
        """Convert a DataFrame of landmarks to the (N, C, T, V) ST-GCN tensor."""
        num_frames = df['frame'].nunique()
        num_joints = self.metadata["binary_model"]["num_joints"] # Assuming same for both models
        num_channels = 3
        X = np.zeros((1, num_channels, num_frames, num_joints), dtype=np.float32)
        
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
        """Make hierarchical predictions on a DataFrame of landmarks."""
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Convert DataFrame to tensor
        X_tensor = self._csv_to_stgcn_tensor(input_df).to(self.device)
        
        with torch.no_grad():
            # --- Binary Prediction ---
            binary_logits = self.binary_model(X_tensor)
            binary_prob = torch.sigmoid(binary_logits).item()
            binary_prediction = "abnormal" if binary_prob >= self.metadata["binary_model"]["threshold"] else "normal"
            
            # --- Multi-Label Prediction (only if abnormal) ---
            multi_label_predictions = []
            if binary_prediction == "abnormal":
                multi_logits = self.multi_model(X_tensor)
                multi_probs = torch.sigmoid(multi_logits).cpu().numpy()[0]
                
                # Get anomaly columns from metadata
                anomaly_cols = self.metadata["multilabel_model"]["anomaly_cols"]
                
                # Apply threshold and map indices to names
                for i, prob in enumerate(multi_probs):
                    if prob >= 0.5:
                        multi_label_predictions.append(anomaly_cols[i])
            
            return {
                "binary_prediction": binary_prediction,
                "binary_probability": float(binary_prob),
                "multilabel_predictions": multi_label_predictions,
                "model_type": "hierarchical_stgcn"
            }

def main():
    parser = argparse.ArgumentParser(description="Predict gait using hierarchical ST-GCN models.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to the input CSV file with landmarks.")
    parser.add_argument("--models-dir", "-m", type=Path, default="models", help="Directory containing models.")
    args = parser.parse_args()

    predictor = MultiLabelSTGCPredictor(models_dir=args.models_dir)
    if not predictor.load_models():
        sys.exit(1)

    try:
        df = pd.read_csv(args.input)
        results = predictor.predict(df)
        
        print(f"\nPrediction for: {args.input.name}")
        print(f"Overall gait: {results['binary_prediction']} (prob={results['binary_probability']:.3f})")
        
        if results['multilabel_predictions']:
            print("Detected anomalies:")
            for anomaly in results['multilabel_predictions']:
                print(f" - {anomaly}")
        else:
            print("No specific anomalies detected.")

    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()