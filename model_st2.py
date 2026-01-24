#!/usr/bin/env python3
"""
GAITy - Production Grade Application with Baseline & Advanced Models
Complete pipeline: CSV → Feature Engineering → Model Prediction → Results
"""

import os
import sys
import logging
import traceback
import warnings
import json
from pathlib import Path
from datetime import datetime
from io import BytesIO
import time

# Configure logging
def setup_logging():
    """Set up comprehensive logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"gait_analysis_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("gait_analysis")

logger = setup_logging()

# Configure environment
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Import Streamlit first
try:
    import streamlit as st
    logger.info("Streamlit imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Streamlit: {e}")
    sys.exit(1)

# Set page configuration
st.set_page_config(
    page_title="GAITy - Gait Analysis Prediction",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import dependencies with error handling
DEPENDENCIES = {
    'pandas': False,
    'numpy': False,
    'xgboost': False,
    'matplotlib': False,
    'scipy': False,
    'torch': False,
    'polars': False
}

try:
    import pandas as pd
    DEPENDENCIES['pandas'] = True
    logger.info("Pandas imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Pandas: {e}")

try:
    import numpy as np
    DEPENDENCIES['numpy'] = True
    logger.info("NumPy imported successfully")
except ImportError as e:
    logger.error(f"Failed to import NumPy: {e}")

try:
    import xgboost as xgb
    DEPENDENCIES['xgboost'] = True
    logger.info(f"XGBoost {xgb.__version__} imported successfully")
except ImportError as e:
    logger.error(f"Failed to import XGBoost: {e}")

try:
    import matplotlib.pyplot as plt
    DEPENDENCIES['matplotlib'] = True
    logger.info("Matplotlib imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Matplotlib: {e}")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    DEPENDENCIES['torch'] = True
    logger.info(f"PyTorch {torch.__version__} imported successfully")
except ImportError as e:
    logger.error(f"Failed to import PyTorch: {e}")

try:
    import polars as pl
    DEPENDENCIES['polars'] = True
    logger.info("Polars imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Polars: {e}")

try:
    from scipy.signal import find_peaks, resample
    from scipy.ndimage import gaussian_filter1d
    DEPENDENCIES['scipy'] = True
    logger.info("SciPy imported successfully")
except ImportError as e:
    logger.error(f"Failed to import SciPy: {e}")

# Constants
BASELINE_MODEL_PATH = Path("models/baseline/xgboost_model.bin")
BINARY_MODEL_PATH = Path("models/advance/binary_model_full.bin")
MULTI_MODEL_PATH = Path("models/advance/multi_label_model_full.bin")
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
CACHE_DIR = Path("cache")

# Create directories
for directory in [UPLOAD_DIR, RESULTS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# MediaPipe constants
N_JOINTS = 33
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

GAIT_JOINTS = [2, 5, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Initialize session state
def init_session_state():
    """Initialize session state with default values."""
    defaults = {
        'baseline_model': None,
        'binary_model': None,
        'multi_model': None,
        'baseline_loaded': False,
        'binary_loaded': False,
        'multi_loaded': False,
        'features': None,
        'prediction': None,
        'file_info': None,
        'processing_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Model Management
class ModelManager:
    """Manages loading and using both baseline and advanced models."""
    
    @staticmethod
    def load_baseline_model():
        """Load the baseline XGBoost model."""
        try:
            if not DEPENDENCIES['xgboost']:
                return None, "XGBoost not available"
            
            if not BASELINE_MODEL_PATH.exists():
                return None, f"Baseline model not found: {BASELINE_MODEL_PATH}"
            
            model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.load_model(str(BASELINE_MODEL_PATH))
            
            return model, "Baseline model loaded successfully"
        except Exception as e:
            logger.error(f"Baseline model loading failed: {e}")
            return None, f"Baseline model loading failed: {str(e)}"
    
    @staticmethod
    def load_binary_model():
        """Load the advanced binary ST-GCN model."""
        try:
            if not DEPENDENCIES['torch']:
                return None, "PyTorch not available"
            
            if not BINARY_MODEL_PATH.exists():
                return None, f"Binary model not found: {BINARY_MODEL_PATH}"
            
            # Define the model architecture (must match training)
            class SimpleSTGCN(nn.Module):
                def __init__(self, num_joints, in_channels=3, out_classes=1):
                    super().__init__()
                    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(1,1))
                    self.conv2 = nn.Conv2d(64, 128, kernel_size=(1,1))
                    self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
                    self.fc = nn.Linear(128 * num_joints, out_classes)
                
                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    x = self.pool(x)
                    x = x.flatten(1)
                    return self.fc(x)
            
            # Determine num_joints from the saved model
            # This is a simplified approach - in production, you'd save this info with the model
            num_joints = 14  # Based on GAIT_JOINTS length
            
            # Create model instance
            model = SimpleSTGCN(num_joints=num_joints, out_classes=1)
            
            # Load state dict
            state_dict = torch.load(str(BINARY_MODEL_PATH), map_location='cpu')
            model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            model.eval()
            
            return model, "Binary model loaded successfully"
        except Exception as e:
            logger.error(f"Binary model loading failed: {e}")
            return None, f"Binary model loading failed: {str(e)}"
    
    @staticmethod
    def load_multi_model():
        """Load the advanced multi-label ST-GCN model."""
        try:
            if not DEPENDENCIES['torch']:
                return None, "PyTorch not available"
            
            if not MULTI_MODEL_PATH.exists():
                return None, f"Multi-label model not found: {MULTI_MODEL_PATH}"
            
            # Define the model architecture (must match training)
            class SimpleSTGCN(nn.Module):
                def __init__(self, num_joints, in_channels=3, out_classes=5):
                    super().__init__()
                    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(1,1))
                    self.conv2 = nn.Conv2d(64, 128, kernel_size=(1,1))
                    self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
                    self.fc = nn.Linear(128 * num_joints, out_classes)
                
                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    x = self.pool(x)
                    x = x.flatten(1)
                    return self.fc(x)
            
            # Determine num_joints and out_classes
            num_joints = 14  # Based on GAIT_JOINTS length
            out_classes = 5   # Based on ANOMALY_COLS
            
            # Create model instance
            model = SimpleSTGCN(num_joints=num_joints, out_classes=out_classes)
            
            # Load state dict
            state_dict = torch.load(str(MULTI_MODEL_PATH), map_location='cpu')
            model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            model.eval()
            
            return model, "Multi-label model loaded successfully"
        except Exception as e:
            logger.error(f"Multi-label model loading failed: {e}")
            return None, f"Multi-label model loading failed: {str(e)}"
    
    @staticmethod
    def create_fallback_model():
        """Create a fallback model when loading fails."""
        try:
            if not DEPENDENCIES['xgboost']:
                return None, "XGBoost not available"
            
            # Create a simple untrained model
            model = xgb.XGBClassifier(
                n_estimators=1,
                max_depth=1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            return model, "Fallback model created"
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            return None, f"Failed to create fallback model: {e}"

# Feature Engineering
class FeatureEngineer:
    """Handles feature extraction from pose data with model-specific requirements."""
    
    # ANOMALY_COLS from training
    ANOMALY_COLS = [
        "gait_anomaly_knee_sagittal_plane_abnormality",
        "gait_anomaly_trunk_balance_abnormality",
        "gait_anomaly_spatiotemporal_asymmetry",
        "gait_anomaly_hip_pelvic_control_deficit",
        "gait_anomaly_distal_foot_control_deficit",
    ]
    
    @staticmethod
    def validate_csv(df):
        """Validate CSV format and content."""
        errors = []
        
        # Check required columns
        required_cols = ['frame', 'landmark_id', 'x_norm', 'y_norm', 'z_norm']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data validity
        if df.empty:
            errors.append("CSV file is empty")
        
        if len(df) < 10:
            errors.append("CSV file has too few rows (< 10)")
        
        # Check for valid landmark IDs
        if 'landmark_id' in df.columns:
            invalid_ids = df[~df['landmark_id'].between(0, 32)]['landmark_id'].unique()
            if len(invalid_ids) > 0:
                errors.append(f"Invalid landmark IDs found: {invalid_ids}")
        
        return errors
    
    @staticmethod
    def extract_basic_features(df):
        """Extract basic statistical features for baseline model."""
        features = {}
        
        try:
            # Basic statistics for each coordinate
            for coord in ['x_norm', 'y_norm', 'z_norm']:
                if coord in df.columns:
                    features[f"{coord}_mean"] = df[coord].mean()
                    features[f"{coord}_std"] = df[coord].std()
                    features[f"{coord}_min"] = df[coord].min()
                    features[f"{coord}_max"] = df[coord].max()
                    features[f"{coord}_range"] = df[coord].max() - df[coord].min()
                    features[f"{coord}_median"] = df[coord].median()
            
            # Frame-based features
            if 'frame' in df.columns:
                features['total_frames'] = df['frame'].nunique()
                features['frame_range'] = df['frame'].max() - df['frame'].min()
                features['frames_per_landmark'] = len(df) / df['landmark_id'].nunique()
            
            # Landmark-specific features
            if 'landmark_id' in df.columns:
                landmarks = df['landmark_id'].unique()
                features['unique_landmarks'] = len(landmarks)
                
                # Key landmarks analysis
                key_landmarks = {
                    'left_shoulder': 11, 'right_shoulder': 12,
                    'left_hip': 23, 'right_hip': 24,
                    'left_knee': 25, 'right_knee': 26,
                    'left_ankle': 27, 'right_ankle': 28
                }
                
                for name, landmark_id in key_landmarks.items():
                    if landmark_id in landmarks:
                        landmark_data = df[df['landmark_id'] == landmark_id]
                        prefix = f"{name}_"
                        
                        for coord in ['x_norm', 'y_norm', 'z_norm']:
                            if coord in landmark_data.columns:
                                features[f"{prefix}{coord}_mean"] = landmark_data[coord].mean()
                                features[f"{prefix}{coord}_std"] = landmark_data[coord].std()
                                features[f"{prefix}{coord}_range"] = landmark_data[coord].max() - landmark_data[coord].min()
            
            # Inter-landmark features
            if all(landmark in landmarks for landmark in [11, 12]):  # Shoulders
                left_shoulder = df[df['landmark_id'] == 11]
                right_shoulder = df[df['landmark_id'] == 12]
                
                if not left_shoulder.empty and not right_shoulder.empty:
                    if 'x_norm' in left_shoulder.columns and 'x_norm' in right_shoulder.columns:
                        shoulder_width = np.abs(left_shoulder['x_norm'].mean() - right_shoulder['x_norm'].mean())
                        features['shoulder_width'] = shoulder_width
            
            if all(landmark in landmarks for landmark in [23, 24]):  # Hips
                left_hip = df[df['landmark_id'] == 23]
                right_hip = df[df['landmark_id'] == 24]
                
                if not left_hip.empty and not right_hip.empty:
                    if 'x_norm' in left_hip.columns and 'x_norm' in right_hip.columns:
                        hip_width = np.abs(left_hip['x_norm'].mean() - right_hip['x_norm'].mean())
                        features['hip_width'] = hip_width
            
            # Movement features
            if len(df) > 1:
                for coord in ['x_norm', 'y_norm', 'z_norm']:
                    if coord in df.columns:
                        # Calculate velocity (change between consecutive frames)
                        df_sorted = df.sort_values('frame')
                        coord_diff = df_sorted[coord].diff().dropna()
                        features[f"{coord}_velocity_mean"] = coord_diff.mean()
                        features[f"{coord}_velocity_std"] = coord_diff.std()
                        features[f"{coord}_velocity_max"] = coord_diff.max()
            
            return features, "Features extracted successfully"
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None, f"Feature extraction failed: {str(e)}"
    
    @staticmethod
    def extract_stgcn_features(df):
        """Extract features for ST-GCN models (tensor format)."""
        try:
            # First extract basic features
            basic_features, status = FeatureEngineer.extract_basic_features(df)
            if not basic_features:
                return None, status
            
            # For ST-GCN, we need to create the tensor format
            # This is a simplified version - in production, you'd use the full pipeline
            
            # Create pose tensor (T, J, 3)
            if 'frame' in df.columns and 'landmark_id' in df.columns:
                # Sort by frame
                df_sorted = df.sort_values('frame')
                
                # Get unique frames and landmarks
                frames = np.sort(df_sorted['frame'].unique())
                landmarks = np.sort(df_sorted['landmark_id'].unique())
                
                # Filter to GAIT_JOINTS
                gait_landmarks = [l for l in landmarks if l in GAIT_JOINTS]
                
                # Create tensor
                T = len(frames)
                J = len(gait_landmarks)
                pose_tensor = np.zeros((T, J, 3), dtype=np.float32)
                
                # Fill tensor
                frame_to_idx = {f: i for i, f in enumerate(frames)}
                landmark_to_idx = {l: i for i, l in enumerate(gait_landmarks)}
                
                for _, row in df_sorted.iterrows():
                    f_idx = frame_to_idx[row['frame']]
                    l_idx = landmark_to_idx[row['landmark_id']]
                    
                    pose_tensor[f_idx, l_idx, 0] = row['x_norm']
                    pose_tensor[f_idx, l_idx, 1] = row['y_norm']
                    pose_tensor[f_idx, l_idx, 2] = row['z_norm']
                
                # Normalize pose tensor (simplified)
                # In production, you'd use the full normalization pipeline
                pelvis = (pose_tensor[:, 23, :] + pose_tensor[:, 24, :]) / 2
                pose_centered = pose_tensor - pelvis[:, None, :]
                
                torso = (pose_centered[:, 11, :] + pose_centered[:, 12, :]) / 2
                scale = np.linalg.norm(torso, axis=1).mean()
                pose_normalized = pose_centered / scale
                
                # Transpose to ST-GCN format: (N, C, T, V)
                # For single window, N=1
                stgcn_tensor = np.transpose(pose_normalized, (2, 0, 1))  # (C, T, V)
                stgcn_tensor = stgcn_tensor[np.newaxis, ...]  # (1, C, T, V)
                
                return {
                    'stgcn_tensor': stgcn_tensor,
                    'basic_features': basic_features,
                    'tensor_shape': stgcn_tensor.shape
                }, "ST-GCN features extracted successfully"
            
            return None, "Required columns not available"
            
        except Exception as e:
            logger.error(f"ST-GCN feature extraction failed: {e}")
            return None, f"ST-GCN feature extraction failed: {str(e)}"
    
    @staticmethod
    def prepare_features_for_baseline(features):
        """Prepare features for baseline XGBoost model."""
        try:
            if not features:
                return None, "No features to prepare"
            
            # Create feature vector with all the features from training
            feature_vector = []
            
            # Basic coordinate features
            for coord in ['x_norm', 'y_norm', 'z_norm']:
                for stat in ['mean', 'std', 'min', 'max', 'range', 'median']:
                    feature_vector.append(features.get(f"{coord}_{stat}", 0))
            
            # Frame features
            feature_vector.append(features.get('total_frames', 0))
            feature_vector.append(features.get('frame_range', 0))
            feature_vector.append(features.get('frames_per_landmark', 0))
            
            # Landmark features
            key_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 
                              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
            
            for landmark in key_landmarks:
                for coord in ['x_norm', 'y_norm', 'z_norm']:
                    for stat in ['mean', 'std', 'range']:
                        feature_vector.append(features.get(f"{landmark}_{coord}_{stat}", 0))
            
            # Inter-landmark features
            feature_vector.append(features.get('shoulder_width', 0))
            feature_vector.append(features.get('hip_width', 0))
            
            # Movement features
            for coord in ['x_norm', 'y_norm', 'z_norm']:
                for stat in ['velocity_mean', 'velocity_std', 'velocity_max']:
                    feature_vector.append(features.get(f"{coord}_{stat}", 0))
            
            # Convert to numpy array
            feature_array = np.array(feature_vector, dtype=np.float32)
            
            return feature_array.reshape(1, -1), "Features prepared for baseline model"
            
        except Exception as e:
            logger.error(f"Baseline feature preparation failed: {e}")
            return None, f"Baseline feature preparation failed: {str(e)}"
    
    @staticmethod
    def prepare_features_for_stgcn(features_dict):
        """Prepare features for ST-GCN models."""
        try:
            if not features_dict or 'stgcn_tensor' not in features_dict:
                return None, "No ST-GCN features available"
            
            # Get the tensor directly
            stgcn_tensor = features_dict['stgcn_tensor']
            
            # Ensure it's in the right format: (1, C, T, V)
            if stgcn_tensor.ndim == 4:
                return stgcn_tensor, "Features prepared for ST-GCN model"
            elif stgcn_tensor.ndim == 3:
                # Add batch dimension
                return stgcn_tensor[np.newaxis, ...], "Features prepared for ST-GCN model"
            else:
                return None, f"Invalid tensor shape: {stgcn_tensor.shape}"
            
        except Exception as e:
            logger.error(f"ST-GCN feature preparation failed: {e}")
            return None, f"ST-GCN feature preparation failed: {str(e)}"

# Prediction Engine
class PredictionEngine:
    """Handles predictions from all model types."""
    
    # ANOMALY_COLS for multi-label model
    ANOMALY_COLS = [
        "gait_anomaly_knee_sagittal_plane_abnormality",
        "gait_anomaly_trunk_balance_abnormality",
        "gait_anomaly_spatiotemporal_asymmetry",
        "gait_anomaly_hip_pelvic_control_deficit",
        "gait_anomaly_distal_foot_control_deficit",
    ]
    
    @staticmethod
    def predict_with_baseline(model, features_array):
        """Make prediction with baseline XGBoost model."""
        try:
            if model is None or features_array is None:
                return None, "No model or features available"
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            probabilities = model.predict_proba(features_array)[0]
            
            result = {
                'prediction': int(prediction),
                'label': 'Normal' if prediction == 0 else 'Abnormal',
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'Normal': float(probabilities[0]),
                    'Abnormal': float(probabilities[1])
                },
                'timestamp': datetime.now().isoformat(),
                'model_type': 'baseline'
            }
            
            return result, "Baseline prediction successful"
            
        except Exception as e:
            logger.error(f"Baseline prediction failed: {e}")
            return None, f"Baseline prediction failed: {str(e)}"
    
    @staticmethod
    def predict_with_binary(model, stgcn_tensor):
        """Make prediction with binary ST-GCN model."""
        try:
            if model is None or stgcn_tensor is None:
                return None, "No model or features available"
            
            if not DEPENDENCIES['torch']:
                return None, "PyTorch not available"
            
            # Ensure tensor is on the right device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            tensor = torch.tensor(stgcn_tensor, dtype=torch.float32).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(tensor)
                if output.shape[1] == 1:
                    output = output.squeeze(1)
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(output).cpu().numpy()
                prediction = (probabilities > 0.5).astype(int)
            
            result = {
                'prediction': int(prediction[0]),
                'label': 'Normal' if prediction[0] == 0 else 'Abnormal',
                'confidence': float(max(probabilities[0])),
                'probabilities': {
                    'Normal': float(1 - probabilities[0]) if prediction[0] == 1 else float(probabilities[0]),
                    'Abnormal': float(probabilities[0]) if prediction[0] == 1 else float(1 - probabilities[0])
                },
                'timestamp': datetime.now().isoformat(),
                'model_type': 'binary_stgcn'
            }
            
            return result, "Binary ST-GCN prediction successful"
            
        except Exception as e:
            logger.error(f"Binary ST-GCN prediction failed: {e}")
            return None, f"Binary ST-GCN prediction failed: {str(e)}"
    
    @staticmethod
    def predict_with_multi(model, stgcn_tensor):
        """Make prediction with multi-label ST-GCN model."""
        try:
            if model is None or stgcn_tensor is None:
                return None, "No model or features available"
            
            if not DEPENDENCIES['torch']:
                return None, "PyTorch not available"
            
            # Ensure tensor is on the right device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            tensor = torch.tensor(stgcn_tensor, dtype=torch.float32).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(tensor)
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(output).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
            
            # Create result with all anomaly types
            result = {
                'prediction': predictions[0].tolist(),
                'label': 'Multi-label prediction',
                'confidence': float(np.mean(probabilities[0])),
                'probabilities': {
                    col: float(prob[0]) for col, prob in zip(PredictionEngine.ANOMALY_COLS, probabilities[0])
                },
                'anomaly_detected': any(predictions[0]),
                'detected_anomalies': [
                    col for col, pred in zip(PredictionEngine.ANOMALY_COLS, predictions[0]) if pred
                ],
                'timestamp': datetime.now().isoformat(),
                'model_type': 'multi_label_stgcn'
            }
            
            return result, "Multi-label ST-GCN prediction successful"
            
        except Exception as e:
            logger.error(f"Multi-label ST-GCN prediction failed: {e}")
            return None, f"Multi-label ST-GCN prediction failed: {str(e)}"
    
    @staticmethod
    def create_fallback_prediction():
        """Create a fallback prediction when all models fail."""
        import random
        
        prediction = random.choice([0, 1])
        confidence = random.uniform(0.6, 0.9)
        
        return {
            'prediction': prediction,
            'label': 'Normal' if prediction == 0 else 'Abnormal',
            'confidence': confidence,
            'probabilities': {
                'Normal': 1 - confidence if prediction == 1 else confidence,
                'Abnormal': confidence if prediction == 1 else 1 - confidence
            },
            'timestamp': datetime.now().isoformat(),
            'model_type': 'fallback'
        }

# Visualization
class Visualizer:
    """Handles visualization creation for all model types."""
    
    @staticmethod
    def create_features_chart(features):
        """Create a chart visualizing extracted features."""
        try:
            if not DEPENDENCIES['matplotlib']:
                return None, "Matplotlib not available"
            
            if not features:
                return None, "No features to visualize"
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Select top features to display
            feature_names = list(features.keys())[:15]
            feature_values = [features[name] for name in feature_names]
            
            # Create bar chart
            colors = ['#1f77b4' if v > 0 else '#ff7f0e' for v in feature_values]
            bars = ax.bar(range(len(feature_names)), feature_values, color=colors, alpha=0.7)
            
            # Customize plot
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels([name[:20] for name in feature_names], rotation=45, ha='right')
            ax.set_ylabel('Feature Value')
            ax.set_title('Extracted Feature Values')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            return fig, "Feature chart created successfully"
            
        except Exception as e:
            logger.error(f"Feature chart creation failed: {e}")
            return None, f"Feature chart creation failed: {str(e)}"
    
    @staticmethod
    def create_prediction_chart(prediction_result):
        """Create prediction probability chart."""
        try:
            if not DEPENDENCIES['matplotlib']:
                return None, "Matplotlib not available"
            
            if not prediction_result:
                return None, "No prediction result"
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if prediction_result['model_type'] == 'multi_label_stgcn':
                # Multi-label visualization
                probs = prediction_result['probabilities']
                labels = list(probs.keys())
                values = list(probs.values())
                colors = ['red' if 'abnormal' in label.lower() else 'green' for label in labels]
                
                bars = ax.bar(labels, values, color=colors, alpha=0.7)
                ax.set_title('Multi-label Anomaly Probabilities')
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                # Add prediction info
                detected = prediction_result.get('anomaly_detected', False)
                anomalies = prediction_result.get('detected_anomalies', [])
                
                ax.text(0.5, 0.95, f'Anomaly Detected: {detected}', 
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='red' if detected else 'green', alpha=0.8))
                
                if anomalies:
                    ax.text(0.5, 0.85, f'Anomalies: {", ".join(anomalies)}', 
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                # Binary visualization
                probs = prediction_result['probabilities']
                labels = list(probs.keys())
                values = list(probs.values())
                colors = ['green' if label == 'Normal' else 'red' for label in labels]
                
                bars = ax.bar(labels, values, color=colors, alpha=0.7)
                ax.set_title('Prediction Probabilities')
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Add prediction label
            pred_label = prediction_result['label']
            confidence = prediction_result['confidence']
            model_type = prediction_result['model_type']
            
            ax.text(0.5, 0.95, f'Prediction: {pred_label} ({confidence:.1%} confidence)', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.text(0.5, 0.88, f'Model: {model_type}', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            return fig, "Chart created successfully"
            
        except Exception as e:
            logger.error(f"Chart creation failed: {e}")
            return None, f"Chart creation failed: {str(e)}"
    
    @staticmethod
    def create_model_comparison_chart(baseline_result, binary_result, multi_result):
        """Create comparison chart for all models."""
        try:
            if not DEPENDENCIES['matplotlib']:
                return None, "Matplotlib not available"
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Baseline model
            if baseline_result:
                probs = baseline_result['probabilities']
                labels = list(probs.keys())
                values = list(probs.values())
                
                axes[0].bar(labels, values, color=['green', 'red'], alpha=0.7)
                axes[0].set_title('Baseline XGBoost Model')
                axes[0].set_ylabel('Probability')
                axes[0].set_ylim(0, 1)
                axes[0].text(0.5, 0.95, f"Pred: {baseline_result['label']}", 
                         transform=axes[0].transAxes, ha='center', va='top')
            else:
                axes[0].text(0.5, 0.5, "Baseline Model\nNot Available", 
                         transform=axes[0].transAxes, ha='center', va='center')
                axes[0].set_title('Baseline XGBoost Model')
            
            # Binary ST-GCN model
            if binary_result:
                probs = binary_result['probabilities']
                labels = list(probs.keys())
                values = list(probs.values())
                
                axes[1].bar(labels, values, color=['green', 'red'], alpha=0.7)
                axes[1].set_title('Binary ST-GCN Model')
                axes[1].set_ylabel('Probability')
                axes[1].set_ylim(0, 1)
                axes[1].text(0.5, 0.95, f"Pred: {binary_result['label']}", 
                         transform=axes[1].transAxes, ha='center', va='top')
            else:
                axes[1].text(0.5, 0.5, "Binary ST-GCN\nNot Available", 
                         transform=axes[1].transAxes, ha='center', va='center')
                axes[1].set_title('Binary ST-GCN Model')
            
            # Multi-label ST-GCN model
            if multi_result:
                probs = multi_result['probabilities']
                labels = list(probs.keys())
                values = list(probs.values())
                colors = ['red' if 'abnormal' in label.lower() else 'green' for label in labels]
                
                axes[2].bar(labels, values, color=colors, alpha=0.7)
                axes[2].set_title('Multi-label ST-GCN Model')
                axes[2].set_ylabel('Probability')
                axes[2].set_ylim(0, 1)
                axes[2].tick_params(axis='x', rotation=45)
                
                detected = multi_result.get('anomaly_detected', False)
                anomalies = multi_result.get('detected_anomalies', [])
                
                axes[2].text(0.5, 0.95, f"Anomaly: {detected}", 
                         transform=axes[2].transAxes, ha='center', va='top',
                         bbox=dict(boxstyle='round', facecolor='red' if detected else 'green', alpha=0.8))
                
                if anomalies:
                    axes[2].text(0.5, 0.85, f"{', '.join(anomalies[:3])}", 
                             transform=axes[2].transAxes, ha='center', va='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                axes[2].text(0.5, 0.5, "Multi-label ST-GCN\nNot Available", 
                         transform=axes[2].transAxes, ha='center', va='center')
                axes[2].set_title('Multi-label ST-GCN Model')
            
            plt.tight_layout()
            return fig, "Comparison chart created successfully"
            
        except Exception as e:
            logger.error(f"Comparison chart creation failed: {e}")
            return None, f"Comparison chart creation failed: {str(e)}"

# File Management
class FileManager:
    """Handles file operations."""
    
    @staticmethod
    def save_uploaded_file(uploaded_file):
        """Save uploaded file with proper naming."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_data_{timestamp}_{uploaded_file.name}"
            file_path = UPLOAD_DIR / filename
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            file_info = {
                'original_name': uploaded_file.name,
                'saved_name': filename,
                'path': str(file_path),
                'size': uploaded_file.size,
                'upload_time': datetime.now().isoformat()
            }
            
            return file_path, file_info
            
        except Exception as e:
            logger.error(f"File save failed: {e}")
            return None, f"File save failed: {str(e)}"
    
    @staticmethod
    def save_results(prediction_result, features, file_info, model_type):
        """Save results to CSV."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"gait_results_{model_type}_{timestamp}.csv"
            results_path = RESULTS_DIR / results_filename
            
            # Create results DataFrame
            results_data = {
                'timestamp': prediction_result['timestamp'],
                'model_type': prediction_result['model_type'],
                'prediction': prediction_result.get('prediction', 'N/A'),
                'label': prediction_result.get('label', 'N/A'),
                'confidence': prediction_result.get('confidence', 0),
                'normal_probability': prediction_result.get('probabilities', {}).get('Normal', 0),
                'abnormal_probability': prediction_result.get('probabilities', {}).get('Abnormal', 0),
                'anomaly_detected': prediction_result.get('anomaly_detected', False)
            }
            
            # Add file info
            if file_info:
                results_data['original_filename'] = file_info['original_name']
                results_data['file_size'] = file_info['size']
            
            # Add features
            if features:
                if isinstance(features, dict):
                    for key, value in features.items():
                        results_data[f"feature_{key}"] = value
                elif 'basic_features' in features:
                    for key, value in features['basic_features'].items():
                        results_data[f"feature_{key}"] = value
            
            # Add multi-label results if available
            if prediction_result.get('detected_anomalies'):
                results_data['detected_anomalies'] = ', '.join(prediction_result['detected_anomalies'])
            
            # Create DataFrame and save
            results_df = pd.DataFrame([results_data])
            results_df.to_csv(results_path, index=False)
            
            return results_path, results_df
            
        except Exception as e:
            logger.error(f"Results save failed: {e}")
            return None, f"Results save failed: {str(e)}"

# Main Application
def main():
    """Main application function."""
    try:
        # Initialize session state
        init_session_state()
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        .model-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🚶 GAITy - Advanced Gait Analysis</h1>', unsafe_allow_html=True)
        st.markdown("### Complete Pipeline with Baseline & Advanced Models")
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 Model Management")
            
            # Model status cards
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.write("**Baseline Model (XGBoost)**")
            if st.session_state.baseline_loaded:
                st.markdown('<p class="status-success">✅ Loaded</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-error">❌ Not loaded</p>', unsafe_allow_html=True)
                if st.button("Load Baseline", key="load_baseline"):
                    with st.spinner("Loading baseline model..."):
                        model, status = ModelManager.load_baseline_model()
                        st.session_state.baseline_model = model
                        st.session_state.baseline_loaded = model is not None
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.write("**Binary Model (ST-GCN)**")
            if st.session_state.binary_loaded:
                st.markdown('<p class="status-success">✅ Loaded</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-error">❌ Not loaded</p>', unsafe_allow_html=True)
                if st.button("Load Binary", key="load_binary"):
                    with st.spinner("Loading binary model..."):
                        model, status = ModelManager.load_binary_model()
                        st.session_state.binary_model = model
                        st.session_state.binary_loaded = model is not None
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.write("**Multi-label Model (ST-GCN)**")
            if st.session_state.multi_loaded:
                st.markdown('<p class="status-success">✅ Loaded</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-error">❌ Not loaded</p>', unsafe_allow_html=True)
                if st.button("Load Multi-label", key="load_multi"):
                    with st.spinner("Loading multi-label model..."):
                        model, status = ModelManager.load_multi_model()
                        st.session_state.multi_model = model
                        st.session_state.multi_loaded = model is not None
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Dependency status
            with st.expander("System Dependencies", expanded=False):
                st.write("**Dependencies:**")
                for dep, status in DEPENDENCIES.items():
                    st.write(f"- {dep.title()}: {'✅' if status else '❌'}")
                
                if DEPENDENCIES['torch']:
                    st.write(f"PyTorch version: {torch.__version__}")
                    st.write(f"CUDA available: {torch.cuda.is_available()}")
                
                if DEPENDENCIES['xgboost']:
                    st.write(f"XGBoost version: {xgb.__version__}")
            
            # Processing history
            if st.session_state.processing_history:
                st.write("**Recent Activity:**")
                for item in st.session_state.processing_history[-5:]:
                    st.write(f"- {item}")
        
        # Main content area
        st.header("📁 Step 1: Upload Pose Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with MediaPipe pose landmarks",
            type=['csv'],
            help="CSV should contain: frame, landmark_id, x_norm, y_norm, z_norm",
            key="pose_upload"
        )
        
        if uploaded_file:
            # Save file
            file_path, file_info = FileManager.save_uploaded_file(uploaded_file)
            
            if file_path:
                st.success(f"✅ File uploaded: {file_info['original_name']} ({file_info['size']/1024:.1f} KB)")
                st.session_state.file_info = file_info
                st.session_state.processing_history.append(f"Uploaded {file_info['original_name']}")
                
                # Read and validate CSV
                try:
                    df = pd.read_csv(file_path)
                    
                    # Display file info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Unique Landmarks", df['landmark_id'].nunique() if 'landmark_id' in df.columns else 'N/A')
                    
                    # Validate CSV
                    validation_errors = FeatureEngineer.validate_csv(df)
                    
                    if validation_errors:
                        st.error("❌ Validation Errors:")
                        for error in validation_errors:
                            st.write(f"- {error}")
                    else:
                        st.success("✅ CSV validation passed")
                        
                        # Data preview
                        with st.expander("📊 Data Preview", expanded=True):
                            st.dataframe(df.head(10))
                        
                        # Step 2: Feature Engineering
                        st.header("🔬 Step 2: Feature Engineering")
                        
                        # Feature extraction method selection
                        feature_method = st.radio(
                            "Select Feature Extraction Method:",
                            ["Basic (for Baseline)", "ST-GCN (for Advanced Models)"],
                            key="feature_method"
                        )
                        
                        if st.button("⚡ Extract Features", key="extract_features"):
                            with st.spinner("Extracting features..."):
                                if feature_method == "Basic (for Baseline)":
                                    features, status = FeatureEngineer.extract_basic_features(df)
                                    st.session_state.features = features
                                    st.session_state.features_type = "basic"
                                else:
                                    features_dict, status = FeatureEngineer.extract_stgcn_features(df)
                                    st.session_state.features = features_dict
                                    st.session_state.features_type = "stgcn"
                                
                                if features:
                                    st.success(f"✅ {status}")
                                    st.session_state.processing_history.append("Features extracted")
                                    
                                    # Display features
                                    if st.session_state.features_type == "basic":
                                        st.write("**Basic Features:**")
                                        basic_features = {k: v for k, v in features.items() if 'mean' in k or 'std' in k}
                                        for key, value in list(basic_features.items())[:10]:
                                            st.write(f"• {key}: {value:.4f}")
                                    else:
                                        st.write("**ST-GCN Features:**")
                                        if 'tensor_shape' in features:
                                            st.write(f"Tensor Shape: {features['tensor_shape']}")
                                        
                                        st.write("**Basic Features:**")
                                        if 'basic_features' in features:
                                            basic_features = features['basic_features']
                                            for key, value in list(basic_features.items())[:10]:
                                                st.write(f"• {key}: {value:.4f}")
                                    
                                    # Features visualization
                                    fig, chart_status = Visualizer.create_features_chart(
                                        features if st.session_state.features_type == "basic" else features.get('basic_features', {})
                                    )
                                    if fig:
                                        st.pyplot(fig)
                                        plt.close()
                                    else:
                                        st.info(chart_status)
                                    
                                    # Step 3: Model Prediction
                                    st.header("🤖 Step 3: Model Prediction")
                                    
                                    # Model selection
                                    available_models = []
                                    if st.session_state.baseline_loaded:
                                        available_models.append("Baseline XGBoost")
                                    if st.session_state.binary_loaded:
                                        available_models.append("Binary ST-GCN")
                                    if st.session_state.multi_loaded:
                                        available_models.append("Multi-label ST-GCN")
                                    
                                    if not available_models:
                                        st.warning("⚠️ No models loaded. Please load models in the sidebar.")
                                        st.info("Use fallback prediction for demonstration.")
                                    
                                    model_choice = st.selectbox(
                                        "Select Model for Prediction:",
                                        available_models if available_models else ["Fallback Prediction"],
                                        key="model_choice"
                                    )
                                    
                                    if st.button("🎯 Make Prediction", key="make_prediction"):
                                        with st.spinner("Making prediction..."):
                                            prediction_result = None
                                            
                                            if model_choice == "Baseline XGBoost":
                                                if st.session_state.features_type == "basic":
                                                    features_array, prep_status = FeatureEngineer.prepare_features_for_baseline(
                                                        st.session_state.features
                                                    )
                                                    
                                                    if features_array is not None:
                                                        prediction_result, pred_status = PredictionEngine.predict_with_baseline(
                                                            st.session_state.baseline_model, features_array
                                                        )
                                                        st.session_state.prediction = prediction_result
                                                        st.success(f"✅ {pred_status}")
                                                        st.session_state.processing_history.append("Baseline prediction completed")
                                                    else:
                                                        st.error(f"❌ {prep_status}")
                                                else:
                                                    st.error("❌ Basic features required for baseline model")
                                            
                                            elif model_choice == "Binary ST-GCN":
                                                if st.session_state.features_type == "stgcn":
                                                    stgcn_tensor, prep_status = FeatureEngineer.prepare_features_for_stgcn(
                                                        st.session_state.features
                                                    )
                                                    
                                                    if stgcn_tensor is not None:
                                                        prediction_result, pred_status = PredictionEngine.predict_with_binary(
                                                            st.session_state.binary_model, stgcn_tensor
                                                        )
                                                        st.session_state.prediction = prediction_result
                                                        st.success(f"✅ {pred_status}")
                                                        st.session_state.processing_history.append("Binary ST-GCN prediction completed")
                                                    else:
                                                        st.error(f"❌ {prep_status}")
                                                else:
                                                    st.error("❌ ST-GCN features required for binary model")
                                            
                                            elif model_choice == "Multi-label ST-GCN":
                                                if st.session_state.features_type == "stgcn":
                                                    stgcn_tensor, prep_status = FeatureEngineer.prepare_features_for_stgcn(
                                                        st.session_state.features
                                                    )
                                                    
                                                    if stgcn_tensor is not None:
                                                        prediction_result, pred_status = PredictionEngine.predict_with_multi(
                                                            st.session_state.multi_model, stgcn_tensor
                                                        )
                                                        st.session_state.prediction = prediction_result
                                                        st.success(f"✅ {pred_status}")
                                                        st.session_state.processing_history.append("Multi-label ST-GCN prediction completed")
                                                    else:
                                                        st.error(f"❌ {prep_status}")
                                                else:
                                                    st.error("❌ ST-GCN features required for multi-label model")
                                            else:
                                                # Use fallback prediction
                                                prediction_result = PredictionEngine.create_fallback_prediction()
                                                st.warning("⚠️ Using fallback prediction")
                                                st.session_state.prediction = prediction_result
                                                st.session_state.processing_history.append("Fallback prediction completed")
                                                st.session_state.prediction = prediction_result
                                                st.session_state.processing_history.append("Fallback prediction completed")
                                            
                                            # Display results
                                            st.header("📊 Prediction Results")
                                            
                                            # Main metrics
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Prediction", prediction_result['label'])
                                            with col2:
                                                st.metric("Confidence", f"{prediction_result['confidence']:.2%}")
                                            with col3:
                                                model_type = prediction_result['model_type']
                                                if model_type == "fallback":
                                                    st.metric("Model", "🔄 Fallback")
                                                else:
                                                    st.metric("Model", model_type.replace('_', ' ').title())
                                            
                                            # Probability chart
                                            fig, chart_status = Visualizer.create_prediction_chart(prediction_result)
                                            if fig:
                                                st.pyplot(fig)
                                                plt.close()
                                            else:
                                                st.info(chart_status)
                                            
                                            # Detailed results
                                            with st.expander("📋 Detailed Results"):
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.write("**Prediction Info:**")
                                                    st.json(prediction_result)
                                                with col2:
                                                    st.write("**Extracted Features:**")
                                                    if st.session_state.features_type == "basic":
                                                        st.json(st.session_state.features)
                                                    elif 'basic_features' in st.session_state.features:
                                                        st.json(st.session_state.features['basic_features'])
                                            
                                            # Step 4: Download Results
                                            st.header("💾 Step 4: Download Results")
                                            
                                            # Save results
                                            results_path, results_df = FileManager.save_results(
                                                prediction_result, 
                                                st.session_state.features if st.session_state.features_type == "basic" else st.session_state.features.get('basic_features', {}),
                                                file_info,
                                                prediction_result['model_type']
                                            )
                                            
                                            if results_path:
                                                # Create download button
                                                csv = results_df.to_csv(index=False)
                                                st.download_button(
                                                    label="📥 Download Complete Results (CSV)",
                                                    data=csv,
                                                    file_name=results_path.name,
                                                    mime="text/csv"
                                                )
                                            
                                            # Model comparison (if multiple models are loaded)
                                            loaded_models = []
                                            if st.session_state.baseline_loaded:
                                                loaded_models.append("baseline")
                                            if st.session_state.binary_loaded:
                                                loaded_models.append("binary")
                                            if st.session_state.multi_loaded:
                                                loaded_models.append("multi")
                                            
                                            if len(loaded_models) > 1:
                                                st.header("📈 Model Comparison")
                                                
                                                # Get predictions from all loaded models
                                                all_results = {}
                                                
                                                if "baseline" in loaded_models and st.session_state.features_type == "basic":
                                                    features_array, _ = FeatureEngineer.prepare_features_for_baseline(
                                                        st.session_state.features
                                                    )
                                                    if features_array is not None:
                                                        all_results["baseline"] = PredictionEngine.predict_with_baseline(
                                                            st.session_state.baseline_model, features_array
                                                        )
                                                
                                                if "binary" in loaded_models and st.session_state.features_type == "stgcn":
                                                    stgcn_tensor, _ = FeatureEngineer.prepare_features_for_stgcn(
                                                        st.session_state.features
                                                    )
                                                    if stgcn_tensor is not None:
                                                        all_results["binary"] = PredictionEngine.predict_with_binary(
                                                            st.session_state.binary_model, stgcn_tensor
                                                        )
                                                
                                                if "multi" in loaded_models and st.session_state.features_type == "stgcn":
                                                    stgcn_tensor, _ = FeatureEngineer.prepare_features_for_stgcn(
                                                        st.session_state.features
                                                    )
                                                    if stgcn_tensor is not None:
                                                        all_results["multi"] = PredictionEngine.predict_with_multi(
                                                            st.session_state.multi_model, stgcn_tensor
                                                        )
                                                
                                                if all_results:
                                                    fig, chart_status = Visualizer.create_model_comparison_chart(
                                                        all_results.get("baseline"),
                                                        all_results.get("binary"),
                                                        all_results.get("multi")
                                                    )
                                                    if fig:
                                                        st.pyplot(fig)
                                                        plt.close()
                                                    else:
                                                        st.info(chart_status)
                                            
                                            # Processing summary
                                            st.header("📈 Processing Summary")
                                            
                                            summary_data = {
                                                'File Name': file_info['original_name'],
                                                'File Size (KB)': f"{file_info['size']/1024:.1f}",
                                                'Feature Method': st.session_state.features_type.title(),
                                                'Features Extracted': len(st.session_state.features) if isinstance(st.session_state.features, dict) else len(st.session_state.features.get('basic_features', {})),
                                                'Prediction': prediction_result['label'],
                                                'Confidence': f"{prediction_result['confidence']:.2%}",
                                                'Model Type': prediction_result['model_type'],
                                                'Models Available': ', '.join(loaded_models)
                                            }
                                            
                                            st.table(pd.DataFrame(list(summary_data.items()), 
                                                             columns=['Metric', 'Value']))
            
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    logger.error(f"File processing error: {e}")
                    logger.error(traceback.format_exc())
        
        # Instructions
        st.write("---")
        st.header("📖 Instructions & Information")
        
        with st.expander("📋 Model Information", expanded=True):
            st.markdown("""
            ### Model Types Available:
            
            **Baseline XGBoost Model**
            - Traditional machine learning approach
            - Uses handcrafted features
            - Fast and lightweight
            - Good for initial analysis
            
            **Binary ST-GCN Model**
            - Deep learning approach
            - Uses spatial-temporal graph convolution
            - Learns spatial relationships between joints
            - Better for complex gait patterns
            
            **Multi-label ST-GCN Model**
            - Detects specific anomaly types
            - Identifies which gait abnormalities are present
            - Provides detailed diagnostic information
            
            ### Feature Engineering Methods:
            
            **Basic Features**
            - Statistical measures (mean, std, min, max)
            - Movement patterns
            - Inter-landmark relationships
            - Compatible with baseline model
            
            **ST-GCN Features**
            - Tensor representation of pose data
            - Preserves spatial and temporal information
            - Required for advanced models
            """)
        
        with st.expander("🔧 Troubleshooting", expanded=False):
            st.markdown("""
            ### Common Issues and Solutions:
            
            **Model Loading Errors**
            - Ensure model files exist in the correct directories
            - Check if PyTorch is installed for advanced models
            - Verify model compatibility with your Python version
            
            **Feature Extraction Errors**
            - Ensure CSV has required columns: frame, landmark_id, x_norm, y_norm, z_norm
            - Check for valid landmark IDs (0-32)
            - Verify data quality and completeness
            
            **Prediction Errors**
            - Use basic features with baseline model
            - Use ST-GCN features with advanced models
            - Check feature dimensions match model expectations
            """)
        
        with st.expander("📚 System Information", expanded=False):
            st.write(f"Python version: {sys.version}")
            st.write(f"Working directory: {os.getcwd()}")
            st.write(f"Model paths:")
            st.write(f"- Baseline: {BASELINE_MODEL_PATH} (exists: {BASELINE_MODEL_PATH.exists()})")
            st.write(f"- Binary: {BINARY_MODEL_PATH} (exists: {BINARY_MODEL_PATH.exists()})")
            st.write(f"- Multi-label: {MULTI_MODEL_PATH} (exists: {MULTI_MODEL_PATH.exists()})")
            
            if DEPENDENCIES['torch']:
                st.write(f"PyTorch version: {torch.__version__}")
                st.write(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    st.write(f"CUDA device: {torch.cuda.get_device_name()}")
            
            if DEPENDENCIES['xgboost']:
                st.write(f"XGBoost version: {xgb.__version__}")
    
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        
        with st.expander("🐛 Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()