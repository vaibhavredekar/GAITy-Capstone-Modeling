#!/usr/bin/env python3
"""
GAITy - Production Grade Gait Analysis Application
Complete pipeline with baseline (XGBoost) and advanced (ST-GCN) models
"""

import os
import sys
import logging
import traceback
import warnings
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
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

# Import dependencies
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

# Decorators for better error handling and logging
def log_execution(func):
    """Decorator to log function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Successfully executed {func.__name__} in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {elapsed:.2f}s: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

def handle_errors(default_return=None):
    """Decorator to handle errors gracefully."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                if default_return is not None:
                    return default_return
                raise
        return wrapper
    return decorator

# Constants
MODEL_PATHS = {
    'baseline': Path("models/baseline/xgboost_model.bin"),
    'baseline_ubj': Path("models/baseline/xgboost_model.ubj"),
    'binary': Path("models/advance/binary_model_full.bin"),
    'multi': Path("models/advance/multi_label_model_full.bin")
}

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
CACHE_DIR = Path("cache")
MODEL_CACHE_DIR = CACHE_DIR / "models"

# Create directories
for directory in [UPLOAD_DIR, RESULTS_DIR, CACHE_DIR, MODEL_CACHE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# MediaPipe constants - MATCHING TRAINING
N_JOINTS = 33
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# CRITICAL: GAIT_JOINTS must match training pipeline
GAIT_JOINTS = [2, 5, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Feature configuration matching training
@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature extraction matching training."""
    smooth_sigma: float = 1.0
    speed_thresh: float = 0.02
    min_step_time: float = 0.3
    auto_normalize_if_needed: bool = True

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
        'features_type': None,
        'prediction': None,
        'file_info': None,
        'processing_history': [],
        'model_info': {},
        'available_models': [],
        'feature_columns': None,  # Store expected feature columns
        'imputation_values': None  # Store median imputation values
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Model Management - FIXED BASELINE MODEL LOADING
class ModelManager:
    """Manages loading and using all models."""
    
    @staticmethod
    @log_execution
    def load_baseline_model():
        """Load the baseline XGBoost model with proper feature handling."""
        try:
            if not DEPENDENCIES['xgboost']:
                return None, "XGBoost not available"
            
            # Try to find model file
            model_path = None
            for path in [MODEL_PATHS['baseline'], MODEL_PATHS['baseline_ubj']]:
                if path.exists():
                    model_path = path
                    break
            
            if not model_path:
                return None, f"Baseline model not found in {MODEL_PATHS['baseline']} or {MODEL_PATHS['baseline_ubj']}"
            
            # Check cache
            cache_key = f"baseline_{model_path.stem}"
            cache_file = MODEL_CACHE_DIR / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)
                    # Verify checksum
                    with open(model_path, 'rb') as f:
                        current_hash = hashlib.md5(f.read()).hexdigest()
                    if cached.get('hash') == current_hash:
                        logger.info(f"Loading baseline model from cache: {cache_file}")
                        return cached['model'], "Baseline model loaded from cache"
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}")
            
            # Load model with EXACT training parameters
            # IMPORTANT: These must match the training parameters exactly
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=8.0,  # From training: 14259/1789 ≈ 8.0
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                verbosity=0
            )
            
            if model_path.suffix == '.bin':
                model.load_model(str(model_path))
            elif model_path.suffix == '.ubj':
                # Try to load UBJSON format
                booster = xgb.Booster()
                booster.load_model(str(model_path))
                model._Booster = booster
            else:
                return None, f"Unknown model format: {model_path.suffix}"
            
            # Store expected feature columns from training
            st.session_state.feature_columns = [
                "step_height_L", "step_height_R", "step_length_L", "step_length_R",
                "pelvis_drop_mean", "pelvis_drop_std", "trunk_lean_mean", "trunk_lean_std",
                "heel_range_L", "heel_range_R", "step_height_symmetry", "step_length_symmetry",
                "knee_L_moving_time_sec", "knee_L_still_time_sec", "knee_L_moving_fraction",
                "knee_L_still_fraction", "knee_L_mean_speed", "knee_L_max_speed",
                "knee_L_total_time_sec", "knee_R_moving_time_sec", "knee_R_still_time_sec",
                "knee_R_moving_fraction", "knee_R_still_fraction", "knee_R_mean_speed",
                "knee_R_max_speed", "knee_R_total_time_sec", "knee_L_rom_y", "knee_R_rom_y",
                "hip_L_rom_y", "hip_R_rom_y", "shoulder_L_rom_x", "shoulder_R_rom_x",
                "ankle_L_rom_y", "ankle_R_rom_y", "knee_rom_asym", "hip_rom_asym",
                "shoulder_rom_asym", "ankle_rom_asym", "ankle_L_moving_fraction",
                "ankle_L_still_fraction", "ankle_R_moving_fraction", "ankle_R_still_fraction",
                "stance_ratio_L", "stance_ratio_R", "stance_ratio_asym", "knee_angle_L_mean",
                "knee_angle_L_std", "knee_angle_L_rom", "knee_angle_R_mean", "knee_angle_R_std",
                "knee_angle_R_rom", "hip_angle_L_mean", "hip_angle_L_std", "hip_angle_L_rom",
                "hip_angle_R_mean", "hip_angle_R_std", "hip_angle_R_rom", "ankle_angle_L_mean",
                "ankle_angle_L_std", "ankle_angle_L_rom", "ankle_angle_R_mean",
                "ankle_angle_R_std", "ankle_angle_R_rom", "knee_angle_rom_asym",
                "hip_angle_rom_asym", "ankle_angle_rom_asym", "step_L_mean_step_time",
                "step_L_std_step_time", "step_L_cadence", "step_L_mean_stride_time",
                "step_L_std_stride_time", "step_L_step_time_cv", "step_R_mean_step_time",
                "step_R_std_step_time", "step_R_cadence", "step_R_mean_stride_time",
                "step_R_std_stride_time", "step_R_step_time_cv", "step_time_asym",
                "cadence_asym", "step_width_mean", "step_width_std"
            ]
            
            # Store default imputation values (you should replace these with actual medians from training)
            st.session_state.imputation_values = {col: 0.0 for col in st.session_state.feature_columns}
            
            # Cache the model
            try:
                with open(model_path, 'rb') as f:
                    model_hash = hashlib.md5(f.read()).hexdigest()
                with open(cache_file, 'wb') as f:
                    pickle.dump({'model': model, 'hash': model_hash}, f)
                logger.info(f"Cached baseline model to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache model: {e}")
            
            return model, "Baseline model loaded successfully"
            
        except Exception as e:
            logger.error(f"Baseline model loading failed: {e}")
            logger.error(traceback.format_exc())
            return None, f"Baseline model loading failed: {str(e)}"
    
    @staticmethod
    @log_execution
    def load_binary_model():
        """Load binary ST-GCN model."""
        try:
            if not DEPENDENCIES['torch']:
                return None, "PyTorch not available"
            
            if not MODEL_PATHS['binary'].exists():
                return None, f"Binary model not found: {MODEL_PATHS['binary']}"
            
            # Check cache
            cache_key = "binary_stgcn"
            cache_file = MODEL_CACHE_DIR / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)
                    with open(MODEL_PATHS['binary'], 'rb') as f:
                        current_hash = hashlib.md5(f.read()).hexdigest()
                    if cached.get('hash') == current_hash:
                        logger.info(f"Loading binary model from cache: {cache_file}")
                        return cached['model'], "Binary model loaded from cache"
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}")
            
            # Load PyTorch model
            state_dict = torch.load(str(MODEL_PATHS['binary']), map_location='cpu')
            
            # Define model architecture (must match training)
            class BinarySTGCN(nn.Module):
                def __init__(self, in_channels=3, num_joints=14, out_classes=1):
                    super().__init__()
                    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
                    self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
                    self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
                    self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
                    self.fc = nn.Linear(256 * num_joints, out_classes)
                    
                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    x = F.relu(self.conv3(x))
                    x = self.pool(x)
                    x = x.flatten(1)
                    return self.fc(x)
            
            model = BinarySTGCN(num_joints=14, out_classes=1)
            
            # Handle different state dict formats
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            
            # Cache the model
            try:
                with open(MODEL_PATHS['binary'], 'rb') as f:
                    model_hash = hashlib.md5(f.read()).hexdigest()
                with open(cache_file, 'wb') as f:
                    pickle.dump({'model': model, 'hash': model_hash}, f)
                logger.info(f"Cached binary model to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache model: {e}")
            
            return model, "Binary model loaded successfully"
            
        except Exception as e:
            logger.error(f"Binary model loading failed: {e}")
            return None, f"Binary model loading failed: {str(e)}"
    
    @staticmethod
    @log_execution
    def load_multi_model():
        """Load multi-label ST-GCN model."""
        try:
            if not DEPENDENCIES['torch']:
                return None, "PyTorch not available"
            
            if not MODEL_PATHS['multi'].exists():
                return None, f"Multi-label model not found: {MODEL_PATHS['multi']}"
            
            # Check cache
            cache_key = "multi_stgcn"
            cache_file = MODEL_CACHE_DIR / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)
                    with open(MODEL_PATHS['multi'], 'rb') as f:
                        current_hash = hashlib.md5(f.read()).hexdigest()
                    if cached.get('hash') == current_hash:
                        logger.info(f"Loading multi-label model from cache: {cache_file}")
                        return cached['model'], "Multi-label model loaded from cache"
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}")
            
            # Load PyTorch model
            state_dict = torch.load(str(MODEL_PATHS['multi']), map_location='cpu')
            
            # Define model architecture
            class MultiSTGCN(nn.Module):
                def __init__(self, in_channels=3, num_joints=14, out_classes=5):
                    super().__init__()
                    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
                    self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
                    self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
                    self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
                    self.fc = nn.Linear(256 * num_joints, out_classes)
                    
                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    x = F.relu(self.conv3(x))
                    x = self.pool(x)
                    x = x.flatten(1)
                    return self.fc(x)
            
            model = MultiSTGCN(num_joints=14, out_classes=5)
            
            # Handle state dict
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            
            # Cache the model
            try:
                with open(MODEL_PATHS['multi'], 'rb') as f:
                    model_hash = hashlib.md5(f.read()).hexdigest()
                with open(cache_file, 'wb') as f:
                    pickle.dump({'model': model, 'hash': model_hash}, f)
                logger.info(f"Cached multi-label model to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache model: {e}")
            
            return model, "Multi-label model loaded successfully"
            
        except Exception as e:
            logger.error(f"Multi-label model loading failed: {e}")
            return None, f"Multi-label model loading failed: {str(e)}"

# Feature Engineering matching training pipeline - FIXED FOR BASELINE
class FeatureEngineer:
    """Feature extraction matching the training pipeline."""
    
    # Constants matching training
    ANOMALY_COLS = [
        "gait_anomaly_knee_sagittal_plane_abnormality",
        "gait_anomaly_trunk_balance_abnormality",
        "gait_anomaly_spatiotemporal_asymmetry",
        "gait_anomaly_hip_pelvic_control_deficit",
        "gait_anomaly_distal_foot_control_deficit",
    ]
    
    # Expected feature columns from training - MUST MATCH EXACTLY
    BASELINE_FEATURE_COLS = [
        "step_height_L", "step_height_R", "step_length_L", "step_length_R",
        "pelvis_drop_mean", "pelvis_drop_std", "trunk_lean_mean", "trunk_lean_std",
        "heel_range_L", "heel_range_R", "step_height_symmetry", "step_length_symmetry",
        "knee_L_moving_time_sec", "knee_L_still_time_sec", "knee_L_moving_fraction",
        "knee_L_still_fraction", "knee_L_mean_speed", "knee_L_max_speed",
        "knee_L_total_time_sec", "knee_R_moving_time_sec", "knee_R_still_time_sec",
        "knee_R_moving_fraction", "knee_R_still_fraction", "knee_R_mean_speed",
        "knee_R_max_speed", "knee_R_total_time_sec", "knee_L_rom_y", "knee_R_rom_y",
        "hip_L_rom_y", "hip_R_rom_y", "shoulder_L_rom_x", "shoulder_R_rom_x",
        "ankle_L_rom_y", "ankle_R_rom_y", "knee_rom_asym", "hip_rom_asym",
        "shoulder_rom_asym", "ankle_rom_asym", "ankle_L_moving_fraction",
        "ankle_L_still_fraction", "ankle_R_moving_fraction", "ankle_R_still_fraction",
        "stance_ratio_L", "stance_ratio_R", "stance_ratio_asym", "knee_angle_L_mean",
        "knee_angle_L_std", "knee_angle_L_rom", "knee_angle_R_mean", "knee_angle_R_std",
        "knee_angle_R_rom", "hip_angle_L_mean", "hip_angle_L_std", "hip_angle_L_rom",
        "hip_angle_R_mean", "hip_angle_R_std", "hip_angle_R_rom", "ankle_angle_L_mean",
        "ankle_angle_L_std", "ankle_angle_L_rom", "ankle_angle_R_mean",
        "ankle_angle_R_std", "ankle_angle_R_rom", "knee_angle_rom_asym",
        "hip_angle_rom_asym", "ankle_angle_rom_asym", "step_L_mean_step_time",
        "step_L_std_step_time", "step_L_cadence", "step_L_mean_stride_time",
        "step_L_std_stride_time", "step_L_step_time_cv", "step_R_mean_step_time",
        "step_R_std_step_time", "step_R_cadence", "step_R_mean_stride_time",
        "step_R_std_stride_time", "step_R_step_time_cv", "step_time_asym",
        "cadence_asym", "step_width_mean", "step_width_std"
    ]
    
    @staticmethod
    @log_execution
    def validate_csv(df):
        """Validate CSV format and content."""
        errors = []
        
        required_cols = ['frame', 'landmark_id', 'x_norm', 'y_norm', 'z_norm']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            errors.append("CSV file is empty")
        
        if len(df) < 10:
            errors.append("CSV file has too few rows (< 10)")
        
        if 'landmark_id' in df.columns:
            invalid_ids = df[~df['landmark_id'].between(0, 32)]['landmark_id'].unique()
            if len(invalid_ids) > 0:
                errors.append(f"Invalid landmark IDs found: {invalid_ids}")
        
        return errors
    
    @staticmethod
    def normalize_pose_3d(pose: np.ndarray) -> np.ndarray:
        """Normalize pose sequence by pelvis centering and torso-length scaling."""
        pose = np.asarray(pose, dtype=float)
        
        # Get pelvis (average of hips)
        pelvis = (pose[:, LEFT_HIP] + pose[:, RIGHT_HIP]) / 2.0
        pose_centered = pose - pelvis[:, None, :]
        
        # Get torso (average of shoulders)
        torso = (pose_centered[:, LEFT_SHOULDER] + pose_centered[:, RIGHT_SHOULDER]) / 2.0
        scale = np.linalg.norm(torso, axis=1).mean()
        
        if scale == 0 or not np.isfinite(scale):
            scale = 1.0
        
        return pose_centered / scale
    
    @staticmethod
    def joint_speed(pose_norm: np.ndarray, joint_idx: int, fps: float, smooth_sigma: float = 1.0) -> np.ndarray:
        """Calculate joint speed."""
        joint_traj = pose_norm[:, joint_idx, :]
        
        if smooth_sigma and smooth_sigma > 0:
            joint_traj = gaussian_filter1d(joint_traj, sigma=smooth_sigma, axis=0)
        
        diffs = np.diff(joint_traj, axis=0)
        disp = np.linalg.norm(diffs, axis=1)
        return disp * fps
    
    @staticmethod
    def moving_and_still_times(pose_norm: np.ndarray, joint_idx: int, fps: float, 
                               speed_thresh: float = 0.02, smooth_sigma: float = 1.0) -> dict:
        """Calculate moving and still times for a joint."""
        speed = FeatureEngineer.joint_speed(pose_norm, joint_idx, fps, smooth_sigma)
        
        moving_mask = speed >= speed_thresh
        still_mask = ~moving_mask
        
        moving_time_sec = moving_mask.sum() / fps
        still_time_sec = still_mask.sum() / fps
        total_time_sec = len(speed) / fps if fps > 0 else np.nan
        
        return {
            "moving_time_sec": float(moving_time_sec),
            "still_time_sec": float(still_time_sec),
            "moving_fraction": float(moving_mask.mean()),
            "still_fraction": float(still_mask.mean()),
            "mean_speed": float(speed.mean()),
            "max_speed": float(speed.max() if len(speed) > 0 else 0.0),
            "total_time_sec": float(total_time_sec),
        }
    
    @staticmethod
    def range_of_motion(pose_norm: np.ndarray, joint_idx: int, axis: str = None) -> dict:
        """Calculate range of motion for a joint."""
        traj = pose_norm[:, joint_idx, :]
        
        if axis is None:
            mean_pos = traj.mean(axis=0)
            dist = np.linalg.norm(traj - mean_pos, axis=1)
            rom_3d = dist.max() - dist.min()
            return {"rom_3d": float(rom_3d)}
        
        axis_to_idx = {"x": 0, "y": 1, "z": 2}
        idx = axis_to_idx[axis]
        coord = traj[:, idx]
        rom_axis = coord.max() - coord.min()
        return {f"rom_{axis}": float(rom_axis)}
    
    @staticmethod
    def asymmetry(L: float, R: float, eps: float = 1e-6) -> float:
        """Calculate asymmetry index."""
        return float((L - R) / (L + R + eps))
    
    @staticmethod
    def joint_angle(p_prox: np.ndarray, p_joint: np.ndarray, p_dist: np.ndarray) -> np.ndarray:
        """Calculate joint angle in degrees."""
        v1 = p_prox - p_joint
        v2 = p_dist - p_joint
        
        num = np.einsum("ij,ij->i", v1, v2)
        den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6
        
        cosang = np.clip(num / den, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosang))
        
        return angles
    
    @staticmethod
    def detect_step_events_from_ankle(ankle_y: np.ndarray, fps: float, min_step_time: float = 0.3) -> np.ndarray:
        """Detect step events from ankle vertical trajectory."""
        ankle_y = np.asarray(ankle_y, dtype=float)
        if ankle_y.size < 3 or fps <= 0:
            return np.array([], dtype=int)
        
        inv = -ankle_y
        min_distance = max(1, int(min_step_time * fps))
        peaks, _ = find_peaks(inv, distance=min_distance)
        return peaks
    
    @staticmethod
    def step_temporal_features(ankle_y: np.ndarray, fps: float, min_step_time: float = 0.3) -> dict:
        """Extract temporal gait features from ankle trajectory."""
        ankle_y = np.asarray(ankle_y, dtype=float)
        peaks = FeatureEngineer.detect_step_events_from_ankle(ankle_y, fps, min_step_time)
        
        if peaks.size < 2 or fps <= 0:
            return {
                "mean_step_time": np.nan,
                "std_step_time": np.nan,
                "cadence": np.nan,
                "mean_stride_time": np.nan,
                "std_stride_time": np.nan,
                "step_time_cv": np.nan,
            }
        
        times = peaks / fps
        step_intervals = np.diff(times)
        
        mean_step = float(step_intervals.mean())
        std_step = float(step_intervals.std())
        cadence = 60.0 / mean_step if mean_step > 0 else np.nan
        
        if times.size >= 3:
            stride_intervals = times[2:] - times[:-2]
            mean_stride = float(stride_intervals.mean())
            std_stride = float(stride_intervals.std())
        else:
            mean_stride = np.nan
            std_stride = np.nan
        
        step_time_cv = (std_step / mean_step) if mean_step > 0 else np.nan
        
        return {
            "mean_step_time": mean_step,
            "std_step_time": std_step,
            "cadence": float(cadence),
            "mean_stride_time": mean_stride,
            "std_stride_time": std_stride,
            "step_time_cv": float(step_time_cv),
        }
    
    @staticmethod
    @log_execution
    def extract_baseline_features(df: pd.DataFrame, fps: float = 30.0, cfg: FeatureConfig = None) -> Tuple[Dict, str]:
        """Extract comprehensive baseline features matching training pipeline."""
        try:
            cfg = cfg or FeatureConfig()
            
            # Preprocess dataframe - MATCH TRAINING
            df = df.copy()
            df['frame'] = df['frame'].astype(int)
            df['landmark_id'] = df['landmark_id'].astype(int)
            
            # Important: Ensure frames start at 0 (matching training)
            min_frame = df['frame'].min()
            df['frame'] = df['frame'] - min_frame
            
            df = df.sort_values(['frame', 'landmark_id']).reset_index(drop=True)
            
            # Create pose tensor (T, 33, 3)
            frames = np.sort(df['frame'].unique())
            T = len(frames)
            pose = np.zeros((T, N_JOINTS, 3), dtype=np.float32)
            
            frame_to_idx = {f: i for i, f in enumerate(frames)}
            for _, row in df.iterrows():
                f_idx = frame_to_idx[row['frame']]
                j = int(row['landmark_id'])
                if j < N_JOINTS:
                    pose[f_idx, j, :] = [row['x_norm'], row['y_norm'], row['z_norm']]
            
            # CRITICAL: Only use GAIT joints (14 joints) for feature extraction
            # Convert to (T, 14, 3) tensor for gait joints
            gait_pose = np.zeros((T, len(GAIT_JOINTS), 3), dtype=np.float32)
            for i, joint_idx in enumerate(GAIT_JOINTS):
                gait_pose[:, i, :] = pose[:, joint_idx, :]
            
            # Normalize pose
            pose_norm = FeatureEngineer.normalize_pose_3d(pose)  # Normalize full pose
            # Then extract gait joints from normalized pose
            gait_pose_norm = np.zeros((T, len(GAIT_JOINTS), 3), dtype=np.float32)
            for i, joint_idx in enumerate(GAIT_JOINTS):
                gait_pose_norm[:, i, :] = pose_norm[:, joint_idx, :]
            
            # Initialize features dictionary
            feats = {}
            
            # Map joint indices in the gait joints array
            joint_map = {}
            for i, joint_id in enumerate(GAIT_JOINTS):
                joint_map[joint_id] = i
            
            # Helper function to get joint index in gait joints array
            def get_gait_idx(joint_id):
                return joint_map.get(joint_id, -1)
            
            # 1. Basic spatial features
            # Step height (ankle vertical range)
            left_ankle_idx = get_gait_idx(LEFT_ANKLE)
            right_ankle_idx = get_gait_idx(RIGHT_ANKLE)
            
            if left_ankle_idx >= 0:
                left_ankle_y = gait_pose_norm[:, left_ankle_idx, 1]
                feats["step_height_L"] = float(left_ankle_y.max() - left_ankle_y.min())
            else:
                feats["step_height_L"] = 0.0
            
            if right_ankle_idx >= 0:
                right_ankle_y = gait_pose_norm[:, right_ankle_idx, 1]
                feats["step_height_R"] = float(right_ankle_y.max() - right_ankle_y.min())
            else:
                feats["step_height_R"] = 0.0
            
            # Step length (ankle horizontal range)
            if left_ankle_idx >= 0:
                left_ankle_x = gait_pose_norm[:, left_ankle_idx, 0]
                feats["step_length_L"] = float(left_ankle_x.max() - left_ankle_x.min())
            else:
                feats["step_length_L"] = 0.0
            
            if right_ankle_idx >= 0:
                right_ankle_x = gait_pose_norm[:, right_ankle_idx, 0]
                feats["step_length_R"] = float(right_ankle_x.max() - right_ankle_x.min())
            else:
                feats["step_length_R"] = 0.0
            
            # Pelvic drop (using hips from gait joints)
            left_hip_idx = get_gait_idx(LEFT_HIP)
            right_hip_idx = get_gait_idx(RIGHT_HIP)
            
            if left_hip_idx >= 0 and right_hip_idx >= 0:
                left_hip_y = gait_pose_norm[:, left_hip_idx, 1]
                right_hip_y = gait_pose_norm[:, right_hip_idx, 1]
                pelvis_diff = left_hip_y - right_hip_y
                feats["pelvis_drop_mean"] = float(pelvis_diff.mean())
                feats["pelvis_drop_std"] = float(pelvis_diff.std())
            else:
                feats["pelvis_drop_mean"] = 0.0
                feats["pelvis_drop_std"] = 0.0
            
            # Trunk lean (using shoulders from gait joints)
            left_sh_idx = get_gait_idx(LEFT_SHOULDER)
            right_sh_idx = get_gait_idx(RIGHT_SHOULDER)
            
            if left_sh_idx >= 0 and right_sh_idx >= 0:
                left_sh_x = gait_pose_norm[:, left_sh_idx, 0]
                right_sh_x = gait_pose_norm[:, right_sh_idx, 0]
                trunk_lean = left_sh_x - right_sh_x
                feats["trunk_lean_mean"] = float(trunk_lean.mean())
                feats["trunk_lean_std"] = float(trunk_lean.std())
            else:
                feats["trunk_lean_mean"] = 0.0
                feats["trunk_lean_std"] = 0.0
            
            # Heel clearance (using heels from gait joints)
            left_heel_idx = get_gait_idx(LEFT_HEEL)
            right_heel_idx = get_gait_idx(RIGHT_HEEL)
            
            if left_heel_idx >= 0:
                left_heel_y = gait_pose_norm[:, left_heel_idx, 1]
                feats["heel_range_L"] = float(left_heel_y.max() - left_heel_y.min())
            else:
                feats["heel_range_L"] = 0.0
            
            if right_heel_idx >= 0:
                right_heel_y = gait_pose_norm[:, right_heel_idx, 1]
                feats["heel_range_R"] = float(right_heel_y.max() - right_heel_y.min())
            else:
                feats["heel_range_R"] = 0.0
            
            # Symmetry indices
            eps = 1e-6
            hL, hR = feats["step_height_L"], feats["step_height_R"]
            lL, lR = feats["step_length_L"], feats["step_length_R"]
            
            feats["step_height_symmetry"] = float((hL - hR) / (hL + hR + eps))
            feats["step_length_symmetry"] = float((lL - lR) / (lL + lR + eps))
            
            # 2. Knee motion features
            left_knee_idx = get_gait_idx(LEFT_KNEE)
            right_knee_idx = get_gait_idx(RIGHT_KNEE)
            
            if left_knee_idx >= 0:
                # Use the full normalized pose for knee features (not just gait joints)
                left_knee_move = FeatureEngineer.moving_and_still_times(pose_norm, LEFT_KNEE, fps, cfg.speed_thresh, cfg.smooth_sigma)
                for k, v in left_knee_move.items():
                    feats[f"knee_L_{k}"] = v
                feats["knee_L_rom_y"] = FeatureEngineer.range_of_motion(pose_norm, LEFT_KNEE, axis="y")["rom_y"]
            else:
                # Set default values
                for k in ["moving_time_sec", "still_time_sec", "moving_fraction", "still_fraction", 
                         "mean_speed", "max_speed", "total_time_sec"]:
                    feats[f"knee_L_{k}"] = 0.0
                feats["knee_L_rom_y"] = 0.0
            
            if right_knee_idx >= 0:
                right_knee_move = FeatureEngineer.moving_and_still_times(pose_norm, RIGHT_KNEE, fps, cfg.speed_thresh, cfg.smooth_sigma)
                for k, v in right_knee_move.items():
                    feats[f"knee_R_{k}"] = v
                feats["knee_R_rom_y"] = FeatureEngineer.range_of_motion(pose_norm, RIGHT_KNEE, axis="y")["rom_y"]
            else:
                for k in ["moving_time_sec", "still_time_sec", "moving_fraction", "still_fraction", 
                         "mean_speed", "max_speed", "total_time_sec"]:
                    feats[f"knee_R_{k}"] = 0.0
                feats["knee_R_rom_y"] = 0.0
            
            # 3. Joint ROM features
            # Hip ROM
            if left_hip_idx >= 0:
                hip_L_rom_y = FeatureEngineer.range_of_motion(pose_norm, LEFT_HIP, axis="y")["rom_y"]
                feats["hip_L_rom_y"] = hip_L_rom_y
            else:
                feats["hip_L_rom_y"] = 0.0
            
            if right_hip_idx >= 0:
                hip_R_rom_y = FeatureEngineer.range_of_motion(pose_norm, RIGHT_HIP, axis="y")["rom_y"]
                feats["hip_R_rom_y"] = hip_R_rom_y
            else:
                feats["hip_R_rom_y"] = 0.0
            
            # Shoulder ROM
            if left_sh_idx >= 0:
                shoulder_L_rom_x = FeatureEngineer.range_of_motion(pose_norm, LEFT_SHOULDER, axis="x")["rom_x"]
                feats["shoulder_L_rom_x"] = shoulder_L_rom_x
            else:
                feats["shoulder_L_rom_x"] = 0.0
            
            if right_sh_idx >= 0:
                shoulder_R_rom_x = FeatureEngineer.range_of_motion(pose_norm, RIGHT_SHOULDER, axis="x")["rom_x"]
                feats["shoulder_R_rom_x"] = shoulder_R_rom_x
            else:
                feats["shoulder_R_rom_x"] = 0.0
            
            # Ankle ROM
            if left_ankle_idx >= 0:
                ankle_L_rom_y = FeatureEngineer.range_of_motion(pose_norm, LEFT_ANKLE, axis="y")["rom_y"]
                feats["ankle_L_rom_y"] = ankle_L_rom_y
            else:
                feats["ankle_L_rom_y"] = 0.0
            
            if right_ankle_idx >= 0:
                ankle_R_rom_y = FeatureEngineer.range_of_motion(pose_norm, RIGHT_ANKLE, axis="y")["rom_y"]
                feats["ankle_R_rom_y"] = ankle_R_rom_y
            else:
                feats["ankle_R_rom_y"] = 0.0
            
            # ROM asymmetries
            feats["knee_rom_asym"] = FeatureEngineer.asymmetry(feats["knee_L_rom_y"], feats["knee_R_rom_y"])
            feats["hip_rom_asym"] = FeatureEngineer.asymmetry(feats["hip_L_rom_y"], feats["hip_R_rom_y"])
            feats["shoulder_rom_asym"] = FeatureEngineer.asymmetry(feats["shoulder_L_rom_x"], feats["shoulder_R_rom_x"])
            feats["ankle_rom_asym"] = FeatureEngineer.asymmetry(feats["ankle_L_rom_y"], feats["ankle_R_rom_y"])
            
            # 4. Stance/ratio features
            if left_ankle_idx >= 0:
                ankle_L_move = FeatureEngineer.moving_and_still_times(pose_norm, LEFT_ANKLE, fps, cfg.speed_thresh, cfg.smooth_sigma)
                feats["ankle_L_moving_fraction"] = ankle_L_move["moving_fraction"]
                feats["ankle_L_still_fraction"] = ankle_L_move["still_fraction"]
            else:
                feats["ankle_L_moving_fraction"] = 0.0
                feats["ankle_L_still_fraction"] = 0.0
            
            if right_ankle_idx >= 0:
                ankle_R_move = FeatureEngineer.moving_and_still_times(pose_norm, RIGHT_ANKLE, fps, cfg.speed_thresh, cfg.smooth_sigma)
                feats["ankle_R_moving_fraction"] = ankle_R_move["moving_fraction"]
                feats["ankle_R_still_fraction"] = ankle_R_move["still_fraction"]
            else:
                feats["ankle_R_moving_fraction"] = 0.0
                feats["ankle_R_still_fraction"] = 0.0
            
            stance_ratio_L = feats["ankle_L_still_fraction"] / (feats["ankle_L_moving_fraction"] + 1e-6)
            stance_ratio_R = feats["ankle_R_still_fraction"] / (feats["ankle_R_moving_fraction"] + 1e-6)
            
            feats["stance_ratio_L"] = float(stance_ratio_L)
            feats["stance_ratio_R"] = float(stance_ratio_R)
            feats["stance_ratio_asym"] = FeatureEngineer.asymmetry(stance_ratio_L, stance_ratio_R)
            
            # 5. Joint angle features
            # Knee angles
            if left_hip_idx >= 0 and left_knee_idx >= 0 and left_ankle_idx >= 0:
                knee_angle_L = FeatureEngineer.joint_angle(
                    gait_pose_norm[:, left_hip_idx, :],
                    gait_pose_norm[:, left_knee_idx, :],
                    gait_pose_norm[:, left_ankle_idx, :],
                )
                feats["knee_angle_L_mean"] = float(knee_angle_L.mean())
                feats["knee_angle_L_std"] = float(knee_angle_L.std())
                feats["knee_angle_L_rom"] = float(knee_angle_L.max() - knee_angle_L.min())
            else:
                feats["knee_angle_L_mean"] = 0.0
                feats["knee_angle_L_std"] = 0.0
                feats["knee_angle_L_rom"] = 0.0
            
            if right_hip_idx >= 0 and right_knee_idx >= 0 and right_ankle_idx >= 0:
                knee_angle_R = FeatureEngineer.joint_angle(
                    gait_pose_norm[:, right_hip_idx, :],
                    gait_pose_norm[:, right_knee_idx, :],
                    gait_pose_norm[:, right_ankle_idx, :],
                )
                feats["knee_angle_R_mean"] = float(knee_angle_R.mean())
                feats["knee_angle_R_std"] = float(knee_angle_R.std())
                feats["knee_angle_R_rom"] = float(knee_angle_R.max() - knee_angle_R.min())
            else:
                feats["knee_angle_R_mean"] = 0.0
                feats["knee_angle_R_std"] = 0.0
                feats["knee_angle_R_rom"] = 0.0
            
            # Hip angles
            if left_sh_idx >= 0 and left_hip_idx >= 0 and left_knee_idx >= 0:
                hip_angle_L = FeatureEngineer.joint_angle(
                    gait_pose_norm[:, left_sh_idx, :],
                    gait_pose_norm[:, left_hip_idx, :],
                    gait_pose_norm[:, left_knee_idx, :],
                )
                feats["hip_angle_L_mean"] = float(hip_angle_L.mean())
                feats["hip_angle_L_std"] = float(hip_angle_L.std())
                feats["hip_angle_L_rom"] = float(hip_angle_L.max() - hip_angle_L.min())
            else:
                feats["hip_angle_L_mean"] = 0.0
                feats["hip_angle_L_std"] = 0.0
                feats["hip_angle_L_rom"] = 0.0
            
            if right_sh_idx >= 0 and right_hip_idx >= 0 and right_knee_idx >= 0:
                hip_angle_R = FeatureEngineer.joint_angle(
                    gait_pose_norm[:, right_sh_idx, :],
                    gait_pose_norm[:, right_hip_idx, :],
                    gait_pose_norm[:, right_knee_idx, :],
                )
                feats["hip_angle_R_mean"] = float(hip_angle_R.mean())
                feats["hip_angle_R_std"] = float(hip_angle_R.std())
                feats["hip_angle_R_rom"] = float(hip_angle_R.max() - hip_angle_R.min())
            else:
                feats["hip_angle_R_mean"] = 0.0
                feats["hip_angle_R_std"] = 0.0
                feats["hip_angle_R_rom"] = 0.0
            
            # Ankle angles
            left_foot_idx = get_gait_idx(LEFT_FOOT_INDEX)
            right_foot_idx = get_gait_idx(RIGHT_FOOT_INDEX)
            
            if left_knee_idx >= 0 and left_ankle_idx >= 0 and left_foot_idx >= 0:
                ankle_angle_L = FeatureEngineer.joint_angle(
                    gait_pose_norm[:, left_knee_idx, :],
                    gait_pose_norm[:, left_ankle_idx, :],
                    gait_pose_norm[:, left_foot_idx, :],
                )
                feats["ankle_angle_L_mean"] = float(ankle_angle_L.mean())
                feats["ankle_angle_L_std"] = float(ankle_angle_L.std())
                feats["ankle_angle_L_rom"] = float(ankle_angle_L.max() - ankle_angle_L.min())
            else:
                feats["ankle_angle_L_mean"] = 0.0
                feats["ankle_angle_L_std"] = 0.0
                feats["ankle_angle_L_rom"] = 0.0
            
            if right_knee_idx >= 0 and right_ankle_idx >= 0 and right_foot_idx >= 0:
                ankle_angle_R = FeatureEngineer.joint_angle(
                    gait_pose_norm[:, right_knee_idx, :],
                    gait_pose_norm[:, right_ankle_idx, :],
                    gait_pose_norm[:, right_foot_idx, :],
                )
                feats["ankle_angle_R_mean"] = float(ankle_angle_R.mean())
                feats["ankle_angle_R_std"] = float(ankle_angle_R.std())
                feats["ankle_angle_R_rom"] = float(ankle_angle_R.max() - ankle_angle_R.min())
            else:
                feats["ankle_angle_R_mean"] = 0.0
                feats["ankle_angle_R_std"] = 0.0
                feats["ankle_angle_R_rom"] = 0.0
            
            # Angle ROM asymmetries
            feats["knee_angle_rom_asym"] = FeatureEngineer.asymmetry(
                feats["knee_angle_L_rom"], feats["knee_angle_R_rom"]
            )
            feats["hip_angle_rom_asym"] = FeatureEngineer.asymmetry(
                feats["hip_angle_L_rom"], feats["hip_angle_R_rom"]
            )
            feats["ankle_angle_rom_asym"] = FeatureEngineer.asymmetry(
                feats["ankle_angle_L_rom"], feats["ankle_angle_R_rom"]
            )
            
            # 6. Temporal gait features
            if left_ankle_idx >= 0:
                left_temporal = FeatureEngineer.step_temporal_features(left_ankle_y, fps)
                for k, v in left_temporal.items():
                    feats[f"step_L_{k}"] = float(v) if not np.isnan(v) else 0.0
            else:
                for k in ["mean_step_time", "std_step_time", "cadence", "mean_stride_time", 
                         "std_stride_time", "step_time_cv"]:
                    feats[f"step_L_{k}"] = 0.0
            
            if right_ankle_idx >= 0:
                right_temporal = FeatureEngineer.step_temporal_features(right_ankle_y, fps)
                for k, v in right_temporal.items():
                    feats[f"step_R_{k}"] = float(v) if not np.isnan(v) else 0.0
            else:
                for k in ["mean_step_time", "std_step_time", "cadence", "mean_stride_time", 
                         "std_stride_time", "step_time_cv"]:
                    feats[f"step_R_{k}"] = 0.0
            
            # Temporal asymmetries
            if not np.isnan(feats["step_L_mean_step_time"]) and not np.isnan(feats["step_R_mean_step_time"]):
                feats["step_time_asym"] = FeatureEngineer.asymmetry(
                    feats["step_L_mean_step_time"], feats["step_R_mean_step_time"]
                )
            else:
                feats["step_time_asym"] = 0.0
            
            if not np.isnan(feats["step_L_cadence"]) and not np.isnan(feats["step_R_cadence"]):
                feats["cadence_asym"] = FeatureEngineer.asymmetry(
                    feats["step_L_cadence"], feats["step_R_cadence"]
                )
            else:
                feats["cadence_asym"] = 0.0
            
            # 7. Step width features
            if left_ankle_idx >= 0 and right_ankle_idx >= 0:
                step_width_series = np.abs(left_ankle_x - right_ankle_x)
                feats["step_width_mean"] = float(step_width_series.mean())
                feats["step_width_std"] = float(step_width_series.std())
            else:
                feats["step_width_mean"] = 0.0
                feats["step_width_std"] = 0.0
            
            # CRITICAL: Ensure all expected features are present and in correct order
            final_features = {}
            for col in FeatureEngineer.BASELINE_FEATURE_COLS:
                if col in feats:
                    final_features[col] = feats[col]
                else:
                    logger.warning(f"Missing feature: {col}, setting to 0.0")
                    final_features[col] = 0.0
            
            # Handle NaN values
            for col in final_features:
                if np.isnan(final_features[col]) or not np.isfinite(final_features[col]):
                    final_features[col] = 0.0
            
            return final_features, "Baseline features extracted successfully"
            
        except Exception as e:
            logger.error(f"Baseline feature extraction failed: {e}")
            logger.error(traceback.format_exc())
            return None, f"Baseline feature extraction failed: {str(e)}"
    
    @staticmethod
    @log_execution
    def extract_stgcn_features(df: pd.DataFrame, fps: float = 30.0) -> Tuple[Dict, str]:
        """Extract ST-GCN features."""
        try:
            # First extract baseline features for consistency
            baseline_features, status = FeatureEngineer.extract_baseline_features(df, fps)
            if not baseline_features:
                return None, status
            
            # Create pose tensor for ST-GCN
            df = df.copy()
            df['frame'] = df['frame'].astype(int)
            df['landmark_id'] = df['landmark_id'].astype(int)
            
            # Ensure frames start at 0
            min_frame = df['frame'].min()
            df['frame'] = df['frame'] - min_frame
            
            df = df.sort_values(['frame', 'landmark_id']).reset_index(drop=True)
            
            frames = np.sort(df['frame'].unique())
            T = len(frames)
            
            # Create tensor with GAIT joints
            pose_tensor = np.zeros((T, len(GAIT_JOINTS), 3), dtype=np.float32)
            
            frame_to_idx = {f: i for i, f in enumerate(frames)}
            landmark_to_idx = {l: i for i, l in enumerate(GAIT_JOINTS)}
            
            for _, row in df.iterrows():
                f_idx = frame_to_idx[row['frame']]
                l = int(row['landmark_id'])
                if l in landmark_to_idx:
                    l_idx = landmark_to_idx[l]
                    pose_tensor[f_idx, l_idx, 0] = row['x_norm']
                    pose_tensor[f_idx, l_idx, 1] = row['y_norm']
                    pose_tensor[f_idx, l_idx, 2] = row['z_norm']
            
            # Normalize
            hip_left_idx = GAIT_JOINTS.index(LEFT_HIP) if LEFT_HIP in GAIT_JOINTS else 0
            hip_right_idx = GAIT_JOINTS.index(RIGHT_HIP) if RIGHT_HIP in GAIT_JOINTS else 1
            
            pelvis = (pose_tensor[:, hip_left_idx, :] + pose_tensor[:, hip_right_idx, :]) / 2.0
            pose_centered = pose_tensor - pelvis[:, np.newaxis, :]
            
            shoulder_left_idx = GAIT_JOINTS.index(LEFT_SHOULDER) if LEFT_SHOULDER in GAIT_JOINTS else 2
            shoulder_right_idx = GAIT_JOINTS.index(RIGHT_SHOULDER) if RIGHT_SHOULDER in GAIT_JOINTS else 3
            
            left_shoulder = pose_centered[:, shoulder_left_idx, :]
            right_shoulder = pose_centered[:, shoulder_right_idx, :]
            shoulder_distances = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
            scale = np.mean(shoulder_distances)
            
            if scale < 1e-6:
                scale = 1.0
            
            pose_normalized = pose_centered / scale
            pose_normalized = np.nan_to_num(pose_normalized, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Transpose to ST-GCN format: (N, C, T, V)
            stgcn_tensor = np.transpose(pose_normalized, (2, 0, 1))  # (C, T, V)
            stgcn_tensor = stgcn_tensor[np.newaxis, ...]  # (1, C, T, V)
            
            return {
                'stgcn_tensor': stgcn_tensor,
                'baseline_features': baseline_features,
                'tensor_shape': stgcn_tensor.shape,
                'num_frames': T,
                'num_joints': len(GAIT_JOINTS)
            }, "ST-GCN features extracted successfully"
            
        except Exception as e:
            logger.error(f"ST-GCN feature extraction failed: {e}")
            logger.error(traceback.format_exc())
            return None, f"ST-GCN feature extraction failed: {str(e)}"
    
    @staticmethod
    @handle_errors(default_return=(None, "Feature preparation failed"))
    def prepare_features_for_baseline(features: Dict) -> Tuple[np.ndarray, str]:
        """Prepare features for baseline model prediction."""
        try:
            if not features:
                return None, "No features to prepare"
            
            # Create feature vector in the expected order
            feature_vector = []
            for col in FeatureEngineer.BASELINE_FEATURE_COLS:
                if col in features:
                    feature_vector.append(features[col])
                else:
                    logger.warning(f"Missing feature {col} in features dict")
                    feature_vector.append(0.0)  # Default value
            
            feature_array = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
            
            # Handle NaN values - CRITICAL: match training imputation
            feature_array = np.nan_to_num(feature_array, nan=0.0)
            
            # Apply imputation values if available
            if hasattr(st.session_state, 'imputation_values') and st.session_state.imputation_values:
                for i, col in enumerate(FeatureEngineer.BASELINE_FEATURE_COLS):
                    if feature_array[0, i] == 0.0 and col in st.session_state.imputation_values:
                        feature_array[0, i] = st.session_state.imputation_values[col]
            
            return feature_array, "Features prepared for baseline model"
            
        except Exception as e:
            logger.error(f"Baseline feature preparation failed: {e}")
            return None, f"Baseline feature preparation failed: {str(e)}"
    
    @staticmethod
    @handle_errors(default_return=(None, "Feature preparation failed"))
    def prepare_features_for_stgcn(features_dict: Dict) -> Tuple[np.ndarray, str]:
        """Prepare features for ST-GCN models."""
        try:
            if not features_dict or 'stgcn_tensor' not in features_dict:
                return None, "No ST-GCN features available"
            
            stgcn_tensor = features_dict['stgcn_tensor']
            
            if stgcn_tensor.ndim == 4:
                return stgcn_tensor, "Features prepared for ST-GCN model"
            elif stgcn_tensor.ndim == 3:
                return stgcn_tensor[np.newaxis, ...], "Features prepared for ST-GCN model"
            else:
                return None, f"Invalid tensor shape: {stgcn_tensor.shape}"
            
        except Exception as e:
            logger.error(f"ST-GCN feature preparation failed: {e}")
            return None, f"ST-GCN feature preparation failed: {str(e)}"

# Prediction Engine - FIXED FOR BASELINE
class PredictionEngine:
    """Handles predictions from all model types."""
    
    ANOMALY_COLS = FeatureEngineer.ANOMALY_COLS
    
    @staticmethod
    @log_execution
    def predict_with_baseline(model, features_array: np.ndarray) -> Tuple[Dict, str]:
        """Make prediction with baseline XGBoost model."""
        try:
            if model is None or features_array is None:
                return None, "No model or features available"
            
            # CRITICAL: Verify feature dimensions
            expected_features = len(FeatureEngineer.BASELINE_FEATURE_COLS)
            if features_array.shape[1] != expected_features:
                logger.error(f"Feature mismatch: Expected {expected_features}, got {features_array.shape[1]}")
                # Try to pad or truncate
                if features_array.shape[1] < expected_features:
                    # Pad with zeros
                    padded = np.zeros((1, expected_features), dtype=np.float32)
                    padded[0, :features_array.shape[1]] = features_array
                    features_array = padded
                else:
                    # Truncate
                    features_array = features_array[:, :expected_features]
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            probabilities = model.predict_proba(features_array)[0]
            
            # Ensure probabilities sum to 1
            if probabilities.sum() > 0:
                probabilities = probabilities / probabilities.sum()
            
            result = {
                'prediction': int(prediction),
                'label': 'Normal' if prediction == 0 else 'Abnormal',
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'Normal': float(probabilities[0]) if len(probabilities) > 0 else 0.5,
                    'Abnormal': float(probabilities[1]) if len(probabilities) > 1 else 0.5
                },
                'timestamp': datetime.now().isoformat(),
                'model_type': 'baseline_xgboost',
                'feature_count': features_array.shape[1],
                'expected_features': expected_features
            }
            
            return result, "Baseline prediction successful"
            
        except Exception as e:
            logger.error(f"Baseline prediction failed: {e}")
            logger.error(traceback.format_exc())
            return None, f"Baseline prediction failed: {str(e)}"
    
    @staticmethod
    @log_execution
    def predict_with_binary(model, stgcn_tensor: np.ndarray) -> Tuple[Dict, str]:
        """Make prediction with binary ST-GCN model."""
        try:
            if model is None or stgcn_tensor is None:
                return None, "No model or features available"
            
            if not DEPENDENCIES['torch']:
                return None, "PyTorch not available"
            
            # Ensure tensor is in correct format
            if stgcn_tensor.shape[1] != 3:
                if stgcn_tensor.shape[-1] == 3:
                    stgcn_tensor = np.transpose(stgcn_tensor, (0, 3, 1, 2))
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            tensor = torch.tensor(stgcn_tensor, dtype=torch.float32).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(tensor)
                
                if output.shape[1] == 1:
                    # Binary classification
                    probabilities = torch.sigmoid(output).cpu().numpy()
                    prediction = (probabilities > 0.5).astype(int)
                    normal_prob = 1 - probabilities[0, 0]
                    abnormal_prob = probabilities[0, 0]
                else:
                    # Multi-class treated as binary
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()
                    prediction = np.argmax(probabilities, axis=1)
                    normal_prob = probabilities[0, 0] if probabilities.shape[1] > 0 else 0.5
                    abnormal_prob = 1 - normal_prob
            
            result = {
                'prediction': int(prediction[0]),
                'label': 'Normal' if prediction[0] == 0 else 'Abnormal',
                'confidence': float(max(normal_prob, abnormal_prob)),
                'probabilities': {
                    'Normal': float(normal_prob),
                    'Abnormal': float(abnormal_prob)
                },
                'timestamp': datetime.now().isoformat(),
                'model_type': 'binary_stgcn',
                'tensor_shape': stgcn_tensor.shape
            }
            
            return result, "Binary ST-GCN prediction successful"
            
        except Exception as e:
            logger.error(f"Binary ST-GCN prediction failed: {e}")
            return None, f"Binary ST-GCN prediction failed: {str(e)}"
    
    @staticmethod
    @log_execution
    def predict_with_multi(model, stgcn_tensor: np.ndarray) -> Tuple[Dict, str]:
        """Make prediction with multi-label ST-GCN model."""
        try:
            if model is None or stgcn_tensor is None:
                return None, "No model or features available"
            
            if not DEPENDENCIES['torch']:
                return None, "PyTorch not available"
            
            # Ensure tensor format
            if stgcn_tensor.shape[1] != 3:
                if stgcn_tensor.shape[-1] == 3:
                    stgcn_tensor = np.transpose(stgcn_tensor, (0, 3, 1, 2))
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            tensor = torch.tensor(stgcn_tensor, dtype=torch.float32).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.sigmoid(output).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
            
            # Create result
            result = {
                'prediction': predictions[0].tolist(),
                'label': 'Multi-label prediction',
                'confidence': float(np.mean(probabilities[0])),
                'probabilities': {
                    col: float(prob) for col, prob in zip(PredictionEngine.ANOMALY_COLS, probabilities[0])
                },
                'anomaly_detected': bool(any(predictions[0])),
                'detected_anomalies': [
                    col for col, pred in zip(PredictionEngine.ANOMALY_COLS, predictions[0]) if pred
                ],
                'timestamp': datetime.now().isoformat(),
                'model_type': 'multi_label_stgcn',
                'tensor_shape': stgcn_tensor.shape
            }
            
            return result, "Multi-label ST-GCN prediction successful"
            
        except Exception as e:
            logger.error(f"Multi-label ST-GCN prediction failed: {e}")
            return None, f"Multi-label ST-GCN prediction failed: {str(e)}"

# Visualization
class Visualizer:
    """Handles visualization creation."""
    
    @staticmethod
    def create_features_chart(features: Dict) -> Tuple[Optional[plt.Figure], str]:
        """Create chart visualizing extracted features."""
        try:
            if not DEPENDENCIES['matplotlib']:
                return None, "Matplotlib not available"
            
            if not features:
                return None, "No features to visualize"
            
            # Select top features
            feature_names = list(features.keys())[:20]
            feature_values = [features[name] for name in feature_names]
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Create bar chart with color coding
            colors = []
            for name in feature_names:
                if 'asym' in name.lower():
                    colors.append('#ff7f0e')  # Orange for asymmetry
                elif 'std' in name.lower():
                    colors.append('#2ca02c')  # Green for variability
                elif 'mean' in name.lower():
                    colors.append('#1f77b4')  # Blue for means
                else:
                    colors.append('#d62728')  # Red for others
            
            bars = ax.bar(range(len(feature_names)), feature_values, color=colors, alpha=0.7)
            
            # Customize
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels([name[:25] + '...' if len(name) > 25 else name for name in feature_names], 
                              rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Feature Value')
            ax.set_title('Extracted Gait Features (Top 20)')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, feature_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            return fig, "Feature chart created successfully"
            
        except Exception as e:
            logger.error(f"Feature chart creation failed: {e}")
            return None, f"Feature chart creation failed: {str(e)}"
    
    @staticmethod
    def create_prediction_chart(prediction_result: Dict) -> Tuple[Optional[plt.Figure], str]:
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
                
                # Color based on anomaly presence
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
            
            # Add general info
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

# File Management
class FileManager:
    """Handles file operations."""
    
    @staticmethod
    @log_execution
    def save_uploaded_file(uploaded_file) -> Tuple[Optional[Path], Optional[Dict]]:
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
    @log_execution
    def save_results(prediction_result: Dict, features: Dict, file_info: Dict, model_type: str) -> Tuple[Optional[Path], Optional[pd.DataFrame]]:
        """Save results to CSV."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"gait_results_{model_type}_{timestamp}.csv"
            results_path = RESULTS_DIR / results_filename
            
            # Create results data
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
            
            # Add features (top 20 to keep CSV manageable)
            if features:
                if isinstance(features, dict):
                    for idx, (key, value) in enumerate(list(features.items())[:20]):
                        results_data[f"feature_{key}"] = value
                elif isinstance(features, dict) and 'baseline_features' in features:
                    for idx, (key, value) in enumerate(list(features['baseline_features'].items())[:20]):
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

# Main Application - MAINTAIN EXISTING STRUCTURE
def main():
    """Main application function."""
    try:
        # Initialize session state
        init_session_state()
        
        # Custom CSS (keep existing)
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
        .feature-badge {
            display: inline-block;
            padding: 2px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 0.8em;
            background-color: #e9ecef;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🚶 GAITy - Advanced Gait Analysis</h1>', unsafe_allow_html=True)
        st.markdown("### Complete Pipeline with Baseline & Advanced Models")
        
        # Sidebar - Model Management
        with st.sidebar:
            st.header("🔧 Model Management")
            
            # Model loading controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Load Baseline", key="load_baseline", help="Load XGBoost baseline model"):
                    with st.spinner("Loading baseline model..."):
                        model, status = ModelManager.load_baseline_model()
                        st.session_state.baseline_model = model
                        st.session_state.baseline_loaded = model is not None
                        if model:
                            st.success("✅ Baseline loaded")
                            st.session_state.processing_history.append("Baseline model loaded")
                        else:
                            st.error(f"❌ {status}")
                        st.rerun()
            
            with col2:
                if st.button("Load Binary", key="load_binary", help="Load binary ST-GCN model"):
                    with st.spinner("Loading binary model..."):
                        model, status = ModelManager.load_binary_model()
                        st.session_state.binary_model = model
                        st.session_state.binary_loaded = model is not None
                        if model:
                            st.success("✅ Binary loaded")
                            st.session_state.processing_history.append("Binary ST-GCN model loaded")
                        else:
                            st.error(f"❌ {status}")
                        st.rerun()
            
            with col3:
                if st.button("Load Multi", key="load_multi", help="Load multi-label ST-GCN model"):
                    with st.spinner("Loading multi-label model..."):
                        model, status = ModelManager.load_multi_model()
                        st.session_state.multi_model = model
                        st.session_state.multi_loaded = model is not None
                        if model:
                            st.success("✅ Multi-label loaded")
                            st.session_state.processing_history.append("Multi-label ST-GCN model loaded")
                        else:
                            st.error(f"❌ {status}")
                        st.rerun()
            
            # Model status display
            st.markdown("### Model Status")
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.write("**Baseline XGBoost**")
            if st.session_state.baseline_loaded:
                st.markdown('<p class="status-success">✅ Loaded</p>', unsafe_allow_html=True)
                if st.session_state.feature_columns:
                    st.caption(f"{len(st.session_state.feature_columns)} features expected")
            else:
                st.markdown('<p class="status-error">❌ Not loaded</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.write("**Binary ST-GCN**")
            if st.session_state.binary_loaded:
                st.markdown('<p class="status-success">✅ Loaded</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-error">❌ Not loaded</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.write("**Multi-label ST-GCN**")
            if st.session_state.multi_loaded:
                st.markdown('<p class="status-success">✅ Loaded</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-error">❌ Not loaded</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Available models
            available_models = []
            if st.session_state.baseline_loaded:
                available_models.append("Baseline XGBoost")
            if st.session_state.binary_loaded:
                available_models.append("Binary ST-GCN")
            if st.session_state.multi_loaded:
                available_models.append("Multi-label ST-GCN")
            
            st.session_state.available_models = available_models
            
            # System info
            with st.expander("System Information", expanded=False):
                st.write("**Dependencies:**")
                for dep, status in DEPENDENCIES.items():
                    st.write(f"- {dep.title()}: {'✅' if status else '❌'}")
                
                if DEPENDENCIES['torch']:
                    st.write(f"PyTorch version: {torch.__version__}")
                    st.write(f"CUDA available: {torch.cuda.is_available()}")
                
                if DEPENDENCIES['xgboost']:
                    st.write(f"XGBoost version: {xgb.__version__}")
                
                st.write("**Model Files:**")
                for name, path in MODEL_PATHS.items():
                    exists = path.exists()
                    st.write(f"- {name}: {'✅' if exists else '❌'} {path}")
            
            # Processing history
            if st.session_state.processing_history:
                with st.expander("Recent Activity", expanded=False):
                    for item in st.session_state.processing_history[-10:]:
                        st.write(f"- {item}")
        
        # Main content area - KEEP EXISTING STRUCTURE
        # ... (rest of the main function remains the same as your original code)
        # The key changes are in ModelManager.load_baseline_model() and FeatureEngineer.extract_baseline_features()
        
        # Note: The main UI code below remains largely the same, but ensures FPS is passed correctly
        
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
                
                # Read and process CSV
                try:
                    df = pd.read_csv(file_path)
                    
                    # Display file info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Frames", df['frame'].nunique() if 'frame' in df.columns else 'N/A')
                    with col4:
                        st.metric("Landmarks", df['landmark_id'].nunique() if 'landmark_id' in df.columns else 'N/A')
                    
                    # Validate CSV
                    validation_errors = FeatureEngineer.validate_csv(df)
                    
                    if validation_errors:
                        st.error("❌ Validation Errors:")
                        for error in validation_errors:
                            st.write(f"- {error}")
                    else:
                        st.success("✅ CSV validation passed")
                        
                        # Data preview
                        with st.expander("📊 Data Preview", expanded=False):
                            st.dataframe(df.head(20))
                        
                        # FPS input - IMPORTANT FOR FEATURE EXTRACTION
                        fps = st.number_input(
                            "Video FPS (frames per second)",
                            min_value=1.0,
                            max_value=120.0,
                            value=30.0,
                            step=1.0,
                            help="Frame rate of the original video - CRITICAL for temporal features"
                        )
                        
                        # Step 2: Feature Engineering
                        st.header("🔬 Step 2: Feature Engineering")
                        
                        # Feature extraction method selection
                        feature_method = st.radio(
                            "Select Feature Extraction Method:",
                            ["Baseline (XGBoost Features)", "ST-GCN (Advanced Models)"],
                            key="feature_method"
                        )
                        
                        if st.button("⚡ Extract Features", key="extract_features", type="primary"):
                            with st.spinner("Extracting features..."):
                                if feature_method == "Baseline (XGBoost Features)":
                                    features, status = FeatureEngineer.extract_baseline_features(df, fps)
                                    st.session_state.features = features
                                    st.session_state.features_type = "baseline"
                                else:
                                    features_dict, status = FeatureEngineer.extract_stgcn_features(df, fps)
                                    st.session_state.features = features_dict
                                    st.session_state.features_type = "stgcn"
                                
                                if features or features_dict:
                                    st.success(f"✅ {status}")
                                    st.session_state.processing_history.append(f"Features extracted ({feature_method})")
                                    
                                    # Display feature summary
                                    if st.session_state.features_type == "baseline":
                                        st.write(f"**Extracted {len(features)} baseline features**")
                                        
                                        # Show top features
                                        with st.expander("📋 Top Feature Values", expanded=False):
                                            top_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
                                            for name, value in top_features:
                                                st.write(f"**{name}:** {value:.6f}")
                                        
                                        # Features visualization
                                        fig, chart_status = Visualizer.create_features_chart(features)
                                        if fig:
                                            st.pyplot(fig)
                                            plt.close()
                                        else:
                                            st.info(chart_status)
                                        
                                    else:
                                        st.write(f"**Extracted ST-GCN features**")
                                        if 'tensor_shape' in features_dict:
                                            st.write(f"Tensor Shape: {features_dict['tensor_shape']}")
                                        
                                        if 'baseline_features' in features_dict:
                                            st.write(f"**Also extracted {len(features_dict['baseline_features'])} baseline features**")
                                    
                                    # Step 3: Model Prediction
                                    st.header("🤖 Step 3: Model Prediction")
                                    
                                    if not st.session_state.available_models:
                                        st.warning("⚠️ No models loaded. Please load models in the sidebar.")
                                    else:
                                        # Model selection
                                        model_choice = st.selectbox(
                                            "Select Model for Prediction:",
                                            st.session_state.available_models,
                                            key="model_choice"
                                        )
                                        
                                        if st.button("🎯 Make Prediction", key="make_prediction", type="primary"):
                                            with st.spinner("Making prediction..."):
                                                prediction_result = None
                                                
                                                if model_choice == "Baseline XGBoost":
                                                    if st.session_state.features_type == "baseline":
                                                        features_array, prep_status = FeatureEngineer.prepare_features_for_baseline(
                                                            st.session_state.features
                                                        )
                                                        
                                                        if features_array is not None:
                                                            prediction_result, pred_status = PredictionEngine.predict_with_baseline(
                                                                st.session_state.baseline_model, features_array
                                                            )
                                                            if prediction_result:
                                                                st.session_state.prediction = prediction_result
                                                                st.success(f"✅ {pred_status}")
                                                                st.session_state.processing_history.append("Baseline prediction completed")
                                                            else:
                                                                st.error(f"❌ {pred_status}")
                                                        else:
                                                            st.error(f"❌ {prep_status}")
                                                    else:
                                                        st.error("❌ Baseline features required for baseline model")
                                                
                                                elif model_choice == "Binary ST-GCN":
                                                    if st.session_state.features_type == "stgcn":
                                                        stgcn_tensor, prep_status = FeatureEngineer.prepare_features_for_stgcn(
                                                            st.session_state.features
                                                        )
                                                        
                                                        if stgcn_tensor is not None:
                                                            prediction_result, pred_status = PredictionEngine.predict_with_binary(
                                                                st.session_state.binary_model, stgcn_tensor
                                                            )
                                                            if prediction_result:
                                                                st.session_state.prediction = prediction_result
                                                                st.success(f"✅ {pred_status}")
                                                                st.session_state.processing_history.append("Binary ST-GCN prediction completed")
                                                            else:
                                                                st.error(f"❌ {pred_status}")
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
                                                            if prediction_result:
                                                                st.session_state.prediction = prediction_result
                                                                st.success(f"✅ {pred_status}")
                                                                st.session_state.processing_history.append("Multi-label ST-GCN prediction completed")
                                                            else:
                                                                st.error(f"❌ {pred_status}")
                                                        else:
                                                            st.error(f"❌ {prep_status}")
                                                    else:
                                                        st.error("❌ ST-GCN features required for multi-label model")
                                                
                                                # Display results if prediction was successful
                                                if prediction_result:
                                                    st.header("📊 Prediction Results")
                                                    
                                                    # Main metrics
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        st.metric("Prediction", prediction_result['label'])
                                                    with col2:
                                                        st.metric("Confidence", f"{prediction_result['confidence']:.2%}")
                                                    with col3:
                                                        st.metric("Model", model_choice)
                                                    
                                                    # Probability chart
                                                    fig, chart_status = Visualizer.create_prediction_chart(prediction_result)
                                                    if fig:
                                                        st.pyplot(fig)
                                                        plt.close()
                                                    else:
                                                        st.info(chart_status)
                                                    
                                                    # Step 4: Download Results
                                                    st.header("💾 Step 4: Download Results")
                                                    
                                                    # Save results
                                                    features_to_save = st.session_state.features if st.session_state.features_type == "baseline" else st.session_state.features.get('baseline_features', {})
                                                    results_path, results_df = FileManager.save_results(
                                                        prediction_result, 
                                                        features_to_save,
                                                        file_info,
                                                        prediction_result['model_type']
                                                    )
                                                    
                                                    if results_path and results_df is not None:
                                                        # Create download button
                                                        csv = results_df.to_csv(index=False)
                                                        st.download_button(
                                                            label="📥 Download Complete Results (CSV)",
                                                            data=csv,
                                                            file_name=results_path.name,
                                                            mime="text/csv",
                                                            type="primary"
                                                        )
                                                        st.success(f"✅ Results saved to {results_path.name}")
                                                    
                                                    # Processing summary
                                                    st.header("📈 Processing Summary")
                                                    
                                                    summary_data = {
                                                        'File Name': file_info['original_name'],
                                                        'File Size (KB)': f"{file_info['size']/1024:.1f}",
                                                        'FPS Used': fps,
                                                        'Feature Method': feature_method,
                                                        'Features Extracted': len(st.session_state.features) if st.session_state.features_type == "baseline" else len(st.session_state.features.get('baseline_features', {})),
                                                        'Prediction': prediction_result['label'],
                                                        'Confidence': f"{prediction_result['confidence']:.2%}",
                                                        'Model Used': model_choice,
                                                        'Models Available': ', '.join(st.session_state.available_models)
                                                    }
                                                    
                                                    st.table(pd.DataFrame(list(summary_data.items()), 
                                                                     columns=['Metric', 'Value']))
                                                else:
                                                    st.error("❌ Prediction failed. Check logs for details.")
                                else:
                                    st.error(f"❌ Feature extraction failed: {status}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    logger.error(f"File processing error: {e}")
                    logger.error(traceback.format_exc())
        
        # Instructions section
        st.write("---")
        st.header("📖 Instructions & Information")
        
        with st.expander("📋 How to Use", expanded=True):
            st.markdown("""
            1. **Load Models**: Click the buttons in the sidebar to load the models you want to use
            2. **Upload CSV**: Upload a CSV file with MediaPipe pose landmarks
            3. **Set FPS**: Enter the frame rate of your video (default: 30)
            4. **Extract Features**: Choose feature extraction method and extract features
            5. **Make Prediction**: Select a model and get predictions
            
            **CSV Format Requirements:**
            - Columns: `frame`, `landmark_id`, `x_norm`, `y_norm`, `z_norm`
            - Frame numbers should be integers
            - Landmark IDs: 0-32 (MediaPipe pose landmarks)
            - Coordinates should be normalized (0-1 range)
            """)
        
        with st.expander("🔧 Troubleshooting", expanded=False):
            st.markdown("""
            **Common Issues:**
            
            1. **Model Loading Failed**
               - Check model files exist in correct locations
               - Ensure PyTorch/XGBoost are installed
               - Check file permissions
            
            2. **Feature Extraction Failed**
               - Verify CSV has required columns
               - Check landmark IDs are 0-32
               - Ensure data is not empty
            
            3. **Prediction Failed**
               - Match feature type with model type
               - Check feature dimensions
               - Ensure models are loaded
            """)
        
        with st.expander("📊 Feature Information", expanded=False):
            st.markdown("""
            **Baseline Model Features (82 features):**
            
            *Spatial Features*
            - Step height (vertical ankle range)
            - Step length (horizontal ankle range)
            - Pelvic drop (hip height asymmetry)
            - Trunk lean (shoulder asymmetry)
            - Heel clearance (vertical heel range)
            
            *Temporal Features*
            - Step time and cadence
            - Stride time variability
            - Temporal asymmetries
            
            *Joint Features*
            - Range of motion (ROM)
            - Joint angles (hip, knee, ankle)
            - Movement fractions
            - Asymmetry indices
            
            **Total: 82 features matching training pipeline**
            """)
    
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        
        with st.expander("🐛 Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()