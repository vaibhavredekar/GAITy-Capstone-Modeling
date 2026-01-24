#!/usr/bin/env python3
"""
GAITy - Production Grade Gait Analysis Application
Complete pipeline: CSV → Feature Engineering → Model Prediction → Results
Version: 2.0.0
Author: Production AI Systems
"""

import os
import sys
import logging
import traceback
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging() -> logging.Logger:
    """Set up comprehensive logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"gait_{datetime.now().strftime('%Y%m%d')}.log"
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger("gait_analysis")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Environment configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

# ============================================================================
# DEPENDENCY IMPORTS
# ============================================================================

DEPENDENCIES = {
    'streamlit': False, 'pandas': False, 'numpy': False,
    'xgboost': False, 'matplotlib': False, 'scipy': False,
    'torch': False
}

try:
    import streamlit as st
    DEPENDENCIES['streamlit'] = True
    logger.info("Streamlit imported")
except ImportError as e:
    logger.critical(f"Streamlit import failed: {e}")
    sys.exit(1)

st.set_page_config(
    page_title="GAITy - Gait Analysis",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    import pandas as pd
    DEPENDENCIES['pandas'] = True
except ImportError:
    logger.error("Pandas not available")

try:
    import numpy as np
    DEPENDENCIES['numpy'] = True
except ImportError:
    logger.error("NumPy not available")

try:
    import xgboost as xgb
    DEPENDENCIES['xgboost'] = True
except ImportError:
    logger.error("XGBoost not available")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    DEPENDENCIES['matplotlib'] = True
except ImportError:
    logger.error("Matplotlib not available")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEPENDENCIES['torch'] = True
    logger.info(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError:
    logger.error("PyTorch not available")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration."""
    BASELINE_MODEL_PATH = Path("models/baseline/xgboost_model.bin")
    BINARY_MODEL_PATH = Path("models/advance/binary_model_full.bin")
    MULTI_MODEL_PATH = Path("models/advance/multi_label_model_full.bin")
    UPLOAD_DIR = Path("uploads")
    RESULTS_DIR = Path("results")
    
    N_JOINTS = 33
    GAIT_JOINTS = [2, 5, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    STGCN_NUM_JOINTS = 14
    
    ANOMALY_COLS = [
        "knee_sagittal_plane", "trunk_balance", "spatiotemporal_asymmetry",
        "hip_pelvic_control", "distal_foot_control"
    ]
    
    MIN_ROWS = 10
    MIN_FRAMES = 5
    MAX_FILE_SIZE_MB = 100

for directory in [Config.UPLOAD_DIR, Config.RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
    """Initialize session state."""
    defaults = {
        'baseline_model': None, 'binary_model': None, 'multi_model': None,
        'baseline_loaded': False, 'binary_loaded': False, 'multi_loaded': False,
        'features': None, 'features_type': None, 'prediction': None,
        'file_info': None, 'dataframe': None, 'history': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def log_activity(msg: str):
    """Log activity to session history."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append(f"[{timestamp}] {msg}")
    logger.info(msg)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SimpleSTGCN(nn.Module):
    """ST-GCN for gait analysis."""
    def __init__(self, num_joints: int, in_channels: int = 3, out_classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
        self.fc = nn.Linear(128 * num_joints, out_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """Manages model loading."""
    
    @staticmethod
    def load_baseline_model() -> Tuple[Optional[Any], str]:
        try:
            if not DEPENDENCIES['xgboost']:
                return None, "XGBoost not available"
            if not Config.BASELINE_MODEL_PATH.exists():
                return None, f"Model not found: {Config.BASELINE_MODEL_PATH}"
            
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.load_model(str(Config.BASELINE_MODEL_PATH))
            logger.info("Baseline model loaded")
            return model, "Baseline model loaded"
        except Exception as e:
            logger.error(f"Baseline load failed: {e}")
            return None, f"Load failed: {str(e)}"
    
    @staticmethod
    def load_binary_model() -> Tuple[Optional[nn.Module], str]:
        try:
            if not DEPENDENCIES['torch']:
                return None, "PyTorch not available"
            if not Config.BINARY_MODEL_PATH.exists():
                return None, f"Model not found: {Config.BINARY_MODEL_PATH}"
            
            model = SimpleSTGCN(Config.STGCN_NUM_JOINTS, 3, 1)
            state_dict = torch.load(str(Config.BINARY_MODEL_PATH), map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Binary model loaded")
            return model, "Binary model loaded"
        except Exception as e:
            logger.error(f"Binary load failed: {e}")
            return None, f"Load failed: {str(e)}"
    
    @staticmethod
    def load_multi_model() -> Tuple[Optional[nn.Module], str]:
        try:
            if not DEPENDENCIES['torch']:
                return None, "PyTorch not available"
            if not Config.MULTI_MODEL_PATH.exists():
                return None, f"Model not found: {Config.MULTI_MODEL_PATH}"
            
            model = SimpleSTGCN(Config.STGCN_NUM_JOINTS, 3, 5)
            state_dict = torch.load(str(Config.MULTI_MODEL_PATH), map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Multi-label model loaded")
            return model, "Multi-label model loaded"
        except Exception as e:
            logger.error(f"Multi load failed: {e}")
            return None, f"Load failed: {str(e)}"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Feature extraction and validation."""
    
    @staticmethod
    def validate_csv(df: pd.DataFrame) -> List[str]:
        errors = []
        required = ['frame', 'landmark_id', 'x_norm', 'y_norm', 'z_norm']
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")
        if df.empty:
            errors.append("Empty CSV")
        if len(df) < Config.MIN_ROWS:
            errors.append(f"Too few rows: {len(df)}")
        if 'landmark_id' in df.columns:
            invalid = df[~df['landmark_id'].between(0, 32)]['landmark_id'].unique()
            if len(invalid) > 0:
                errors.append(f"Invalid landmark IDs: {invalid.tolist()}")
        if 'frame' in df.columns and df['frame'].nunique() < Config.MIN_FRAMES:
            errors.append(f"Too few frames: {df['frame'].nunique()}")
        return errors
    
    @staticmethod
    def extract_basic_features(df: pd.DataFrame) -> Tuple[Optional[Dict], str]:
        try:
            features = {}
            for coord in ['x_norm', 'y_norm', 'z_norm']:
                if coord not in df.columns:
                    continue
                data = df[coord].dropna()
                features[f"{coord}_mean"] = float(data.mean())
                features[f"{coord}_std"] = float(data.std())
                features[f"{coord}_min"] = float(data.min())
                features[f"{coord}_max"] = float(data.max())
                features[f"{coord}_range"] = float(data.max() - data.min())
                features[f"{coord}_median"] = float(data.median())
            
            if 'frame' in df.columns:
                features['total_frames'] = int(df['frame'].nunique())
                features['frame_range'] = int(df['frame'].max() - df['frame'].min())
            
            if 'landmark_id' in df.columns:
                landmarks = df['landmark_id'].unique()
                key_lms = {
                    'left_shoulder': 11, 'right_shoulder': 12,
                    'left_hip': 23, 'right_hip': 24,
                    'left_knee': 25, 'right_knee': 26
                }
                
                for name, lm_id in key_lms.items():
                    if lm_id not in landmarks:
                        continue
                    lm_data = df[df['landmark_id'] == lm_id]
                    for coord in ['x_norm', 'y_norm', 'z_norm']:
                        if coord in lm_data.columns:
                            data = lm_data[coord].dropna()
                            features[f"{name}_{coord}_mean"] = float(data.mean())
                            features[f"{name}_{coord}_std"] = float(data.std())
            
            logger.info(f"Extracted {len(features)} basic features")
            return features, "Features extracted"
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None, f"Extraction failed: {str(e)}"
    
    @staticmethod
    def extract_stgcn_features(df: pd.DataFrame) -> Tuple[Optional[Dict], str]:
        try:
            basic, status = FeatureEngineer.extract_basic_features(df)
            if not basic:
                return None, status
            
            df_sorted = df.sort_values(['frame', 'landmark_id']).copy()
            frames = np.sort(df_sorted['frame'].unique())
            df_gait = df_sorted[df_sorted['landmark_id'].isin(Config.GAIT_JOINTS)]
            gait_lms = np.sort(df_gait['landmark_id'].unique())
            
            T, J = len(frames), len(gait_lms)
            pose = np.zeros((T, J, 3), dtype=np.float32)
            
            frame_idx = {f: i for i, f in enumerate(frames)}
            lm_idx = {l: i for i, l in enumerate(gait_lms)}
            
            for _, row in df_gait.iterrows():
                try:
                    f_i = frame_idx[row['frame']]
                    l_i = lm_idx[row['landmark_id']]
                    pose[f_i, l_i, :] = [row['x_norm'], row['y_norm'], row['z_norm']]
                except (KeyError, IndexError):
                    continue
            
            # Normalize
            try:
                lh_i = lm_idx.get(23)
                rh_i = lm_idx.get(24)
                if lh_i is not None and rh_i is not None:
                    pelvis = (pose[:, lh_i, :] + pose[:, rh_i, :]) / 2
                else:
                    pelvis = pose.mean(axis=1)
                pose_centered = pose - pelvis[:, None, :]
                
                ls_i = lm_idx.get(11)
                rs_i = lm_idx.get(12)
                if ls_i is not None and rs_i is not None:
                    torso = (pose_centered[:, ls_i, :] + pose_centered[:, rs_i, :]) / 2
                    scale = np.linalg.norm(torso, axis=1).mean()
                    pose_norm = pose_centered / max(scale, 1e-6)
                else:
                    pose_norm = pose_centered
            except:
                pose_norm = pose
            
            tensor = np.transpose(pose_norm, (2, 0, 1))[np.newaxis, ...]
            
            result = {
                'stgcn_tensor': tensor,
                'basic_features': basic,
                'tensor_shape': tensor.shape,
                'num_frames': T,
                'num_joints': J
            }
            logger.info(f"ST-GCN tensor: {tensor.shape}")
            return result, "ST-GCN features extracted"
        except Exception as e:
            logger.error(f"ST-GCN extraction failed: {e}")
            return None, f"Extraction failed: {str(e)}"
    
    @staticmethod
    def prepare_baseline(features: Dict) -> Tuple[Optional[np.ndarray], str]:
        try:
            vec = []
            for coord in ['x_norm', 'y_norm', 'z_norm']:
                for stat in ['mean', 'std', 'min', 'max', 'range', 'median']:
                    vec.append(features.get(f"{coord}_{stat}", 0.0))
            
            vec.extend([features.get('total_frames', 0.0), features.get('frame_range', 0.0)])
            
            for lm in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
                for coord in ['x_norm', 'y_norm', 'z_norm']:
                    for stat in ['mean', 'std']:
                        vec.append(features.get(f"{lm}_{coord}_{stat}", 0.0))
            
            arr = np.array(vec, dtype=np.float32).reshape(1, -1)
            logger.info(f"Baseline array: {arr.shape}")
            return arr, "Features prepared"
        except Exception as e:
            logger.error(f"Baseline prep failed: {e}")
            return None, f"Prep failed: {str(e)}"
    
    @staticmethod
    def prepare_stgcn(feat_dict: Dict) -> Tuple[Optional[np.ndarray], str]:
        try:
            if 'stgcn_tensor' not in feat_dict:
                return None, "No tensor"
            tensor = feat_dict['stgcn_tensor']
            if tensor.ndim == 3:
                tensor = tensor[np.newaxis, ...]
            logger.info(f"ST-GCN prepared: {tensor.shape}")
            return tensor, "Features prepared"
        except Exception as e:
            logger.error(f"ST-GCN prep failed: {e}")
            return None, f"Prep failed: {str(e)}"

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class PredictionEngine:
    """Prediction handling."""
    
    @staticmethod
    def predict_baseline(model, feat_arr: np.ndarray) -> Tuple[Optional[Dict], str]:
        try:
            pred = model.predict(feat_arr)[0]
            probs = model.predict_proba(feat_arr)[0]
            result = {
                'prediction': int(pred),
                'label': 'Normal' if pred == 0 else 'Abnormal',
                'confidence': float(max(probs)),
                'probabilities': {'Normal': float(probs[0]), 'Abnormal': float(probs[1])},
                'timestamp': datetime.now().isoformat(),
                'model_type': 'baseline_xgboost'
            }
            logger.info(f"Baseline: {result['label']} ({result['confidence']:.2%})")
            return result, "Prediction successful"
        except Exception as e:
            logger.error(f"Baseline pred failed: {e}")
            return None, f"Prediction failed: {str(e)}"
    
    @staticmethod
    def predict_binary(model: nn.Module, tensor: np.ndarray) -> Tuple[Optional[Dict], str]:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device).eval()
            t = torch.tensor(tensor, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                out = model(t)
                if out.shape[-1] == 1:
                    out = out.squeeze(-1)
                prob = torch.sigmoid(out).cpu().numpy()[0]
                pred = int(prob > 0.5)
            
            result = {
                'prediction': pred,
                'label': 'Normal' if pred == 0 else 'Abnormal',
                'confidence': float(max(prob, 1-prob)),
                'probabilities': {
                    'Normal': float(1-prob) if pred == 1 else float(prob),
                    'Abnormal': float(prob) if pred == 1 else float(1-prob)
                },
                'timestamp': datetime.now().isoformat(),
                'model_type': 'binary_stgcn'
            }
            logger.info(f"Binary: {result['label']} ({result['confidence']:.2%})")
            return result, "Prediction successful"
        except Exception as e:
            logger.error(f"Binary pred failed: {e}")
            return None, f"Prediction failed: {str(e)}"
    
    @staticmethod
    def predict_multi(model: nn.Module, tensor: np.ndarray) -> Tuple[Optional[Dict], str]:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device).eval()
            t = torch.tensor(tensor, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                out = model(t)
                probs = torch.sigmoid(out).cpu().numpy()[0]
                preds = (probs > 0.5).astype(int)
            
            anomaly_probs = {col: float(prob) for col, prob in zip(Config.ANOMALY_COLS, probs)}
            detected = [col for col, pred in zip(Config.ANOMALY_COLS, preds) if pred]
            
            result = {
                'prediction': preds.tolist(),
                'label': 'Multi-label prediction',
                'confidence': float(np.mean(probs)),
                'probabilities': anomaly_probs,
                'anomaly_detected': bool(any(preds)),
                'detected_anomalies': detected,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'multi_label_stgcn'
            }
            logger.info(f"Multi: {len(detected)} anomalies")
            return result, "Prediction successful"
        except Exception as e:
            logger.error(f"Multi pred failed: {e}")
            return None, f"Prediction failed: {str(e)}"

# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Visualization creation."""
    
    @staticmethod
    def create_prediction_chart(pred: Dict) -> Tuple[Optional[Any], str]:
        try:
            if not DEPENDENCIES['matplotlib']:
                return None, "Matplotlib unavailable"
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if pred['model_type'] == 'multi_label_stgcn':
                probs = pred['probabilities']
                labels = [l.replace('_', ' ').title()[:20] for l in probs.keys()]
                values = list(probs.values())
                colors = ['crimson' if v > 0.5 else 'seagreen' for v in values]
                
                bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.7)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                ax.set_title('Multi-label Anomaly Probabilities')
                ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
                
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            else:
                probs = pred['probabilities']
                labels = list(probs.keys())
                values = list(probs.values())
                colors = ['seagreen' if l == 'Normal' else 'crimson' for l in labels]
                
                bars = ax.bar(labels, values, color=colors, alpha=0.7, width=0.5)
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                ax.set_title(f'Prediction - {pred["model_type"].upper()}')
                
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1%}', ha='center', va='bottom', fontsize=12)
            
            plt.tight_layout()
            return fig, "Chart created"
        except Exception as e:
            logger.error(f"Chart failed: {e}")
            return None, f"Chart failed: {str(e)}"

# ============================================================================
# FILE MANAGEMENT
# ============================================================================

class FileManager:
    """File operations."""
    
    @staticmethod
    def save_file(uploaded) -> Tuple[Optional[Path], Optional[Dict]]:
        try:
            size_mb = uploaded.size / (1024 * 1024)
            if size_mb > Config.MAX_FILE_SIZE_MB:
                return None, None
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in uploaded.name if c.isalnum() or c in '._-')[:50]
            filename = f"pose_{ts}_{safe_name}"
            path = Config.UPLOAD_DIR / filename
            
            with open(path, 'wb') as f:
                f.write(uploaded.getvalue())
            
            info = {
                'original_name': uploaded.name,
                'saved_name': filename,
                'path': str(path),
                'size_mb': size_mb,
                'upload_time': datetime.now().isoformat()
            }
            logger.info(f"File saved: {filename}")
            return path, info
        except Exception as e:
            logger.error(f"File save failed: {e}")
            return None, None
    
    @staticmethod
    def save_results(pred: Dict, feat: Dict, file_info: Dict) -> Tuple[Optional[Path], Optional[pd.DataFrame]]:
        try:
            if not DEPENDENCIES['pandas']:
                return None, None
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{pred['model_type']}_{ts}.csv"
            path = Config.RESULTS_DIR / filename
            
            data = {
                'timestamp': pred['timestamp'],
                'model_type': pred['model_type'],
                'prediction': str(pred.get('prediction')),
                'label': pred.get('label'),
                'confidence': pred.get('confidence'),
                'filename': file_info.get('original_name')
            }
            
            for k, v in pred.get('probabilities', {}).items():
                data[f'prob_{k}'] = v
            
            df = pd.DataFrame([data])
            df.to_csv(path, index=False)
            logger.info(f"Results saved: {filename}")
            return path, df
        except Exception as e:
            logger.error(f"Results save failed: {e}")
            return None, None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application."""
    try:
        init_session_state()
        
        st.markdown("""
        <style>
        .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center; font-weight: bold;}
        .metric-card {padding: 20px; border-radius: 10px; background: #f8f9fa; border: 1px solid #dee2e6;}
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">🚶 GAITy - Gait Analysis System</h1>', unsafe_allow_html=True)
        st.markdown("### Advanced AI-Powered Gait Analysis", unsafe_allow_html=False)
        
        # Sidebar
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            st.subheader("Models")
            
            # Baseline
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Baseline XGBoost**")
            with col2:
                st.write("✅" if st.session_state.baseline_loaded else "❌")
            if not st.session_state.baseline_loaded:
                if st.button("Load", key="lb", use_container_width=True):
                    model, status = ModelManager.load_baseline_model()
                    st.session_state.baseline_model = model
                    st.session_state.baseline_loaded = model is not None
                    log_activity(status)
                    st.rerun()
            
            # Binary
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Binary ST-GCN**")
            with col2:
                st.write("✅" if st.session_state.binary_loaded else "❌")
            if not st.session_state.binary_loaded:
                if st.button("Load", key="lbin", use_container_width=True):
                    model, status = ModelManager.load_binary_model()
                    st.session_state.binary_model = model
                    st.session_state.binary_loaded = model is not None
                    log_activity(status)
                    st.rerun()
            
            # Multi
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Multi-label ST-GCN**")
            with col2:
                st.write("✅" if st.session_state.multi_loaded else "❌")
            if not st.session_state.multi_loaded:
                if st.button("Load", key="lm", use_container_width=True):
                    model, status = ModelManager.load_multi_model()
                    st.session_state.multi_model = model
                    st.session_state.multi_loaded = model is not None
                    log_activity(status)
                    st.rerun()
            
            if st.session_state.history:
                with st.expander("Activity Log"):
                    for item in reversed(st.session_state.history[-10:]):
                        st.text(item)
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📤 Upload", "🔬 Analysis", "📊 Results"])
        
        with tab1:
            st.header("Upload Pose Data")
            uploaded = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded:
                path, info = FileManager.save_file(uploaded)
                if path and info:
                    st.success(f"✅ Uploaded: {info['original_name']}")
                    st.session_state.file_info = info
                    log_activity(f"Uploaded: {info['original_name']}")
                    
                    try:
                        df = pd.read_csv(path)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", f"{len(df):,}")
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            if 'landmark_id' in df.columns:
                                st.metric("Landmarks", df['landmark_id'].nunique())
                        
                        errors = FeatureEngineer.validate_csv(df)
                        if errors:
                            st.error("❌ Validation Errors:")
                            for err in errors:
                                st.write(f"- {err}")
                        else:
                            st.success("✅ Validation passed")
                            with st.expander("Data Preview"):
                                st.dataframe(df.head(20), use_container_width=True)
                            st.session_state.dataframe = df
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with tab2:
            st.header("Feature Extraction & Prediction")
            
            if 'dataframe' not in st.session_state:
                st.info("👈 Upload CSV first")
            else:
                df = st.session_state.dataframe
                
                st.subheader("1️⃣ Extract Features")
                method = st.radio(
                    "Feature method:",
                    ["Basic (Baseline)", "ST-GCN (Advanced)"]
                )
                
                if st.button("🔄 Extract", type="primary"):
                    with st.spinner("Extracting..."):
                        if "Basic" in method:
                            feat, status = FeatureEngineer.extract_basic_features(df)
                            st.session_state.features = feat
                            st.session_state.features_type = "basic"
                        else:
                            feat, status = FeatureEngineer.extract_stgcn_features(df)
                            st.session_state.features = feat
                            st.session_state.features_type = "stgcn"
                        
                        if feat:
                            st.success(f"✅ {status}")
                            log_activity(f"Features: {st.session_state.features_type}")
                        else:
                            st.error(f"❌ {status}")
                
                if st.session_state.features:
                    with st.expander("Features"):
                        if st.session_state.features_type == "basic":
                            st.write(f"Total: {len(st.session_state.features)}")
                            df_f = pd.DataFrame(
                                list(st.session_state.features.items()),
                                columns=['Feature', 'Value']
                            )
                            st.dataframe(df_f.head(20))
                        else:
                            st.write(f"Shape: {st.session_state.features['tensor_shape']}")
                            st.write(f"Frames: {st.session_state.features['num_frames']}")
                    
                    st.divider()
                    st.subheader("2️⃣ Predict")
                    
                    avail = []
                    if st.session_state.baseline_loaded and st.session_state.features_type == "basic":
                        avail.append("Baseline XGBoost")
                    if st.session_state.binary_loaded and st.session_state.features_type == "stgcn":
                        avail.append("Binary ST-GCN")
                    if st.session_state.multi_loaded and st.session_state.features_type == "stgcn":
                        avail.append("Multi-label ST-GCN")
                    
                    if not avail:
                        st.warning("⚠️ No compatible models")
                    
                    choice = st.selectbox("Model:", avail if avail else ["None"])
                    
                    if st.button("🎯 Predict", type="primary", disabled=not avail):
                        with st.spinner("Predicting..."):
                            pred = None
                            
                            if choice == "Baseline XGBoost":
                                arr, _ = FeatureEngineer.prepare_baseline(st.session_state.features)
                                if arr is not None:
                                    pred, status = PredictionEngine.predict_baseline(
                                        st.session_state.baseline_model, arr
                                    )
                            elif choice == "Binary ST-GCN":
                                tens, _ = FeatureEngineer.prepare_stgcn(st.session_state.features)
                                if tens is not None:
                                    pred, status = PredictionEngine.predict_binary(
                                        st.session_state.binary_model, tens
                                    )
                            elif choice == "Multi-label ST-GCN":
                                tens, _ = FeatureEngineer.prepare_stgcn(st.session_state.features)
                                if tens is not None:
                                    pred, status = PredictionEngine.predict_multi(
                                        st.session_state.multi_model, tens
                                    )
                            
                            if pred:
                                st.session_state.prediction = pred
                                st.success(f"✅ {status}")
                                log_activity(f"Prediction: {pred['label']}")
                            else:
                                st.error(f"❌ {status}")
        
        with tab3:
            st.header("Results")
            
            if st.session_state.prediction is None:
                st.info("👈 Make a prediction first")
            else:
                pred = st.session_state.prediction
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", pred['label'])
                with col2:
                    st.metric("Confidence", f"{pred['confidence']:.1%}")
                with col3:
                    st.metric("Model", pred['model_type'].replace('_', ' ').title())
                
                fig, status = Visualizer.create_prediction_chart(pred)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                
                with st.expander("Details"):
                    st.json(pred)
                
                if st.session_state.file_info:
                    path, df_res = FileManager.save_results(
                        pred,
                        st.session_state.features if st.session_state.features_type == "basic" 
                        else st.session_state.features.get('basic_features', {}),
                        st.session_state.file_info
                    )
                    
                    if path:
                        csv = df_res.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            path.name,
                            "text/csv"
                        )
    
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"App error: {e}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()