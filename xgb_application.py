#!/usr/bin/env python3
"""
MEDIAPIPE POSE DETECTION & MODELLING PIPELINE - PRODUCTION GRADE (FINAL FIX)
Fixed: Video Loading, UI Visibility, Config Update, Logging
"""

import os
import sys
import warnings
import logging
import time
import traceback
import hashlib
from io import BytesIO
from typing import Optional, Dict, Tuple, List, Any, Callable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Environment configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

# Third-party imports
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import json
import subprocess
import zipfile
import importlib.util
import glob

# Scientific imports
from scipy.signal import find_peaks, resample
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Polygon, Patch, Circle
from mpl_toolkits.mplot3d import Axes3D

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING & DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    """Configure intensive logging for application."""
    log_file = Path("gait_app.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def log_execution(func: Callable) -> Callable:
    """Decorator to log function execution start, end, and exceptions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"START: {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"END: {func_name} | Success")
            return result
        except Exception as e:
            logger.error(f"ERROR: {func_name} | {str(e)}")
            logger.error(traceback.format_exc())
            raise e
    return wrapper

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION & PATHS
# ═════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.absolute()
CONFIG_PATH = PROJECT_ROOT / "config.json"
MEDIAPIPE_SCRIPT = PROJECT_ROOT / "pre-processing-models" / "mediapipe" / "pre_mediapipe.py"

# Directories
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
GAIT_CYCLES_DIR = PROJECT_ROOT / "data" / "gait_cycles"
MODELS_DIR = PROJECT_ROOT / "models" / "baseline"

# Model Paths
BINARY_MODEL_PATH = MODELS_DIR / "xgboost_model.bin"
BINARY_FEATURES_PATH = MODELS_DIR / "feature_names.json"
MULTICLASS_MODEL_PATH = MODELS_DIR / "xgboost_gait_5class.bin"
MULTICLASS_METADATA_PATH = MODELS_DIR / "xgboost_gait_5class_metadata.json"

for directory in [UPLOAD_DIR, OUTPUT_DIR, FEATURES_DIR, GAIT_CYCLES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════
# CORE LOGIC CLASSES
# ═════════════════════════════════════════════════════════════════════════

class VideoConverter:
    @staticmethod
    def check_ffmpeg() -> bool:
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5, text=True)
            return result.returncode == 0
        except: return False
    
    @staticmethod
    def get_video_codec(video_path: Path) -> Optional[str]:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            cap.release()
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            return codec.strip()
        except: return None
    
    @staticmethod
    def is_web_compatible(codec: str) -> bool:
        if not codec: return False
        return any(c in codec.upper() for c in ['AVC1', 'H264', 'X264'])
    
    @staticmethod
    def convert_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
        try:
            cmd = ['ffmpeg', '-i', str(input_path), '-c:v', 'libx264', '-preset', 'medium',
                    '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    '-c:a', 'aac', '-b:a', '128k', '-y', str(output_path)]
            result = subprocess.run(cmd, capture_output=True, timeout=300, text=True)
            return result.returncode == 0 and output_path.exists()
        except: return False

    @staticmethod
    def ensure_web_compatible(video_path: Path) -> Path:
        if not video_path or not video_path.exists(): return video_path
        web_path = video_path.parent / f"{video_path.stem}_h264.mp4"
        if web_path.exists(): return web_path
        
        if VideoConverter.is_web_compatible(VideoConverter.get_video_codec(video_path)):
            return video_path
        
        if VideoConverter.convert_with_ffmpeg(video_path, web_path): return web_path
        return video_path

class FileManager:
    @staticmethod
    def save_uploaded_video(uploaded_file) -> Tuple[Optional[Path], bool]:
        try:
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            clean_name = Path(uploaded_file.name).stem
            ext = Path(uploaded_file.name).suffix
            video_path = UPLOAD_DIR / f"{clean_name}{ext}"
            
            if video_path.exists():
                with open(video_path, 'rb') as f:
                    if hashlib.md5(f.read()).hexdigest() == file_hash:
                        return video_path, True
            
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                video_path.unlink()
                return None, False
            cap.release()
            
            return video_path, False
        except Exception as e:
            logger.error(f"Save video failed: {e}")
            return None, False

    @staticmethod
    def find_output_videos(video_path: Path) -> Dict[str, Optional[Path]]:
        """
        ROBUST SEARCH:
        Instead of checking exact names, we glob for any file containing the video stem.
        This fixes the "Not able to load" issue if names differ slightly.
        """
        results = {'annotated': None, 'skeleton': None, 'csv': None}
        stem = video_path.stem # e.g., "video_name"
        
        # Get all files in output dir
        output_files = list(OUTPUT_DIR.glob("*"))
        
        # Helper to find match
        def find_match(patterns):
            for f in output_files:
                if stem in f.stem: # Fuzzy match: if video name is part of filename
                    for p in patterns:
                        if p in f.stem:
                            return f
            return None

        # Find CSV
        csv_match = find_match(['landmarks'])
        if csv_match: results['csv'] = csv_match
        elif any('landmarks' in f.stem for f in output_files):
             # Fallback for any landmarks file
             results['csv'] = [f for f in output_files if 'landmarks' in f.stem][0]

        # Find Annotated
        ann_match = find_match(['annotated'])
        if ann_match: results['annotated'] = ann_match

        # Find Skeleton
        skel_match = find_match(['skeleton'])
        if skel_match: results['skeleton'] = skel_match

        logger.info(f"File Search Results for '{stem}': {results}")
        return results

class PipelineManager:
    @staticmethod
    def load_config() -> Optional[dict]:
        if not CONFIG_PATH.exists(): return {}
        try:
            with open(CONFIG_PATH, 'r') as f: return json.load(f)
        except: return None
    
    @staticmethod
    def save_config(config: dict) -> bool:
        try:
            with open(CONFIG_PATH, 'w') as f: json.dump(config, f, indent=2)
            return True
        except: return False

    @staticmethod
    def update_config_with_video(video_path: Path) -> bool:
        try:
            config = PipelineManager.load_config()
            config["input_paths"] = [str(video_path)]
            config["output_dir"] = "data/output"
            return PipelineManager.save_config(config)
        except: return False

    @staticmethod
    def run_pipeline():
        """Runs MediaPipe script with robust error handling for DLL/Framework issues."""
        if not MEDIAPIPE_SCRIPT.exists():
            logger.error("MediaPipe script not found")
            st.error("MediaPipe script not found. Check path.")
            return None

        try:
            # Critical Fix: Catch DLL/Framework initialization errors
            spec = importlib.util.spec_from_file_location("mediapipe_module", MEDIAPIPE_SCRIPT)
            if spec is None or spec.loader is None:
                raise ImportError("Could not load MediaPipe module spec")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            logger.info("MediaPipe module loaded successfully")
            
            # Assuming script runs automatically or has a main entry point
            if hasattr(module, 'main'):
                return module.main()
            return {"status": "loaded"}
            
        except ImportError as ie:
            if "DLL load failed" in str(ie) or "_framework_bindings" in str(ie):
                error_msg = (
                    "⚠️ **CRITICAL SYSTEM ERROR:** Failed to load TensorFlow/MediaPipe DLLs.\n\n"
                    "This is a dependency issue, not a code issue.\n"
                    "1. Reinstall TensorFlow: `pip uninstall tensorflow -y && pip install tensorflow`.\n"
                    "2. Reinstall MediaPipe: `pip uninstall mediapipe -y && pip install mediapipe`."
                )
                st.error(error_msg)
                logger.critical(f"DLL/Framework Error: {ie}")
            else:
                st.error(f"Import Error: {ie}")
            return None
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            st.error(f"Pipeline failed: {str(e)}")
            return None

# ═════════════════════════════════════════════════════════════════════════════
# GAIT ANALYSIS ENGINE (PREPROCESSING & FEATURE EXTRACTION)
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureConfig:
    smooth_sigma: float = 1.0
    speed_thresh: float = 0.02
    min_step_time: float = 0.3
    auto_normalize_if_needed: bool = True

@dataclass
class PredictionResult:
    sample_indices: List[int]
    predictions: List[str]
    probabilities: List[float]
    timestamps: List[str]
    details: Optional[pd.DataFrame] = None

class GaitAnalysisEngine:
    """Complete gait analysis engine."""
    
    N_JOINTS = 33
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28
    LEFT_HEEL, RIGHT_HEEL = 29, 30
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32
    
    GAIT_JOINTS = [2, 5, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    GAIT_JOINT_INDEX = {joint_id: i for i, joint_id in enumerate(GAIT_JOINTS)}
    
    ANOMALY_COLS = [
        "gait_anomaly_knee_sagittal_plane_abnormality",
        "gait_anomaly_trunk_balance_abnormality",
        "gait_anomaly_spatiotemporal_asymmetry",
        "gait_anomaly_hip_pelvic_control_deficit",
        "gait_anomaly_distal_foot_control_deficit",
    ]

    @staticmethod
    @log_execution
    def interpolate_pose(pose: np.ndarray) -> np.ndarray:
        T, J, C = pose.shape
        pose_interp = pose.copy()
        for j in range(J):
            for c in range(C):
                coord = pose[:, j, c]
                missing = coord == 0
                if missing.all(): continue
                valid_idx = np.where(~missing)[0]
                pose_interp[:, j, c] = np.interp(np.arange(T), valid_idx, coord[valid_idx])
        return pose_interp

    @staticmethod
    @log_execution
    def normalize_pose_3d(pose: np.ndarray) -> np.ndarray:
        pelvis = (pose[:, GaitAnalysisEngine.LEFT_HIP] + pose[:, GaitAnalysisEngine.RIGHT_HIP]) / 2
        pose_centered = pose - pelvis[:, None, :]
        torso = (pose_centered[:, GaitAnalysisEngine.LEFT_SHOULDER] + pose_centered[:, GaitAnalysisEngine.RIGHT_SHOULDER]) / 2
        scale = np.linalg.norm(torso, axis=1).mean()
        if scale == 0 or not np.isfinite(scale): scale = 1.0
        return pose_centered / scale

    @staticmethod
    @log_execution
    def add_pose_column(df: pd.DataFrame) -> pd.DataFrame:
        pose_rows = []
        meta_cols = ['video_id', 'dataset', 'fps', 'movement_type', 'side', 'gait_markers']
        for video_id in df["video_id"].unique():
            group = df[df["video_id"] == video_id].sort_values("frame")
            frames = np.sort(group["frame"].unique())
            frame_to_idx = {f: i for i, f in enumerate(frames)}
            T = len(frames)
            pose = np.zeros((T, GaitAnalysisEngine.N_JOINTS,3), dtype=np.float32)
            
            for _, row in group.iterrows():
                f_idx = frame_to_idx[row["frame"]]
                j = int(row["landmark_id"])
                if j < GaitAnalysisEngine.N_JOINTS:
                    pose[f_idx, j, :] = [row["x_norm"], row["y_norm"], row["z_norm"]]
            
            pose = GaitAnalysisEngine.interpolate_pose(pose)
            
            base_row = {col: group.iloc[0].get(col) for col in meta_cols if col in group.columns}
            base_row["pose"] = pose
            pose_rows.append(base_row)
        return pd.DataFrame(pose_rows)

    @staticmethod
    @log_execution
    def extract_sliding_windows(pose, fps=30, window_seconds=2.0, overlap=0.5):
        T, J, C = pose.shape
        window_frames = int(window_seconds * fps)
        step_frames = int(window_frames * (1 - overlap))
        if window_frames > T: return [pose]
        windows = []
        start = 0
        while start + window_frames <= T:
            windows.append(pose[start:start + window_frames])
            start += step_frames
        return windows

    @staticmethod
    @log_execution
    def preprocess_gait_sliding_windows(df_video: pd.DataFrame, window_seconds=2.0, overlap=0.5, resample_frames=60):
        all_windows = []
        all_ids = []
        for _, row in df_video.iterrows():
            pose = row.get("pose")
            if pose is None: continue
            pose = np.asarray(pose)
            if pose.ndim != 3: continue
            
            fps = row.get("fps", 30)
            try: pose_norm = GaitAnalysisEngine.normalize_pose_3d(pose)
            except: continue
            
            windows = GaitAnalysisEngine.extract_sliding_windows(pose_norm, fps, window_seconds, overlap)
            for i, w in enumerate(windows):
                if w.shape[0] < 2: continue
                w_resampled = resample(w, resample_frames, axis=0)
                all_windows.append(w_resampled)
                all_ids.append(f"{row['video_id']}_win{i}")
        return np.array(all_windows), all_ids

    @staticmethod
    @log_execution
    def qc_gait_window(window, fps=60):
        L_HIP, R_HIP = GaitAnalysisEngine.LEFT_HIP, GaitAnalysisEngine.RIGHT_HIP
        L_KNEE = GaitAnalysisEngine.LEFT_KNEE
        L_ANKLE = GaitAnalysisEngine.LEFT_ANKLE
        L_SHOULDER = GaitAnalysisEngine.LEFT_SHOULDER
        
        qc = {}
        qc["n_frames"] = window.shape[0]
        
        pelvis = (window[:, L_HIP] + window[:, R_HIP]) / 2
        qc["pelvis_offset"] = np.linalg.norm(pelvis.mean(axis=0))
        qc["flag_off_center"] = qc["pelvis_offset"] > 0.1
        
        torso = window[:, L_SHOULDER] - pelvis
        qc["torso_std"] = np.linalg.norm(torso, axis=1).std()
        qc["flag_unstable"] = qc["torso_std"] > 0.15
        
        ankle_y = window[:, L_ANKLE, 1]
        peaks, _ = find_peaks(ankle_y, distance=int(0.4 * fps))
        qc["n_peaks"] = len(peaks)
        qc["flag_no_periodicity"] = qc["n_peaks"] < 1
        
        qc["qc_fail"] = any([qc["flag_off_center"], qc["flag_unstable"], qc["flag_no_periodicity"]])
        return qc

    @staticmethod
    @log_execution
    def apply_qc_windows(X_windows, window_ids):
        X_clean = []
        ids_clean = []
        for i, window in enumerate(X_windows):
            qc = GaitAnalysisEngine.qc_gait_window(window)
            if not qc["qc_fail"]:
                X_clean.append(window)
                ids_clean.append(window_ids[i])
        return np.array(X_clean), ids_clean

    @staticmethod
    def compute_window_features(window: np.ndarray, fps: float) -> dict:
        # Auto-normalize if needed
        pelvis = (window[:, GaitAnalysisEngine.LEFT_HIP] + window[:, GaitAnalysisEngine.RIGHT_HIP]) / 2
        if np.linalg.norm(pelvis.mean(axis=0)) > 1e-2:
            window = GaitAnalysisEngine.normalize_pose_3d(window)
            
        feats = {}
        def rom(joint, axis): return float(window[:, joint, axis].max() - window[:, joint, axis].min())
        
        # Spatial
        feats["step_height_L"] = rom(GaitAnalysisEngine.LEFT_ANKLE, 1)
        feats["step_height_R"] = rom(GaitAnalysisEngine.RIGHT_ANKLE, 1)
        feats["step_length_L"] = rom(GaitAnalysisEngine.LEFT_ANKLE, 0)
        feats["step_length_R"] = rom(GaitAnalysisEngine.RIGHT_ANKLE, 0)
        
        eps = 1e-6
        hL, hR = feats["step_height_L"], feats["step_height_R"]
        feats["step_height_symmetry"] = (hL - hR) / (hL + hR + eps)
        
        # Temporal
        def get_speed(j):
            traj = gaussian_filter1d(window[:, j, :], sigma=1.0, axis=0)
            return np.linalg.norm(np.diff(traj, axis=0), axis=1) * fps
            
        speed_L = get_speed(GaitAnalysisEngine.LEFT_ANKLE)
        feats["ankle_L_moving_fraction"] = float((speed_L > 0.02).mean())
        
        # Angles
        def angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cosang = np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            
        knee_angle_L = angle(window[:, GaitAnalysisEngine.LEFT_HIP], window[:, GaitAnalysisEngine.LEFT_KNEE], window[:, GaitAnalysisEngine.LEFT_ANKLE])
        feats["knee_angle_L_mean"] = float(knee_angle_L.mean())
        feats["knee_angle_L_rom"] = float(knee_angle_L.max() - knee_angle_L.min())
        
        return feats

    @staticmethod
    @log_execution
    def extract_features_from_windows(X_windows, fps):
        rows = []
        for i in range(len(X_windows)):
            feats = GaitAnalysisEngine.compute_window_features(X_windows[i], fps)
            rows.append(feats)
        return pd.DataFrame(rows)

    @staticmethod
    @log_execution
    def extract_features_from_csv(csv_path: Path) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        df = pd.read_csv(csv_path)
        if 'video_id' not in df.columns: df['video_id'] = 'default'
        df_video = GaitAnalysisEngine.add_pose_column(df)
        X_windows, window_ids = GaitAnalysisEngine.preprocess_gait_sliding_windows(df_video)
        X_clean, ids_clean = GaitAnalysisEngine.apply_qc_windows(X_windows, window_ids)
        
        if len(X_clean) == 0: return pd.DataFrame(), None
        df_features = GaitAnalysisEngine.extract_features_from_windows(X_clean, fps=60)
        df_features['window_id'] = ids_clean
        return df_features, X_clean

# ═════════════════════════════════════════════════════════════════════════════
# MODEL PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ModelPredictor:
    def __init__(self):
        self.binary_model = None
        self.multiclass_model = None
        self.binary_features = None
        self.multiclass_metadata = None
        self.multiclass_classes = []

    @log_execution
    def load_binary_model(self):
        if not BINARY_MODEL_PATH.exists(): return False
        if not BINARY_FEATURES_PATH.exists(): return False
        try:
            self.binary_model = xgb.XGBClassifier()
            self.binary_model.load_model(BINARY_MODEL_PATH)
            with open(BINARY_FEATURES_PATH, 'r') as f: self.binary_features = json.load(f)
            return True
        except: return False

    @log_execution
    def load_multiclass_model(self):
        if not MULTICLASS_MODEL_PATH.exists(): return False
        if not MULTICLASS_METADATA_PATH.exists(): return False
        try:
            self.multiclass_model = xgb.XGBClassifier()
            self.multiclass_model.load_model(MULTICLASS_MODEL_PATH)
            with open(MULTICLASS_METADATA_PATH, 'r') as f: self.multiclass_metadata = json.load(f)
            id_to_class = self.multiclass_metadata.get("id_to_class", {})
            self.multiclass_classes = [id_to_class[str(i)] for i in sorted(id_to_class.keys())]
            return True
        except: return False

    def align_features(self, df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
        missing = set(required_features) - set(df.columns)
        if missing:
            for m in missing: df[m] = 0.0
        return df[required_features]

    @log_execution
    def predict_binary(self, df_features: pd.DataFrame) -> PredictionResult:
        if not self.binary_model or not self.binary_features: return None
        df_aligned = self.align_features(df_features.copy(), self.binary_features)
        df_filled = df_aligned.fillna(df_aligned.median())
        probs = self.binary_model.predict_proba(df_filled)
        preds = self.binary_model.predict(df_filled)
        labels = ["Normal" if p == 0 else "Abnormal" for p in preds]
        confidences = [max(p) for p in probs]
        
        details_df = df_features.copy()
        details_df['prediction'] = labels
        details_df['confidence'] = confidences
        
        return PredictionResult(list(range(len(df_features))), labels, confidences, [datetime.now().isoformat()]*len(df_features), details_df)

    @log_execution
    def predict_multiclass(self, df_features: pd.DataFrame) -> PredictionResult:
        if not self.multiclass_model or not self.multiclass_metadata: return None
        required_features = self.multiclass_metadata.get("feature_cols", [])
        df_aligned = self.align_features(df_features.copy(), required_features)
        df_filled = df_aligned.fillna(df_aligned.median())
        probs = self.multiclass_model.predict_proba(df_filled)
        preds = self.multiclass_model.predict(df_filled)
        labels = [self.multiclass_classes[p] for p in preds]
        confidences = [probs[i, p] for i, p in enumerate(preds)]
        
        details_df = df_features.copy()
        details_df['prediction'] = labels
        details_df['confidence'] = confidences
        return PredictionResult(list(range(len(df_features))), labels, confidences, [datetime.now().isoformat()]*len(df_features), details_df)

# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION MANAGER (RESTORED OLD VISUALIZATIONS)
# ═════════════════════════════════════════════════════════════════════════════════

class GaitVisualizer:
    """Contains all visualization methods from original application."""
    
    @staticmethod
    def create_gait_score_dashboard(features_df):
        fig, ax = plt.subplots(figsize=(12, 8))
        gait_metrics = {
            'Step Symmetry': ('step_height_symmetry', 0.15, 0.25),
            'Knee Flexion': ('knee_angle_L_rom', 30, 60), 
            'Ankle Control': ('ankle_L_moving_fraction', 0.3, 0.7)
        }
        
        positions = [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8)]
        
        for i, (metric_name, (feature_key, min_good, max_good)) in enumerate(gait_metrics.items()):
            x, y = positions[i]
            if feature_key in features_df.columns:
                value = features_df.iloc[0][feature_key]
                
                if min_good <= value <= max_good: color = 'green'
                else: color = 'red'
                
                circle = Circle((x, y), 0.08, color=color, alpha=0.8)
                ax.add_patch(circle)
                ax.text(x, y - 0.15, metric_name, ha='center', fontsize=10, weight='bold')
                ax.text(x, y, f'{value:.2f}', ha='center', va='center', fontsize=9, color='white', weight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Gait Health Assessment Dashboard', fontsize=16, weight='bold')
        ax.axis('off')
        return fig

    @staticmethod
    def create_movement_flow_chart(features_df):
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(figsize=(14, 8))
        phases = [('Initial Contact', 0.1, 0.8), ('Loading Response', 0.3, 0.8)]
        
        box_width = 0.15
        for phase_name, x, y in phases:
            rect = mpatches.Rectangle((x - box_width/2, y - box_width/2), box_width, box_width, 
                                  facecolor='lightblue', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, phase_name, ha='center', va='center', fontsize=9, weight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Gait Cycle Movement Pattern Flow', fontsize=16, weight='bold')
        ax.axis('off')
        return fig

    @staticmethod
    def create_3d_joint_trajectory(gait_cycles):
        if gait_cycles is None or len(gait_cycles) == 0: return None
        fig = plt.figure(figsize=(15, 10))
        
        avg_cycle = np.mean(gait_cycles, axis=0)
        joint_indices = [27, 28, 25, 26, 23, 24] 
        joint_names = ['L Ankle', 'R Ankle', 'L Knee', 'R Knee', 'L Hip', 'R Hip']
        
        views = [(0, 0), (0, 90), (90, 0), (30, 45)]
        view_labels = ['Front View', 'Side View', 'Top View', '3D View']
        
        for idx, ((elev, azim), label) in enumerate(zip(views, view_labels)):
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
            for j_idx, j_name in zip(joint_indices, joint_names):
                trajectory = avg_cycle[:, j_idx, :]
                color = 'blue' if 'L' in j_name else 'red'
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, linewidth=2, label=j_name)
            
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title(label); ax.view_init(elev=elev, azim=azim)
        
        plt.suptitle('3D Joint Trajectories During Gait Cycle', fontsize=16)
        plt.tight_layout()
        return fig

    @staticmethod
    def create_gait_stability_index(features_df):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        stability_metrics = [
            ('Dynamic Balance', 'pelvis_offset', 0.1),
            ('Step Consistency', 'step_height_symmetry', 0.15),
            ('Movement Control', 'ankle_L_moving_fraction', 0.2)
        ]
        
        labels = [m[0] for m in stability_metrics]
        values = []
        for _, f_key, _ in stability_metrics:
            if f_key in features_df.columns: values.append(abs(features_df.iloc[0][f_key]))
            else: values.append(0.5)
            
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=3)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title('Gait Stability Index', size=16)
        return fig

    @staticmethod
    def create_temporal_gait_heatmap(gait_cycles):
        if gait_cycles is None or len(gait_cycles) == 0: return None
        avg_cycle = np.mean(gait_cycles, axis=0)
        joint_data = {
            'Left Ankle': avg_cycle[:, 27, 1], 'Right Ankle': avg_cycle[:, 28, 1],
            'Left Knee': avg_cycle[:, 25, 1], 'Right Knee': avg_cycle[:, 26, 1]
        }
        
        df_heatmap = pd.DataFrame(joint_data)
        fig, ax = plt.subplots(figsize=(14, 8))
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(df_heatmap.T, cmap=cmap, center=0, ax=ax)
        ax.set_title('Temporal Gait Pattern Heatmap', fontsize=16)
        return fig

    @staticmethod
    def create_prediction_charts(result: PredictionResult, model_type="binary"):
        if result is None: return None
        
        if model_type == "binary":
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            labels, counts = np.unique(result.predictions, return_counts=True)
            colors = ['green' if 'Normal' in l else 'red' for l in labels]
            ax[0].bar(labels, counts, color=colors)
            ax[0].set_title("Prediction Distribution")
            
            ax[1].hist(result.probabilities, bins=10, color='skyblue', edgecolor='black')
            ax[1].set_title("Model Confidence")
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            labels, counts = np.unique(result.predictions, return_counts=True)
            ax[0].barh(labels, counts, color='teal')
            ax[0].set_title("Prediction Distribution")
            
            if hasattr(result, 'details') and result.details is not None:
                prob_cols = [c for c in result.details.columns if c.startswith('prob_')]
                if prob_cols:
                    avg_probs = result.details[prob_cols].mean()
                    avg_probs.plot(kind='bar', ax=ax[1], color='coral')
                    ax[1].set_title("Average Class Probability")
                    ax[1].tick_params(axis='x', rotation=45)
            plt.tight_layout()
        return fig

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Robust initialization to prevent KeyError crashes."""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = ModelPredictor()
    
    defaults = {
        'uploaded_video_path': None,
        'processing_complete': False,
        'output_videos': {},
        'features_df': None, 
        'gait_cycles': None,
        'csv_features_df': None, 
        'pred_binary': None,
        'pred_multiclass': None,
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def main():
    # PLATINUM THEME - User Friendly UI
    st.set_page_config(
        page_title="Gait Studio", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    /* Platinum Grey Theme */
    body {
        background-color: #F5F5F7; /* Light Platinum Grey */
        color: #2C3E50; /* Dark Blue/Grey for high readability */
    }
    .stApp {
        background-color: #F5F5F7;
    }
    h1, h2, h3 {
        color: #1F2937; /* Even Darker for headers */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #2C3E50; /* Dark element */
        color: white;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1); /* Slight depth */
    }
    .stDataFrame {
        background-color: white;
        border: 1px solid #e0e0e0;
    }
    .metric-container {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize State First (Crucial Fix)
    init_session_state()
    
    st.title("🚶 Production-Grade Gait Analysis & AI Modelling")
    st.markdown("### Complete Pipeline: Processing, Feature Engineering, Analysis & Prediction")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        st.markdown(f"**Models:** `{MODELS_DIR}`")
        if BINARY_MODEL_PATH.exists(): st.success("✅ Binary Model")
        else: st.warning("⚠️ Binary Model Missing")
        if MULTICLASS_MODEL_PATH.exists(): st.success("✅ Multi-class Model")
        else: st.warning("⚠️ Multi-class Model Missing")
        
        st.markdown("---")
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # NEW: System Logs Viewer in Sidebar
        with st.expander("📋 System Logs (Extensive Logging)"):
            try:
                with open("gait_app.log", "r", encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                    st.text_area("Logs", value=log_content[-2000:], height=400, key="sidebar_logs")
            except:
                st.text("Log file not found yet.")

    # Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs(["📤 Upload", "⚙️ Process", "🎬 Landmarker", "📊 Feature Engineering", "🔬 Detailed Analysis", "🤖 AI Modelling"])

    # TAB 1: UPLOAD
    with t1:
        uploaded_file = st.file_uploader("Upload Gait Video", type=["mp4", "mov", "avi"])
        if uploaded_file:
            video_path, is_dup = FileManager.save_uploaded_video(uploaded_file)
            if video_path:
                st.session_state.uploaded_video_path = video_path
                st.success(f"✅ Video Saved: {video_path.name}")
                if is_dup: st.info("⚠️ Duplicate detected.")
                web_video = VideoConverter.ensure_web_compatible(video_path)
                with open(web_video, 'rb') as vid: st.video(vid.read())

    # TAB 2: PROCESS
    with t2:
        if not st.session_state.uploaded_video_path:
            st.warning("Please upload a video first.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"Ready: {st.session_state.uploaded_video_path.name}")
                # RESTORED: Update Config Button
                if st.button("📝 Update Config", type="primary"):
                    if PipelineManager.update_config_with_video(st.session_state.uploaded_video_path):
                        st.success("Config Updated")
                    else:
                        st.error("Config Update Failed")
            with col2:
                if st.button("▶️ Run Pipeline", type="primary"):
                    with st.spinner("Processing..."):
                        PipelineManager.update_config_with_video(st.session_state.uploaded_video_path)
                        results = PipelineManager.run_pipeline()
                        if results:
                            st.session_state.processing_complete = True
                            # FIX: Force refresh of file list
                            st.session_state.output_videos = FileManager.find_output_videos(st.session_state.uploaded_video_path)
                            st.success("Processing Complete")
                            st.balloons()
                        else:
                            st.error("Pipeline execution returned no results.")

    # TAB 3: LANDMARKER
    with t3:
        if not st.session_state.processing_complete:
            st.warning("Process a video in Tab 2 first.")
        else:
            # NEW: Manual Refresh Button for robustness
            if st.button("🔄 Refresh Output Files"):
                if st.session_state.uploaded_video_path:
                    st.session_state.output_videos = FileManager.find_output_videos(st.session_state.uploaded_video_path)
                    st.rerun()
            
            vids = st.session_state.output_videos
            
            # Debug Info (Hidden unless needed, but helpful for "Not able to load" issues)
            with st.expander("🐛 Debug: File Search Results"):
                st.json(vids)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Annotated Video")
                if vids.get('annotated'):
                    web_vid = VideoConverter.ensure_web_compatible(vids['annotated'])
                    with open(web_vid, 'rb') as f: st.video(f.read())
                else:
                    st.warning("Annotated video not found.")
            with c2:
                st.subheader("Skeleton Video")
                if vids.get('skeleton'):
                    web_vid = VideoConverter.ensure_web_compatible(vids['skeleton'])
                    with open(web_vid, 'rb') as f: st.video(f.read())
                else:
                    st.warning("Skeleton video not found.")
            
            if vids.get('csv'):
                st.subheader("Landmarks CSV")
                df = pd.read_csv(vids['csv'])
                st.dataframe(df.head())
            else:
                st.error("Landmarks CSV not found.")

    # TAB 4: FEATURE ENGINEERING (Restored Sub-tabs)
    with t4:
        subtab1, subtab2 = st.tabs(["From Processed Video", "From CSV Upload"])
        
        # Subtab 1: From Processed Video
        with subtab1:
            if st.session_state.processing_complete:
                csv_path = st.session_state.output_videos.get('csv')
                if csv_path:
                    if st.button("🚀 Extract Features from Video"):
                        with st.spinner("Extracting..."):
                            try:
                                features_df, cycles = GaitAnalysisEngine.extract_features_from_csv(csv_path)
                                if not features_df.empty:
                                    st.session_state.features_df = features_df
                                    st.session_state.gait_cycles = cycles
                                    st.success(f"✅ Extracted {len(features_df)} windows.")
                                else: st.error("❌ Extraction failed.")
                            except Exception as e:
                                st.error(f"Error: {e}")
            else:
                st.warning("Process a video in Tab 2 first.")
            
            if st.session_state.features_df is not None:
                st.dataframe(st.session_state.features_df)

        # Subtab 2: From CSV Upload (Flexibility)
        with subtab2:
            uploaded_csv = st.file_uploader("Upload CSV for Feature Analysis", type=["csv"])
            if uploaded_csv:
                try:
                    df = pd.read_csv(uploaded_csv)
                    st.session_state.csv_features_df = df
                    st.success(f"✅ Loaded CSV: {len(df)} rows")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Invalid CSV: {e}")

    # TAB 5: DETAILED ANALYSIS
    with t5:
        # Use features from Video or CSV if available
        target_df = st.session_state.features_df if st.session_state.features_df is not None else st.session_state.csv_features_df
        cycles = st.session_state.gait_cycles

        if target_df is None:
            st.warning("⚠️ No features available. Extract features in Tab 4 first.")
        else:
            st.success(f"✅ Analyzing {len(target_df)} feature vectors.")
            
            viz_type = st.selectbox(
                "Select Visualization:",
                ["Gait Health Dashboard", "Movement Flow", "3D Trajectories", "Stability Index", "Temporal Heatmap"]
            )
            
            if viz_type == "Gait Health Dashboard":
                fig = GaitVisualizer.create_gait_score_dashboard(target_df)
                st.pyplot(fig)
            elif viz_type == "Movement Flow":
                fig = GaitVisualizer.create_movement_flow_chart(target_df)
                st.pyplot(fig)
            elif viz_type == "3D Trajectories":
                if cycles is not None:
                    fig = GaitVisualizer.create_3d_joint_trajectory(cycles)
                    st.pyplot(fig)
            elif viz_type == "Stability Index":
                fig = GaitVisualizer.create_gait_stability_index(target_df)
                st.pyplot(fig)
            elif viz_type == "Temporal Heatmap":
                if cycles is not None:
                    fig = GaitVisualizer.create_temporal_gait_heatmap(cycles)
                    st.pyplot(fig)

    # TAB 6: AI MODELLING (New)
    with t6:
        st.header("🤖 AI Model Predictions")
        
        # Determine which features to use
        features = st.session_state.features_df
        if features is None: features = st.session_state.csv_features_df
        
        if features is None:
            st.warning("⚠️ No features available. Extract features in Tab 4 first.")
        else:
            # Binary Section
            st.markdown("### 🔵 Binary Classification (Normal vs Abnormal)")
            c1, c2 = st.columns([1, 1])
            
            with c1:
                if st.button("Load Binary Model"):
                    if st.session_state.predictor.load_binary_model(): st.success("Loaded")
            with c2:
                if st.button("Predict Binary"):
                    res = st.session_state.predictor.predict_binary(features)
                    if res:
                        st.session_state.pred_binary = res
                        st.dataframe(res.details)
                        fig = GaitVisualizer.create_prediction_charts(res, "binary")
                        st.pyplot(fig)

            # Multi
            st.markdown("### 🟢 Multi-class Classification (5 Classes)")
            c3, c4 = st.columns([1, 1])
            
            with c3:
                if st.button("Load Multi-class Model"):
                    if st.session_state.predictor.load_multiclass_model(): st.success("Loaded")
            with c4:
                if st.button("Predict Multi-class"):
                    res = st.session_state.predictor.predict_multiclass(features)
                    if res:
                        st.session_state.pred_multiclass = res
                        st.dataframe(res.details)
                        fig = GaitVisualizer.create_prediction_charts(res, "multiclass")
                        st.pyplot(fig)

if __name__ == "__main__":
    main()