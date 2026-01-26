#--------------- Multi-class slightly better improvements ------------##

#!/usr/bin/env python3
"""
MEDIAPIPE POSE DETECTION & MODELLING PIPELINE - PRODUCTION GRADE (ALL-IN-ONE)
Includes: Processing, Feature Engineering, Visualization, and AI Modelling.
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
import cv2

# Scientific imports
from scipy.signal import find_peaks, resample
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Polygon, Patch, Circle
from mpl_toolkits.mplot3d import Axes3D

# ═════════════════════════════════════════════════════════════════════════════
# LOGGING & DECORATORS
# ═════════════════════════════════════════════════════════════════════════════════

def setup_logging():
    """Configure intensive logging for application."""
    # Force UTF-8 encoding for logs to handle special characters on Windows
    # If this still fails on Windows CMD, the code below handles encoding errors gracefully
    try:
        log_file = Path("gait_app.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                # StreamHandler might still fail on Windows CMD if it uses cp1252
                # We will use a wrapper for stream output or rely on file logs
            ]
        )
    except Exception as e:
        print(f"Warning: Could not set up UTF-8 logging: {e}")
        logging.basicConfig(level=logging.INFO)
        
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

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & PATHS
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═════════════════════════════════════════════════════════════════════════════════
# CORE LOGIC CLASSES
# ══════════════════════════════════════════════════════════════════════════════════

class VideoConverter:
    """Production-grade video converter with multiple fallback strategies"""
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    @staticmethod
    def get_video_codec(video_path: Path) -> Optional[str]:
        """Get codec information from video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened(): return None
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            cap.release()
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            return codec.strip()
        except Exception as e:
            logger.error(f"Failed to get codec: {e}")
            return None
    
    @staticmethod
    def is_web_compatible(codec: str) -> bool:
        """Check if codec is web browser compatible"""
        if not codec: return False
        return any(c in codec.upper() for c in ['AVC1', 'H264', 'X264'])
    
    @staticmethod
    def convert_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
        """Convert video using FFmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-c:v', 'libx264', '-preset', 'medium',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                '-c:a', 'aac', '-b:a', '128k', '-y', str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300, text=True)
            
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"FFmpeg conversion successful: {output_path.name}")
                return True
            else:
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"FFmpeg error: {e}")
            return False
    
    @staticmethod
    def convert_with_opencv(input_path: Path, output_path: Path) -> bool:
        """Fallback: Convert using OpenCV"""
        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                return False
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Codec options
            codec_options = [('avc1', 'H.264'), ('H264', 'H.264'), ('X264', 'X264'), ('mp4v', 'MPEG-4')]
            
            for codec_str, codec_name in codec_options:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_str)
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    
                    if not out.isOpened():
                        continue
                    
                    logger.info(f"Using {codec_name} for conversion")
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                    
                    cap.release()
                    out.release()
                    
                    if output_path.exists() and output_path.stat().st_size > 0:
                        logger.info(f"OpenCV conversion successful with {codec_name}")
                        return True
                except Exception:
                    continue
            
            cap.release()
            return False
        except Exception as e:
            logger.error(f"OpenCV error: {e}")
            return False
    
    @classmethod
    def ensure_web_compatible(cls, video_path: Path) -> Path:
        """Ensure video is web-compatible, convert if necessary"""
        if not video_path or not video_path.exists(): return video_path
        
        web_path = video_path.parent / f"{video_path.stem}_h264.mp4"
        if web_path.exists() and web_path.stat().st_size > 0:
            logger.info(f"Using cached H.264: {web_path.name}")
            return web_path
        
        codec = cls.get_video_codec(video_path)
        logger.info(f"Video codec: {codec}")
        
        if cls.is_web_compatible(codec):
            logger.info(f"Already compatible: {video_path.name}")
            return video_path
        
        logger.warning(f"Converting from {codec} to H.264")
        
        if cls.check_ffmpeg():
            if cls.convert_with_ffmpeg(video_path, web_path): return web_path
        
        if cls.convert_with_opencv(video_path, web_path): return web_path
        
        logger.error(f"All conversions failed for {video_path.name}")
        return video_path

class FileManager:
    """Production-grade file management"""
    
    @staticmethod
    def get_file_hash(file_bytes: bytes) -> str:
        return hashlib.md5(file_bytes).hexdigest()
    
    @staticmethod
    def cleanup_old_uploads(keep_latest: int = 1) -> None:
        try:
            uploads = sorted(UPLOAD_DIR.glob("*.*"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_file in uploads[keep_latest:]:
                try:
                    old_file.unlink()
                    logger.info(f"Deleted: {old_file.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {old_file.name}: {e}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    @staticmethod
    def cleanup_old_outputs(video_stem: str) -> None:
        try:
            patterns = [
                f"*{video_stem}*_annotated*.mp4",
                f"*{video_stem}*_skeleton*.mp4",
                f"*{video_stem}*_landmarks*.csv",
                f"*{video_stem}*_h264*.mp4",
            ]
            for pattern in patterns:
                for old_file in OUTPUT_DIR.glob(pattern):
                    try:
                        old_file.unlink()
                        logger.info(f"Deleted: {old_file.name}")
                    except Exception as e:
                        logger.warning(f"Could not delete {old_file.name}: {e}")
        except Exception as e:
            logger.error(f"Output cleanup failed: {e}")
    
    @staticmethod
    def save_uploaded_video(uploaded_file) -> Tuple[Optional[Path], bool]:
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension not in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']:
            logger.error(f"Unsupported: {file_extension}")
            return None, False
        
        file_bytes = uploaded_file.getvalue()
        file_hash = FileManager.get_file_hash(file_bytes)
        
        for existing_file in UPLOAD_DIR.glob(f"*{file_extension}"):
            try:
                with open(existing_file, 'rb') as f:
                    if FileManager.get_file_hash(f.read()) == file_hash:
                        logger.info(f"Duplicate: {existing_file.name}")
                        return existing_file, True
            except:
                pass
        
        FileManager.cleanup_old_uploads(keep_latest=0)
        
        clean_name = Path(uploaded_file.name).stem
        video_path = UPLOAD_DIR / f"{clean_name}{file_extension}"
        
        if video_path.exists():
            video_path.unlink()
        
        FileManager.cleanup_old_outputs(clean_name)
        
        with open(video_path, 'wb') as f:
            f.write(file_bytes)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error("Invalid video")
                video_path.unlink()
                return None, False
            cap.release()
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            if video_path.exists():
                video_path.unlink()
            return None, False
        
        logger.info(f"Saved: {video_path.name}")
        return video_path, False
    
    @staticmethod
    def find_output_videos(video_path: Path) -> Dict[str, Optional[Path]]:
        """
        ROBUST SEARCH:
        Instead of checking exact names, we glob for any file containing the video stem.
        """
        results = {'annotated': None, 'skeleton': None, 'csv': None}
        stem = video_path.stem 
        
        # Normalize stem if it has timestamp
        if '_' in stem:
            parts = stem.split('_')
            if len(parts[0]) == 8 and parts[0].isdigit():
                if len(parts) > 2:
                    stem = '_'.join(parts[2:])
                elif len(parts) > 1:
                    stem = parts[1]
        
        logger.info(f"Searching for outputs related to: {stem}")
        
        candidates = {
            'annotated': [OUTPUT_DIR / f"{stem}_annotated.mp4", OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"],
            'skeleton': [OUTPUT_DIR / f"{stem}_skeleton.mp4", OUTPUT_DIR / f"{video_path.stem}_skeleton.mp4"],
            'csv': [OUTPUT_DIR / f"{stem}_landmarks.csv", OUTPUT_DIR / f"{video_path.stem}_landmarks.csv"]
        }
        
        for key, paths in candidates.items():
            for candidate in paths:
                if candidate.exists() and candidate.stat().st_size > 0:
                    results[key] = candidate
                    logger.info(f"Found {key}: {candidate.name}")
                    break
        
        # Fuzzy search
        for file in OUTPUT_DIR.iterdir():
            name_lower = file.name.lower()
            stem_lower = stem.lower()
            
            if stem_lower in name_lower or video_path.stem.lower() in name_lower:
                if 'annotated' in name_lower and file.suffix == '.mp4' and not results['annotated']:
                    results['annotated'] = file
                elif 'skeleton' in name_lower and file.suffix == '.mp4' and not results['skeleton']:
                    results['skeleton'] = file
                elif 'landmark' in name_lower and file.suffix == '.csv' and not results['csv']:
                    results['csv'] = file
        
        # Last resort: Most recent files
        if not results['annotated']:
            annotated = sorted(OUTPUT_DIR.glob("*annotated*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
            if annotated: results['annotated'] = annotated[0]
        
        if not results['skeleton']:
            skeleton = sorted(OUTPUT_DIR.glob("*skeleton*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
            if skeleton: results['skeleton'] = skeleton[0]
        
        if not results['csv']:
            csv_files = sorted(OUTPUT_DIR.glob("*landmark*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
            if csv_files: results['csv'] = csv_files[0]
        
        return results

class PipelineManager:
    
    @staticmethod
    def load_config() -> Optional[dict]:
        if not CONFIG_PATH.exists():
            return {}
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Load config failed: {e}")
            return {}
    
    @staticmethod
    def save_config(config: dict) -> bool:
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Save config failed: {e}")
            return False
    
    @staticmethod
    def update_config_with_video(video_path: Path) -> bool:
        config = PipelineManager.load_config()
        try:
            rel_path = str(video_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_path = str(video_path)
        
        config["input_paths"] = [rel_path]
        config["output_dir"] = "data/output"
        return PipelineManager.save_config(config)
    
    @staticmethod
    def load_mediapipe_module():
        if not MEDIAPIPE_SCRIPT.exists():
            logger.error(f"Script not found: {MEDIAPIPE_SCRIPT}")
            return None
        
        try:
            spec = importlib.util.spec_from_file_location("mediapipe_module", MEDIAPIPE_SCRIPT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info("MediaPipe loaded")
            return module
        except Exception as e:
            logger.error(f"Load module failed: {e}")
            return None
    
    @staticmethod
    def run_pipeline() -> Optional[list]:
        mp_module = PipelineManager.load_mediapipe_module()
        if not mp_module: return None
        
        try:
            # Assuming external script follows a specific interface
            config = mp_module.PipelineConfig.from_json(CONFIG_PATH)
            pipeline = mp_module.PoseDetectionPipeline(config)
            results = pipeline.run()
            logger.info(f"Pipeline complete: {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return None

class VideoDisplay:
    
    @staticmethod
    def get_video_info(video_path: Path) -> Optional[dict]:
        try:
            cap = cv2.VideoCapture(str(video_path))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)]).strip()
            
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS) or 30,
                'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 30),
                'size_mb': video_path.stat().st_size / (1024 * 1024),
                'codec': codec
            }
            cap.release()
            return info
        except Exception as e:
            return None
    
    @staticmethod
    def display_video_with_download(video_path: Optional[Path], label: str, key_suffix: str) -> None:
        st.markdown(f"**{label}**")
        
        if not video_path or not video_path.exists():
            st.warning("⚠️ Video not found")
            return
        
        try:
            web_video = VideoConverter.ensure_web_compatible(video_path)
            
            with open(web_video, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                info = VideoDisplay.get_video_info(web_video)
                if info:
                    codec_status = "✅" if VideoConverter.is_web_compatible(info['codec']) else "⚠️"
                    st.caption(f"📊 {info['width']}×{info['height']} | {info['fps']:.0f} FPS | {info['duration']:.1f}s | {info['size_mb']:.1f} MB | {codec_status} {info['codec']}")
            
            with col2:
                file_data_key = f"data_{key_suffix}_{web_video.stem}_{web_video.stat().st_mtime}"
                widget_key = f"widget_{key_suffix}_{web_video.stem}_{web_video.stat().st_mtime}"
                
                if file_data_key not in st.session_state:
                    try:
                        with open(web_video, 'rb') as f:
                            st.session_state[file_data_key] = f.read()
                    except Exception:
                        st.session_state[file_data_key] = None
                
                if st.session_state[file_data_key]:
                    st.download_button(
                        "📥 Download",
                        data=st.session_state[file_data_key],
                        file_name=web_video.name,
                        mime="video/mp4",
                        key=widget_key,
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Display error: {str(e)}")

class ExportManager:
    
    @staticmethod
    def create_results_zip(video_path: Path, output_files: dict) -> Optional[BytesIO]:
        try:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                if video_path and video_path.exists():
                    zf.write(video_path, f"original/{video_path.name}")
                
                for key, path in output_files.items():
                    if path and path.exists():
                        folder = "data" if key == 'csv' else "videos"
                        zf.write(path, f"{folder}/{path.name}")
                
                metadata = {
                    'generated_at': datetime.now().isoformat(),
                    'original_video': video_path.name if video_path else None,
                    'files': {key: path.name if path else None for key, path in output_files.items()}
                }
                zf.writestr('metadata.json', json.dumps(metadata, indent=2))
            
            zip_buffer.seek(0)
            return zip_buffer
        except Exception as e:
            logger.error(f"ZIP failed: {e}")
            return None

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PREDICTOR (FIXED ENCODING & CLASS MISMATCH)
# ═══════════════════════════════════════════════════════════════════════════════

class ModelPredictor:
    """Handles loading and predicting with XGBoost models."""
    
    def __init__(self):
        self.binary_model = None
        self.multiclass_model = None
        self.binary_features = None
        self.multiclass_metadata = None
        self.multiclass_classes = []
        
    def load_binary_model(self):
        if not BINARY_MODEL_PATH.exists() or not BINARY_FEATURES_PATH.exists():
            raise FileNotFoundError("Binary model or metadata files missing.")
        try:
            self.binary_model = xgb.XGBClassifier()
            self.binary_model.load_model(BINARY_MODEL_PATH)
            with open(BINARY_FEATURES_PATH, 'r') as f: 
                self.binary_features = json.load(f)
            logger.info("Binary model loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load binary model: {e}")
            raise e

    def load_multiclass_model(self):
        if not MULTICLASS_MODEL_PATH.exists() or not MULTICLASS_METADATA_PATH.exists():
            raise FileNotFoundError("Multiclass model or metadata files missing.")
        try:
            self.multiclass_model = xgb.XGBClassifier()
            self.multiclass_model.load_model(MULTICLASS_MODEL_PATH)
            with open(MULTICLASS_METADATA_PATH, 'r') as f: 
                self.multiclass_metadata = json.load(f)
            id_to_class = self.multiclass_metadata.get("id_to_class", {})
            # Initialize class list from metadata
            self.multiclass_classes = [id_to_class[str(i)] for i in sorted(id_to_class.keys())]
            logger.info(f"Multiclass model loaded. Metadata claims {len(self.multiclass_classes)} classes.")
            return True
        except Exception as e:
            logger.error(f"Failed to load multiclass model: {e}")
            raise e

    def align_features(self, df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
        """Aligns dataframe columns to match model requirements, filling missing with 0."""
        missing = set(required_features) - set(df.columns)
        if missing:
            # Safe log message (no emojis)
            logger.warning(f"Adding {len(missing)} missing features with 0.0: {list(missing)[:5]}...")
            for m in missing: df[m] = 0.0
        return df[required_features]

    def predict_binary(self, df_features: pd.DataFrame):
        if not self.binary_model: raise ValueError("Binary model not loaded.")
        
        df_aligned = self.align_features(df_features.copy(), self.binary_features)
        df_filled = df_aligned.fillna(df_aligned.median())
        
        probs = self.binary_model.predict_proba(df_filled)
        preds = self.binary_model.predict(df_filled)
        
        labels = ["Normal" if p == 0 else "Abnormal" for p in preds]
        confidences = [max(p) for p in probs]
        
        details_df = df_features.copy()
        details_df['prediction'] = labels
        details_df['confidence'] = confidences
        details_df['prob_normal'] = probs[:, 0]
        details_df['prob_abnormal'] = probs[:, 1]
        return details_df

    def predict_multiclass(self, df_features: pd.DataFrame):
        if not self.multiclass_model: raise ValueError("Multiclass model not loaded.")
            
        required_features = self.multiclass_metadata.get("feature_cols", [])
        df_aligned = self.align_features(df_features.copy(), required_features)
        df_filled = df_aligned.fillna(df_aligned.median())
        
        probs = self.multiclass_model.predict_proba(df_filled)
        preds = self.multiclass_model.predict(df_filled)
        
        # --- FIX FOR INDEX ERROR & ENCODING ---
        # Detect the actual number of classes output by the model
        actual_n_classes = probs.shape[1]
        expected_n_classes = len(self.multiclass_classes)
        
        if actual_n_classes != expected_n_classes:
            # CRITICAL FIX: Do NOT use emojis in logger.warning() to avoid Windows UnicodeEncodeError
            # Use ASCII characters for logs, Emojis only for Streamlit UI
            log_msg = f"Model Mismatch: Model output contains {actual_n_classes} classes, but metadata lists {expected_n_classes}. Adjusting to {actual_n_classes}."
            ui_msg = f"⚠️ **Model Mismatch**: Model output contains {actual_n_classes} classes, but metadata lists {expected_n_classes}. Adjusting to {actual_n_classes}."
            
            logger.warning(log_msg)
            st.warning(ui_msg)
            
            # Slice the class list to match the model's actual output
            self.multiclass_classes = self.multiclass_classes[:actual_n_classes]
        
        labels = []
        confidences = []
        
        # Map predictions safely
        for i, p in enumerate(preds):
            if p < len(self.multiclass_classes):
                labels.append(self.multiclass_classes[p])
                confidences.append(probs[i, p])
            else:
                labels.append("Unknown")
                confidences.append(0.0)
        
        details_df = df_features.copy()
        details_df['prediction'] = labels
        details_df['confidence'] = confidences
        
        # Add probability columns dynamically based on actual output
        for i in range(actual_n_classes):
            cls_name = self.multiclass_classes[i] if i < len(self.multiclass_classes) else f"Class_{i}"
            details_df[f'prob_{cls_name}'] = probs[:, i]
            
        return details_df

# ═══════════════════════════════════════════════════════════════════════════════
# GAIT ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class GaitAnalysisEngine:
    """Complete gait analysis engine"""
    
    N_JOINTS = 33
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28
    LEFT_HEEL, RIGHT_HEEL = 29, 30
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32
    
    LABEL_MAP = {"normal": 0, "abnormal": 1}
    ANOMALY_COLS = [
        "gait_anomaly_knee_sagittal_plane_abnormality",
        "gait_anomaly_trunk_balance_abnormality",
        "gait_anomaly_spatiotemporal_asymmetry",
        "gait_anomaly_hip_pelvic_control_deficit",
        "gait_anomaly_distal_foot_control_deficit",
    ]
    
    @staticmethod
    def normalize_pose_3d(pose):
        pelvis = (pose[:, GaitAnalysisEngine.LEFT_HIP] + pose[:, GaitAnalysisEngine.RIGHT_HIP]) / 2
        pose_centered = pose - pelvis[:, None, :]
        torso = (pose_centered[:, GaitAnalysisEngine.LEFT_SHOULDER] + pose_centered[:, GaitAnalysisEngine.RIGHT_SHOULDER]) / 2
        scale = np.linalg.norm(torso, axis=1).mean()
        if scale < 1e-6: scale = 1.0
        return pose_centered / scale
    
    @staticmethod
    def interpolate_pose(pose):
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
    def add_pose_column(df):
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
    def preprocess_gait_sliding_windows(df_video, window_seconds=2.0, overlap=0.5, resample_frames=60):
        all_windows = []
        all_binary_labels = []
        all_multilabels = []
        all_window_ids = []

        for idx, row in df_video.iterrows():
            pose = row.get("pose")
            if pose is None: continue
            pose = np.asarray(pose)
            if pose.size == 0 or pose.ndim != 3: continue

            fps_val = row.get("fps", 30) or 30
            label_str = str(row.get("dataset", "none")).strip().lower()
            binary_label = GaitAnalysisEngine.LABEL_MAP.get(label_str, 0)
            multilabel = np.array([int(row.get(col, 0)) for col in GaitAnalysisEngine.ANOMALY_COLS], dtype=np.int32)

            try:
                pose_norm = GaitAnalysisEngine.normalize_pose_3d(pose)
            except Exception:
                continue

            windows = GaitAnalysisEngine.extract_sliding_windows(pose_norm, fps=fps_val, window_seconds=window_seconds, overlap=overlap)
            video_id = row.get("video_id", f"vid{idx}")

            for win_idx, w in enumerate(windows):
                if w.shape[0] < 2: continue
                w_resampled = resample(w, resample_frames, axis=0)
                
                start_frame = win_idx * int(window_seconds * fps_val * (1 - overlap))
                end_frame = start_frame + w_resampled.shape[0] - 1
                window_id = f"{video_id}_win{win_idx:03d}_f{start_frame}-{end_frame}"

                all_windows.append(w_resampled)
                all_binary_labels.append(binary_label)
                all_multilabels.append(multilabel)
                all_window_ids.append(window_id)

        return np.asarray(all_windows, dtype=np.float32), \
               np.asarray(all_binary_labels, dtype=np.int32), \
               np.asarray(all_multilabels, dtype=np.int32), \
               all_window_ids

    @staticmethod
    def qc_gait_window(window, fps=60):
        L_HIP, R_HIP = GaitAnalysisEngine.LEFT_HIP, GaitAnalysisEngine.RIGHT_HIP
        L_SHOULDER = GaitAnalysisEngine.LEFT_SHOULDER
        L_KNEE = GaitAnalysisEngine.LEFT_KNEE
        L_ANKLE = GaitAnalysisEngine.LEFT_ANKLE
        qc = {}
        
        qc["n_frames"] = window.shape[0]
        qc["duration_s"] = window.shape[0] / fps
        qc["flag_short"] = qc["duration_s"] < 1.0
        
        pelvis = (window[:, L_HIP] + window[:, R_HIP]) / 2
        qc["pelvis_offset"] = np.linalg.norm(pelvis.mean(axis=0))
        qc["flag_off_center"] = qc["pelvis_offset"] > 0.1
        
        torso = window[:, L_SHOULDER] - pelvis
        torso_std = np.linalg.norm(torso, axis=1).std()
        qc["torso_std_length"] = torso_std
        qc["flag_torso_unstable"] = torso_std > 0.15
        
        ankle_y = window[:, L_ANKLE, 1]
        ankle_y_vel = np.diff(ankle_y)
        qc["ankle_y_velocity_std"] = np.std(ankle_y_vel)
        qc["flag_jitter"] = qc["ankle_y_velocity_std"] > 0.2
        
        knee_y = window[:, L_KNEE, 1]
        min_peak_dist = int(0.4 * fps)
        peaks, _ = find_peaks(knee_y, distance=min_peak_dist)
        qc["n_peaks"] = len(peaks)
        qc["flag_no_periodicity"] = qc["n_peaks"] < 1
        
        ankle_z = window[:, L_ANKLE, 2]
        qc["ankle_z_range"] = ankle_z.max() - ankle_z.min()
        qc["flag_flat_depth"] = qc["ankle_z_range"] < 0.05
        
        qc["qc_fail"] = any([qc["flag_short"], qc["flag_off_center"], qc["flag_jitter"], qc["flag_no_periodicity"], qc["flag_flat_depth"], qc["flag_torso_unstable"]])
        return qc
    
    @staticmethod
    def apply_qc_windows(X_windows, y_binary, y_multilabel, window_ids, fps=60):
        qc_rows = []
        for i, window in enumerate(X_windows):
            qc_rows.append(GaitAnalysisEngine.qc_gait_window(window, fps=fps))
        qc_df = pd.DataFrame(qc_rows)
        qc_df["binary_label"] = y_binary
        qc_df["window_id"] = window_ids
        
        keep_mask = ~qc_df["qc_fail"].values
        return X_windows[keep_mask], y_binary[keep_mask], y_multilabel[keep_mask], np.array(window_ids)[keep_mask], qc_df
    
    @staticmethod
    def joint_speed(pose_norm, joint_idx, fps, smooth_sigma=1.0):
        joint_traj = pose_norm[:, joint_idx, :]
        if smooth_sigma > 0:
            joint_traj = gaussian_filter1d(joint_traj, sigma=smooth_sigma, axis=0)
        diffs = np.diff(joint_traj, axis=0)
        return np.linalg.norm(diffs, axis=1) * fps
    
    @staticmethod
    def moving_and_still_times(pose_norm, joint_idx, fps, speed_thresh=0.02, smooth_sigma=1.0):
        speed = GaitAnalysisEngine.joint_speed(pose_norm, joint_idx, fps, smooth_sigma=smooth_sigma)
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
            "total_time_sec": float(total_time_sec), # Added explicit total_time
        }
    
    @staticmethod
    def range_of_motion(pose_norm, joint_idx, axis=None):
        traj = pose_norm[:, joint_idx, :]
        if axis is None:
            mean_pos = traj.mean(axis=0)
            dist = np.linalg.norm(traj - mean_pos, axis=1)
            rom_3d = dist.max() - dist.min()
            return {"rom_3d": float(rom_3d)}
        axis_to_idx = {"x": 0, "y": 1, "z": 2}
        idx = axis_to_idx[axis]
        coord = traj[:, idx]
        return {f"rom_{axis}": float(coord.max() - coord.min())}

    @staticmethod
    def asymmetry(L: float, R: float, eps: float=1e-6) -> float:
        return float((L - R) / (L + R + eps))

    @staticmethod
    def joint_angle(p_prox, p_joint, p_dist):
        v1 = p_prox - p_joint
        v2 = p_dist - p_joint
        cosang = np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1) + 1e-6)
        angles = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        return angles

    @staticmethod
    def detect_step_events_from_ankle(ankle_y, fps, min_step_time=0.3):
        ankle_y = np.asarray(ankle_y, dtype=float)
        if ankle_y.size < 3 or fps <= 0: return np.array([], dtype=int)
        inv = -ankle_y
        min_distance = max(1, int(min_step_time * fps))
        peaks, _ = find_peaks(inv, distance=min_distance)
        return peaks

    @staticmethod
    def step_temporal_features(ankle_y, fps, min_step_time=0.3):
        ankle_y = np.asarray(ankle_y, dtype=float)
        peaks = GaitAnalysisEngine.detect_step_events_from_ankle(ankle_y, fps, min_step_time=min_step_time)

        if peaks.size < 2 or fps <= 0:
            return {"mean_step_time": np.nan, "std_step_time": np.nan, "cadence": np.nan, 
                    "mean_stride_time": np.nan, "std_stride_time": np.nan, "step_time_cv": np.nan}

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

        return {
            "mean_step_time": mean_step, "std_step_time": std_step, "cadence": float(cadence),
            "mean_stride_time": mean_stride, "std_stride_time": std_stride,
            "step_time_cv": (std_step / mean_step) if mean_step > 0 else np.nan,
        }
    
    @staticmethod
    def compute_window_features(window: np.ndarray, fps: float) -> dict[str, float]:
        window = np.asarray(window)
        if window.ndim != 3 or window.shape[1] != GaitAnalysisEngine.N_JOINTS or window.shape[2] != 3:
            raise ValueError(f"Invalid window shape: {window.shape}")

        # Auto-normalize
        pelvis = (window[:, GaitAnalysisEngine.LEFT_HIP] + window[:, GaitAnalysisEngine.RIGHT_HIP]) / 2
        if np.linalg.norm(pelvis.mean(axis=0)) > 1e-2:
            window = GaitAnalysisEngine.normalize_pose_3d(window)
        
        feats: dict[str, float] = {}

        # Spatial Features
        left_ankle_y = window[:, GaitAnalysisEngine.LEFT_ANKLE, 1]
        right_ankle_y = window[:, GaitAnalysisEngine.RIGHT_ANKLE, 1]
        left_ankle_x = window[:, GaitAnalysisEngine.LEFT_ANKLE, 0]
        right_ankle_x = window[:, GaitAnalysisEngine.RIGHT_ANKLE, 0]

        feats["step_height_L"] = float(left_ankle_y.max() - left_ankle_y.min())
        feats["step_height_R"] = float(right_ankle_y.max() - right_ankle_y.min())
        feats["step_length_L"] = float(left_ankle_x.max() - left_ankle_x.min())
        feats["step_length_R"] = float(right_ankle_x.max() - right_ankle_x.min())

        left_hip_y = window[:, GaitAnalysisEngine.LEFT_HIP, 1]
        right_hip_y = window[:, GaitAnalysisEngine.RIGHT_HIP, 1]
        pelvis_diff = left_hip_y - right_hip_y
        feats["pelvis_drop_mean"] = float(pelvis_diff.mean())
        feats["pelvis_drop_std"] = float(pelvis_diff.std())

        left_sh_x = window[:, GaitAnalysisEngine.LEFT_SHOULDER, 0]
        right_sh_x = window[:, GaitAnalysisEngine.RIGHT_SHOULDER, 0]
        trunk_lean = left_sh_x - right_sh_x
        feats["trunk_lean_mean"] = float(trunk_lean.mean())
        feats["trunk_lean_std"] = float(trunk_lean.std())

        # Heel Range (Added explicitly to match metadata)
        left_heel_y = window[:, GaitAnalysisEngine.LEFT_HEEL, 1]
        right_heel_y = window[:, GaitAnalysisEngine.RIGHT_HEEL, 1]
        feats["heel_range_L"] = float(left_heel_y.max() - left_heel_y.min())
        feats["heel_range_R"] = float(right_heel_y.max() - right_heel_y.min())

        eps = 1e-6
        hL, hR = feats["step_height_L"], feats["step_height_R"]
        lL, lR = feats["step_length_L"], feats["step_length_R"]
        feats["step_height_symmetry"] = float((hL - hR) / (hL + hR + eps))
        feats["step_length_symmetry"] = float((lL - lR) / (lL + lR + eps))
        
        # ROM & Speed
        left_knee_move = GaitAnalysisEngine.moving_and_still_times(window, GaitAnalysisEngine.LEFT_KNEE, fps)
        right_knee_move = GaitAnalysisEngine.moving_and_still_times(window, GaitAnalysisEngine.RIGHT_KNEE, fps)
        
        for k, v in left_knee_move.items(): feats[f"knee_L_{k}"] = v
        for k, v in right_knee_move.items(): feats[f"knee_R_{k}"] = v
        
        feats["knee_L_rom_y"] = GaitAnalysisEngine.range_of_motion(window, GaitAnalysisEngine.LEFT_KNEE, axis="y")["rom_y"]
        feats["knee_R_rom_y"] = GaitAnalysisEngine.range_of_motion(window, GaitAnalysisEngine.RIGHT_KNEE, axis="y")["rom_y"]

        hip_L_rom_y = GaitAnalysisEngine.range_of_motion(window, GaitAnalysisEngine.LEFT_HIP, axis="y")["rom_y"]
        hip_R_rom_y = GaitAnalysisEngine.range_of_motion(window, GaitAnalysisEngine.RIGHT_HIP, axis="y")["rom_y"]
        feats["hip_L_rom_y"] = hip_L_rom_y
        feats["hip_R_rom_y"] = hip_R_rom_y

        shoulder_L_rom_x = GaitAnalysisEngine.range_of_motion(window, GaitAnalysisEngine.LEFT_SHOULDER, axis="x")["rom_x"]
        shoulder_R_rom_x = GaitAnalysisEngine.range_of_motion(window, GaitAnalysisEngine.RIGHT_SHOULDER, axis="x")["rom_x"]
        feats["shoulder_L_rom_x"] = shoulder_L_rom_x
        feats["shoulder_R_rom_x"] = shoulder_R_rom_x

        ankle_L_rom_y = GaitAnalysisEngine.range_of_motion(window, GaitAnalysisEngine.LEFT_ANKLE, axis="y")["rom_y"]
        ankle_R_rom_y = GaitAnalysisEngine.range_of_motion(window, GaitAnalysisEngine.RIGHT_ANKLE, axis="y")["rom_y"]
        feats["ankle_L_rom_y"] = ankle_L_rom_y
        feats["ankle_R_rom_y"] = ankle_R_rom_y

        feats["knee_rom_asym"] = GaitAnalysisEngine.asymmetry(feats["knee_L_rom_y"], feats["knee_R_rom_y"])
        feats["hip_rom_asym"] = GaitAnalysisEngine.asymmetry(hip_L_rom_y, hip_R_rom_y)
        feats["shoulder_rom_asym"] = GaitAnalysisEngine.asymmetry(shoulder_L_rom_x, shoulder_R_rom_x)
        feats["ankle_rom_asym"] = GaitAnalysisEngine.asymmetry(ankle_L_rom_y, ankle_R_rom_y)

        ankle_L_move = GaitAnalysisEngine.moving_and_still_times(window, GaitAnalysisEngine.LEFT_ANKLE, fps)
        ankle_R_move = GaitAnalysisEngine.moving_and_still_times(window, GaitAnalysisEngine.RIGHT_ANKLE, fps)

        feats["ankle_L_moving_fraction"] = ankle_L_move["moving_fraction"]
        feats["ankle_L_still_fraction"] = ankle_L_move["still_fraction"]
        feats["ankle_R_moving_fraction"] = ankle_R_move["moving_fraction"]
        feats["ankle_R_still_fraction"] = ankle_R_move["still_fraction"]

        stance_ratio_L = ankle_L_move["still_fraction"] / (ankle_L_move["moving_fraction"] + 1e-6)
        stance_ratio_R = ankle_R_move["still_fraction"] / (ankle_R_move["moving_fraction"] + 1e-6)

        feats["stance_ratio_L"] = float(stance_ratio_L)
        feats["stance_ratio_R"] = float(stance_ratio_R)
        feats["stance_ratio_asym"] = GaitAnalysisEngine.asymmetry(stance_ratio_L, stance_ratio_R)

        # Joint Angles
        knee_angle_L = GaitAnalysisEngine.joint_angle(window[:, GaitAnalysisEngine.LEFT_HIP, :], window[:, GaitAnalysisEngine.LEFT_KNEE, :], window[:, GaitAnalysisEngine.LEFT_ANKLE, :])
        knee_angle_R = GaitAnalysisEngine.joint_angle(window[:, GaitAnalysisEngine.RIGHT_HIP, :], window[:, GaitAnalysisEngine.RIGHT_KNEE, :], window[:, GaitAnalysisEngine.RIGHT_ANKLE, :])

        feats["knee_angle_L_mean"] = float(knee_angle_L.mean())
        feats["knee_angle_L_std"] = float(knee_angle_L.std())
        feats["knee_angle_L_rom"] = float(knee_angle_L.max() - knee_angle_L.min())
        feats["knee_angle_R_mean"] = float(knee_angle_R.mean())
        feats["knee_angle_R_std"] = float(knee_angle_R.std())
        feats["knee_angle_R_rom"] = float(knee_angle_R.max() - knee_angle_R.min())

        hip_angle_L = GaitAnalysisEngine.joint_angle(window[:, GaitAnalysisEngine.LEFT_SHOULDER, :], window[:, GaitAnalysisEngine.LEFT_HIP, :], window[:, GaitAnalysisEngine.LEFT_KNEE, :])
        hip_angle_R = GaitAnalysisEngine.joint_angle(window[:, GaitAnalysisEngine.RIGHT_SHOULDER, :], window[:, GaitAnalysisEngine.RIGHT_HIP, :], window[:, GaitAnalysisEngine.RIGHT_KNEE, :])

        feats["hip_angle_L_mean"] = float(hip_angle_L.mean())
        feats["hip_angle_L_std"] = float(hip_angle_L.std())
        feats["hip_angle_L_rom"] = float(hip_angle_L.max() - hip_angle_L.min())
        feats["hip_angle_R_mean"] = float(hip_angle_R.mean())
        feats["hip_angle_R_std"] = float(hip_angle_R.std())
        feats["hip_angle_R_rom"] = float(hip_angle_R.max() - hip_angle_R.min())

        ankle_angle_L = GaitAnalysisEngine.joint_angle(window[:, GaitAnalysisEngine.LEFT_KNEE, :], window[:, GaitAnalysisEngine.LEFT_ANKLE, :], window[:, GaitAnalysisEngine.LEFT_FOOT_INDEX, :])
        ankle_angle_R = GaitAnalysisEngine.joint_angle(window[:, GaitAnalysisEngine.RIGHT_KNEE, :], window[:, GaitAnalysisEngine.RIGHT_ANKLE, :], window[:, GaitAnalysisEngine.RIGHT_FOOT_INDEX, :])

        feats["ankle_angle_L_mean"] = float(ankle_angle_L.mean())
        feats["ankle_angle_L_std"] = float(ankle_angle_L.std())
        feats["ankle_angle_L_rom"] = float(ankle_angle_L.max() - ankle_angle_L.min())
        feats["ankle_angle_R_mean"] = float(ankle_angle_R.mean())
        feats["ankle_angle_R_std"] = float(ankle_angle_R.std())
        feats["ankle_angle_R_rom"] = float(ankle_angle_R.max() - ankle_angle_R.min())

        feats["knee_angle_rom_asym"] = GaitAnalysisEngine.asymmetry(feats["knee_angle_L_rom"], feats["knee_angle_R_rom"])
        feats["hip_angle_rom_asym"] = GaitAnalysisEngine.asymmetry(feats["hip_angle_L_rom"], feats["hip_angle_R_rom"])
        feats["ankle_angle_rom_asym"] = GaitAnalysisEngine.asymmetry(feats["ankle_L_rom_y"], feats["ankle_R_rom_y"])

        # Temporal
        left_temporal = GaitAnalysisEngine.step_temporal_features(left_ankle_y, fps)
        right_temporal = GaitAnalysisEngine.step_temporal_features(right_ankle_y, fps)

        for k, v in left_temporal.items(): feats[f"step_L_{k}"] = float(v) if v is not None else np.nan
        for k, v in right_temporal.items(): feats[f"step_R_{k}"] = float(v) if v is not None else np.nan

        if not np.isnan(left_temporal["mean_step_time"]) and not np.isnan(right_temporal["mean_step_time"]):
            feats["step_time_asym"] = GaitAnalysisEngine.asymmetry(left_temporal["mean_step_time"], right_temporal["mean_step_time"])
        else: feats["step_time_asym"] = np.nan
        
        if not np.isnan(left_temporal["cadence"]) and not np.isnan(right_temporal["cadence"]):
            feats["cadence_asym"] = GaitAnalysisEngine.asymmetry(left_temporal["cadence"], right_temporal["cadence"])
        else: feats["cadence_asym"] = np.nan

        step_width_series = np.abs(left_ankle_x - right_ankle_x)
        feats["step_width_mean"] = float(step_width_series.mean())
        feats["step_width_std"] = float(step_width_series.std())

        return feats
    
    @staticmethod
    def extract_features_from_windows(X_windows, fps, gait_pattern, movement_type, side, source_file):
        X_windows = np.asarray(X_windows)
        N = X_windows.shape[0]
        rows: list[dict[str, any]] = []

        for i in range(N):
            feats = GaitAnalysisEngine.compute_window_features(X_windows[i], fps=fps)
            feats["label_fine"] = gait_pattern[i] if gait_pattern else None
            feats["movement_type"] = movement_type[i] if movement_type else None
            feats["side"] = side[i] if side else None
            feats["source_file"] = source_file[i] if source_file else None
            rows.append(feats)
        return pd.DataFrame(rows)
    
    @staticmethod
    def extract_features_from_csv(csv_path: Path, video_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['frame', 'landmark_id', 'x_norm', 'y_norm', 'z_norm']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"CSV missing required columns: {required_columns}")
                return pd.DataFrame(), None
            
            fps = 30.0
            if video_path and video_path.exists():
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened(): fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    cap.release()
                except: pass
            
            if "movement_type" in df.columns:
                df = df.fillna({"movement_type": "N"})
                df = df[df["movement_type"] != "SLOWMOTION"]
            if "gait_markers" in df.columns:
                df = df.fillna({"gait_markers": "NA"})
            if "video_id" not in df.columns:
                df["video_id"] = "default_video"
            
            df_video = GaitAnalysisEngine.add_pose_column(df)
            if df_video.empty: return pd.DataFrame(), None
            
            X_windows, y_binary, y_multilabel, window_ids = GaitAnalysisEngine.preprocess_gait_sliding_windows(
                df_video, window_seconds=2.0, overlap=0.5, resample_frames=60
            )
            
            X_clean, y_binary_clean, y_multilabel_clean, window_ids_clean, qc_df = GaitAnalysisEngine.apply_qc_windows(
                X_windows, y_binary, y_multilabel, window_ids, fps=60
            )
            
            df_features = GaitAnalysisEngine.extract_features_from_windows(
                X_clean, fps=60, gait_pattern=None, movement_type=None, side=None, source_file=[csv_path.name] * len(X_clean)
            )
            
            df_features["binary_label"] = y_binary_clean
            for i, col in enumerate(GaitAnalysisEngine.ANOMALY_COLS):
                df_features[col] = y_multilabel_clean[:, i]
            
            return df_features, X_clean
        
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            return pd.DataFrame(), None

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_gait_score_dashboard(features_df):
    fig, ax = plt.subplots(figsize=(12, 8))
    gait_metrics = {
        'Step Symmetry': ('step_height_symmetry', 0.15, 0.25),
        'Knee Flexion (L)': ('knee_L_rom_y', 30, 60), 
        'Ankle Control (L)': ('ankle_L_moving_fraction', 0.3, 0.7)
    }
    positions = [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8)]
    
    for i, (metric_name, (feature_key, min_good, max_good)) in enumerate(gait_metrics.items()):
        x, y = positions[i]
        if feature_key in features_df.columns:
            value = features_df.iloc[0][feature_key]
            if 'symmetry' in feature_key.lower():
                color = 'green' if abs(value) <= min_good else 'orange' if abs(value) <= max_good else 'red'
            else:
                color = 'green' if min_good <= value <= max_good else 'orange'
            circle = Circle((x, y), 0.08, color=color, alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y - 0.15, metric_name, ha='center', fontsize=10, weight='bold')
            ax.text(x, y, f'{value:.2f}', ha='center', va='center', fontsize=9, color='white', weight='bold')
    
    legend_elements = [
        plt.scatter([], [], c='green', s=100, label='Normal'),
        plt.scatter([], [], c='orange', s=100, label='Caution'),
        plt.scatter([], [], c='red', s=100, label='Attention')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title('Gait Health Assessment Dashboard', fontsize=16, weight='bold'); ax.axis('off')
    return fig

def create_movement_flow_chart(features_df):
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(14, 8))
    phases = [
        ('Initial Contact', 0.1, 0.8), ('Loading Response', 0.3, 0.8),
        ('Mid Stance', 0.5, 0.8), ('Terminal Stance', 0.7, 0.8),
        ('Pre-Swing', 0.9, 0.8), ('Initial Swing', 0.1, 0.4),
        ('Mid Swing', 0.5, 0.4), ('Terminal Swing', 0.9, 0.4)
    ]
    box_width, box_height = 0.15, 0.1
    
    for phase_name, x, y in phases:
        efficiency = features_df.iloc[0].get('stance_ratio_L', 0.6) if 'stance' in phase_name.lower() else 1.0
        color = plt.cm.RdYlGn(efficiency)
        rect = mpatches.Rectangle((x - box_width/2, y - box_height/2), box_width, box_height, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, phase_name, ha='center', va='center', fontsize=9, weight='bold')
    
    arrows = [
        ((0.175, 0.8), (0.225, 0.8)), ((0.375, 0.8), (0.425, 0.8)),
        ((0.575, 0.8), (0.625, 0.8)), ((0.775, 0.8), (0.825, 0.8)),
        ((0.925, 0.8), (0.925, 0.45)), ((0.925, 0.35), (0.525, 0.35)),
        ((0.475, 0.35), (0.075, 0.35)), ((0.025, 0.35), (0.025, 0.75))
    ]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    ax.text(0.5, 0.95, 'STANCE PHASE', ha='center', fontsize=12, weight='bold', bbox=dict(boxstyle="round", facecolor='lightblue'))
    ax.text(0.5, 0.25, 'SWING PHASE', ha='center', fontsize=12, weight='bold', bbox=dict(boxstyle="round", facecolor='lightgreen'))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title('Gait Cycle Movement Pattern Flow', fontsize=16, weight='bold'); ax.axis('off')
    return fig

def create_3d_joint_trajectory(gait_cycles):
    if gait_cycles is None or len(gait_cycles) == 0: return None
    fig = plt.figure(figsize=(15, 10))
    joint_names = ['Left Ankle', 'Right Ankle', 'Left Knee', 'Right Knee', 'Left Hip', 'Right Hip']
    joint_indices = [27, 28, 25, 26, 23, 24]
    avg_cycle = np.mean(gait_cycles, axis=0)
    
    views = [(0, 0), (0, 90), (90, 0), (30, 45)]
    view_labels = ['Front View', 'Side View', 'Top View', '3D View']
    
    for idx, (view, label, angles) in enumerate(zip(views, view_labels, [(0,0), (0,90), (90,0), (30,45)])):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        for j_name, j_idx in zip(joint_names, joint_indices):
            traj = avg_cycle[:, j_idx, :]
            color = 'blue' if 'Left' in j_name else 'red'
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=2, label=j_name)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(label); ax.view_init(elev=angles[0], azim=angles[1])
    plt.suptitle('3D Joint Trajectories', fontsize=16, weight='bold'); plt.tight_layout()
    return fig

def create_gait_stability_index(features_df):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    stability_metrics = [
        ('Dynamic Balance', 'pelvis_drop_std', 0.1), ('Step Consistency', 'step_L_step_time_cv', 0.1),
        ('Joint Coordination', 'knee_angle_rom_asym', 0.15), ('Movement Control', 'ankle_L_moving_fraction', 0.2),
        ('Rhythm Regularity', 'cadence_asym', 0.1), ('Postural Stability', 'trunk_lean_std', 0.1)
    ]
    labels, values = [], []
    for name, key, _ in stability_metrics:
        labels.append(name)
        values.append(abs(features_df.iloc[0].get(key, 0)))
    
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=3); ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
    ax.set_title('Gait Stability Index', size=16, weight='bold', pad=20)
    return fig

def create_temporal_gait_heatmap(gait_cycles):
    if gait_cycles is None or len(gait_cycles) == 0: return None
    avg_cycle = np.mean(gait_cycles, axis=0)
    joint_data = {
        'Left Ankle': avg_cycle[:, 27, 1], 'Right Ankle': avg_cycle[:, 28, 1],
        'Left Knee': avg_cycle[:, 25, 1], 'Right Knee': avg_cycle[:, 26, 1],
        'Left Hip': avg_cycle[:, 23, 1], 'Right Hip': avg_cycle[:, 24, 1],
    }
    df_heatmap = pd.DataFrame(joint_data)
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(df_heatmap.T, cmap=cmap, center=0, cbar_kws={'label': 'Vertical Position'}, ax=ax)
    ax.set_title('Temporal Gait Pattern Heatmap', fontsize=16); ax.set_xlabel('Timeline'); ax.set_ylabel('Joints')
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="MediaPipe Gait Analysis", page_icon="🎥", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""<style>
    body { background-color: #F0F2F5; color: #2C3E50; }
    .stButton>button { background-color: #2C3E50; color: white; border-radius: 5px; height: 3em; font-weight: bold; }
    h1, h2, h3 { color: #1F2937; font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    </style>""", unsafe_allow_html=True)
    
    st.title("🚶 Production-Grade Gait Analysis & AI Modelling")
    st.markdown("**Complete Pipeline: Processing, Feature Engineering, Analysis & Prediction**")
    st.markdown("---")
    
    # Session State Initialization
    if 'predictor' not in st.session_state:
        st.session_state.predictor = ModelPredictor()
    
    default_state = {
        'uploaded_video_path': None, 'processing_complete': False, 'output_videos': {},
        'features_df': None, 'gait_cycles': None, 
        'pred_binary_results': None, 'pred_multiclass_results': None,
        'csv_features_df': None # For standalone CSV upload
    }
    
    for key, val in default_state.items():
        if key not in st.session_state: st.session_state[key] = val

    # Sidebar
    with st.sidebar:
        st.header("📋 System Status")
        def status_icon(path): return "✅" if path.exists() else "❌"
        st.write(f"Config: {status_icon(CONFIG_PATH)}")
        st.write(f"MediaPipe: {status_icon(MEDIAPIPE_SCRIPT)}")
        st.write(f"Binary Model: {status_icon(BINARY_MODEL_PATH)}")
        st.write(f"Binary Meta: {status_icon(BINARY_FEATURES_PATH)}")
        st.write(f"Multi Model: {status_icon(MULTICLASS_MODEL_PATH)}")
        st.write(f"Multi Meta: {status_icon(MULTICLASS_METADATA_PATH)}")
        
        if st.button("🔄 Reset App"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs(["📤 Upload", "⚙️ Process", "🎬 Videos", "📊 Features", "🔬 Analysis", "🤖 AI Modelling"])

    # TAB 1: UPLOAD
    with t1:
        st.subheader("Upload Video for Analysis")
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        if uploaded_file:
            st.info(f"📄 {uploaded_file.name} | 📏 {uploaded_file.size / (1024*1024):.2f} MB")
            with st.spinner("Saving video..."):
                video_path, is_duplicate = FileManager.save_uploaded_video(uploaded_file)
            
            if video_path:
                if st.session_state.uploaded_video_path != video_path:
                    st.session_state.uploaded_video_path = video_path
                    st.session_state.processing_complete = False
                    st.session_state.output_videos = {}
                    st.session_state.features_df = None
                    st.session_state.gait_cycles = None
                    st.session_state.pred_binary_results = None
                    st.session_state.pred_multiclass_results = None
                
                if is_duplicate: st.warning("⚠️ Duplicate video detected")
                else: st.success("✅ Video saved")
                
                if st.session_state.uploaded_video_path:
                    VideoDisplay.display_video_with_download(st.session_state.uploaded_video_path, "Original Video", "tab1_orig")

    # TAB 2: PROCESS
    with t2:
        st.subheader("Process Video with MediaPipe")
        if not st.session_state.uploaded_video_path:
            st.info("Please upload a video first.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"Target: {st.session_state.uploaded_video_path.name}")
            with col2:
                if st.button("▶️ Run Pipeline", type="primary"):
                    with st.spinner("Preparing pipeline..."):
                        PipelineManager.update_config_with_video(st.session_state.uploaded_video_path)
                    
                    st.markdown("---")
                    st.subheader("⏳ Processing...")
                    progress_bar = st.progress(0)
                    
                    try:
                        progress_bar.progress(20)
                        results = PipelineManager.run_pipeline()
                        progress_bar.progress(80)
                        
                        if results:
                            output_files = FileManager.find_output_videos(st.session_state.uploaded_video_path)
                            if output_files.get('annotated'): VideoConverter.ensure_web_compatible(output_files['annotated'])
                            if output_files.get('skeleton'): VideoConverter.ensure_web_compatible(output_files['skeleton'])
                            
                            st.session_state.output_videos = output_files
                            st.session_state.processing_complete = True
                            progress_bar.progress(100)
                            st.success("✅ Pipeline Complete")
                        else:
                            st.error("❌ Pipeline failed")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    # TAB 3: VIDEOS
    with t3:
        st.subheader("Landmarker Videos")
        if not st.session_state.processing_complete:
            st.info("Process a video first.")
        else:
            output_files = st.session_state.output_videos
            c1, c2, c3 = st.columns(3)
            with c1: VideoDisplay.display_video_with_download(st.session_state.uploaded_video_path, "Original", "t3_orig")
            with c2: VideoDisplay.display_video_with_download(output_files.get('annotated'), "Annotated", "t3_anno")
            with c3: VideoDisplay.display_video_with_download(output_files.get('skeleton'), "Skeleton", "t3_sk")
            
            if output_files.get('csv'):
                csv_path = output_files['csv']
                with st.expander("👁️ Preview CSV"):
                    st.dataframe(pd.read_csv(csv_path).head(10))

    # TAB 4: FEATURES
    with t4:
        st.subheader("Feature Engineering")
        # Option A: From Processed Video
        if st.session_state.processing_complete and st.session_state.output_videos.get('csv'):
            st.markdown("#### From Processed Video")
            if st.button("🚀 Extract Features (Video)", key="extract_video"):
                csv_path = st.session_state.output_videos['csv']
                with st.spinner("Extracting..."):
                    df_feats, cycles = GaitAnalysisEngine.extract_features_from_csv(csv_path, st.session_state.uploaded_video_path)
                    if not df_feats.empty:
                        st.session_state.features_df = df_feats
                        st.session_state.gait_cycles = cycles
                        st.success(f"✅ Extracted {len(df_feats)} windows")
                    else:
                        st.error("❌ Extraction failed")
        
        # Option B: Upload CSV
        st.markdown("---")
        st.markdown("#### Standalone CSV Upload")
        csv_file = st.file_uploader("Upload Landmarks CSV", type=['csv'], key='t4_csv')
        if csv_file:
            csv_path = FEATURES_DIR / f"upload_{csv_file.name}"
            with open(csv_path, 'wb') as f: f.write(csv_file.getbuffer())
            
            if st.button("🚀 Extract Features (CSV)", key="extract_csv"):
                with st.spinner("Extracting..."):
                    df_feats, cycles = GaitAnalysisEngine.extract_features_from_csv(csv_path)
                    if not df_feats.empty:
                        st.session_state.csv_features_df = df_feats
                        st.session_state.csv_gait_cycles = cycles
                        st.success(f"✅ Extracted {len(df_feats)} windows")
                    else:
                        st.error("❌ Extraction failed")

    # TAB 5: ANALYSIS
    with t5:
        st.subheader("Detailed Gait Analysis")
        # Determine source of features
        source_df = None
        source_cycles = None
        source_name = ""
        
        if st.session_state.features_df is not None:
            source_df = st.session_state.features_df
            source_cycles = st.session_state.gait_cycles
            source_name = "Processed Video"
        elif st.session_state.csv_features_df is not None:
            source_df = st.session_state.csv_features_df
            source_cycles = st.session_state.csv_gait_cycles
            source_name = "Uploaded CSV"
            
        if source_df is None:
            st.info("Extract features in Tab 4 first.")
        else:
            st.metric("Source", source_name)
            st.metric("Windows Analyzed", len(source_df))
            
            viz_type = st.selectbox("Visualization", ["Dashboard", "Flow Chart", "3D Trajectory", "Stability Index", "Heatmap"])
            
            if viz_type == "Dashboard": fig = create_gait_score_dashboard(source_df)
            elif viz_type == "Flow Chart": fig = create_movement_flow_chart(source_df)
            elif viz_type == "3D Trajectory": fig = create_3d_joint_trajectory(source_cycles)
            elif viz_type == "Stability Index": fig = create_gait_stability_index(source_df)
            elif viz_type == "Heatmap": fig = create_temporal_gait_heatmap(source_cycles)
            
            if fig: st.pyplot(fig); plt.close()

    # ═════════════════════════════════════════════════════════════════════════════════
    # TAB 6: AI MODELLING (FIXED UNICODE LOGGING)
    # ═════════════════════════════════════════════════════════════════════════════════

    with t6:
        st.subheader("AI Modelling & Prediction")
        
        # 1. Determine Data Source
        model_df = None
        data_source_name = ""
        
        if st.session_state.features_df is not None:
            model_df = st.session_state.features_df
            data_source_name = "Processed Video"
        elif st.session_state.csv_features_df is not None:
            model_df = st.session_state.csv_features_df
            data_source_name = "Uploaded CSV"
            
        if model_df is None:
            st.warning("⚠️ No features available. Please extract features in Tab 4.")
        else:
            st.info(f"📂 Using features from: **{data_source_name}** ({len(model_df)} windows)")
            
            # 2. VALIDATION PHASE: Check Models and Metadata
            st.markdown("---")
            st.subheader("1. Model & Data Validation")
            
            col1, col2 = st.columns(2)
            
            # --- Binary Check ---
            with col1:
                st.markdown("**Binary Classification Model**")
                b_model_ok = BINARY_MODEL_PATH.exists()
                b_meta_ok = BINARY_FEATURES_PATH.exists()
                
                if b_model_ok and b_meta_ok:
                    st.success("✅ Files present")
                    with open(BINARY_FEATURES_PATH, 'r') as f:
                        binary_features = json.load(f)
                    
                    st.metric("Expected Features", len(binary_features))
                    
                    # Check overlap
                    current_features = set(model_df.columns)
                    required_features = set(binary_features)
                    missing = list(required_features - current_features)
                    
                    if missing:
                        st.warning(f"⚠️ Missing {len(missing)} features (will be filled with 0)")
                        with st.expander("View Missing Features"):
                            st.write(missing[:10]) # Show first 10
                    else:
                        st.success("✅ All expected features found")
                else:
                    st.error("❌ Model or Metadata missing")
                    if not b_model_ok: st.write("Missing: `xgboost_model.bin`")
                    if not b_meta_ok: st.write("Missing: `feature_names.json`")

            # --- Multiclass Check ---
            with col2:
                st.markdown("**Multiclass Classification Model (5-Class)**")
                m_model_ok = MULTICLASS_MODEL_PATH.exists()
                m_meta_ok = MULTICLASS_METADATA_PATH.exists()
                
                if m_model_ok and m_meta_ok:
                    st.success("✅ Files present")
                    with open(MULTICLASS_METADATA_PATH, 'r') as f:
                        multi_meta = json.load(f)
                    
                    multi_features = multi_meta.get('feature_cols', [])
                    st.metric("Expected Features", len(multi_features))
                    
                    current_features = set(model_df.columns)
                    required_features = set(multi_features)
                    missing = list(required_features - current_features)
                    
                    if missing:
                        st.warning(f"⚠️ Missing {len(missing)} features (will be filled with 0)")
                    else:
                        st.success("✅ All expected features found")
                else:
                    st.error("❌ Model or Metadata missing")
                    if not m_model_ok: st.write("Missing: `xgboost_gait_5class.bin`")
                    if not m_meta_ok: st.write("Missing: `xgboost_gait_5class_metadata.json`")

            # 3. PREDICTION PHASE
            st.markdown("---")
            st.subheader("2. Run Predictions")
            
            # Binary Prediction
            st.markdown("### Binary Prediction (Normal vs Abnormal)")
            if b_model_ok and b_meta_ok:
                if st.button("▶️ Run Binary Prediction", key="run_binary"):
                    with st.spinner("Loading Model..."):
                        try:
                            # Reload model to ensure fresh state
                            st.session_state.predictor.load_binary_model()
                            
                            with st.spinner("Aligning features..."):
                                pass # Logic handled inside predict, but spinner gives feedback
                                
                            with st.spinner("Predicting..."):
                                res = st.session_state.predictor.predict_binary(model_df)
                                st.session_state.pred_binary_results = res
                                st.success("✅ Binary Prediction Complete")
                        except Exception as e:
                            st.error(f"❌ Prediction Failed: {e}")
                            st.exception(e)
            
            # Multiclass Prediction
            st.markdown("### Multiclass Prediction (Specific Anomalies)")
            if m_model_ok and m_meta_ok:
                if st.button("▶️ Run Multiclass Prediction", key="run_multi"):
                    with st.spinner("Loading Model..."):
                        try:
                            # Reload model
                            st.session_state.predictor.load_multiclass_model()
                            
                            with st.spinner("Aligning features..."):
                                pass
                                
                            with st.spinner("Predicting..."):
                                res = st.session_state.predictor.predict_multiclass(model_df)
                                st.session_state.pred_multiclass_results = res
                                st.success("✅ Multiclass Prediction Complete")
                        except Exception as e:
                            st.error(f"❌ Prediction Failed: {e}")
                            st.exception(e)

            # 4. RESULTS DISPLAY
            st.markdown("---")
            st.subheader("3. Prediction Results")

            # Binary Results
            if st.session_state.pred_binary_results is not None:
                st.markdown("#### Binary Classification Results")
                res_df = st.session_state.pred_binary_results
                
                # Aggregate stats
                counts = res_df['prediction'].value_counts()
                st.write("**Prediction Distribution:**")
                st.bar_chart(counts)
                
                # Detailed View
                with st.expander("View Detailed Probability Breakdown"):
                    display_cols = ['prediction', 'confidence', 'prob_normal', 'prob_abnormal']
                    st.dataframe(res_df[display_cols])

            # Multiclass Results
            if st.session_state.pred_multiclass_results is not None:
                st.markdown("#### Multiclass Classification Results")
                res_df = st.session_state.pred_multiclass_results
                
                # Aggregate stats
                counts = res_df['prediction'].value_counts()
                st.write("**Prediction Distribution (Detected Classes):**")
                st.bar_chart(counts)
                
                # Detailed View
                # Find probability columns dynamically
                prob_cols = [c for c in res_df.columns if c.startswith('prob_')]
                display_cols = ['prediction', 'confidence'] + prob_cols
                
                with st.expander("View Detailed Probability Breakdown"):
                    st.dataframe(res_df[display_cols])

if __name__ == "__main__":
    main()
