#!/usr/bin/env python3
"""
MEDIAPIPE POSE DETECTION PIPELINE - PRODUCTION GRADE APPLICATION
Complete implementation with robust video rendering and export functionality
"""

import os
import sys
import warnings
import logging
import pickle
import joblib
import traceback
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

# Environment configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gait_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import streamlit as st
import json
import importlib.util
from pathlib import Path
import hashlib
import subprocess
import time
import zipfile
from io import BytesIO
from typing import Optional, Dict, Tuple, List, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.absolute()
CONFIG_PATH = PROJECT_ROOT / "config.json"
MEDIAPIPE_SCRIPT = PROJECT_ROOT / "pre-processing-models" / "mediapipe" / "pre_mediapipe.py"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
GAIT_CYCLES_DIR = PROJECT_ROOT / "data" / "gait_cycles"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
ADVANCED_MODELS_DIR = MODELS_DIR / "advance"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, FEATURES_DIR, GAIT_CYCLES_DIR, 
                 PREDICTIONS_DIR, LOGS_DIR, BASELINE_MODELS_DIR, ADVANCED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Set up detailed logging for model operations
model_logger = logging.getLogger("model_operations")
model_handler = logging.FileHandler(LOGS_DIR / f"model_operations_{datetime.now().strftime('%Y%m%d')}.log")
model_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
model_logger.addHandler(model_handler)
model_logger.setLevel(logging.DEBUG)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Model features - these are the columns expected by the baseline model
BASELINE_MODEL_FEATURES = [
    'step_height_L', 'step_height_R', 'step_length_L', 'step_length_R',
    'pelvis_drop_mean', 'pelvis_drop_std', 'trunk_lean_mean', 'trunk_lean_std',
    'heel_range_L', 'heel_range_R', 'step_height_symmetry', 'step_length_symmetry',
    'knee_L_moving_time_sec', 'knee_L_still_time_sec', 'knee_L_moving_fraction',
    'knee_L_still_fraction', 'knee_L_mean_speed', 'knee_L_max_speed',
    'knee_L_total_time_sec', 'knee_R_moving_time_sec', 'knee_R_still_time_sec',
    'knee_R_moving_fraction', 'knee_R_still_fraction', 'knee_R_mean_speed',
    'knee_R_max_speed', 'knee_R_total_time_sec', 'knee_L_rom_y', 'knee_R_rom_y',
    'hip_L_rom_y', 'hip_R_rom_y', 'shoulder_L_rom_x', 'shoulder_R_rom_x',
    'ankle_L_rom_y', 'ankle_R_rom_y', 'knee_rom_asym', 'hip_rom_asym',
    'shoulder_rom_asym', 'ankle_rom_asym', 'ankle_L_moving_fraction',
    'ankle_L_still_fraction', 'ankle_R_moving_fraction', 'ankle_R_still_fraction',
    'stance_ratio_L', 'stance_ratio_R', 'stance_ratio_asym', 'knee_angle_L_mean',
    'knee_angle_L_std', 'knee_angle_L_rom', 'knee_angle_R_mean', 'knee_angle_R_std',
    'knee_angle_R_rom', 'hip_angle_L_mean', 'hip_angle_L_std', 'hip_angle_L_rom',
    'hip_angle_R_mean', 'hip_angle_R_std', 'hip_angle_R_rom', 'ankle_angle_L_mean',
    'ankle_angle_L_std', 'ankle_angle_L_rom', 'ankle_angle_R_mean',
    'ankle_angle_R_std', 'ankle_angle_R_rom', 'knee_angle_rom_asym',
    'hip_angle_rom_asym', 'ankle_angle_rom_asym', 'step_L_mean_step_time',
    'step_L_std_step_time', 'step_L_cadence', 'step_L_mean_stride_time',
    'step_L_std_stride_time', 'step_L_step_time_cv', 'step_R_mean_step_time',
    'step_R_std_step_time', 'step_R_cadence', 'step_R_mean_stride_time',
    'step_R_std_stride_time', 'step_R_step_time_cv', 'step_time_asym',
    'cadence_asym', 'step_width_mean', 'step_width_std'
]

# Class labels for the models
BASELINE_CLASS_LABELS = {
    0: "Normal",
    1: "Mild Impairment",
    2: "Moderate Impairment",
    3: "Severe Impairment"
}

ADVANCED_BINARY_LABELS = {
    0: "Normal",
    1: "Abnormal"
}

ADVANCED_MULTILABEL_LABELS = {
    0: "Normal",
    1: "Foot Drop",
    2: "Knee Stiffness",
    3: "Hip Weakness",
    4: "Poor Balance"
}

# ═══════════════════════════════════════════════════════════════════════════
# VIDEO CODEC DETECTION & CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

class VideoConverter:
    """Production-grade video converter with multiple fallback strategies"""
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5, text=True)
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def get_video_codec(video_path: Path) -> Optional[str]:
        """Get codec information from video"""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            cap.release()
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            return codec.strip()
        except Exception as e:
            logger.error(f"Failed to get codec: {e}")
            return None
    
    @staticmethod
    def is_web_compatible(codec: str) -> bool:
        """Check if codec is web browser compatible"""
        if not codec:
            return False
        codec_upper = codec.upper()
        compatible = ['AVC1', 'H264', 'X264']
        incompatible = ['MP4V', 'XVID', 'DIVX', 'FMP4']
        
        if any(c in codec_upper for c in compatible):
            return True
        if any(c in codec_upper for c in incompatible):
            return False
        return False
    
    @staticmethod
    def convert_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
        """Convert video using FFmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
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
            import cv2
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                return False
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            codec_options = [('avc1', 'H.264'), ('H264', 'H.264'), ('X264', 'X264'), ('mp4v', 'MPEG-4')]
            
            for codec_str, codec_name in codec_options:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_str)
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    
                    if not out.isOpened():
                        continue
                    
                    logger.info(f"Using {codec_name} for conversion")
                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                        frame_count += 1
                    
                    cap.release()
                    out.release()
                    
                    if output_path.exists() and output_path.stat().st_size > 0:
                        logger.info(f"OpenCV conversion successful with {codec_name}")
                        return True
                except:
                    continue
            
            cap.release()
            return False
        except Exception as e:
            logger.error(f"OpenCV error: {e}")
            return False
    
    @classmethod
    def ensure_web_compatible(cls, video_path: Path) -> Path:
        """Ensure video is web-compatible, convert if necessary"""
        if not video_path or not video_path.exists():
            return video_path
        
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
            if cls.convert_with_ffmpeg(video_path, web_path):
                return web_path
        
        if cls.convert_with_opencv(video_path, web_path):
            return web_path
        
        logger.error(f"All conversions failed for {video_path.name}")
        return video_path


# ═══════════════════════════════════════════════════════════════════════════
# FILE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

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
            import cv2
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
        video_stem = video_path.stem
        
        if '_' in video_stem:
            parts = video_stem.split('_')
            if len(parts[0]) == 8 and parts[0].isdigit():
                if len(parts) > 2:
                    video_stem = '_'.join(parts[2:])
                elif len(parts) > 1:
                    video_stem = parts[1]
        
        logger.info(f"Searching for: {video_stem}")
        
        results = {'annotated': None, 'skeleton': None, 'csv': None}
        
        candidates = {
            'annotated': [
                OUTPUT_DIR / f"{video_stem}_annotated.mp4",
                OUTPUT_DIR / f"{video_path.stem}_annotated.mp4",
            ],
            'skeleton': [
                OUTPUT_DIR / f"{video_stem}_skeleton.mp4",
                OUTPUT_DIR / f"{video_path.stem}_skeleton.mp4",
            ],
            'csv': [
                OUTPUT_DIR / f"{video_stem}_landmarks.csv",
                OUTPUT_DIR / f"{video_path.stem}_landmarks.csv",
            ]
        }
        
        for key, paths in candidates.items():
            for candidate in paths:
                if candidate.exists() and candidate.stat().st_size > 0:
                    results[key] = candidate
                    logger.info(f"Found {key}: {candidate.name}")
                    break
        
        for file in OUTPUT_DIR.iterdir():
            name_lower = file.name.lower()
            stem_lower = video_stem.lower()
            
            if stem_lower in name_lower or video_path.stem.lower() in name_lower:
                if 'annotated' in name_lower and file.suffix == '.mp4' and not results['annotated']:
                    results['annotated'] = file
                    logger.info(f"Fuzzy found annotated: {file.name}")
                elif 'skeleton' in name_lower and file.suffix == '.mp4' and not results['skeleton']:
                    results['skeleton'] = file
                    logger.info(f"Fuzzy found skeleton: {file.name}")
                elif 'landmark' in name_lower and file.suffix == '.csv' and not results['csv']:
                    results['csv'] = file
                    logger.info(f"Fuzzy found CSV: {file.name}")
        
        if not results['annotated']:
            annotated = sorted(OUTPUT_DIR.glob("*annotated*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
            if annotated:
                results['annotated'] = annotated[0]
                logger.info(f"Most recent annotated: {annotated[0].name}")
        
        if not results['skeleton']:
            skeleton = sorted(OUTPUT_DIR.glob("*skeleton*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
            if skeleton:
                results['skeleton'] = skeleton[0]
                logger.info(f"Most recent skeleton: {skeleton[0].name}")
        
        if not results['csv']:
            csv_files = sorted(OUTPUT_DIR.glob("*landmark*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
            if csv_files:
                results['csv'] = csv_files[0]
                logger.info(f"Most recent CSV: {csv_files[0].name}")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class PipelineManager:
    
    @staticmethod
    def load_config() -> Optional[dict]:
        if not CONFIG_PATH.exists():
            logger.error(f"Config not found: {CONFIG_PATH}")
            return None
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("Config loaded")
            return config
        except Exception as e:
            logger.error(f"Load config failed: {e}")
            return None
    
    @staticmethod
    def save_config(config: dict) -> bool:
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            logger.info("Config saved")
            return True
        except Exception as e:
            logger.error(f"Save config failed: {e}")
            return False
    
    @staticmethod
    def update_config_with_video(video_path: Path) -> bool:
        config = PipelineManager.load_config()
        if not config:
            return False
        
        try:
            rel_path = str(video_path.relative_to(PROJECT_ROOT))
        except:
            rel_path = str(video_path)
        
        config["input_paths"] = [rel_path]
        config.setdefault("output_dir", "data/output")
        
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
        if not mp_module:
            return None
        
        try:
            config = mp_module.PipelineConfig.from_json(CONFIG_PATH)
            pipeline = mp_module.PoseDetectionPipeline(config)
            results = pipeline.run()
            logger.info(f"Pipeline complete: {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            return None


# ═══════════════════════════════════════════════════════════════════════════
# VIDEO DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

class VideoDisplay:
    
    @staticmethod
    def get_video_info(video_path: Path) -> Optional[dict]:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
                'size_mb': video_path.stat().st_size / (1024 * 1024),
                'codec': codec
            }
            cap.release()
            return info
        except Exception as e:
            logger.error(f"Get info failed: {e}")
            return None
    
    @staticmethod
    def display_video_with_download(video_path: Optional[Path], label: str, key_suffix: str) -> None:
        if not video_path or not video_path.exists():
            st.warning(f"⚠️ {label} not available")
            return
        
        try:
            web_video = VideoConverter.ensure_web_compatible(video_path)
            
            # Display video
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
                # Create a download button with proper key handling
                try:
                    # Create unique keys for session state and widget
                    file_data_key = f"data_{key_suffix}_{web_video.stem}_{web_video.stat().st_mtime}"
                    widget_key = f"widget_{key_suffix}_{web_video.stem}_{web_video.stat().st_mtime}"
                    
                    # Store file data in session state with a different key
                    if file_data_key not in st.session_state:
                        try:
                            with open(web_video, 'rb') as f:
                                st.session_state[file_data_key] = f.read()
                        except Exception as e:
                            logger.error(f"Failed to read file for download: {e}")
                            st.session_state[file_data_key] = None
                    
                    if st.session_state[file_data_key]:
                        st.download_button(
                            "📥 Download",
                            data=st.session_state[file_data_key],
                            file_name=web_video.name,
                            mime="video/mp4",
                            key=widget_key,  # Use different key for the widget
                            use_container_width=True
                        )
                    else:
                        st.error("❌ File not available")
                except Exception as e:
                    st.error("❌ Download failed")
                    logger.error(f"Download button error: {e}")
        
        except Exception as e:
            st.error(f"❌ Display error: {str(e)}")
            logger.error(f"Display error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class ExportManager:
    
    @staticmethod
    def create_results_zip(video_path: Path, output_files: dict) -> Optional[BytesIO]:
        try:
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                if video_path and video_path.exists():
                    zf.write(video_path, f"original/{video_path.name}")
                    logger.info(f"Added to ZIP: {video_path.name}")
                
                for key, path in output_files.items():
                    if path and path.exists():
                        if key == 'csv':
                            zf.write(path, f"data/{path.name}")
                        else:
                            zf.write(path, f"videos/{path.name}")
                        logger.info(f"Added to ZIP: {path.name}")
                
                metadata = {
                    'generated_at': datetime.now().isoformat(),
                    'original_video': video_path.name if video_path else None,
                    'files': {key: path.name if path else None for key, path in output_files.items()}
                }
                zf.writestr('metadata.json', json.dumps(metadata, indent=2))
            
            zip_buffer.seek(0)
            logger.info("ZIP created")
            return zip_buffer
        except Exception as e:
            logger.error(f"ZIP failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# GAIT ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class GaitAnalysisEngine:
    """Complete gait analysis engine with preprocessing and feature extraction"""
    
    # Constants for MediaPipe pose
    N_JOINTS = 33
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HEEL, RIGHT_HEEL = 29, 30
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32
    
    # Gait joints for analysis
    GAIT_JOINTS = [
        2, 5,     # eyes
        11, 12,   # shoulders
        23, 24,   # hips
        25, 26,   # knees
        27, 28,   # ankles
        29, 30,   # heels
        31, 32    # foot index
    ]
    
    @staticmethod
    def normalize_pose_3d(pose):
        """
        Pelvis-centered, torso-scaled normalization
        pose: (T, 33, 3)
        """
        pelvis = (pose[:, GaitAnalysisEngine.LEFT_HIP] + pose[:, GaitAnalysisEngine.RIGHT_HIP]) / 2
        pose_centered = pose - pelvis[:, None, :]

        torso = (pose_centered[:, GaitAnalysisEngine.LEFT_SHOULDER] + pose_centered[:, GaitAnalysisEngine.RIGHT_SHOULDER]) / 2
        scale = np.linalg.norm(torso, axis=1).mean()

        pose_scaled = pose_centered / scale
        return pose_scaled
    
    @staticmethod
    def select_joints(pose, joint_indices):
        """Select specific joints from pose data"""
        pose = np.asarray(pose)
        assert pose.ndim == 3 and pose.shape[1] >= max(joint_indices) + 1
        return pose[:, joint_indices, :]
    
    @staticmethod
    def find_heel_strikes(pose, foot="left", fps=30, min_time_between_steps=0.5, smooth_sigma=1):
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d
        
        heel_idx = GaitAnalysisEngine.LEFT_HEEL if foot == "left" else GaitAnalysisEngine.RIGHT_HEEL
        y = pose[:, heel_idx, 1]

        y_smooth = gaussian_filter1d(y, sigma=smooth_sigma)
        min_distance_frames = int(min_time_between_steps * fps)

        peaks, _ = find_peaks(-y_smooth, distance=min_distance_frames)
        return peaks
    
    @staticmethod
    def extract_gait_cycle_clips(pose, fps, cycles=1, min_time_between_steps=0.5, 
                               min_frames=40, max_frames=150, resample_frames=60):
        """
        Extract gait cycle clips from pose data
        """
        from scipy.signal import resample
        
        clips = []

        left_events = GaitAnalysisEngine.find_heel_strikes(
            pose, "left", fps, min_time_between_steps
        )
        right_events = GaitAnalysisEngine.find_heel_strikes(
            pose, "right", fps, min_time_between_steps
        )

        for foot_events in [left_events, right_events]:
            for i in range(len(foot_events) - cycles):
                s = foot_events[i]
                e = foot_events[i + cycles]
                clip = pose[s:e]

                if min_frames <= len(clip) <= max_frames:
                    clip = resample(clip, resample_frames, axis=0)
                    clips.append(clip)

        return clips
    
    @staticmethod
    def joint_speed(pose_norm, joint_idx, fps, smooth_sigma=1.0):
        """
        Frame-to-frame 3D speed of one joint in a normalized clip.
        """
        from scipy.ndimage import gaussian_filter1d
        
        joint_traj = pose_norm[:, joint_idx, :]  # (T, 3)

        if smooth_sigma and smooth_sigma > 0:
            joint_traj = gaussian_filter1d(joint_traj, sigma=smooth_sigma, axis=0)

        diffs = np.diff(joint_traj, axis=0)      # (T-1, 3)
        disp = np.linalg.norm(diffs, axis=1)     # (T-1,)
        speed = disp * fps
        return speed
    
    @staticmethod
    def moving_and_still_times(pose_norm, joint_idx, fps, speed_thresh=0.02, smooth_sigma=1.0):
        """
        How long a joint is moving vs not moving.
        """
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
            "total_time_sec": float(total_time_sec),
        }
    
    @staticmethod
    def range_of_motion(pose_norm, joint_idx, axis=None):
        """
        Range of motion (ROM) of a joint.
        """
        traj = pose_norm[:, joint_idx, :]  # (T, 3)

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
    def asymmetry(L, R, eps=1e-6):
        """
        Generic left-right asymmetry index:
            (L - R) / (L + R + eps)
        """
        return float((L - R) / (L + R + eps))
    
    @staticmethod
    def joint_angle(p_prox, p_joint, p_dist):
        """
        Joint angle in degrees over time.
        """
        v1 = p_prox - p_joint        # (T, 3)
        v2 = p_dist - p_joint        # (T, 3)

        num = np.einsum("ij,ij->i", v1, v2)  # (T,)
        den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6

        cosang = np.clip(num / den, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosang))  # (T,)

        return angles
    
    @staticmethod
    def detect_step_events_from_ankle(ankle_y, fps, min_step_time=0.3):
        """
        Coarse step detection from an ankle vertical trajectory.
        """
        from scipy.signal import find_peaks
        
        ankle_y = np.asarray(ankle_y, dtype=float)
        if ankle_y.size < 3 or fps <= 0:
            return np.array([], dtype=int)

        inv = -ankle_y
        min_distance = max(1, int(min_step_time * fps))
        peaks, _ = find_peaks(inv, distance=min_distance)
        return peaks
    
    @staticmethod
    def step_temporal_features(ankle_y, fps, min_step_time=0.3):
        """
        Temporal gait features from one ankle trajectory.
        """
        ankle_y = np.asarray(ankle_y, dtype=float)
        peaks = GaitAnalysisEngine.detect_step_events_from_ankle(ankle_y, fps, min_step_time=min_step_time)

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
        step_intervals = np.diff(times)  # step durations in seconds

        mean_step = float(step_intervals.mean())
        std_step = float(step_intervals.std())
        cadence = 60.0 / mean_step if mean_step > 0 else np.nan

        # Stride = two steps
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
    def compute_clip_features(clip, fps):
        """
        Compute gait features from a single clip.
        """
        clip = np.asarray(clip)
        if clip.ndim != 3 or clip.shape[1] != GaitAnalysisEngine.N_JOINTS:
            raise ValueError(f"clip must be of shape (T, {GaitAnalysisEngine.N_JOINTS}, 3), got {clip.shape}")

        # Normalize if needed (pelvis should be near 0)
        pelvis = (clip[:, GaitAnalysisEngine.LEFT_HIP] + clip[:, GaitAnalysisEngine.RIGHT_HIP]) / 2
        pelvis_mean_norm = np.linalg.norm(pelvis.mean(axis=0))

        if pelvis_mean_norm > 1e-2:
            pose_norm = GaitAnalysisEngine.normalize_pose_3d(clip)
        else:
            pose_norm = clip

        feats = {}

        # Basic spatial features
        left_ankle_y = pose_norm[:, GaitAnalysisEngine.LEFT_ANKLE, 1]
        right_ankle_y = pose_norm[:, GaitAnalysisEngine.RIGHT_ANKLE, 1]

        feats["step_height_L"] = float(left_ankle_y.max() - left_ankle_y.min())
        feats["step_height_R"] = float(right_ankle_y.max() - right_ankle_y.min())

        left_ankle_x = pose_norm[:, GaitAnalysisEngine.LEFT_ANKLE, 0]
        right_ankle_x = pose_norm[:, GaitAnalysisEngine.RIGHT_ANKLE, 0]

        feats["step_length_L"] = float(left_ankle_x.max() - left_ankle_x.min())
        feats["step_length_R"] = float(right_ankle_x.max() - right_ankle_x.min())

        left_hip_y = pose_norm[:, GaitAnalysisEngine.LEFT_HIP, 1]
        right_hip_y = pose_norm[:, GaitAnalysisEngine.RIGHT_HIP, 1]
        pelvis_diff = left_hip_y - right_hip_y

        feats["pelvis_drop_mean"] = float(pelvis_diff.mean())
        feats["pelvis_drop_std"] = float(pelvis_diff.std())

        left_sh_x = pose_norm[:, GaitAnalysisEngine.LEFT_SHOULDER, 0]
        right_sh_x = pose_norm[:, GaitAnalysisEngine.RIGHT_SHOULDER, 0]
        trunk_lean = left_sh_x - right_sh_x

        feats["trunk_lean_mean"] = float(trunk_lean.mean())
        feats["trunk_lean_std"] = float(trunk_lean.std())

        left_heel_y = pose_norm[:, GaitAnalysisEngine.LEFT_HEEL, 1]
        right_heel_y = pose_norm[:, GaitAnalysisEngine.RIGHT_HEEL, 1]

        feats["heel_range_L"] = float(left_heel_y.max() - left_heel_y.min())
        feats["heel_range_R"] = float(right_heel_y.max() - right_heel_y.min())

        eps = 1e-6
        hL, hR = feats["step_height_L"], feats["step_height_R"]
        lL, lR = feats["step_length_L"], feats["step_length_R"]

        feats["step_height_symmetry"] = float((hL - hR) / (hL + hR + eps))
        feats["step_length_symmetry"] = float((lL - lR) / (lL + lR + eps))

        # Knee motion
        left_knee_move = GaitAnalysisEngine.moving_and_still_times(pose_norm, GaitAnalysisEngine.LEFT_KNEE, fps)
        right_knee_move = GaitAnalysisEngine.moving_and_still_times(pose_norm, GaitAnalysisEngine.RIGHT_KNEE, fps)

        for k, v in left_knee_move.items():
            feats[f"knee_L_{k}"] = v
        for k, v in right_knee_move.items():
            feats[f"knee_R_{k}"] = v

        feats["knee_L_rom_y"] = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.LEFT_KNEE, axis="y")["rom_y"]
        feats["knee_R_rom_y"] = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.RIGHT_KNEE, axis="y")["rom_y"]

        # Joint ROM
        hip_L_rom_y = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.LEFT_HIP, axis="y")["rom_y"]
        hip_R_rom_y = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.RIGHT_HIP, axis="y")["rom_y"]
        feats["hip_L_rom_y"] = hip_L_rom_y
        feats["hip_R_rom_y"] = hip_R_rom_y

        shoulder_L_rom_x = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.LEFT_SHOULDER, axis="x")["rom_x"]
        shoulder_R_rom_x = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.RIGHT_SHOULDER, axis="x")["rom_x"]
        feats["shoulder_L_rom_x"] = shoulder_L_rom_x
        feats["shoulder_R_rom_x"] = shoulder_R_rom_x

        ankle_L_rom_y = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.LEFT_ANKLE, axis="y")["rom_y"]
        ankle_R_rom_y = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.RIGHT_ANKLE, axis="y")["rom_y"]
        feats["ankle_L_rom_y"] = ankle_L_rom_y
        feats["ankle_R_rom_y"] = ankle_R_rom_y

        # ROM asymmetries
        feats["knee_rom_asym"] = GaitAnalysisEngine.asymmetry(feats["knee_L_rom_y"], feats["knee_R_rom_y"])
        feats["hip_rom_asym"] = GaitAnalysisEngine.asymmetry(hip_L_rom_y, hip_R_rom_y)
        feats["shoulder_rom_asym"] = GaitAnalysisEngine.asymmetry(shoulder_L_rom_x, shoulder_R_rom_x)
        feats["ankle_rom_asym"] = GaitAnalysisEngine.asymmetry(ankle_L_rom_y, ankle_R_rom_y)

        # Stance / swing ratio
        ankle_L_move = GaitAnalysisEngine.moving_and_still_times(pose_norm, GaitAnalysisEngine.LEFT_ANKLE, fps)
        ankle_R_move = GaitAnalysisEngine.moving_and_still_times(pose_norm, GaitAnalysisEngine.RIGHT_ANKLE, fps)

        feats["ankle_L_moving_fraction"] = ankle_L_move["moving_fraction"]
        feats["ankle_L_still_fraction"] = ankle_L_move["still_fraction"]
        feats["ankle_R_moving_fraction"] = ankle_R_move["moving_fraction"]
        feats["ankle_R_still_fraction"] = ankle_R_move["still_fraction"]

        stance_ratio_L = ankle_L_move["still_fraction"] / (ankle_L_move["moving_fraction"] + 1e-6)
        stance_ratio_R = ankle_R_move["still_fraction"] / (ankle_R_move["moving_fraction"] + 1e-6)

        feats["stance_ratio_L"] = float(stance_ratio_L)
        feats["stance_ratio_R"] = float(stance_ratio_R)
        feats["stance_ratio_asym"] = GaitAnalysisEngine.asymmetry(stance_ratio_L, stance_ratio_R)

        # Joint angles
        knee_angle_L = GaitAnalysisEngine.joint_angle(
            pose_norm[:, GaitAnalysisEngine.LEFT_HIP, :],
            pose_norm[:, GaitAnalysisEngine.LEFT_KNEE, :],
            pose_norm[:, GaitAnalysisEngine.LEFT_ANKLE, :],
        )
        knee_angle_R = GaitAnalysisEngine.joint_angle(
            pose_norm[:, GaitAnalysisEngine.RIGHT_HIP, :],
            pose_norm[:, GaitAnalysisEngine.RIGHT_KNEE, :],
            pose_norm[:, GaitAnalysisEngine.RIGHT_ANKLE, :],
        )

        feats["knee_angle_L_mean"] = float(knee_angle_L.mean())
        feats["knee_angle_L_std"] = float(knee_angle_L.std())
        feats["knee_angle_L_rom"] = float(knee_angle_L.max() - knee_angle_L.min())

        feats["knee_angle_R_mean"] = float(knee_angle_R.mean())
        feats["knee_angle_R_std"] = float(knee_angle_R.std())
        feats["knee_angle_R_rom"] = float(knee_angle_R.max() - knee_angle_R.min())

        hip_angle_L = GaitAnalysisEngine.joint_angle(
            pose_norm[:, GaitAnalysisEngine.LEFT_SHOULDER, :],
            pose_norm[:, GaitAnalysisEngine.LEFT_HIP, :],
            pose_norm[:, GaitAnalysisEngine.LEFT_KNEE, :],
        )
        hip_angle_R = GaitAnalysisEngine.joint_angle(
            pose_norm[:, GaitAnalysisEngine.RIGHT_SHOULDER, :],
            pose_norm[:, GaitAnalysisEngine.RIGHT_HIP, :],
            pose_norm[:, GaitAnalysisEngine.RIGHT_KNEE, :],
        )

        feats["hip_angle_L_mean"] = float(hip_angle_L.mean())
        feats["hip_angle_L_std"] = float(hip_angle_L.std())
        feats["hip_angle_L_rom"] = float(hip_angle_L.max() - hip_angle_L.min())

        feats["hip_angle_R_mean"] = float(hip_angle_R.mean())
        feats["hip_angle_R_std"] = float(hip_angle_R.std())
        feats["hip_angle_R_rom"] = float(hip_angle_R.max() - hip_angle_R.min())

        ankle_angle_L = GaitAnalysisEngine.joint_angle(
            pose_norm[:, GaitAnalysisEngine.LEFT_KNEE, :],
            pose_norm[:, GaitAnalysisEngine.LEFT_ANKLE, :],
            pose_norm[:, GaitAnalysisEngine.LEFT_FOOT_INDEX, :],
        )
        ankle_angle_R = GaitAnalysisEngine.joint_angle(
            pose_norm[:, GaitAnalysisEngine.RIGHT_KNEE, :],
            pose_norm[:, GaitAnalysisEngine.RIGHT_ANKLE, :],
            pose_norm[:, GaitAnalysisEngine.RIGHT_FOOT_INDEX, :],
        )

        feats["ankle_angle_L_mean"] = float(ankle_angle_L.mean())
        feats["ankle_angle_L_std"] = float(ankle_angle_L.std())
        feats["ankle_angle_L_rom"] = float(ankle_angle_L.max() - ankle_angle_L.min())

        feats["ankle_angle_R_mean"] = float(ankle_angle_R.mean())
        feats["ankle_angle_R_std"] = float(ankle_angle_R.std())
        feats["ankle_angle_R_rom"] = float(ankle_angle_R.max() - ankle_angle_R.min())

        # Angle-based ROM asymmetries
        feats["knee_angle_rom_asym"] = GaitAnalysisEngine.asymmetry(
            feats["knee_angle_L_rom"], feats["knee_angle_R_rom"]
        )
        feats["hip_angle_rom_asym"] = GaitAnalysisEngine.asymmetry(
            feats["hip_angle_L_rom"], feats["hip_angle_R_rom"]
        )
        feats["ankle_angle_rom_asym"] = GaitAnalysisEngine.asymmetry(
            feats["ankle_angle_L_rom"], feats["ankle_angle_R_rom"]
        )

        # Temporal gait features
        left_temporal = GaitAnalysisEngine.step_temporal_features(left_ankle_y, fps)
        right_temporal = GaitAnalysisEngine.step_temporal_features(right_ankle_y, fps)

        for k, v in left_temporal.items():
            feats[f"step_L_{k}"] = float(v) if v is not None else np.nan
        for k, v in right_temporal.items():
            feats[f"step_R_{k}"] = float(v) if v is not None else np.nan

        # Asymmetries from temporal features
        if not np.isnan(left_temporal["mean_step_time"]) and not np.isnan(right_temporal["mean_step_time"]):
            feats["step_time_asym"] = GaitAnalysisEngine.asymmetry(
                left_temporal["mean_step_time"], right_temporal["mean_step_time"]
            )
        else:
            feats["step_time_asym"] = np.nan

        if not np.isnan(left_temporal["cadence"]) and not np.isnan(right_temporal["cadence"]):
            feats["cadence_asym"] = GaitAnalysisEngine.asymmetry(
                left_temporal["cadence"], right_temporal["cadence"]
            )
        else:
            feats["cadence_asym"] = np.nan

        # Step width proxy
        ankle_L_x = pose_norm[:, GaitAnalysisEngine.LEFT_ANKLE, 0]
        ankle_R_x = pose_norm[:, GaitAnalysisEngine.RIGHT_ANKLE, 0]
        step_width_series = np.abs(ankle_L_x - ankle_R_x)

        feats["step_width_mean"] = float(step_width_series.mean())
        feats["step_width_std"] = float(step_width_series.std())

        return feats
    
    @staticmethod
    def extract_features_from_csv(csv_path: Path, video_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Extract features from a CSV file containing MediaPipe landmarks.
        Returns both features DataFrame and gait cycles (if available)
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Check if the CSV has the expected columns
            required_columns = ['frame', 'landmark_id', 'x_norm', 'y_norm', 'z_norm']
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV file missing required columns: {required_columns}")
                return pd.DataFrame(), None
            
            # Get video info if available
            fps = 30.0  # Default FPS
            if video_path and video_path.exists():
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                except:
                    logger.warning(f"Could not determine FPS from video, using default: {fps}")
            
            # Convert long format to pose tensor
            pose_rows = []
            for source_file, group in df.groupby("source_file" if "source_file" in df.columns else lambda x: 0):
                T = group["frame"].nunique()
                pose = np.zeros((T, GaitAnalysisEngine.N_JOINTS, 3), dtype=np.float32)

                for _, r in group.iterrows():
                    f = int(r.frame)
                    j = int(r.landmark_id)
                    pose[f, j] = [r.x_norm, r.y_norm, r.z_norm]

                pose_rows.append(pose)
            
            if not pose_rows:
                st.error("No valid pose data found in CSV")
                return pd.DataFrame(), None
            
            # Extract features for each pose
            all_features = []
            gait_cycles = []
            
            for i, pose in enumerate(pose_rows):
                try:
                    # Normalize pose
                    pose_norm = GaitAnalysisEngine.normalize_pose_3d(pose)
                    
                    # Extract gait cycles
                    cycles = GaitAnalysisEngine.extract_gait_cycle_clips(
                        pose_norm, fps, cycles=1, resample_frames=60
                    )
                    gait_cycles.extend(cycles)
                    
                    # Extract features from full pose
                    features = GaitAnalysisEngine.compute_clip_features(pose_norm, fps)
                    features["clip_id"] = i
                    all_features.append(features)
                except Exception as e:
                    logger.error(f"Error extracting features from clip {i}: {e}")
            
            if not all_features:
                st.error("Failed to extract features from any clips")
                return pd.DataFrame(), None
            
            # Convert to DataFrame
            df_features = pd.DataFrame(all_features)
            
            # Convert gait cycles to numpy array
            gait_cycles_array = np.array(gait_cycles) if gait_cycles else None
            
            return df_features, gait_cycles_array
        
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            logger.error(f"Error processing CSV file: {e}")
            return pd.DataFrame(), None
    
    @staticmethod
    def save_features(df_features: pd.DataFrame, video_path: Path) -> Optional[Path]:
        """
        Save extracted features to a CSV file.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_stem = video_path.stem
            features_path = FEATURES_DIR / f"{video_stem}_features_{timestamp}.csv"
            
            df_features.to_csv(features_path, index=False)
            logger.info(f"Features saved to: {features_path}")
            return features_path
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            return None
    
    @staticmethod
    def save_gait_cycles(gait_cycles: np.ndarray, video_path: Path) -> Optional[Path]:
        """
        Save gait cycles to a numpy file.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_stem = video_path.stem
            cycles_path = GAIT_CYCLES_DIR / f"{video_stem}_gait_cycles_{timestamp}.npz"
            
            np.savez_compressed(cycles_path, gait_cycles=gait_cycles)
            logger.info(f"Gait cycles saved to: {cycles_path}")
            return cycles_path
        except Exception as e:
            logger.error(f"Error saving gait cycles: {e}")
            return None
    
    @staticmethod
    def extract_gait_cycles_for_advanced_model(gait_cycles: np.ndarray, 
                                              window_size: int = 60, 
                                              stride: int = 30) -> np.ndarray:
        """
        Extract sliding windows from gait cycles for the advanced model
        """
        try:
            if gait_cycles is None or len(gait_cycles) == 0:
                model_logger.warning("No gait cycles available for advanced model")
                return np.array([])
            
            # Average across all gait cycles
            avg_cycle = np.mean(gait_cycles, axis=0)
            
            # Extract sliding windows
            windows = []
            for i in range(0, len(avg_cycle) - window_size + 1, stride):
                window = avg_cycle[i:i+window_size]
                windows.append(window)
            
            if not windows:
                model_logger.warning("No windows extracted from gait cycles")
                return np.array([])
            
            # Convert to numpy array
            X = np.array(windows)
            
            # Select only the GAIT_JOINTS
            X = X[:, GaitAnalysisEngine.GAIT_JOINTS, :]
            
            model_logger.info(f"Extracted {len(X)} windows of shape {X.shape} for advanced model")
            return X
        
        except Exception as e:
            model_logger.error(f"Error extracting gait cycles for advanced model: {e}")
            model_logger.error(traceback.format_exc())
            return np.array([])


# ═══════════════════════════════════════════════════════════════════════════
# MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class ModelManager:
    """Production-grade model management for gait classification"""
    
    @staticmethod
    def load_model(model_path: Optional[Path] = None, model_type: str = "baseline") -> Optional[object]:
        """Load the model from file with extensive error handling and logging"""
        try:
            if model_path is None:
                # Try to find the model in the default location
                if model_type == "baseline":
                    model_files = list(BASELINE_MODELS_DIR.glob("*.bin")) + list(BASELINE_MODELS_DIR.glob("*.pkl")) + list(BASELINE_MODELS_DIR.glob("*.joblib"))
                    if not model_files:
                        model_logger.error(f"No baseline model files found in {BASELINE_MODELS_DIR}")
                        return None
                    model_path = model_files[0]
                    model_logger.info(f"Using baseline model: {model_path}")
                elif model_type == "advanced":
                    model_files = list(ADVANCED_MODELS_DIR.glob("*.h5")) + list(ADVANCED_MODELS_DIR.glob("*.pth")) + list(ADVANCED_MODELS_DIR.glob("*.pt"))
                    if not model_files:
                        model_logger.error(f"No advanced model files found in {ADVANCED_MODELS_DIR}")
                        return None
                    model_path = model_files[0]
                    model_logger.info(f"Using advanced model: {model_path}")
                else:
                    model_logger.error(f"Unknown model type: {model_type}")
                    return None
            
            if not model_path.exists():
                model_logger.error(f"Model file not found: {model_path}")
                return None
            
            model_logger.info(f"Loading {model_type} model from {model_path}")
            
            # Try different loading methods based on file extension and model type
            if model_type == "baseline":
                if model_path.suffix == '.bin':
                    model = xgb.XGBClassifier()
                    model.load_model(str(model_path))
                    model_logger.info("Loaded XGBoost model from .bin file")
                elif model_path.suffix in ['.pkl', '.pickle']:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    model_logger.info("Loaded model from pickle file")
                elif model_path.suffix == '.joblib':
                    model = joblib.load(model_path)
                    model_logger.info("Loaded model from joblib file")
                else:
                    model_logger.error(f"Unsupported baseline model file format: {model_path.suffix}")
                    return None
            elif model_type == "advanced":
                # Try to load ST-GCN or other advanced models
                try:
                    import torch
                    if model_path.suffix in ['.h5', '.hdf5']:
                        # Try loading Keras/TensorFlow model
                        try:
                            import tensorflow as tf
                            model = tf.keras.models.load_model(str(model_path))
                            model_logger.info("Loaded TensorFlow/Keras model")
                        except:
                            model_logger.warning("Failed to load with TensorFlow, trying PyTorch")
                    if model_path.suffix in ['.pt', '.pth']:
                        model = torch.load(str(model_path))
                        model_logger.info("Loaded PyTorch model")
                    else:
                        model_logger.error(f"Unsupported advanced model file format: {model_path.suffix}")
                        return None
                except ImportError:
                    model_logger.error("PyTorch/TensorFlow not available for advanced model")
                    return None
            else:
                model_logger.error(f"Unknown model type: {model_type}")
                return None
            
            model_logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            model_logger.error(f"Failed to load {model_type} model: {e}")
            model_logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def prepare_features_for_baseline_model(df_features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for baseline model prediction with extensive error handling"""
        try:
            model_logger.info("Preparing features for baseline model prediction")
            
            # Create a copy to avoid modifying the original
            df = df_features.copy()
            
            # Ensure all required features are present
            missing_features = [f for f in BASELINE_MODEL_FEATURES if f not in df.columns]
            if missing_features:
                model_logger.warning(f"Missing features: {missing_features}")
                # Add missing features with NaN values
                for feature in missing_features:
                    df[feature] = np.nan
            
            # Select only the required features in the correct order
            df_model = df[BASELINE_MODEL_FEATURES].copy()
            
            # Handle NaN values - fill with median of the column
            for col in df_model.columns:
                if df_model[col].isna().any():
                    median_val = df_model[col].median()
                    if not np.isnan(median_val):
                        df_model[col].fillna(median_val, inplace=True)
                        model_logger.debug(f"Filled NaN values in {col} with median: {median_val}")
                    else:
                        df_model[col].fillna(0, inplace=True)
                        model_logger.debug(f"Filled NaN values in {col} with 0 (median was NaN)")
            
            model_logger.info(f"Prepared {len(df_model)} samples with {len(df_model.columns)} features for baseline model")
            return df_model
        except Exception as e:
            model_logger.error(f"Failed to prepare features for baseline model: {e}")
            model_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    @staticmethod
    def prepare_features_for_advanced_model(gait_cycles: np.ndarray) -> Optional[np.ndarray]:
        """Prepare features for advanced model prediction with extensive error handling"""
        try:
            model_logger.info("Preparing features for advanced model prediction")
            
            if gait_cycles is None or len(gait_cycles) == 0:
                model_logger.error("No gait cycles available for advanced model")
                return None
            
            # Extract sliding windows
            X = GaitAnalysisEngine.extract_gait_cycles_for_advanced_model(gait_cycles)
            
            if X.size == 0:
                model_logger.error("Failed to extract windows from gait cycles")
                return None
            
            # Convert from (N, T, J, 3) → (N, 3, T, J) for ST-GCN
            X_stgcn = X.transpose(0, 3, 1, 2)
            
            model_logger.info(f"Prepared {len(X_stgcn)} samples with shape {X_stgcn.shape} for advanced model")
            return X_stgcn
        except Exception as e:
            model_logger.error(f"Failed to prepare features for advanced model: {e}")
            model_logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def predict_with_baseline_model(model: xgb.XGBClassifier, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the baseline model with extensive error handling"""
        try:
            model_logger.info("Making predictions with baseline model")
            
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(features)
            
            # Get class predictions
            y_pred = model.predict(features)
            
            model_logger.info(f"Baseline model prediction completed: {len(y_pred)} samples")
            return y_pred, y_pred_proba
        except Exception as e:
            model_logger.error(f"Baseline model prediction failed: {e}")
            model_logger.error(traceback.format_exc())
            return np.array([]), np.array([])
    
    @staticmethod
    def predict_with_advanced_model(model, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using the advanced model with extensive error handling"""
        try:
            model_logger.info("Making predictions with advanced model")
            
            # Try different model types
            if hasattr(model, 'predict'):
                # Standard predict method
                try:
                    if hasattr(model, 'predict_proba'):
                        # Has predict_proba (classification)
                        y_pred_proba = model.predict_proba(features)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        model_logger.info(f"Advanced model prediction completed with probabilities: {len(y_pred)} samples")
                        return y_pred, y_pred_proba
                    else:
                        # Only predict (regression or binary classification)
                        y_pred = model.predict(features)
                        model_logger.info(f"Advanced model prediction completed: {len(y_pred)} samples")
                        return y_pred, None
                except Exception as e:
                    model_logger.error(f"Error in standard predict method: {e}")
                    raise
            else:
                # Try PyTorch specific methods
                try:
                    import torch
                    if isinstance(model, torch.nn.Module):
                        model.eval()
                        with torch.no_grad():
                            # Convert to tensor if needed
                            if not isinstance(features, torch.Tensor):
                                features_tensor = torch.tensor(features, dtype=torch.float32)
                            else:
                                features_tensor = features

                            # Add batch dimension if missing
                            if len(features_tensor.shape) == 3:
                                features_tensor = features_tensor.unsqueeze(0)

                            # Forward pass
                            outputs = model(features_tensor)

                            # Handle different output types
                            if isinstance(outputs, tuple):
                                # Multiple outputs (e.g., features and predictions)
                                y_pred = outputs[0].cpu().numpy()
                                if len(outputs) > 1 and hasattr(outputs[1], 'cpu'):
                                    y_pred_proba = outputs[1].cpu().numpy()
                                else:
                                    y_pred_proba = None
                            else:
                                # Single output
                                y_pred = outputs.cpu().numpy()
                                y_pred_proba = None

                            # Convert to class predictions if needed
                            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                                y_pred = np.argmax(y_pred, axis=1)
                            elif y_pred.ndim > 1:
                                y_pred = y_pred.flatten()

                            model_logger.info(f"PyTorch model prediction completed: {len(y_pred)} samples")
                            return y_pred, y_pred_proba
                    else:
                        model_logger.error("Model has no recognizable predict method")
                        raise ValueError("Model has no recognizable predict method")
                except Exception as e:
                    model_logger.error(f"Error in PyTorch predict method: {e}")
                    raise
        except Exception as e:
            model_logger.error(f"Advanced model prediction failed: {e}")
            model_logger.error(traceback.format_exc())
            return np.array([]), None
    
    @staticmethod
    def save_predictions(df_features: pd.DataFrame, y_pred: np.ndarray, 
                        y_pred_proba: np.ndarray, file_prefix: str, 
                        model_type: str = "baseline") -> Path:
        """Save predictions to a CSV file with extensive error handling"""
        try:
            model_logger.info(f"Saving {model_type} model predictions")
            
            # Create a DataFrame with the predictions
            df_predictions = df_features.copy()
            
            # Add predictions
            df_predictions['predicted_class'] = y_pred
            
            # Add appropriate labels based on model type
            if model_type == "baseline":
                df_predictions['predicted_label'] = [BASELINE_CLASS_LABELS.get(c, f"Unknown({c})") for c in y_pred]
                
                # Add probability columns
                for i, label in BASELINE_CLASS_LABELS.items():
                    df_predictions[f'prob_{label.replace(" ", "_").lower()}'] = y_pred_proba[:, i]
            elif model_type == "advanced_binary":
                df_predictions['predicted_label'] = [ADVANCED_BINARY_LABELS.get(c, f"Unknown({c})") for c in y_pred]
                
                # Add probability columns
                for i, label in ADVANCED_BINARY_LABELS.items():
                    df_predictions[f'prob_{label.replace(" ", "_").lower()}'] = y_pred_proba[:, i]
            elif model_type == "advanced_multilabel":
                # For multilabel, we need to handle differently
                df_predictions['predicted_label'] = "Multilabel Prediction"
                
                # Add probability columns
                for i, label in ADVANCED_MULTILABEL_LABELS.items():
                    df_predictions[f'prob_{label.replace(" ", "_").lower()}'] = y_pred_proba[:, i]
            else:
                df_predictions['predicted_label'] = f"Unknown model type: {model_type}"
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            predictions_path = PREDICTIONS_DIR / f"{file_prefix}_{model_type}_predictions_{timestamp}.csv"
            df_predictions.to_csv(predictions_path, index=False)
            
            model_logger.info(f"Predictions saved to: {predictions_path}")
            return predictions_path
        except Exception as e:
            model_logger.error(f"Failed to save {model_type} predictions: {e}")
            model_logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def create_prediction_visualization(df_features: pd.DataFrame, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray, model_type: str = "baseline"):
        """Create visualizations for the prediction results with extensive error handling"""
        try:
            model_logger.info(f"Creating {model_type} model prediction visualization")
            
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(18, 12))
            
            # Get appropriate labels based on model type
            if model_type == "baseline":
                class_labels = BASELINE_CLASS_LABELS
            elif model_type == "advanced_binary":
                class_labels = ADVANCED_BINARY_LABELS
            elif model_type == "advanced_multilabel":
                class_labels = ADVANCED_MULTILABEL_LABELS
            else:
                class_labels = {i: f"Class {i}" for i in range(len(y_pred_proba[0]) if y_pred_proba is not None else len(set(y_pred)))}
            
            # 1. Class distribution
            ax1 = plt.subplot(2, 3, 1)
            class_counts = pd.Series(y_pred).value_counts().sort_index()
            labels = [class_labels.get(i, f"Class {i}") for i in class_counts.index]
            colors = ['green', 'gold', 'orange', 'red', 'purple', 'brown', 'pink', 'gray'][:len(class_counts)]
            ax1.pie(class_counts, labels=labels, autopct='%1.1f%%', colors=colors)
            ax1.set_title('Predicted Class Distribution')
            
            # 2. Probability distribution
            ax2 = plt.subplot(2, 3, 2)
            if y_pred_proba is not None:
                for i, label in class_labels.items():
                    if i < y_pred_proba.shape[1]:
                        ax2.hist(y_pred_proba[:, i], alpha=0.5, label=label, bins=20)
                ax2.set_xlabel('Probability')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Probability Distribution')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No probability data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Probability Distribution')
            
            # 3. Top feature importances (if available)
            ax3 = plt.subplot(2, 3, 3)
            try:
                if model_type == "baseline" and 'model' in st.session_state and st.session_state.model:
                    # Get feature importances from the model
                    importances = st.session_state.model.feature_importances_
                    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
                    ax3.barh(range(len(indices)), importances[indices])
                    ax3.set_yticks(range(len(indices)))
                    ax3.set_yticklabels([BASELINE_MODEL_FEATURES[i] for i in indices])
                    ax3.set_xlabel('Importance')
                    ax3.set_title('Top 10 Feature Importances')
                else:
                    ax3.text(0.5, 0.5, 'Feature importances not available', 
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Feature Importances')
            except Exception as e:
                model_logger.error(f"Error creating feature importance plot: {e}")
                ax3.text(0.5, 0.5, 'Feature importances not available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Feature Importances')
            
            # 4. Radar chart for key metrics
            ax4 = plt.subplot(2, 3, 4, polar=True)
            key_metrics = ['step_height_symmetry', 'step_length_symmetry', 'knee_rom_asym', 
                          'hip_rom_asym', 'ankle_rom_asym', 'step_time_asym']
            
            # Get available metrics
            available_metrics = [m for m in key_metrics if m in df_features.columns]
            if available_metrics:
                values = df_features.iloc[0][available_metrics].values
                # Normalize values for better visualization
                values = np.abs(values) / np.max(np.abs(values)) if np.max(np.abs(values)) > 0 else values
                
                # Number of variables
                N = len(available_metrics)
                
                # Compute angle for each axis
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                values = np.concatenate((values, [values[0]]))  # Complete the loop
                angles += angles[:1]  # Complete the loop
                
                # Plot
                ax4.plot(angles, values, 'o-', linewidth=2)
                ax4.fill(angles, values, alpha=0.25)
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
                ax4.set_ylim(0, 1)
                ax4.set_title("Key Symmetry Metrics")
            else:
                ax4.text(0.5, 0.5, 'Key metrics not available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Key Metrics')
            
            # 5. Prediction confidence
            ax5 = plt.subplot(2, 3, 5)
            if y_pred_proba is not None:
                max_probs = np.max(y_pred_proba, axis=1)
                ax5.hist(max_probs, bins=20, alpha=0.7)
                ax5.set_xlabel('Maximum Probability')
                ax5.set_ylabel('Frequency')
                ax5.set_title('Prediction Confidence')
            else:
                ax5.text(0.5, 0.5, 'No probability data available', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Prediction Confidence')
            
            # 6. Class probability heatmap
            ax6 = plt.subplot(2, 3, 6)
            if y_pred_proba is not None:
                prob_df = pd.DataFrame(y_pred_proba, columns=[class_labels[i] for i in range(len(class_labels))])
                sns.heatmap(prob_df.T, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax6)
                ax6.set_title('Class Probabilities')
                ax6.set_xlabel('Sample')
                ax6.set_ylabel('Class')
            else:
                ax6.text(0.5, 0.5, 'No probability data available', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Class Probabilities')
            
            plt.tight_layout()
            model_logger.info(f"Created {model_type} model prediction visualization")
            return fig
        except Exception as e:
            model_logger.error(f"Failed to create {model_type} model prediction visualization: {e}")
            model_logger.error(traceback.format_exc())
            return None


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_gait_score_dashboard(features_df):
    """Create a traffic-light style dashboard for quick health assessment"""
    from matplotlib.patches import Circle
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define gait parameters and their healthy ranges
    gait_metrics = {
        'Step Symmetry': ('step_height_symmetry', 0.15, 0.25),
        'Cadence': ('step_L_cadence', 100, 140),
        'Step Regularity': ('step_L_step_time_cv', 0.05, 0.15),
        'Knee Flexion': ('knee_L_rom_y', 0.3, 0.7),
        'Hip Movement': ('hip_L_rom_y', 0.2, 0.5),
        'Ankle Control': ('ankle_L_rom_y', 0.4, 0.8)
    }
    
    positions = [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8),
                 (0.2, 0.5), (0.5, 0.5), (0.8, 0.5)]
    
    for i, (metric_name, (feature_key, min_good, max_good)) in enumerate(gait_metrics.items()):
        x, y = positions[i]
        
        if feature_key in features_df.columns:
            value = features_df.iloc[0][feature_key]
            
            # Determine color based on value
            if metric_name == 'Cadence':
                if min_good <= value <= max_good:
                    color = 'green'
                elif value < min_good - 20 or value > max_good + 20:
                    color = 'red'
                else:
                    color = 'orange'
            elif 'symmetry' in feature_key.lower() or 'cv' in feature_key.lower():
                if abs(value) <= min_good:
                    color = 'green'
                elif abs(value) <= max_good:
                    color = 'orange'
                else:
                    color = 'red'
            else:  # ROM features
                if min_good <= value <= max_good:
                    color = 'green'
                elif value < min_good * 0.7 or value > max_good * 1.3:
                    color = 'red'
                else:
                    color = 'orange'
            
            # Draw traffic light
            circle = Circle((x, y), 0.08, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # Add metric name
            ax.text(x, y - 0.15, metric_name, ha='center', fontsize=10, weight='bold')
            
            # Add value
            ax.text(x, y, f'{value:.2f}', ha='center', va='center', 
                   fontsize=9, color='white', weight='bold')
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='green', s=100, label='Normal'),
        plt.scatter([], [], c='orange', s=100, label='Caution'),
        plt.scatter([], [], c='red', s=100, label='Attention Needed')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Gait Health Assessment Dashboard', fontsize=16, weight='bold')
    ax.axis('off')
    
    return fig

def create_movement_flow_chart(features_df):
    """Create a flow chart showing movement patterns through gait cycle"""
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define gait phases and their connections
    phases = [
        ('Initial Contact', 0.1, 0.8),
        ('Loading Response', 0.3, 0.8),
        ('Mid Stance', 0.5, 0.8),
        ('Terminal Stance', 0.7, 0.8),
        ('Pre-Swing', 0.9, 0.8),
        ('Initial Swing', 0.1, 0.4),
        ('Mid Swing', 0.5, 0.4),
        ('Terminal Swing', 0.9, 0.4)
    ]
    
    # Draw phase boxes
    box_width = 0.15
    box_height = 0.1
    
    for phase_name, x, y in phases:
        # Color based on phase efficiency (using stance/swing ratios)
        if 'stance' in phase_name.lower():
            efficiency = features_df.iloc[0].get('stance_ratio_L', 0.6)
            color = plt.cm.RdYlGn(efficiency)
        else:
            efficiency = 1 - features_df.iloc[0].get('stance_ratio_L', 0.6)
            color = plt.cm.RdYlGn(efficiency)
        
        rect = mpatches.Rectangle((x - box_width/2, y - box_height/2), 
                                  box_width, box_height, 
                                  facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(x, y, phase_name, ha='center', va='center', 
               fontsize=9, weight='bold')
    
    # Draw flow arrows
    arrows = [
        ((0.175, 0.8), (0.225, 0.8)),
        ((0.375, 0.8), (0.425, 0.8)),
        ((0.575, 0.8), (0.625, 0.8)),
        ((0.775, 0.8), (0.825, 0.8)),
        ((0.925, 0.8), (0.925, 0.45)),
        ((0.925, 0.35), (0.525, 0.35)),
        ((0.475, 0.35), (0.075, 0.35)),
        ((0.025, 0.35), (0.025, 0.75)),
        ((0.025, 0.85), (0.075, 0.85))
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Add labels for stance and swing phases
    ax.text(0.5, 0.95, 'STANCE PHASE', ha='center', fontsize=12, 
           weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    ax.text(0.5, 0.25, 'SWING PHASE', ha='center', fontsize=12, 
           weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Gait Cycle Movement Pattern Flow', fontsize=16, weight='bold')
    ax.axis('off')
    
    return fig

def create_3d_joint_trajectory(gait_cycles):
    """Create 3D visualization of joint trajectories during gait"""
    from mpl_toolkits.mplot3d import Axes3D
    
    if gait_cycles is None or len(gait_cycles) == 0:
        return None
    
    fig = plt.figure(figsize=(15, 10))
    
    # Select key joints to visualize
    joint_names = ['Left Ankle', 'Right Ankle', 'Left Knee', 'Right Knee', 
                   'Left Hip', 'Right Hip']
    joint_indices = [27, 28, 25, 26, 23, 24]
    
    # Average across all gait cycles
    avg_cycle = np.mean(gait_cycles, axis=0)
    
    # Create subplots for different views
    views = [(0, 0), (0, 1), (1, 0), (1, 1)]
    view_labels = ['Front View', 'Side View', 'Top View', '3D View']
    view_angles = [(0, 0), (0, 90), (90, 0), (30, 45)]
    
    for idx, ((row, col), label, (elev, azim)) in enumerate(zip(views, view_labels, view_angles)):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        
        for joint_name, joint_idx in zip(joint_names, joint_indices):
            trajectory = avg_cycle[:, joint_idx, :]
            
            # Color based on left/right
            color = 'blue' if 'Left' in joint_name else 'red'
            
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                   color=color, linewidth=2, label=joint_name, alpha=0.8)
            
            # Mark start and end points
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                      color=color, s=100, marker='o')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                      color=color, s=100, marker='s')
        
        ax.set_xlabel('X (Forward)')
        ax.set_ylabel('Y (Lateral)')
        ax.set_zlabel('Z (Vertical)')
        ax.set_title(label, fontsize=12, weight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.legend(fontsize=8)
    
    plt.suptitle('3D Joint Trajectories During Gait Cycle', fontsize=16, weight='bold')
    plt.tight_layout()
    
    return fig

def create_gait_stability_index(features_df):
    """Create a comprehensive stability assessment with confidence bands"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Define stability metrics
    stability_metrics = [
        ('Dynamic Balance', 'pelvis_drop_std', 0.1),
        ('Step Consistency', 'step_L_step_time_cv', 0.1),
        ('Joint Coordination', 'knee_angle_rom_asym', 0.15),
        ('Movement Control', 'ankle_L_moving_fraction', 0.2),
        ('Rhythm Regularity', 'cadence_asym', 0.1),
        ('Postural Stability', 'trunk_lean_std', 0.1)
    ]
    
    labels = [metric[0] for metric in stability_metrics]
    values = []
    upper_bounds = []
    lower_bounds = []
    
    for metric_name, feature_key, threshold in stability_metrics:
        if feature_key in features_df.columns:
            value = abs(features_df.iloc[0][feature_key])
            values.append(value)
            
            # Create confidence bands (±20% of threshold)
            upper_bounds.append(min(value + threshold * 0.2, 1.0))
            lower_bounds.append(max(value - threshold * 0.2, 0.0))
        else:
            values.append(0.5)
            upper_bounds.append(0.7)
            lower_bounds.append(0.3)
    
    # Number of variables
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # Complete the loop
    values += values[:1]
    upper_bounds += upper_bounds[:1]
    lower_bounds += lower_bounds[:1]
    angles += angles[:1]
    
    # Create confidence band
    confidence_band = Polygon(list(zip(angles, upper_bounds)), 
                           alpha=0.2, facecolor='green', edgecolor='none')
    ax.add_patch(confidence_band)
    
    confidence_band_lower = Polygon(list(zip(angles, lower_bounds)), 
                             alpha=0.2, facecolor='red', edgecolor='none')
    ax.add_patch(confidence_band_lower)
    
    # Plot main values
    ax.plot(angles, values, 'o-', linewidth=3, color='blue', markersize=8)
    ax.fill(angles, values, alpha=0.25, color='blue')
    
    # Add threshold lines
    threshold_values = [threshold for _, _, threshold in stability_metrics]
    threshold_values += threshold_values[:1]
    ax.plot(angles, threshold_values, '--', linewidth=2, color='orange', 
           label='Optimal Range')
    
    # Formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Gait Stability Index', size=16, weight='bold', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    return fig

def create_temporal_gait_heatmap(gait_cycles):
    """Create a heatmap showing joint movement patterns over time"""
    import seaborn as sns
    
    if gait_cycles is None or len(gait_cycles) == 0:
        return None
    
    # Average across cycles
    avg_cycle = np.mean(gait_cycles, axis=0)
    
    # Select key joints and their vertical movement
    joint_data = {
        'Left Ankle': avg_cycle[:, 27, 1],
        'Right Ankle': avg_cycle[:, 28, 1],
        'Left Knee': avg_cycle[:, 25, 1],
        'Right Knee': avg_cycle[:, 26, 1],
        'Left Hip': avg_cycle[:, 23, 1],
        'Right Hip': avg_cycle[:, 24, 1],
        'Left Shoulder': avg_cycle[:, 11, 1],
        'Right Shoulder': avg_cycle[:, 12, 1]
    }
    
    # Create DataFrame
    df_heatmap = pd.DataFrame(joint_data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create custom colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    sns.heatmap(df_heatmap.T, cmap=cmap, center=0, 
                cbar_kws={'label': 'Vertical Position (Normalized)'},
                ax=ax)
    
    # Add gait phase annotations
    phase_boundaries = [0, 15, 30, 45, 60]  # Approximate frame boundaries
    phase_labels = ['Initial\nContact', 'Loading\nResponse', 'Mid\nStance', 
                   'Terminal\nStance', 'Swing\nPhase']
    
    for i, (boundary, label) in enumerate(zip(phase_boundaries[:-1], phase_labels)):
        ax.axvline(x=boundary, color='white', linestyle='--', alpha=0.5)
        ax.text(boundary + 7.5, -0.5, label, ha='center', fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    ax.set_title('Temporal Gait Pattern Heatmap', fontsize=16, weight='bold')
    ax.set_xlabel('Gait Cycle Timeline (%)')
    ax.set_ylabel('Joints')
    
    # Convert frame numbers to percentage
    ax.set_xticks(np.arange(0, 61, 10))
    ax.set_xticklabels([f'{i}%' for i in range(0, 101, 20)])
    
    plt.tight_layout()
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="MediaPipe Pose Detection",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .stAlert > div { padding: 1rem; }
    .stVideo { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎥 MediaPipe Pose Detection Pipeline")
    st.markdown("**Production-Grade Gait Analysis Application**")
    st.markdown("---")
    
    # Initialize session state with all necessary variables
    if 'uploaded_video_path' not in st.session_state:
        st.session_state.uploaded_video_path = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'output_videos' not in st.session_state:
        st.session_state.output_videos = {}
    if 'last_tab' not in st.session_state:
        st.session_state.last_tab = 0
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None
    if 'features_path' not in st.session_state:
        st.session_state.features_path = None
    if 'gait_cycles' not in st.session_state:
        st.session_state.gait_cycles = None
    if 'gait_cycles_path' not in st.session_state:
        st.session_state.gait_cycles_path = None
    if 'uploaded_csv_path' not in st.session_state:
        st.session_state.uploaded_csv_path = None
    if 'csv_features_df' not in st.session_state:
        st.session_state.csv_features_df = None
    if 'csv_features_path' not in st.session_state:
        st.session_state.csv_features_path = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'advanced_model' not in st.session_state:
        st.session_state.advanced_model = None
    if 'model_predictions' not in st.session_state:
        st.session_state.model_predictions = None
    if 'model_predictions_path' not in st.session_state:
        st.session_state.model_predictions_path = None
    if 'advanced_model_predictions' not in st.session_state:
        st.session_state.advanced_model_predictions = None
    if 'advanced_model_predictions_path' not in st.session_state:
        st.session_state.advanced_model_predictions_path = None
    
    # Sidebar
    with st.sidebar:
        st.header("📋 System Status")
        
        if CONFIG_PATH.exists():
            st.success("✅ Config file")
        else:
            st.error("❌ Config missing")
        
        if MEDIAPIPE_SCRIPT.exists():
            st.success("✅ MediaPipe script")
        else:
            st.error("❌ Script missing")
        
        if VideoConverter.check_ffmpeg():
            st.success("✅ FFmpeg available")
        else:
            st.warning("⚠️ FFmpeg unavailable")
        
        # Check for models
        baseline_model_files = list(BASELINE_MODELS_DIR.glob("*.bin")) + list(BASELINE_MODELS_DIR.glob("*.pkl")) + list(BASELINE_MODELS_DIR.glob("*.joblib"))
        advanced_model_files = list(ADVANCED_MODELS_DIR.glob("*.h5")) + list(ADVANCED_MODELS_DIR.glob("*.pth")) + list(ADVANCED_MODELS_DIR.glob("*.pt"))
        
        if baseline_model_files:
            st.success("✅ Baseline model available")
        else:
            st.error("❌ Baseline model missing")
        
        if advanced_model_files:
            st.success("✅ Advanced model available")
        else:
            st.warning("⚠️ Advanced model missing")
        
        st.markdown("---")
        st.header("📊 Current Session")
        
        if st.session_state.uploaded_video_path:
            st.info(f"📹 {st.session_state.uploaded_video_path.name}")
            if st.session_state.processing_complete:
                st.success("✅ Processed")
        else:
            st.info("No video uploaded")
        
        if st.session_state.features_df is not None:
            st.success("✅ Features extracted")
        
        if st.session_state.gait_cycles is not None:
            st.success("✅ Gait cycles extracted")
        
        if st.session_state.csv_features_df is not None:
            st.success("✅ CSV features extracted")
        
        if st.session_state.model is not None:
            st.success("✅ Baseline model loaded")
        
        if st.session_state.advanced_model is not None:
            st.success("✅ Advanced model loaded")
        
        if st.session_state.model_predictions is not None:
            st.success("✅ Baseline predictions made")
        
        if st.session_state.advanced_model_predictions is not None:
            st.success("✅ Advanced predictions made")
        
        st.markdown("---")
        
        if st.button("🔄 Reset", use_container_width=True):
            # Clear all session state
            keys_to_clear = [
                'uploaded_video_path', 'processing_complete', 'output_videos', 
                'last_tab', 'initialized', 'features_df', 'features_path',
                'gait_cycles', 'gait_cycles_path',
                'uploaded_csv_path', 'csv_features_df', 'csv_features_path',
                'model', 'advanced_model', 'model_predictions', 'model_predictions_path',
                'advanced_model_predictions', 'advanced_model_predictions_path'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear any file data in session state
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith('data_') or k.startswith('widget_')]
            for key in keys_to_clear:
                del st.session_state[key]
            
            st.rerun()
        
        with st.expander("🐛 Debug"):
            st.write(f"Uploads: {len(list(UPLOAD_DIR.glob('*')))}")
            st.write(f"Outputs: {len(list(OUTPUT_DIR.glob('*')))}")
            st.write(f"Features: {len(list(FEATURES_DIR.glob('*')))}")
            st.write(f"Gait cycles: {len(list(GAIT_CYCLES_DIR.glob('*')))}")
            st.write(f"Predictions: {len(list(PREDICTIONS_DIR.glob('*')))}")
            st.write(f"Session keys: {len(st.session_state.keys())}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Upload", "⚙️ Process", "🎬 Landmarker Videos", 
        "📊 Feature Engineering", "🔬 Detailed Analysis", "🤖 Model Prediction"
    ])

    # Track tab changes to preserve state
    current_tab = 0
    if tab1:
        current_tab = 0
    elif tab2:
        current_tab = 1
    elif tab3:
        current_tab = 2
    elif tab4:
        current_tab = 3
    elif tab5:
        current_tab = 4
    elif tab6:
        current_tab = 5
    
    # Only update the session state if the tab actually changed
    if current_tab != st.session_state.last_tab:
        st.session_state.last_tab = current_tab
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: UPLOAD
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.subheader("Upload Video for Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'],
                help="Upload walking/gait video"
            )
            
            if uploaded_file:
                st.info(f"📄 {uploaded_file.name}")
                st.info(f"📏 {uploaded_file.size / (1024*1024):.2f} MB")
                
                with st.spinner("Saving video..."):
                    video_path, is_duplicate = FileManager.save_uploaded_video(uploaded_file)
                
                if video_path:
                    # Only reset states when a new video is uploaded
                    if st.session_state.uploaded_video_path != video_path:
                        st.session_state.uploaded_video_path = video_path
                        st.session_state.processing_complete = False
                        st.session_state.output_videos = {}
                        st.session_state.features_df = None
                        st.session_state.features_path = None
                        st.session_state.gait_cycles = None
                        st.session_state.gait_cycles_path = None
                        st.session_state.model_predictions = None
                        st.session_state.model_predictions_path = None
                        st.session_state.advanced_model_predictions = None
                        st.session_state.advanced_model_predictions_path = None
                    
                    if is_duplicate:
                        st.warning("⚠️ This video was previously uploaded")
                    else:
                        st.success("✅ Video saved successfully!")
                else:
                    st.error("❌ Failed to save video")
        
        with col2:
            if st.session_state.uploaded_video_path:
                st.metric("Status", "Ready for processing")
                st.metric("Video", st.session_state.uploaded_video_path.name)
        
        # Display uploaded video
        if st.session_state.uploaded_video_path:
            st.markdown("---")
            st.subheader("📹 Video Preview")
            VideoDisplay.display_video_with_download(
                st.session_state.uploaded_video_path,
                "Original Video",
                "tab1_original"
            )
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: PROCESS
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.subheader("Process Video with MediaPipe")
        
        if not st.session_state.uploaded_video_path:
            st.info("👈 Please upload a video in the Upload tab first")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"📹 **Ready to process:** {st.session_state.uploaded_video_path.name}")
                
                # Show config preview
                with st.expander("⚙️ View Current Configuration"):
                    config = PipelineManager.load_config()
                    if config:
                        st.json(config)
            
            with col2:
                if st.button("📝 Update Config", use_container_width=True, type="secondary"):
                    with st.spinner("Updating configuration..."):
                        if PipelineManager.update_config_with_video(st.session_state.uploaded_video_path):
                            st.success("✅ Config updated!")
                        else:
                            st.error("❌ Failed to update config")
                
                st.markdown("")
                
                if st.button("🚀 Run Pipeline", use_container_width=True, type="primary"):
                    # Update config first
                    with st.spinner("Preparing pipeline..."):
                        if not PipelineManager.update_config_with_video(st.session_state.uploaded_video_path):
                            st.error("❌ Failed to update configuration")
                            st.stop()
                    
                    # Run pipeline
                    st.markdown("---")
                    st.subheader("⏳ Processing Video...")
                    
                    progress_bar = st.progress(0, text="Initializing MediaPipe...")
                    status_placeholder = st.empty()
                    
                    try:
                        progress_bar.progress(10, text="Loading MediaPipe module...")
                        time.sleep(0.5)
                        
                        progress_bar.progress(20, text="Starting pose detection...")
                        start_time = time.time()
                        
                        results = PipelineManager.run_pipeline()
                        
                        progress_bar.progress(80, text="Processing complete, locating outputs...")
                        
                        if results and len(results) > 0:
                            # Find output files
                            time.sleep(2)  # Give filesystem time to sync
                            output_files = FileManager.find_output_videos(st.session_state.uploaded_video_path)
                            
                            progress_bar.progress(90, text="Converting videos for web compatibility...")
                            
                            # Convert videos to web-compatible format
                            if output_files.get('annotated'):
                                status_placeholder.info("🔄 Converting annotated video...")
                                VideoConverter.ensure_web_compatible(output_files['annotated'])
                            
                            if output_files.get('skeleton'):
                                status_placeholder.info("🔄 Converting skeleton video...")
                                VideoConverter.ensure_web_compatible(output_files['skeleton'])
                            
                            progress_bar.progress(100, text="Complete!")
                            
                            # Update session state
                            st.session_state.output_videos = output_files
                            st.session_state.processing_complete = True
                            
                            end_time = time.time()
                            
                            status_placeholder.empty()
                            st.success(f"✅ Pipeline completed in {end_time - start_time:.1f} seconds!")
                            st.balloons()
                            
                            # Show summary
                            st.markdown("---")
                            st.subheader("📋 Processing Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Frames Processed", results[0].frames_processed if results else "N/A")
                            with col2:
                                st.metric("Landmarks Detected", results[0].landmarks_detected if results else "N/A")
                            with col3:
                                st.metric("Processing Time", f"{end_time - start_time:.1f}s")
                            
                            st.info("✨ Check the Landmarker Videos and Feature Engineering tabs to view results")
                            
                        else:
                            progress_bar.empty()
                            status_placeholder.empty()
                            st.error("❌ Pipeline failed to produce results. Check logs for details.")
                    
                    except Exception as e:
                        progress_bar.empty()
                        status_placeholder.empty()
                        st.error(f"❌ Pipeline error: {str(e)}")
                        logger.error(f"Pipeline execution error: {e}")
                        logger.error(traceback.format_exc())
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: LANDMARKER VIDEOS
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab3:
        st.subheader("Landmarker Videos")
        
        if not st.session_state.processing_complete:
            st.info("⏳ No results available yet. Process a video in the Process tab.")
        else:
            output_files = st.session_state.output_videos
            
            # Refresh button
            if st.button("🔄 Refresh Output Files"):
                with st.spinner("Refreshing..."):
                    output_files = FileManager.find_output_videos(st.session_state.uploaded_video_path)
                    st.session_state.output_videos = output_files
                    st.rerun()
            
            st.markdown("---")
            
            # Display videos in 3 columns
            st.subheader("📹 Video Outputs")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                VideoDisplay.display_video_with_download(
                    st.session_state.uploaded_video_path,
                    "📹 Original Video",
                    "result_original"
                )
            
            with col2:
                VideoDisplay.display_video_with_download(
                    output_files.get('annotated'),
                    "🎨 Annotated Video",
                    "result_annotated"
                )
            
            with col3:
                VideoDisplay.display_video_with_download(
                    output_files.get('skeleton'),
                    "🦴 Skeleton Video",
                    "result_skeleton"
                )
            
            # CSV Data section
            st.markdown("---")
            st.subheader("📄 Landmarks Data")
            
            if output_files.get('csv'):
                csv_path = output_files['csv']
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.info(f"**CSV File:** {csv_path.name}")
                    st.caption(f"📏 Size: {csv_path.stat().st_size / 1024:.1f} KB")
                
                with col2:
                    # Create unique keys for CSV download
                    csv_data_key = f"data_csv_{csv_path.stem}_{csv_path.stat().st_mtime}"
                    csv_widget_key = f"widget_csv_{csv_path.stem}_{csv_path.stat().st_mtime}"
                    
                    # Store CSV data in session state
                    if csv_data_key not in st.session_state:
                        try:
                            with open(csv_path, 'rb') as f:
                                st.session_state[csv_data_key] = f.read()
                        except Exception as e:
                            logger.error(f"Failed to read CSV for download: {e}")
                            st.session_state[csv_data_key] = None
                    
                    if st.session_state[csv_data_key]:
                        st.download_button(
                            "📥 Download CSV",
                            data=st.session_state[csv_data_key],
                            file_name=csv_path.name,
                            mime="text/csv",
                            key=csv_widget_key,
                            use_container_width=True
                        )
                    else:
                        st.error("❌ CSV not available")
                
                # Preview CSV
                with st.expander("👁️ Preview CSV Data"):
                    try:
                        df = pd.read_csv(csv_path)
                        st.dataframe(df.head(10), use_container_width=True)
                        st.caption(f"Showing first 10 rows of {len(df)} total rows")
                    except Exception as e:
                        st.error(f"Could not preview CSV: {e}")
            else:
                st.warning("⚠️ CSV file not found")
            
            # Export all results
            st.markdown("---")
            st.subheader("📦 Export All Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("Download all results (videos + CSV) as a ZIP archive")
            
            with col2:
                if st.button("📦 Create ZIP", use_container_width=True, type="primary"):
                    with st.spinner("Creating ZIP archive..."):
                        zip_buffer = ExportManager.create_results_zip(
                            st.session_state.uploaded_video_path,
                            output_files
                        )
                        
                        if zip_buffer:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            zip_data_key = f"data_zip_{timestamp}"
                            zip_widget_key = f"widget_zip_{timestamp}"
                            zip_data = zip_buffer.getvalue()
                            
                            # Store ZIP data in session state
                            st.session_state[zip_data_key] = zip_data
                            
                            st.download_button(
                                "📥 Download ZIP Archive",
                                data=zip_data,
                                file_name=f"mediapipe_results_{timestamp}.zip",
                                mime="application/zip",
                                key=zip_widget_key,
                                use_container_width=True
                            )
                            st.success("✅ ZIP archive ready!")
                        else:
                            st.error("❌ Failed to create ZIP archive")
            
            # Debug section
            with st.expander("🐛 Debug: File Locations"):
                st.write("**Output files found:**")
                for key, path in output_files.items():
                    if path:
                        st.write(f"- {key}: `{path}` (exists: {path.exists()})")
                    else:
                        st.write(f"- {key}: Not found")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4: FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab4:
        st.subheader("Feature Engineering")
        
        # Create two sub-tabs for different feature extraction methods
        subtab1, subtab2 = st.tabs(["From Processed Video", "From CSV Upload"])
        
        # Sub-tab 1: Extract features from processed video
        with subtab1:
            if not st.session_state.processing_complete:
                st.info("⏳ No results available yet. Process a video in the Process tab.")
            else:
                output_files = st.session_state.output_videos
                
                if not output_files.get('csv'):
                    st.error("❌ No landmarks data available. Please process a video first.")
                else:
                    csv_path = output_files['csv']
                    
                    # Feature extraction section
                    st.markdown("---")
                    st.subheader("🔬 Extract Features from Landmarks")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.info(f"**CSV File:** {csv_path.name}")
                        st.caption(f"📏 Size: {csv_path.stat().st_size / 1024:.1f} KB")
                    
                    with col2:
                        if st.button("🚀 Extract Features", use_container_width=True, type="primary"):
                            with st.spinner("Extracting features..."):
                                try:
                                    # Extract features and gait cycles
                                    df_features, gait_cycles = GaitAnalysisEngine.extract_features_from_csv(
                                        csv_path, 
                                        st.session_state.uploaded_video_path
                                    )
                                    
                                    if not df_features.empty:
                                        # Save features
                                        features_path = GaitAnalysisEngine.save_features(
                                            df_features, 
                                            st.session_state.uploaded_video_path
                                        )
                                        
                                        # Save gait cycles if available
                                        gait_cycles_path = None
                                        if gait_cycles is not None:
                                            gait_cycles_path = GaitAnalysisEngine.save_gait_cycles(
                                                gait_cycles,
                                                st.session_state.uploaded_video_path
                                            )
                                        
                                        # Update session state
                                        st.session_state.features_df = df_features
                                        st.session_state.features_path = features_path
                                        st.session_state.gait_cycles = gait_cycles
                                        st.session_state.gait_cycles_path = gait_cycles_path
                                        
                                        st.success(f"✅ Features extracted successfully! Found {len(df_features)} features.")
                                        if gait_cycles is not None:
                                            st.success(f"✅ Extracted {len(gait_cycles)} gait cycles.")
                                    else:
                                        st.error("❌ Failed to extract features.")
                                except Exception as e:
                                    st.error(f"❌ Error extracting features: {str(e)}")
                                    logger.error(f"Error extracting features: {e}")
                    
                    # Display features if available
                    if st.session_state.features_df is not None and not st.session_state.features_df.empty:
                        st.markdown("---")
                        st.subheader("📊 Extracted Features")
                        
                        # Feature summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Features", len(st.session_state.features_df.columns))
                        with col2:
                            st.metric("Numeric Features", sum(st.session_state.features_df.dtypes.apply(lambda x: x in ['int64', 'float64'])))
                        with col3:
                            st.metric("Clips Analyzed", len(st.session_state.features_df))
                        
                        # Feature preview (vertical format)
                        with st.expander("👁️ Preview Features"):
                            if len(st.session_state.features_df) > 0:
                                features_dict = st.session_state.features_df.iloc[0].to_dict()
                                
                                # Create a DataFrame with feature name and value columns
                                features_display = pd.DataFrame(list(features_dict.items()), 
                                                              columns=['Feature', 'Value'])
                                
                                # Display in a two-column format
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.dataframe(features_display.iloc[:len(features_display)//2], 
                                                use_container_width=True, hide_index=True)
                                
                                with col2:
                                    st.dataframe(features_display.iloc[len(features_display)//2:], 
                                                use_container_width=True, hide_index=True)
                            else:
                                st.warning("No feature data available")
                        
                        # Feature download
                        if st.session_state.features_path:
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.info(f"**Features File:** {st.session_state.features_path.name}")
                                st.caption(f"📏 Size: {st.session_state.features_path.stat().st_size / 1024:.1f} KB")
                            
                            with col2:
                                # Create unique keys for features download
                                features_data_key = f"data_features_{st.session_state.features_path.stem}_{st.session_state.features_path.stat().st_mtime}"
                                features_widget_key = f"widget_features_{st.session_state.features_path.stem}_{st.session_state.features_path.stat().st_mtime}"
                                
                                # Store features data in session state
                                if features_data_key not in st.session_state:
                                    try:
                                        with open(st.session_state.features_path, 'rb') as f:
                                            st.session_state[features_data_key] = f.read()
                                    except Exception as e:
                                        logger.error(f"Failed to read features for download: {e}")
                                        st.session_state[features_data_key] = None
                                
                                if st.session_state[features_data_key]:
                                    st.download_button(
                                        "📥 Download Features",
                                        data=st.session_state[features_data_key],
                                        file_name=st.session_state.features_path.name,
                                        mime="text/csv",
                                        key=features_widget_key,
                                        use_container_width=True
                                    )
                                else:
                                    st.error("❌ Features not available")
                        
                        # Gait cycles info
                        if st.session_state.gait_cycles is not None:
                            st.markdown("---")
                            st.subheader("🚶 Gait Cycles")
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.info(f"**Gait Cycles:** {len(st.session_state.gait_cycles)} cycles extracted")
                                st.caption(f"Each cycle: {st.session_state.gait_cycles.shape[1]} frames × {st.session_state.gait_cycles.shape[2]} joints × 3 coordinates")
                            
                            with col2:
                                if st.session_state.gait_cycles_path:
                                    gait_data_key = f"data_gait_{st.session_state.gait_cycles_path.stem}_{st.session_state.gait_cycles_path.stat().st_mtime}"
                                    gait_widget_key = f"widget_gait_{st.session_state.gait_cycles_path.stem}_{st.session_state.gait_cycles_path.stat().st_mtime}"
                                    
                                    if gait_data_key not in st.session_state:
                                        try:
                                            with open(st.session_state.gait_cycles_path, 'rb') as f:
                                                st.session_state[gait_data_key] = f.read()
                                        except Exception as e:
                                            logger.error(f"Failed to read gait cycles for download: {e}")
                                            st.session_state[gait_data_key] = None
                                    
                                    if st.session_state[gait_data_key]:
                                        st.download_button(
                                            "📥 Download Gait Cycles",
                                            data=st.session_state[gait_data_key],
                                            file_name=st.session_state.gait_cycles_path.name,
                                            mime="application/octet-stream",
                                            key=gait_widget_key,
                                            use_container_width=True
                                        )
                        
                        # Feature visualization
                        st.markdown("---")
                        st.subheader("📈 Feature Visualization")
                        
                        # User-friendly visualizations
                        st.markdown("### 🎯 User-Friendly Gait Analysis")
                        
                        # Select visualization type
                        viz_type = st.selectbox(
                            "Select visualization type:",
                            ["Gait Parameters", "Joint Movements", "Symmetry Analysis", "Temporal Patterns"]
                        )
                        
                        if viz_type == "Gait Parameters":
                            # Create a radar chart for key gait parameters
                            import seaborn as sns
                            
                            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
                            
                            # Select key gait features
                            gait_features = [
                                "step_height_L", "step_height_R", "step_length_L", "step_length_R",
                                "pelvis_drop_mean", "trunk_lean_mean"
                            ]
                            
                            # Filter features that exist in the dataframe
                            available_features = [f for f in gait_features if f in st.session_state.features_df.columns]
                            
                            if available_features and len(st.session_state.features_df) > 0:
                                values = st.session_state.features_df.iloc[0][available_features].values
                                # Normalize values for better visualization
                                values = np.abs(values) / np.max(np.abs(values)) if np.max(np.abs(values)) > 0 else values
                                
                                # Number of variables
                                N = len(available_features)
                                
                                # Compute angle for each axis
                                angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                                values = np.concatenate((values, [values[0]]))  # Complete the loop
                                angles += angles[:1]  # Complete the loop
                                
                                # Plot
                                ax.plot(angles, values, 'o-', linewidth=2)
                                ax.fill(angles, values, alpha=0.25)
                                ax.set_xticks(angles[:-1])
                                ax.set_xticklabels([f.replace('_', ' ').title() for f in available_features])
                                ax.set_ylim(0, 1)
                                ax.set_title("Gait Parameters Profile", size=15)
                                
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.warning("Required gait parameters not available in the extracted features")
                        
                        elif viz_type == "Joint Movements":
                            # Create a bar chart comparing left vs right joint movements
                            import seaborn as sns
                            
                            joint_pairs = [
                                ("Step Height", "step_height_L", "step_height_R"),
                                ("Step Length", "step_length_L", "step_length_R"),
                                ("Knee ROM", "knee_L_rom_y", "knee_R_rom_y"),
                                ("Hip ROM", "hip_L_rom_y", "hip_R_rom_y"),
                                ("Ankle ROM", "ankle_L_rom_y", "ankle_R_rom_y")
                            ]
                            
                            # Filter pairs that exist in the dataframe
                            available_pairs = [(label, left, right) for label, left, right in joint_pairs 
                                              if left in st.session_state.features_df.columns and right in st.session_state.features_df.columns]
                            
                            if available_pairs and len(st.session_state.features_df) > 0:
                                labels = [pair[0] for pair in available_pairs]
                                left_values = [st.session_state.features_df.iloc[0][pair[1]] for pair in available_pairs]
                                right_values = [st.session_state.features_df.iloc[0][pair[2]] for pair in available_pairs]
                                
                                x = np.arange(len(labels))  # the label locations
                                width = 0.35  # the width of the bars
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                rects1 = ax.bar(x - width/2, left_values, width, label='Left')
                                rects2 = ax.bar(x + width/2, right_values, width, label='Right')
                                
                                ax.set_ylabel('Values')
                                ax.set_title('Left vs Right Joint Movements')
                                ax.set_xticks(x)
                                ax.set_xticklabels(labels)
                                ax.legend()
                                
                                # Add value labels on bars
                                def autolabel(rects):
                                    for rect in rects:
                                        height = rect.get_height()
                                        ax.annotate(f'{height:.2f}',
                                                   xy=(rect.get_x() + rect.get_width() / 2, height),
                                                   xytext=(0, 3),  # 3 points vertical offset
                                                   textcoords="offset points",
                                                   ha='center', va='bottom')
                                
                                autolabel(rects1)
                                autolabel(rects2)
                                
                                fig.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.warning("Required joint movement features not available")
                        
                        elif viz_type == "Symmetry Analysis":
                            # Create a visualization for symmetry features
                            import seaborn as sns
                            
                            symmetry_features = [
                                ("Step Height", "step_height_symmetry"),
                                ("Step Length", "step_length_symmetry"),
                                ("Knee ROM", "knee_rom_asym"),
                                ("Hip ROM", "hip_rom_asym"),
                                ("Ankle ROM", "ankle_rom_asym"),
                                ("Step Time", "step_time_asym"),
                                ("Cadence", "cadence_asym")
                            ]
                            
                            # Filter features that exist in the dataframe
                            available_symmetry = [(label, feature) for label, feature in symmetry_features 
                                                 if feature in st.session_state.features_df.columns]
                            
                            if available_symmetry and len(st.session_state.features_df) > 0:
                                labels = [item[0] for item in available_symmetry]
                                values = [st.session_state.features_df.iloc[0][item[1]] for item in available_symmetry]
                                
                                # Create a horizontal bar chart
                                fig, ax = plt.subplots(figsize=(10, 6))
                                y_pos = np.arange(len(labels))
                                
                                # Color bars based on symmetry (green for balanced, red for imbalanced)
                                colors = ['green' if abs(v) < 0.1 else 'orange' if abs(v) < 0.2 else 'red' for v in values]
                                
                                ax.barh(y_pos, values, align='center', color=colors)
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(labels)
                                ax.set_xlabel('Symmetry Index')
                                ax.set_title('Movement Symmetry Analysis')
                                
                                # Add a vertical line at 0 for reference
                                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                                
                                # Add value labels
                                for i, v in enumerate(values):
                                    ax.text(v + 0.01 if v >= 0 else v - 0.01, i, f'{v:.2f}', 
                                           color='black', va='center')
                                
                                # Add a legend for the colors
                                from matplotlib.patches import Patch
                                legend_elements = [
                                    Patch(facecolor='green', label='Balanced (|value| < 0.1)'),
                                    Patch(facecolor='orange', label='Moderate (0.1 ≤ |value| < 0.2)'),
                                    Patch(facecolor='red', label='Imbalanced (|value| ≥ 0.2)')
                                ]
                                ax.legend(handles=legend_elements, loc='upper right')
                                
                                fig.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.warning("Required symmetry features not available")
                        
                        elif viz_type == "Temporal Patterns":
                            # Create a visualization for temporal gait features
                            import seaborn as sns
                            
                            temporal_features = [
                                ("Mean Step Time (s)", "step_L_mean_step_time"),
                                ("Step Time CV", "step_L_step_time_cv"),
                                ("Cadence (steps/min)", "step_L_cadence"),
                                ("Mean Stride Time (s)", "step_L_mean_stride_time"),
                                ("Stance Ratio (L)", "stance_ratio_L"),
                                ("Stance Ratio (R)", "stance_ratio_R")
                            ]
                            
                            # Filter features that exist in the dataframe
                            available_temporal = [(label, feature) for label, feature in temporal_features 
                                                 if feature in st.session_state.features_df.columns]
                            
                            if available_temporal and len(st.session_state.features_df) > 0:
                                labels = [item[0] for item in available_temporal]
                                values = [st.session_state.features_df.iloc[0][item[1]] for item in available_temporal]
                                
                                # Create a gauge-like visualization for each temporal feature
                                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                                axes = axes.flatten()
                                
                                for i, (label, value) in enumerate(zip(labels, values)):
                                    if i < len(axes):
                                        ax = axes[i]
                                        
                                        # Create a simple gauge visualization
                                        theta = np.linspace(0, np.pi, 100)
                                        r = 0.5
                                        
                                        # Background arc
                                        ax.plot(theta, np.ones_like(theta) * r, color='lightgray', linewidth=20)
                                        
                                        # Value arc (normalized between 0 and 1)
                                        if "cadence" in label.lower():
                                            # Normalize cadence (steps/min) - typical range 80-180
                                            norm_value = (value - 80) / 100 if value > 80 else 0
                                            norm_value = min(max(norm_value, 0), 1)
                                        elif "ratio" in label.lower():
                                            # Normalize ratio (0-1)
                                            norm_value = min(max(value, 0), 1)
                                        elif "time" in label.lower():
                                            # Normalize time - typical range 0.3-1.0s for step time
                                            norm_value = (value - 0.3) / 0.7 if value > 0.3 else 0
                                            norm_value = min(max(norm_value, 0), 1)
                                        elif "cv" in label.lower():
                                            # Normalize coefficient of variation - typical range 0-0.3
                                            norm_value = min(max(value / 0.3, 0), 1)
                                        else:
                                            # Default normalization
                                            norm_value = 0.5
                                        
                                        # Value arc
                                        value_theta = np.linspace(0, np.pi * norm_value, 100)
                                        ax.plot(value_theta, np.ones_like(value_theta) * r, color='blue', linewidth=20)
                                        
                                        # Add text
                                        ax.text(0.5, 0.3, f"{value:.2f}", ha='center', va='center', fontsize=12)
                                        ax.text(0.5, 0.1, label, ha='center', va='center', fontsize=10)
                                        
                                        # Remove axes
                                        ax.set_xlim(0, 1)
                                        ax.set_ylim(0, 0.6)
                                        ax.axis('off')
                                
                                # Hide unused subplots
                                for i in range(len(available_temporal), len(axes)):
                                    axes[i].axis('off')
                                
                                plt.suptitle("Temporal Gait Parameters", fontsize=16)
                                fig.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.warning("Required temporal features not available")
                        
                        # Original histogram visualization
                        st.markdown("---")
                        st.markdown("### 📊 Statistical Distribution")
                        
                        # Select feature to visualize
                        numeric_features = st.session_state.features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        
                        if numeric_features:
                            selected_feature = st.selectbox("Select a feature to visualize:", numeric_features)
                            
                            if selected_feature:
                                import seaborn as sns
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(st.session_state.features_df[selected_feature].dropna(), kde=True, ax=ax)
                                ax.set_title(f"Distribution of {selected_feature}")
                                ax.set_xlabel(selected_feature)
                                ax.set_ylabel("Frequency")
                                
                                st.pyplot(fig)
                                plt.close()
                        else:
                            st.warning("⚠️ No numeric features available for visualization.")
                    else:
                        st.info("👆 Click 'Extract Features' to analyze the landmarks data.")
        
        # Sub-tab 2: Upload CSV directly for feature extraction
        with subtab2:
            st.markdown("---")
            st.subheader("📁 Upload CSV for Feature Extraction")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_csv = st.file_uploader(
                    "Upload a CSV file with MediaPipe landmarks",
                    type=['csv'],
                    help="Upload a CSV file containing MediaPipe landmarks data"
                )
                
                if uploaded_csv:
                    st.info(f"📄 {uploaded_csv.name}")
                    st.info(f"📏 {uploaded_csv.size / (1024*1024):.2f} MB")
                    
                    # Save uploaded CSV
                    csv_path = FEATURES_DIR / uploaded_csv.name
                    with open(csv_path, 'wb') as f:
                        f.write(uploaded_csv.getvalue())
                    
                    st.session_state.uploaded_csv_path = csv_path
                    
                    # Preview CSV
                    with st.expander("👁️ Preview CSV Data"):
                        try:
                            df = pd.read_csv(csv_path)
                            st.dataframe(df.head(10), use_container_width=True)
                            st.caption(f"Showing first 10 rows of {len(df)} total rows")
                        except Exception as e:
                            st.error(f"Could not preview CSV: {e}")
            
            with col2:
                if st.session_state.uploaded_csv_path:
                    st.metric("Status", "CSV uploaded")
                    st.metric("File", st.session_state.uploaded_csv_path.name)
            
            # Extract features from uploaded CSV
            if st.session_state.uploaded_csv_path:
                st.markdown("---")
                st.subheader("🔬 Extract Features from Uploaded CSV")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.info(f"**CSV File:** {st.session_state.uploaded_csv_path.name}")
                    st.caption(f"📏 Size: {st.session_state.uploaded_csv_path.stat().st_size / 1024:.1f} KB")
                
                with col2:
                    if st.button("🚀 Extract Features", use_container_width=True, type="primary", key="extract_csv_features"):
                        with st.spinner("Extracting features..."):
                            try:
                                # Extract features
                                df_features, gait_cycles = GaitAnalysisEngine.extract_features_from_csv(
                                    st.session_state.uploaded_csv_path
                                )
                                
                                if not df_features.empty:
                                    # Save features
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    csv_stem = st.session_state.uploaded_csv_path.stem
                                    features_path = FEATURES_DIR / f"{csv_stem}_features_{timestamp}.csv"
                                    
                                    df_features.to_csv(features_path, index=False)
                                    
                                    # Save gait cycles if available
                                    gait_cycles_path = None
                                    if gait_cycles is not None:
                                        gait_cycles_path = GaitAnalysisEngine.save_gait_cycles(
                                            gait_cycles,
                                            st.session_state.uploaded_csv_path
                                        )
                                    
                                    # Update session state
                                    st.session_state.csv_features_df = df_features
                                    st.session_state.csv_features_path = features_path
                                    
                                    st.success(f"✅ Features extracted successfully! Found {len(df_features)} features.")
                                    if gait_cycles is not None:
                                        st.success(f"✅ Extracted {len(gait_cycles)} gait cycles.")
                                else:
                                    st.error("❌ Failed to extract features.")
                            except Exception as e:
                                st.error(f"❌ Error extracting features: {str(e)}")
                                logger.error(f"Error extracting features from CSV: {e}")
                
                # Display features if available
                if st.session_state.csv_features_df is not None and not st.session_state.csv_features_df.empty:
                    st.markdown("---")
                    st.subheader("📊 Extracted Features")
                    
                    # Feature summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Features", len(st.session_state.csv_features_df.columns))
                    with col2:
                        st.metric("Numeric Features", sum(st.session_state.csv_features_df.dtypes.apply(lambda x: x in ['int64', 'float64'])))
                    with col3:
                        st.metric("Clips Analyzed", len(st.session_state.csv_features_df))
                    
                    # Feature preview (vertical format)
                    with st.expander("👁️ Preview Features"):
                        if len(st.session_state.csv_features_df) > 0:
                            features_dict = st.session_state.csv_features_df.iloc[0].to_dict()
                            
                            # Create a DataFrame with feature name and value columns
                            features_display = pd.DataFrame(list(features_dict.items()), 
                                                          columns=['Feature', 'Value'])
                            
                            # Display in a two-column format
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.dataframe(features_display.iloc[:len(features_display)//2], 
                                            use_container_width=True, hide_index=True)
                            
                            with col2:
                                st.dataframe(features_display.iloc[len(features_display)//2:], 
                                            use_container_width=True, hide_index=True)
                        else:
                            st.warning("No feature data available")
                    
                    # Feature download
                    if st.session_state.csv_features_path:
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.info(f"**Features File:** {st.session_state.csv_features_path.name}")
                            st.caption(f"📏 Size: {st.session_state.csv_features_path.stat().st_size / 1024:.1f} KB")
                        
                        with col2:
                            # Create unique keys for features download
                            csv_features_data_key = f"data_csv_features_{st.session_state.csv_features_path.stem}_{st.session_state.csv_features_path.stat().st_mtime}"
                            csv_features_widget_key = f"widget_csv_features_{st.session_state.csv_features_path.stem}_{st.session_state.csv_features_path.stat().st_mtime}"
                            
                            # Store features data in session state
                            if csv_features_data_key not in st.session_state:
                                try:
                                    with open(st.session_state.csv_features_path, 'rb') as f:
                                        st.session_state[csv_features_data_key] = f.read()
                                except Exception as e:
                                    logger.error(f"Failed to read CSV features for download: {e}")
                                    st.session_state[csv_features_data_key] = None
                            
                            if st.session_state[csv_features_data_key]:
                                st.download_button(
                                    "📥 Download Features",
                                    data=st.session_state[csv_features_data_key],
                                    file_name=st.session_state.csv_features_path.name,
                                    mime="text/csv",
                                    key=csv_features_widget_key,
                                    use_container_width=True
                                )
                            else:
                                st.error("❌ Features not available")
                    
                    # Feature visualization
                    st.markdown("---")
                    st.subheader("📈 Feature Visualization")
                    
                    # Select feature to visualize
                    numeric_features = st.session_state.csv_features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    
                    if numeric_features:
                        selected_feature = st.selectbox("Select a feature to visualize:", numeric_features, key="csv_feature_select")
                        
                        if selected_feature:
                            import seaborn as sns
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(st.session_state.csv_features_df[selected_feature].dropna(), kde=True, ax=ax)
                            ax.set_title(f"Distribution of {selected_feature}")
                            ax.set_xlabel(selected_feature)
                            ax.set_ylabel("Frequency")
                            
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.warning("⚠️ No numeric features available for visualization.")
                else:
                    st.info("👆 Click 'Extract Features' to analyze the uploaded CSV data.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5: DETAILED ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab5:
        st.subheader("🔬 Detailed Gait Analysis")
        
        # Check if features are already available from Feature Engineering tab
        features_available = (st.session_state.features_df is not None and 
                            not st.session_state.features_df.empty)
        
        # Create two sub-tabs
        subtab1, subtab2 = st.tabs(["From Feature Engineering", "From CSV Upload"])
        
        # Sub-tab 1: Analysis from Feature Engineering results
        with subtab1:
            if not features_available:
                st.warning("⚠️ No features available. Please extract features in the Feature Engineering tab first.")
                
                # Add a button to go to Feature Engineering tab
                if st.button("Go to Feature Engineering Tab"):
                    st.info("Please navigate to the Feature Engineering tab to extract features first.")
            else:
                st.success("✅ Using features from Feature Engineering tab")
                
                # Feature summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Features", len(st.session_state.features_df.columns))
                with col2:
                    st.metric("Numeric Features", sum(st.session_state.features_df.dtypes.apply(lambda x: x in ['int64', 'float64'])))
                with col3:
                    st.metric("Gait Cycles", len(st.session_state.gait_cycles) if st.session_state.gait_cycles is not None else 0)
                
                # Visualization selection
                st.markdown("---")
                st.subheader("🎯 Visualization Selection")
                
                viz_type = st.selectbox(
                    "Select visualization type:",
                    [
                        "Gait Health Dashboard", 
                        "Movement Pattern Flow", 
                        "3D Joint Trajectories",
                        "Gait Stability Index", 
                        "Temporal Gait Heatmap",
                        "Statistical Distribution"
                    ]
                )
                
                # Display selected visualization
                if viz_type == "Gait Health Dashboard":
                    fig = create_gait_score_dashboard(st.session_state.features_df)
                    st.pyplot(fig)
                    plt.close()
                    
                elif viz_type == "Movement Pattern Flow":
                    fig = create_movement_flow_chart(st.session_state.features_df)
                    st.pyplot(fig)
                    plt.close()
                    
                elif viz_type == "3D Joint Trajectories":
                    if st.session_state.gait_cycles is not None:
                        fig = create_3d_joint_trajectory(st.session_state.gait_cycles)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.warning("Gait cycles not available for 3D visualization")
                        
                elif viz_type == "Gait Stability Index":
                    fig = create_gait_stability_index(st.session_state.features_df)
                    st.pyplot(fig)
                    plt.close()
                    
                elif viz_type == "Temporal Gait Heatmap":
                    if st.session_state.gait_cycles is not None:
                        fig = create_temporal_gait_heatmap(st.session_state.gait_cycles)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.warning("Gait cycles not available for heatmap visualization")
                        
                elif viz_type == "Statistical Distribution":
                    # Statistical distribution visualization
                    numeric_features = st.session_state.features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    
                    if numeric_features:
                        selected_feature = st.selectbox("Select a feature to visualize:", numeric_features)
                        
                        if selected_feature:
                            import seaborn as sns
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(st.session_state.features_df[selected_feature].dropna(), kde=True, ax=ax)
                            ax.set_title(f"Distribution of {selected_feature}")
                            ax.set_xlabel(selected_feature)
                            ax.set_ylabel("Frequency")
                            
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.warning("⚠️ No numeric features available for visualization.")
                
                # Download options
                st.markdown("---")
                st.subheader("📥 Download Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.features_path:
                        # Create unique keys for features download
                        features_data_key = f"data_features_detail_{st.session_state.features_path.stem}_{st.session_state.features_path.stat().st_mtime}"
                        features_widget_key = f"widget_features_detail_{st.session_state.features_path.stem}_{st.session_state.features_path.stat().st_mtime}"
                        
                        # Store features data in session state
                        if features_data_key not in st.session_state:
                            try:
                                with open(st.session_state.features_path, 'rb') as f:
                                    st.session_state[features_data_key] = f.read()
                            except Exception as e:
                                logger.error(f"Failed to read features for download: {e}")
                                st.session_state[features_data_key] = None
                        
                        if st.session_state[features_data_key]:
                            st.download_button(
                                "📥 Download Features",
                                data=st.session_state[features_data_key],
                                file_name=st.session_state.features_path.name,
                                mime="text/csv",
                                key=features_widget_key,
                                use_container_width=True
                            )
                
                with col2:
                    if st.session_state.gait_cycles_path:
                        gait_data_key = f"data_gait_detail_{st.session_state.gait_cycles_path.stem}_{st.session_state.gait_cycles_path.stat().st_mtime}"
                        gait_widget_key = f"widget_gait_detail_{st.session_state.gait_cycles_path.stem}_{st.session_state.gait_cycles_path.stat().st_mtime}"
                        
                        if gait_data_key not in st.session_state:
                            try:
                                with open(st.session_state.gait_cycles_path, 'rb') as f:
                                    st.session_state[gait_data_key] = f.read()
                            except Exception as e:
                                logger.error(f"Failed to read gait cycles for download: {e}")
                                st.session_state[gait_data_key] = None
                        
                        if st.session_state[gait_data_key]:
                            st.download_button(
                                "📥 Download Gait Cycles",
                                data=st.session_state[gait_data_key],
                                file_name=st.session_state.gait_cycles_path.name,
                                mime="application/octet-stream",
                                key=gait_widget_key,
                                use_container_width=True
                            )
        
        # Sub-tab 2: Analysis from uploaded CSV (for standalone use)
        with subtab2:
            st.markdown("---")
            st.subheader("📁 Upload CSV for Detailed Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_csv = st.file_uploader(
                    "Upload a CSV file with MediaPipe landmarks",
                    type=['csv'],
                    help="Upload a CSV file containing MediaPipe landmarks data",
                    key="detailed_csv_upload"
                )
                
                if uploaded_csv:
                    st.info(f"📄 {uploaded_csv.name}")
                    st.info(f"📏 {uploaded_csv.size / (1024*1024):.2f} MB")
                    
                    # Save uploaded CSV
                    csv_path = FEATURES_DIR / f"detailed_{uploaded_csv.name}"
                    with open(csv_path, 'wb') as f:
                        f.write(uploaded_csv.getvalue())
                    
                    st.session_state.detailed_csv_path = csv_path
                    
                    # Preview CSV
                    with st.expander("👁️ Preview CSV Data"):
                        try:
                            df = pd.read_csv(csv_path)
                            st.dataframe(df.head(10), use_container_width=True)
                            st.caption(f"Showing first 10 rows of {len(df)} total rows")
                        except Exception as e:
                            st.error(f"Could not preview CSV: {e}")
            
            with col2:
                if 'detailed_csv_path' in st.session_state and st.session_state.detailed_csv_path:
                    st.metric("Status", "CSV uploaded")
                    st.metric("File", st.session_state.detailed_csv_path.name)
            
            # Extract features from uploaded CSV
            if 'detailed_csv_path' in st.session_state and st.session_state.detailed_csv_path:
                st.markdown("---")
                st.subheader("🔬 Extract Features for Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.info(f"**CSV File:** {st.session_state.detailed_csv_path.name}")
                    st.caption(f"📏 Size: {st.session_state.detailed_csv_path.stat().st_size / 1024:.1f} KB")
                
                with col2:
                    if st.button("🚀 Extract Features", use_container_width=True, type="primary", key="extract_detailed_features"):
                        with st.spinner("Extracting features..."):
                            try:
                                # Extract features
                                df_features, gait_cycles = GaitAnalysisEngine.extract_features_from_csv(
                                    st.session_state.detailed_csv_path
                                )
                                
                                if not df_features.empty:
                                    # Save features
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    csv_stem = st.session_state.detailed_csv_path.stem
                                    features_path = FEATURES_DIR / f"detailed_{csv_stem}_features_{timestamp}.csv"
                                    
                                    df_features.to_csv(features_path, index=False)
                                    
                                    # Save gait cycles if available
                                    gait_cycles_path = None
                                    if gait_cycles is not None:
                                        gait_cycles_path = GaitAnalysisEngine.save_gait_cycles(
                                            gait_cycles,
                                            st.session_state.detailed_csv_path
                                        )
                                    
                                    # Update session state
                                    st.session_state.detailed_features_df = df_features
                                    st.session_state.detailed_features_path = features_path
                                    st.session_state.detailed_gait_cycles = gait_cycles
                                    st.session_state.detailed_gait_cycles_path = gait_cycles_path
                                    
                                    st.success(f"✅ Features extracted successfully! Found {len(df_features)} features.")
                                    if gait_cycles is not None:
                                        st.success(f"✅ Extracted {len(gait_cycles)} gait cycles.")
                                else:
                                    st.error("❌ Failed to extract features.")
                            except Exception as e:
                                st.error(f"❌ Error extracting features: {str(e)}")
                                logger.error(f"Error extracting features from CSV: {e}")
                
                # Display visualizations if features are available
                if 'detailed_features_df' in st.session_state and st.session_state.detailed_features_df is not None and not st.session_state.detailed_features_df.empty:
                    st.markdown("---")
                    st.subheader("📊 Detailed Analysis")
                    
                    # Feature summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Features", len(st.session_state.detailed_features_df.columns))
                    with col2:
                        st.metric("Numeric Features", sum(st.session_state.detailed_features_df.dtypes.apply(lambda x: x in ['int64', 'float64'])))
                    with col3:
                        st.metric("Gait Cycles", len(st.session_state.detailed_gait_cycles) if st.session_state.detailed_gait_cycles is not None else 0)
                    
                    # Visualization selection
                    st.markdown("---")
                    st.subheader("🎯 Visualization Selection")
                    
                    viz_type = st.selectbox(
                        "Select visualization type:",
                        [
                            "Gait Health Dashboard", 
                            "Movement Pattern Flow", 
                            "3D Joint Trajectories",
                            "Gait Stability Index", 
                            "Temporal Gait Heatmap",
                            "Statistical Distribution"
                        ],
                        key="detailed_viz_type"
                    )
                    
                    # Display selected visualization
                    if viz_type == "Gait Health Dashboard":
                        fig = create_gait_score_dashboard(st.session_state.detailed_features_df)
                        st.pyplot(fig)
                        plt.close()
                        
                    elif viz_type == "Movement Pattern Flow":
                        fig = create_movement_flow_chart(st.session_state.detailed_features_df)
                        st.pyplot(fig)
                        plt.close()
                        
                    elif viz_type == "3D Joint Trajectories":
                        if st.session_state.detailed_gait_cycles is not None:
                            fig = create_3d_joint_trajectory(st.session_state.detailed_gait_cycles)
                            if fig:
                                st.pyplot(fig)
                                plt.close()
                        else:
                            st.warning("Gait cycles not available for 3D visualization")
                            
                    elif viz_type == "Gait Stability Index":
                        fig = create_gait_stability_index(st.session_state.detailed_features_df)
                        st.pyplot(fig)
                        plt.close()
                        
                    elif viz_type == "Temporal Gait Heatmap":
                        if st.session_state.detailed_gait_cycles is not None:
                            fig = create_temporal_gait_heatmap(st.session_state.detailed_gait_cycles)
                            if fig:
                                st.pyplot(fig)
                                plt.close()
                        else:
                            st.warning("Gait cycles not available for heatmap visualization")
                            
                    elif viz_type == "Statistical Distribution":
                        # Statistical distribution visualization
                        numeric_features = st.session_state.detailed_features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        
                        if numeric_features:
                            selected_feature = st.selectbox("Select a feature to visualize:", numeric_features, key="detailed_feature_select")
                            
                            if selected_feature:
                                import seaborn as sns
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(st.session_state.detailed_features_df[selected_feature].dropna(), kde=True, ax=ax)
                                ax.set_title(f"Distribution of {selected_feature}")
                                ax.set_xlabel(selected_feature)
                                ax.set_ylabel("Frequency")
                                
                                st.pyplot(fig)
                                plt.close()
                        else:
                            st.warning("⚠️ No numeric features available for visualization.")
                    
                    # Download options
                    st.markdown("---")
                    st.subheader("📥 Download Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.session_state.detailed_features_path:
                            # Create unique keys for features download
                            features_data_key = f"data_features_detailed_{st.session_state.detailed_features_path.stem}_{st.session_state.detailed_features_path.stat().st_mtime}"
                            features_widget_key = f"widget_features_detailed_{st.session_state.detailed_features_path.stem}_{st.session_state.detailed_features_path.stat().st_mtime}"
                            
                            # Store features data in session state
                            if features_data_key not in st.session_state:
                                try:
                                    with open(st.session_state.detailed_features_path, 'rb') as f:
                                        st.session_state[features_data_key] = f.read()
                                except Exception as e:
                                    logger.error(f"Failed to read features for download: {e}")
                                    st.session_state[features_data_key] = None
                            
                            if st.session_state[features_data_key]:
                                st.download_button(
                                    "📥 Download Features",
                                    data=st.session_state[features_data_key],
                                    file_name=st.session_state.detailed_features_path.name,
                                    mime="text/csv",
                                    key=features_widget_key,
                                    use_container_width=True
                                )
                    
                    with col2:
                        if st.session_state.detailed_gait_cycles_path:
                            gait_data_key = f"data_gait_detailed_{st.session_state.detailed_gait_cycles_path.stem}_{st.session_state.detailed_gait_cycles_path.stat().st_mtime}"
                            gait_widget_key = f"widget_gait_detailed_{st.session_state.detailed_gait_cycles_path.stem}_{st.session_state.detailed_gait_cycles_path.stat().st_mtime}"
                            
                            if gait_data_key not in st.session_state:
                                try:
                                    with open(st.session_state.detailed_gait_cycles_path, 'rb') as f:
                                        st.session_state[gait_data_key] = f.read()
                                except Exception as e:
                                    logger.error(f"Failed to read gait cycles for download: {e}")
                                    st.session_state[gait_data_key] = None
                            
                            if st.session_state[gait_data_key]:
                                st.download_button(
                                    "📥 Download Gait Cycles",
                                    data=st.session_state[gait_data_key],
                                    file_name=st.session_state.detailed_gait_cycles_path.name,
                                    mime="application/octet-stream",
                                    key=gait_widget_key,
                                    use_container_width=True
                                )
                else:
                    st.info("👆 Click 'Extract Features' to analyze the uploaded CSV data.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 6: MODEL PREDICTION
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab6:
        st.subheader("🤖 Gait Classification Models")
        
        # Create three sub-tabs for different model types
        subtab1, subtab2, subtab3 = st.tabs(["Baseline Model", "Advanced Model", "Model Comparison"])
        
        # Sub-tab 1: Baseline model prediction
        with subtab1:
            if not st.session_state.processing_complete:
                st.info("⏳ No results available yet. Process a video in the Process tab first.")
            else:
                output_files = st.session_state.output_videos
                
                if not output_files.get('csv'):
                    st.error("❌ No landmarks data available. Please process a video first.")
                else:
                    csv_path = output_files['csv']
                    
                    # Model prediction section
                    st.markdown("---")
                    st.subheader("🔬 Classify Gait with Baseline Model")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.info(f"**CSV File:** {csv_path.name}")
                        st.caption(f"📏 Size: {csv_path.stat().st_size / 1024:.1f} KB")
                    
                    with col2:
                        if st.button("🚀 Run Prediction", use_container_width=True, type="primary"):
                            with st.spinner("Running prediction..."):
                                try:
                                    # Load model if not already loaded
                                    if st.session_state.model is None:
                                        st.session_state.model = ModelManager.load_model(model_type="baseline")
                                        if st.session_state.model is None:
                                            st.error("❌ Failed to load baseline model")
                                            st.stop()
                                    
                                    # Extract features if not already extracted
                                    if st.session_state.features_df is None or st.session_state.features_df.empty:
                                        df_features, gait_cycles = GaitAnalysisEngine.extract_features_from_csv(
                                            csv_path, 
                                            st.session_state.uploaded_video_path
                                        )
                                        
                                        if not df_features.empty:
                                            # Save features
                                            features_path = GaitAnalysisEngine.save_features(
                                                df_features, 
                                                st.session_state.uploaded_video_path
                                            )
                                            
                                            # Save gait cycles if available
                                            gait_cycles_path = None
                                            if gait_cycles is not None:
                                                gait_cycles_path = GaitAnalysisEngine.save_gait_cycles(
                                                    gait_cycles,
                                                    st.session_state.uploaded_video_path
                                                )
                                            
                                            # Update session state
                                            st.session_state.features_df = df_features
                                            st.session_state.features_path = features_path
                                            st.session_state.gait_cycles = gait_cycles
                                            st.session_state.gait_cycles_path = gait_cycles_path
                                        else:
                                            st.error("❌ Failed to extract features.")
                                            st.stop()
                                    
                                    # Prepare features for prediction
                                    df_model = ModelManager.prepare_features_for_baseline_model(st.session_state.features_df)
                                    
                                    # Make predictions
                                    y_pred, y_pred_proba = ModelManager.predict_with_baseline_model(st.session_state.model, df_model)
                                    
                                    if len(y_pred) > 0:
                                        # Save predictions
                                        predictions_path = ModelManager.save_predictions(
                                            st.session_state.features_df, y_pred, y_pred_proba, 
                                            st.session_state.uploaded_video_path.stem,
                                            model_type="baseline"
                                        )
                                        
                                        # Update session state
                                        st.session_state.model_predictions = {
                                            'features': st.session_state.features_df,
                                            'y_pred': y_pred,
                                            'y_pred_proba': y_pred_proba
                                        }
                                        st.session_state.model_predictions_path = predictions_path
                                        
                                        st.success("✅ Baseline model prediction completed successfully!")
                                    else:
                                        st.error("❌ Failed to make predictions.")
                                except Exception as e:
                                    st.error(f"❌ Error during baseline model prediction: {str(e)}")
                                    logger.error(f"Error during baseline model prediction: {e}")
                    
                    # Display predictions if available
                    if st.session_state.model_predictions is not None:
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        
                        # Get the first prediction (most common case)
                        pred_class = st.session_state.model_predictions['y_pred'][0]
                        pred_proba = st.session_state.model_predictions['y_pred_proba'][0]
                        pred_label = BASELINE_CLASS_LABELS.get(pred_class, f"Unknown({pred_class})")
                        
                        # Display the prediction with confidence
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Class", pred_label)
                        
                        with col2:
                            st.metric("Confidence", f"{pred_proba[pred_class]:.2%}")
                        
                        with col3:
                            # Color code the severity
                            if pred_class == 0:
                                st.metric("Severity", "Normal", delta="✅")
                            elif pred_class == 1:
                                st.metric("Severity", "Mild", delta="⚠️")
                            elif pred_class == 2:
                                st.metric("Severity", "Moderate", delta="⚠️")
                            else:
                                st.metric("Severity", "Severe", delta="🚨")
                        
                        # Display probability distribution
                        st.markdown("### Class Probabilities")
                        
                        # Create a DataFrame for the probabilities
                        prob_df = pd.DataFrame({
                            'Class': [BASELINE_CLASS_LABELS[i] for i in range(len(BASELINE_CLASS_LABELS))],
                            'Probability': pred_proba
                        })
                        
                        # Create a bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['green', 'gold', 'orange', 'red']
                        bars = ax.bar(prob_df['Class'], prob_df['Probability'], color=colors)
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.2%}', ha='center', va='bottom')
                        
                        ax.set_xlabel('Class')
                        ax.set_ylabel('Probability')
                        ax.set_title('Class Probability Distribution')
                        ax.set_ylim(0, 1)
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Display prediction visualization
                        st.markdown("### Detailed Analysis")
                        fig = ModelManager.create_prediction_visualization(
                            st.session_state.model_predictions['features'],
                            st.session_state.model_predictions['y_pred'],
                            st.session_state.model_predictions['y_pred_proba'],
                            model_type="baseline"
                        )
                        st.pyplot(fig)
                        plt.close()
                        
                        # Download predictions
                        if st.session_state.model_predictions_path:
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.info(f"**Predictions File:** {st.session_state.model_predictions_path.name}")
                                st.caption(f"📏 Size: {st.session_state.model_predictions_path.stat().st_size / 1024:.1f} KB")
                            
                            with col2:
                                # Create unique keys for predictions download
                                predictions_data_key = f"data_predictions_{st.session_state.model_predictions_path.stem}_{st.session_state.model_predictions_path.stat().st_mtime}"
                                predictions_widget_key = f"widget_predictions_{st.session_state.model_predictions_path.stem}_{st.session_state.model_predictions_path.stat().st_mtime}"
                                
                                # Store predictions data in session state
                                if predictions_data_key not in st.session_state:
                                    try:
                                        with open(st.session_state.model_predictions_path, 'rb') as f:
                                            st.session_state[predictions_data_key] = f.read()
                                    except Exception as e:
                                        logger.error(f"Failed to read predictions for download: {e}")
                                        st.session_state[predictions_data_key] = None
                                
                                if st.session_state[predictions_data_key]:
                                    st.download_button(
                                        "📥 Download Predictions",
                                        data=st.session_state[predictions_data_key],
                                        file_name=st.session_state.model_predictions_path.name,
                                        mime="text/csv",
                                        key=predictions_widget_key,
                                        use_container_width=True
                                    )
                                else:
                                    st.error("❌ Predictions not available")
                    else:
                        st.info("👆 Click 'Run Prediction' to classify the gait.")
        
        # Sub-tab 2: Advanced model prediction
        with subtab2:
            st.markdown("---")
            st.subheader("🔬 Classify Gait with Advanced Model")
            
            # Check if gait cycles are available
            gait_cycles_available = (st.session_state.gait_cycles is not None and len(st.session_state.gait_cycles) > 0)
            
            if not gait_cycles_available and not st.session_state.processing_complete:
                st.info("⏳ No gait cycles available yet. Process a video in the Process tab first.")
            elif not st.session_state.processing_complete:
                st.info("⏳ No results available yet. Process a video in the Process tab first.")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.info(f"**Gait Cycles Available:** {len(st.session_state.gait_cycles)} cycles")
                    st.caption(f"Each cycle: {st.session_state.gait_cycles.shape[1]} frames × {st.session_state.gait_cycles.shape[2]} joints × 3 coordinates")
                    
                    st.markdown("### Advanced Model Configuration")
                    
                    # Model type selection
                    model_type = st.selectbox(
                        "Select advanced model type:",
                        ["Binary Classification", "Multilabel Classification"],
                        help="Select the type of advanced model to use for prediction"
                    )
                    
                    # Window size configuration
                    window_size = st.slider(
                        "Window Size (frames):",
                        min_value=30,
                        max_value=120,
                        value=60,
                        step=10,
                        help="Number of frames in each sliding window"
                    )
                    
                    # Stride configuration
                    stride = st.slider(
                        "Stride (frames):",
                        min_value=10,
                        max_value=60,
                        value=30,
                        step=5,
                        help="Number of frames to skip between windows"
                    )
                    
                    # Model selection
                    advanced_model_files = list(ADVANCED_MODELS_DIR.glob("*.h5")) + list(ADVANCED_MODELS_DIR.glob("*.pth")) + list(ADVANCED_MODELS_DIR.glob("*.pt"))
                    
                    if advanced_model_files:
                        model_options = [f.name for f in advanced_model_files]
                        selected_model = st.selectbox("Select advanced model:", model_options)
                        
                        # Load model if not already loaded
                        if st.session_state.advanced_model is None or selected_model != st.session_state.get('selected_advanced_model', ''):
                            st.session_state.advanced_model = ModelManager.load_model(
                                ADVANCED_MODELS_DIR / selected_model, 
                                model_type="advanced"
                            )
                            st.session_state.selected_advanced_model = selected_model
                    else:
                        st.error("❌ No advanced model files found")
                        st.stop()
                
                with col2:
                    if st.button("🚀 Run Prediction", use_container_width=True, type="primary"):
                        with st.spinner("Running advanced model prediction..."):
                            try:
                                # Load model if not already loaded
                                if st.session_state.advanced_model is None:
                                    st.session_state.advanced_model = ModelManager.load_model(
                                        ADVANCED_MODELS_DIR / st.session_state.selected_advanced_model,
                                        model_type="advanced"
                                    )
                                    if st.session_state.advanced_model is None:
                                        st.error("❌ Failed to load advanced model")
                                        st.stop()
                                
                                # Prepare features for prediction
                                X_stgcn = ModelManager.prepare_features_for_advanced_model(st.session_state.gait_cycles)
                                
                                if X_stgcn is not None:
                                    # Make predictions
                                    y_pred, y_pred_proba = ModelManager.predict_with_advanced_model(
                                        st.session_state.advanced_model, 
                                        X_stgcn
                                    )
                                    
                                    if len(y_pred) > 0:
                                        # Save predictions
                                        predictions_path = ModelManager.save_predictions(
                                            pd.DataFrame(), y_pred, y_pred_proba, 
                                            st.session_state.uploaded_video_path.stem,
                                            model_type="advanced"
                                        )
                                        
                                        # Update session state
                                        st.session_state.advanced_model_predictions = {
                                            'features': pd.DataFrame(),  # Empty dataframe for advanced model
                                            'y_pred': y_pred,
                                            'y_pred_proba': y_pred_proba
                                        }
                                        st.session_state.advanced_model_predictions_path = predictions_path
                                        
                                        st.success("✅ Advanced model prediction completed successfully!")
                                    else:
                                        st.error("❌ Failed to make predictions.")
                                else:
                                    st.error("❌ Failed to prepare features for advanced model")
                            except Exception as e:
                                st.error(f"❌ Error during advanced model prediction: {str(e)}")
                                logger.error(f"Error during advanced model prediction: {e}")
                    
                    # Display predictions if available
                    if st.session_state.advanced_model_predictions is not None:
                        st.markdown("---")
                        st.subheader("📊 Advanced Model Results")
                        
                        # Get the first prediction (most common case)
                        pred_class = st.session_state.advanced_model_predictions['y_pred'][0]
                        pred_proba = st.session_state.advanced_model_predictions['y_pred_proba']
                        
                        # Get appropriate labels based on model type
                        if model_type == "Binary Classification":
                            class_labels = ADVANCED_BINARY_LABELS
                        elif model_type == "Multilabel Classification":
                            class_labels = ADVANCED_MULTILABEL_LABELS
                        else:
                            class_labels = {i: f"Class {i}" for i in range(pred_proba.shape[1])}
                        
                        pred_label = class_labels.get(pred_class, f"Unknown({pred_class})")
                        
                        # Display the prediction with confidence
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Class", pred_label)
                        
                        with col2:
                            if pred_proba is not None:
                                st.metric("Confidence", f"{pred_proba[0][pred_class]:.2%}")
                            else:
                                st.metric("Confidence", "N/A")
                        
                        with col3:
                            # Color code the severity
                            if model_type == "Binary Classification":
                                if pred_class == 0:
                                    st.metric("Severity", "Normal", delta="✅")
                                else:
                                    st.metric("Severity", "Abnormal", delta="⚠️")
                            elif model_type == "Multilabel Classification":
                                if pred_class == 0:
                                    st.metric("Severity", "Normal", delta="✅")
                                elif pred_class == 1:
                                    st.metric("Severity", "Foot Drop", delta="⚠️")
                                elif pred_class == 2:
                                    st.metric("Severity", "Knee Stiffness", delta="⚠️")
                                elif pred_class == 3:
                                    st.metric("Severity", "Hip Weakness", delta="⚠️")
                            else:
                                st.metric("Severity", "Unknown", delta="❓")
                        
                        # Display probability distribution
                        st.markdown("### Class Probabilities")
                        
                        if pred_proba is not None:
                            # Create a DataFrame for the probabilities
                            prob_df = pd.DataFrame({
                                'Class': [class_labels[i] for i in range(len(class_labels))],
                                'Probability': pred_proba[0]
                            })
                            
                            # Create a bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['green', 'gold', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']
                            bars = ax.bar(prob_df['Class'], prob_df['Probability'], color=colors[:len(prob_df)])
                            
                            # Add value labels on bars
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{height:.2%}', ha='center', va='bottom')
                            
                            ax.set_xlabel('Class')
                            ax.set_ylabel('Probability')
                            ax.set_title('Class Probability Distribution')
                            ax.set_ylim(0, 1)
                            
                            st.pyplot(fig)
                            plt.close()
                            
                            # Display prediction visualization
                            st.markdown("### Detailed Analysis")
                            fig = ModelManager.create_prediction_visualization(
                                st.session_state.advanced_model_predictions['features'],
                                st.session_state.advanced_model_predictions['y_pred'],
                                st.session_state.advanced_model_predictions['y_pred_proba'],
                                model_type="advanced"
                            )
                            st.pyplot(fig)
                            plt.close()
                            
                            # Download predictions
                            if st.session_state.advanced_model_predictions_path:
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.info(f"**Predictions File:** {st.session_state.advanced_model_predictions_path.name}")
                                    st.caption(f"📏 Size: {st.session_state.advanced_model_predictions_path.stat().st_size / 1024:.1f} KB")
                                
                                with col2:
                                    # Create unique keys for predictions download
                                    predictions_data_key = f"data_predictions_advanced_{st.session_state.advanced_model_predictions_path.stem}_{st.session_state.advanced_model_predictions_path.stat().st_mtime}"
                                    predictions_widget_key = f"widget_predictions_advanced_{st.session_state.advanced_model_predictions_path.stem}_{st.session_state.advanced_model_predictions_path.stat().st_mtime}"
                                    
                                    # Store predictions data in session state
                                    if predictions_data_key not in st.session_state:
                                        try:
                                            with open(st.session_state.advanced_model_predictions_path, 'rb') as f:
                                                st.session_state[predictions_data_key] = f.read()
                                        except Exception as e:
                                            logger.error(f"Failed to read predictions for download: {e}")
                                            st.session_state[predictions_data_key] = None
                                    
                                    if st.session_state[predictions_data_key]:
                                        st.download_button(
                                            "📥 Download Predictions",
                                            data=st.session_state[predictions_data_key],
                                            file_name=st.session_state.advanced_model_predictions_path.name,
                                            mime="text/csv",
                                            key=predictions_widget_key,
                                            use_container_width=True
                                        )
                                    else:
                                        st.error("❌ Predictions not available")
                        else:
                            st.info("👆 Click 'Run Prediction' to classify the gait.")
        
        # Sub-tab 3: Model comparison
        with subtab3:
            st.markdown("---")
            st.subheader("🔍 Compare Model Performance")
            
            # Check if both models are available
            baseline_available = (st.session_state.model is not None)
            advanced_available = (st.session_state.advanced_model is not None)
            
            if not baseline_available and not advanced_available:
                st.warning("⚠️ No models available for comparison. Please load models in the other tabs.")
            elif not baseline_available:
                st.info("📊 Please load a baseline model in the Baseline Model tab.")
            elif not advanced_available:
                st.info("📊 Please load an advanced model in the Advanced Model tab.")
            else:
                st.success("✅ Both models available for comparison")
                
                # Model comparison section
                st.markdown("### Model Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Baseline Model**")
                    if st.session_state.model is not None:
                        st.success("✅ Loaded")
                        if hasattr(st.session_state.model, 'feature_importances_'):
                            importances = st.session_state.model.feature_importances_
                            st.write(f"Number of features: {len(importances)}")
                            st.write(f"Top 5 features: {importances.argsort()[::-1][:5].tolist()}")
                        else:
                            st.write("No feature importances available")
                    else:
                        st.error("❌ Not loaded")
                
                with col2:
                    st.markdown("**Advanced Model**")
                    if st.session_state.advanced_model is not None:
                        st.success("✅ Loaded")
                        model_type = "Binary Classification" if 'binary' in st.session_state.selected_advanced_model.lower() else "Multilabel Classification"
                        st.write(f"Model type: {model_type}")
                    else:
                        st.error("❌ Not loaded")
                
                # Run comparison if both models are loaded
                if st.button("🔍 Compare Models", use_container_width=True, type="primary"):
                    with st.spinner("Comparing models..."):
                        try:
                            # Get gait cycles for advanced model
                            if st.session_state.gait_cycles is None or len(st.session_state.gait_cycles) == 0:
                                st.error("❌ No gait cycles available for comparison")
                                st.stop()
                            
                            # Prepare features for both models
                            # For baseline model
                            df_baseline = ModelManager.prepare_features_for_baseline_model(st.session_state.features_df)
                            
                            # For advanced model
                            X_stgcn = GaitAnalysisEngine.extract_gait_cycles_for_advanced_model(
                                st.session_state.gait_cycles,
                                window_size=60,
                                stride=30
                            )
                            
                            if X_stgcn is None:
                                st.error("❌ Failed to prepare features for advanced model")
                                st.stop()
                            
                            # Make predictions with both models
                            y_pred_baseline, y_pred_proba_baseline = ModelManager.predict_with_baseline_model(
                                st.session_state.model, df_baseline
                            )
                            
                            y_pred_advanced, y_pred_proba_advanced = ModelManager.predict_with_advanced_model(
                                st.session_state.advanced_model, X_stgcn
                            )
                            
                            if len(y_pred_baseline) > 0 and len(y_pred_advanced) > 0:
                                # Create comparison visualization
                                st.markdown("### Model Comparison Results")
                                
                                # Create a comparison table
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Baseline Model**")
                                    pred_class = y_pred_baseline[0]
                                    pred_proba = y_pred_proba_baseline[0]
                                    pred_label = BASELINE_CLASS_LABELS.get(pred_class, f"Unknown({pred_class})")
                                    st.metric("Predicted Class", pred_label)
                                    st.metric("Confidence", f"{pred_proba[pred_class]:.2%}")
                                    
                                    # Display probability distribution
                                    prob_df = pd.DataFrame({
                                        'Class': [BASELINE_CLASS_LABELS[i] for i in range(len(BASELINE_CLASS_LABELS))],
                                        'Probability': pred_proba
                                    })
                                    
                                    # Create a bar chart
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    colors = ['green', 'gold', 'orange', 'red']
                                    bars = ax.bar(prob_df['Class'], prob_df['Probability'], color=colors)
                                    
                                    # Add value labels on bars
                                    for bar in bars:
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                               f'{height:.2%}', ha='center', va='bottom')
                                    
                                    ax.set_xlabel('Class')
                                    ax.set_ylabel('Probability')
                                    ax.set_title('Baseline Model: Class Probability Distribution')
                                    ax.set_ylim(0, 1)
                                    
                                    st.pyplot(fig)
                                    plt.close()
                                
                                with col2:
                                    st.markdown("**Advanced Model**")
                                    pred_class = y_pred_advanced[0]
                                    pred_proba = y_pred_proba_advanced[0]
                                    
                                    # Get appropriate labels based on model type
                                    model_type = "Binary Classification" if 'binary' in st.session_state.selected_advanced_model.lower() else "Multilabel Classification"
                                    
                                    if model_type == "Binary Classification":
                                        pred_label = ADVANCED_BINARY_LABELS.get(pred_class, f"Unknown({pred_class})")
                                        st.metric("Predicted Class", pred_label)
                                        if pred_proba is not None:
                                            st.metric("Confidence", f"{pred_proba[pred_class]:.2%}")
                                        else:
                                            st.metric("Confidence", "N/A")
                                            
                                        # Display probability distribution
                                        prob_df = pd.DataFrame({
                                            'Class': [ADVANCED_BINARY_LABELS[i] for i in range(len(ADVANCED_BINARY_LABELS))],
                                            'Probability': pred_proba
                                        })
                                        
                                        # Create a bar chart
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        colors = ['green', 'red']
                                        bars = ax.bar(prob_df['Class'], prob_df['Probability'], color=colors)
                                        
                                        # Add value labels on bars
                                        for bar in bars:
                                            height = bar.get_height()
                                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                                   f'{height:.2%}', ha='center', va='bottom')
                                        
                                        ax.set_xlabel('Class')
                                        ax.set_ylabel('Probability')
                                        ax.set_title('Advanced Model: Class Probability Distribution')
                                        ax.set_ylim(0, 1)
                                        
                                        st.pyplot(fig)
                                        plt.close()
                                    else:
                                        # Multilabel classification
                                        pred_label = ADVANCED_MULTILABEL_LABELS.get(pred_class, f"Unknown({pred_class})")
                                        st.metric("Predicted Class", pred_label)
                                        
                                        if pred_proba is not None:
                                            # For multilabel, show top 3 predictions
                                            top_indices = np.argsort(pred_proba[0])[-3:]
                                            top_labels = [ADVANCED_MULTILABEL_LABELS[i] for i in top_indices]
                                            top_probs = pred_proba[0][top_indices]
                                            
                                            st.write("**Top Predictions:**")
                                            for label, prob in zip(top_labels, top_probs):
                                                st.write(f"- {label}: {prob:.2%}")
                                        else:
                                            st.write("No probability data available")
                                
                                # Comparison metrics
                                st.markdown("### Model Comparison Metrics")
                                
                                # Calculate metrics for both models
                                if y_pred_proba_baseline is not None and y_pred_proba_advanced is not None:
                                    # Baseline model metrics
                                    baseline_acc = accuracy_score(
                                        st.session_state.features_df['label_class'] if 'label_class' in st.session_state.features_df.columns else y_pred_baseline
                                    )
                                    
                                    # Advanced model metrics
                                    if 'binary' in st.session_state.selected_advanced_model.lower():
                                        # For binary classification
                                        advanced_acc = accuracy_score(y_pred_advanced)
                                    else:
                                        # For multilabel classification
                                        from sklearn.metrics import multilabel_confusion_matrix
                                        # For multilabel, use average accuracy
                                        advanced_acc = np.mean([
                                            accuracy_score(y_pred_advanced[:, i])
                                            for i in range(y_pred_advanced.shape[1])
                                        ])
                                    
                                    # Display metrics
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Baseline Model Accuracy", f"{baseline_acc:.2%}")
                                    with col2:
                                        st.metric("Advanced Model Accuracy", f"{advanced_acc:.2%}")
                                    
                                    # Create comparison visualization
                                    st.markdown("### Performance Comparison")
                                    
                                    # Create a bar chart comparing model performance
                                    models = ['Baseline', 'Advanced']
                                    accuracies = [baseline_acc, advanced_acc]
                                    colors = ['blue', 'green']
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    bars = ax.bar(models, accuracies, color=colors)
                                    
                                    # Add value labels on bars
                                    for i, (model, acc) in enumerate(zip(models, accuracies)):
                                        ax.text(i, acc + 0.01, f"{acc:.2%}", ha='center', va='bottom')
                                    
                                    ax.set_ylabel('Accuracy')
                                    ax.set_title('Model Comparison')
                                    ax.set_ylim(0, 1)
                                    ax.set_xticks(range(len(models)))
                                    ax.set_xticklabels(models)
                                    
                                    st.pyplot(fig)
                                    plt.close()
                                else:
                                    st.warning("⚠️ Cannot compare models - missing probability data")
                            else:
                                st.warning("⚠️ Cannot compare models - missing predictions")
                        except Exception as e:
                            st.error(f"❌ Error during model comparison: {str(e)}")
                            logger.error(f"Error during model comparison: {e}")
if __name__ == "__main__":
    main()

# ----------

# Luise:
# 38 branch -> 38-m2---13-check-model-for-poses-and-angles-> 1. -> XGBoost_based_on_feature_extraction.ipynb

# ----------

# Marc: 
# video-handling-mission-gait
# modeling_stgcn.ipynb
# pose_p_pasp.py
# Final in one binary !!!

# multi class model -> 
# Last script or bottom of the script. 

# ------------