#!/usr/bin/env python3
"""
MEDIAPIPE POSE DETECTION PIPELINE - PRODUCTION GRADE APPLICATION
Complete implementation with robust video rendering and export functionality
"""

import os
import sys
import warnings
import logging

# Environment configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import streamlit as st
import json
import importlib.util
from pathlib import Path
from datetime import datetime
import traceback
import hashlib
import subprocess
import time
import zipfile
from io import BytesIO
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import xgboost as xgb

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

# Create directories if they don't exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, FEATURES_DIR, GAIT_CYCLES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

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
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)]).strip()
            
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
        st.markdown(f"**{label}**")
        
        if not video_path or not video_path.exists():
            st.warning("⚠️ Video not found")
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
# GAIT ANALYSIS ENGINE - UPDATED WITH NEW IMPLEMENTATION
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
    
    GAIT_JOINT_INDEX = {joint_id: i for i, joint_id in enumerate(GAIT_JOINTS)}
    
    # Target labels
    LABEL_MAP = {
        "normal": 0,
        "abnormal": 1
    }
    
    ANOMALY_COLS = [
        "gait_anomaly_knee_sagittal_plane_abnormality",
        "gait_anomaly_trunk_balance_abnormality",
        "gait_anomaly_spatiotemporal_asymmetry",
        "gait_anomaly_hip_pelvic_control_deficit",
        "gait_anomaly_distal_foot_control_deficit",
    ]
    
    ANOMALY_CLASS_MAP = {
        "gait_anomaly_distal_foot_control_deficit": {
            "Foot Drop",
            "Foot Slap",
            "Inadequate Dorsiflexion",
            "Foot Flat Initial Contact",
            "Excess Pronation",
            "Excess Supination",
            "Reduced Metatarsophalangeal Joint Extension",
            "Absent Heel Rise During Terminal Stance",
            "Early Heel Rise",
            "Steppage Gait",
        },
        "gait_anomaly_knee_sagittal_plane_abnormality": {
            "Knee Extensor Thrust",
            "Knee Hyperextension",
            "Reduced Knee Extension",
            "Reduced Knee Flexion",
            "Knee Valgus",
        },
        "gait_anomaly_hip_pelvic_control_deficit": {
            "Trendelenburg",
            "Hip Hiking",
            "Posterior Pelvic Tilt",
            "Anterior Pelvic Tilt",
            "Reduced Pelvic Rotation",
            "Reduced Hip Extension",
            "Reduced Hip Internal Rotation",
            "Circumduction",
            "Medial Whip",
        },
        "gait_anomaly_trunk_balance_abnormality": {
            "Reduced Arm Swing",
            "Forward Lean",
            "Left Lean",
            "Right Lean",
            "Reduced Trunk Rotation",
            "Imbalance",
            "Cautious Gait",
        },
        "gait_anomaly_spatiotemporal_asymmetry": {
            "Wide Base of Support",
            "Step Length Asymmetry",
            "Reduced Step Length",
            "Reduced Left Weightshift",
        },
    }
    
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
    def interpolate_pose(pose: np.ndarray) -> np.ndarray:
        """
        Linearly interpolate missing frames (zeros) in a pose tensor.
        pose: (T, J, 3)
        """
        T, J, C = pose.shape
        pose_interp = pose.copy()

        for j in range(J):
            for c in range(C):
                coord = pose[:, j, c]
                missing = coord == 0  # detect missing frames
                if missing.all():
                    continue  # skip if all frames missing
                valid_idx = np.where(~missing)[0]
                valid_values = coord[valid_idx]
                # linear interpolation across all T
                pose_interp[:, j, c] = np.interp(np.arange(T), valid_idx, valid_values)
        
        return pose_interp
    
    @staticmethod
    def add_pose_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert long-format pose data to pose column
        """
        pose_rows = []

        for video_id in df["video_id"].unique():
            group = df[df["video_id"] == video_id].sort_values("frame")

            frames = np.sort(group["frame"].unique())
            frame_to_idx = {f: i for i, f in enumerate(frames)}

            T = len(frames)
            pose = np.zeros((T, GaitAnalysisEngine.N_JOINTS, 3), dtype=np.float32)

            for _, row in group.iterrows():
                f_idx = frame_to_idx[row["frame"]]
                j = int(row["landmark_id"])
                if j >= GaitAnalysisEngine.N_JOINTS:
                    continue  # skip any bad IDs
                pose[f_idx, j, :] = [row["x_norm"], row["y_norm"], row["z_norm"]]

            # Interpolation across all 33 joints
            pose = GaitAnalysisEngine.interpolate_pose(pose)

            base_row = group.iloc[0].to_dict()
            base_row["pose"] = pose
            pose_rows.append(base_row)

        return pd.DataFrame(pose_rows)
    
    @staticmethod
    def extract_sliding_windows(
        pose: np.ndarray,
        fps: int = 30,
        window_seconds: float = 2.0,
        overlap: float = 0.5
    ):
        """
        Split pose sequence into sliding windows.
        
        Args:
            pose: (T, J, 3) normalized pose array
            fps: frames per second
            window_seconds: length of window in seconds
            overlap: fraction overlap between windows (0.0-1.0)
        
        Returns:
            List of pose windows: each (window_frames, J, 3)
        """
        T, J, C = pose.shape
        window_frames = int(window_seconds * fps)
        step_frames = int(window_frames * (1 - overlap))

        if window_frames > T:
            # If video is shorter than window, return single padded window
            return [pose]

        windows = []
        start = 0
        while start + window_frames <= T:
            window = pose[start:start + window_frames]
            windows.append(window)
            start += step_frames

        # Optional: include last partial window if remaining frames > 50% of window
        if start < T and (T - start) >= int(0.5 * window_frames):
            window = pose[T - window_frames : T]
            windows.append(window)

        return windows
    
    @staticmethod
    def preprocess_gait_sliding_windows(
        df_video: pd.DataFrame,
        window_seconds: float = 2.0,
        overlap: float = 0.5,
        resample_frames: int = 60
    ):
        """
        Preprocess gait data using sliding windows, resampling each window to a fixed frame count.
        Keep all 33 joints for feature extraction.

        Returns:
            X_windows:        (N_windows, resample_frames, 33, 3)
            y_binary:         (N_windows,)
            y_multilabel:     (N_windows, 5)
            window_ids:       list[str]
        """
        from scipy.signal import resample
        
        all_windows = []
        all_binary_labels = []
        all_multilabels = []
        all_window_ids = []

        for idx, row in df_video.iterrows():
            pose = row["pose"]
            if pose is None:
                continue

            pose = np.asarray(pose)
            if pose.size == 0 or pose.ndim != 3:
                print(f"Row {idx} has invalid pose shape: {pose.shape}")
                continue

            fps = row.get("fps", 30) or 30

            # Label mapping Binary
            label_str = str(row.get("dataset", "none")).strip().lower()
            binary_label = GaitAnalysisEngine.LABEL_MAP.get(label_str, 0)

            # Label mapping Multi-label
            multilabel = np.array(
                [int(row.get(col, 0)) for col in GaitAnalysisEngine.ANOMALY_COLS],
                dtype=np.int32
            )

            # Normalize pose
            try:
                pose_norm = GaitAnalysisEngine.normalize_pose_3d(pose)
            except Exception as e:
                print(f"Skipping row {idx} due to normalization error: {e}")
                continue

            # Extract sliding windows
            windows = GaitAnalysisEngine.extract_sliding_windows(
                pose_norm,
                fps=fps,
                window_seconds=window_seconds,
                overlap=overlap
            )

            # KEEP ALL 33 JOINTS (don't select only gait joints)
            # This ensures compatibility with feature extraction
            
            video_id = row.get("video_id", f"vid{idx}")

            # Loop over windows to assign unique IDs and resample
            for win_idx, w in enumerate(windows):
                # Resample to fixed number of frames
                if w.shape[0] < 2:  # skip trivially short windows
                    continue
                w_resampled = resample(w, resample_frames, axis=0)

                # Generate window ID
                start_frame = win_idx * int(window_seconds * fps * (1 - overlap))
                end_frame = start_frame + w_resampled.shape[0] - 1
                window_id = f"{video_id}_win{win_idx:03d}_f{start_frame}-{end_frame}"

                # Append to lists
                all_windows.append(w_resampled)
                all_binary_labels.append(binary_label)
                all_multilabels.append(multilabel)
                all_window_ids.append(window_id)

        # Convert to NumPy arrays
        X_windows = np.asarray(all_windows, dtype=np.float32)
        y_binary = np.asarray(all_binary_labels, dtype=np.int32)
        y_multilabel = np.asarray(all_multilabels, dtype=np.int32)

        return X_windows, y_binary, y_multilabel, all_window_ids
    
    @staticmethod
    def qc_gait_window(window, fps, visualize=False, title=""):
        """
        QC for a single sliding window.
        
        window: (T, 33, 3) - using all 33 joints
        fps: frames per second for this window
        """
        from scipy.signal import find_peaks
        
        # Use original joint indices (not GAIT_JOINT_INDEX)
        L_HIP = GaitAnalysisEngine.LEFT_HIP
        R_HIP = GaitAnalysisEngine.RIGHT_HIP
        L_SHOULDER = GaitAnalysisEngine.LEFT_SHOULDER
        L_KNEE = GaitAnalysisEngine.LEFT_KNEE
        L_ANKLE = GaitAnalysisEngine.LEFT_ANKLE

        qc = {}

        # -----------------------------
        # 1) Length / duration
        # -----------------------------
        qc["n_frames"] = window.shape[0]
        qc["duration_s"] = window.shape[0] / fps
        qc["flag_short"] = qc["duration_s"] < 1.0   # relaxed vs clip-based QC

        # -----------------------------
        # 2) Pelvis centering
        # -----------------------------
        pelvis = (window[:, L_HIP] + window[:, R_HIP]) / 2
        qc["pelvis_offset"] = np.linalg.norm(pelvis.mean(axis=0))
        qc["flag_off_center"] = qc["pelvis_offset"] > 0.1

        # -----------------------------
        # 3) Torso length stability
        # -----------------------------
        torso = window[:, L_SHOULDER] - pelvis
        torso_len = np.linalg.norm(torso, axis=1)
        qc["torso_mean_length"] = torso_len.mean()
        qc["torso_std_length"] = torso_len.std()
        qc["flag_torso_unstable"] = qc["torso_std_length"] > 0.15

        # -----------------------------
        # 4) Smoothness / jitter (ankle Y)
        # -----------------------------
        ankle_y = window[:, L_ANKLE, 1]
        ankle_y_vel = np.diff(ankle_y)
        qc["ankle_y_velocity_std"] = np.std(ankle_y_vel)
        qc["flag_jitter"] = qc["ankle_y_velocity_std"] > 0.2

        # -----------------------------
        # 5) Weak periodicity (knee Y)
        # -----------------------------
        knee_y = window[:, L_KNEE, 1]
        min_peak_dist = int(0.4 * fps)
        peaks, _ = find_peaks(knee_y, distance=min_peak_dist)

        qc["n_knee_peaks"] = len(peaks)
        qc["flag_no_periodicity"] = qc["n_knee_peaks"] < 1

        # -----------------------------
        # 6) Depth stability (ankle Z)
        # -----------------------------
        ankle_z = window[:, L_ANKLE, 2]
        qc["ankle_z_range"] = ankle_z.max() - ankle_z.min()
        qc["ankle_z_spike"] = np.max(np.abs(np.diff(ankle_z)))
        qc["flag_flat_depth"] = qc["ankle_z_range"] < 0.05

        # -----------------------------
        # Overall QC decision
        # -----------------------------
        qc["qc_fail"] = any([
            qc["flag_short"],
            qc["flag_off_center"],
            qc["flag_jitter"],
            qc["flag_no_periodicity"],
            qc["flag_flat_depth"],
            qc["flag_torso_unstable"]
        ])

        # -----------------------------
        # Optional visualization
        # -----------------------------
        if visualize:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plt.plot(knee_y)
            plt.title(f"{title} – Knee Y")
            plt.subplot(1,2,2)
            plt.plot(ankle_z)
            plt.title(f"{title} – Ankle Z")
            plt.show()

        return qc
    
    @staticmethod
    def apply_qc_windows(X_windows, y_binary, y_multilabel, window_ids, fps=60):
        """
        Apply QC to sliding windows and filter out windows that fail QC.
        
        Returns:
            X_clean, y_binary_clean, y_multilabel_clean, window_ids_clean, qc_df
        """
        
        qc_rows = []
        for i, window in enumerate(X_windows):
            qc = GaitAnalysisEngine.qc_gait_window(window, fps=fps, visualize=False, title=f"Window {i}")
            qc_rows.append(qc)

        qc_df = pd.DataFrame(qc_rows)
        qc_df["binary_label"] = y_binary
        qc_df["window_id"] = window_ids

        # QC-clean mask
        keep_mask = ~qc_df["qc_fail"].values

        # Ensure window_ids is numpy array
        window_ids = np.array(window_ids)
        y_multilabel = np.asarray(y_multilabel)

        # Apply mask
        X_clean = X_windows[keep_mask]
        y_binary_clean = y_binary[keep_mask]
        y_multilabel_clean = y_multilabel[keep_mask]
        window_ids_clean = window_ids[keep_mask]

        print(f"QC-clean windows: {len(X_clean)} / {len(X_windows)} ({100*len(X_clean)/len(X_windows):.2f}%)")
        
        return X_clean, y_binary_clean, y_multilabel_clean, window_ids_clean, qc_df
    
    @staticmethod
    def joint_speed(
        pose_norm: np.ndarray,
        joint_idx: int,
        fps: float,
        smooth_sigma: float = 1.0,
    ) -> np.ndarray:
        """
        Frame-to-frame 3D speed of one joint in a normalized clip.
        -------
        speed : (T-1,) array in 'normalized units per second'
        """
        from scipy.ndimage import gaussian_filter1d
        
        joint_traj = pose_norm[:, joint_idx, :]  # (T, 3)

        if smooth_sigma and smooth_sigma > 0:
            joint_traj = gaussian_filter1d(joint_traj, sigma=smooth_sigma, axis=0)
        diffs = np.diff(joint_traj, axis=0)
        disp = np.linalg.norm(diffs, axis=1)
        return disp * fps

    @staticmethod
    def moving_and_still_times(
        pose_norm: np.ndarray,
        joint_idx: int,
        fps: float,
        speed_thresh: float = 0.02,
        smooth_sigma: float = 1.0,
    ) -> dict:
        """
        How long a joint is moving vs not moving.
        -------
        dict with times in seconds and fractions of the clip.
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
    def range_of_motion(
        pose_norm: np.ndarray,
        joint_idx: int,
        axis: str | None = None,
    ) -> dict:
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
    def asymmetry(L: float, R: float, eps: float = 1e-6) -> float:
        """
        Generic left-right asymmetry index: (L - R) / (L + R + eps)
        """
        return float((L - R) / (L + R + eps))

    @staticmethod
    def joint_angle(
        p_prox: np.ndarray,
        p_joint: np.ndarray,
        p_dist: np.ndarray,
    ) -> np.ndarray:
        """
        Joint angle in degrees over time.
        Angle is between segments (p_prox - p_joint) and (p_dist - p_joint).
        """
        v1 = p_prox - p_joint        # (T, 3)
        v2 = p_dist - p_joint        # (T, 3)

        num = np.einsum("ij,ij->i", v1, v2)  # (T,)
        den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6

        cosang = np.clip(num / den, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosang))  # (T,)

        return angles

    @staticmethod
    def detect_step_events_from_ankle(
        ankle_y: np.ndarray,
        fps: float,
        min_step_time: float = 0.3,
    ) -> np.ndarray:
        """
        - Use local minima (peaks on -ankle_y) as heel strike proxies
        - min_step_time limits unrealistically fast steps
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
    def step_temporal_features(
        ankle_y: np.ndarray,
        fps: float,
        min_step_time: float = 0.3,
    ) -> dict:
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
    def compute_window_features(window: np.ndarray, fps: float) -> dict[str, float]:
        """Compute gait features from a single window.

        Parameters
        ----------
        window : np.ndarray
            Pose window, shape (T, 33, 3). Can be raw or already-normalized.
        fps : float
            Effective FPS for this window time axis (after any resampling).

        Returns
        -------
        dict
            Scalar feature dictionary for this window.
        """

        window = np.asarray(window)
        if window.ndim != 3 or window.shape[1] != GaitAnalysisEngine.N_JOINTS or window.shape[2] != 3:
            raise ValueError(f"window must be of shape (T, {GaitAnalysisEngine.N_JOINTS}, 3), got {window.shape}")

        # Auto-normalize if the pelvis isn't near 0 (same heuristic as before)
        pelvis = (window[:, GaitAnalysisEngine.LEFT_HIP] + window[:, GaitAnalysisEngine.RIGHT_HIP]) / 2
        pelvis_mean_norm = np.linalg.norm(pelvis.mean(axis=0))
        pose_norm = GaitAnalysisEngine.normalize_pose_3d(window) if pelvis_mean_norm > 1e-2 else window

        feats: dict[str, float] = {}

        # ------------------------------------------------------------------
        # Basic spatial features
        # ------------------------------------------------------------------
        # Step height (ankle vertical range)
        left_ankle_y = pose_norm[:, GaitAnalysisEngine.LEFT_ANKLE, 1]
        right_ankle_y = pose_norm[:, GaitAnalysisEngine.RIGHT_ANKLE, 1]

        feats["step_height_L"] = float(left_ankle_y.max() - left_ankle_y.min())
        feats["step_height_R"] = float(right_ankle_y.max() - right_ankle_y.min())

        # Step length (ankle horizontal range)
        left_ankle_x = pose_norm[:, GaitAnalysisEngine.LEFT_ANKLE, 0]
        right_ankle_x = pose_norm[:, GaitAnalysisEngine.RIGHT_ANKLE, 0]

        feats["step_length_L"] = float(left_ankle_x.max() - left_ankle_x.min())
        feats["step_length_R"] = float(right_ankle_x.max() - right_ankle_x.min())

        # Pelvic drop (hip height asymmetry over time)
        left_hip_y = pose_norm[:, GaitAnalysisEngine.LEFT_HIP, 1]
        right_hip_y = pose_norm[:, GaitAnalysisEngine.RIGHT_HIP, 1]
        pelvis_diff = left_hip_y - right_hip_y

        feats["pelvis_drop_mean"] = float(pelvis_diff.mean())
        feats["pelvis_drop_std"] = float(pelvis_diff.std())

        # Trunk lean (horizontal shoulder asymmetry)
        left_sh_x = pose_norm[:, GaitAnalysisEngine.LEFT_SHOULDER, 0]
        right_sh_x = pose_norm[:, GaitAnalysisEngine.RIGHT_SHOULDER, 0]
        trunk_lean = left_sh_x - right_sh_x

        feats["trunk_lean_mean"] = float(trunk_lean.mean())
        feats["trunk_lean_std"] = float(trunk_lean.std())

        # Heel clearance (vertical heel range)
        left_heel_y = pose_norm[:, GaitAnalysisEngine.LEFT_HEEL, 1]
        right_heel_y = pose_norm[:, GaitAnalysisEngine.RIGHT_HEEL, 1]

        feats["heel_range_L"] = float(left_heel_y.max() - left_heel_y.min())
        feats["heel_range_R"] = float(right_heel_y.max() - right_heel_y.min())

        # Simple symmetry indices (step height/length)
        eps = 1e-6
        hL, hR = feats["step_height_L"], feats["step_height_R"]
        lL, lR = feats["step_length_L"], feats["step_length_R"]

        feats["step_height_symmetry"] = float((hL - hR) / (hL + hR + eps))
        feats["step_length_symmetry"] = float((lL - lR) / (lL + lR + eps))
        
        # ------------------------------------------------------------------
        # Knee motion: moving vs still, ROM 
        left_knee_move = GaitAnalysisEngine.moving_and_still_times(pose_norm, GaitAnalysisEngine.LEFT_KNEE, fps)
        right_knee_move = GaitAnalysisEngine.moving_and_still_times(pose_norm, GaitAnalysisEngine.RIGHT_KNEE, fps)

        for k, v in left_knee_move.items():
            feats[f"knee_L_{k}"] = v
        for k, v in right_knee_move.items():
            feats[f"knee_R_{k}"] = v

        feats["knee_L_rom_y"] = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.LEFT_KNEE, axis="y")["rom_y"]
        feats["knee_R_rom_y"] = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.RIGHT_KNEE, axis="y")["rom_y"]

        # ------------------------------------------------------------------
        # Joint ROM (hip / shoulder / ankle) + asymmetries + stance/swing
        
        # Hip ROM (vertical axis)
        hip_L_rom_y = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.LEFT_HIP, axis="y")["rom_y"]
        hip_R_rom_y = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.RIGHT_HIP, axis="y")["rom_y"]
        feats["hip_L_rom_y"] = hip_L_rom_y
        feats["hip_R_rom_y"] = hip_R_rom_y

        # Shoulder ROM (horizontal axis, trunk sway / arm swing proxy)
        shoulder_L_rom_x = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.LEFT_SHOULDER, axis="x")["rom_x"]
        shoulder_R_rom_x = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.RIGHT_SHOULDER, axis="x")["rom_x"]
        feats["shoulder_L_rom_x"] = shoulder_L_rom_x
        feats["shoulder_R_rom_x"] = shoulder_R_rom_x

        # Ankle ROM (vertical axis, dorsiflexion / clearance proxy)
        ankle_L_rom_y = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.LEFT_ANKLE, axis="y")["rom_y"]
        ankle_R_rom_y = GaitAnalysisEngine.range_of_motion(pose_norm, GaitAnalysisEngine.RIGHT_ANKLE, axis="y")["rom_y"]
        feats["ankle_L_rom_y"] = ankle_L_rom_y
        feats["ankle_R_rom_y"] = ankle_R_rom_y

        # ROM asymmetries
        feats["knee_rom_asym"] = GaitAnalysisEngine.asymmetry(feats["knee_L_rom_y"], feats["knee_R_rom_y"])
        feats["hip_rom_asym"] = GaitAnalysisEngine.asymmetry(hip_L_rom_y, hip_R_rom_y)
        feats["shoulder_rom_asym"] = GaitAnalysisEngine.asymmetry(shoulder_L_rom_x, shoulder_R_rom_x)
        feats["ankle_rom_asym"] = GaitAnalysisEngine.asymmetry(ankle_L_rom_y, ankle_R_rom_y)

        # Stance / swing ratio (ankle-based proxy)
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

        # ------------------------------------------------------------------
        # Joint angles (hip / knee / ankle) + angle-based ROM/asym
        
        # Knee angles (hip–knee–ankle)
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

        # Hip angles (shoulder–hip–knee)
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

        # Ankle angles (knee–ankle–foot index)
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

        # ------------------------------------------------------------------
        # Temporal gait features & step width proxy
        
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

        # Step width proxy (mediolateral ankle distance)
        ankle_L_x = pose_norm[:, GaitAnalysisEngine.LEFT_ANKLE, 0]
        ankle_R_x = pose_norm[:, GaitAnalysisEngine.RIGHT_ANKLE, 0]
        step_width_series = np.abs(ankle_L_x - ankle_R_x)

        feats["step_width_mean"] = float(step_width_series.mean())
        feats["step_width_std"] = float(step_width_series.std())

        return feats
    
    @staticmethod
    def extract_features_from_windows(
        X_windows: np.ndarray,
        fps: float,
        gait_pattern: list = None,
        movement_type: list = None,
        side: list = None,
        source_file: list = None,
    ) -> pd.DataFrame:
        """Extract features for each window.

        Parameters
        ----------
        X_windows : np.ndarray
            Window tensor, shape (N, T, 33, 3)
        fps : float
            Effective FPS for the window time axis (after resampling).
        gait_pattern, movement_type, side, source_file : optional iterables
            Per-window metadata. If not provided, values are None.

        Returns
        -------
        pd.DataFrame
            Same columns as the old extract_features_from_df_video output.
        """

        X_windows = np.asarray(X_windows)
        if X_windows.ndim != 4 or X_windows.shape[2] != GaitAnalysisEngine.N_JOINTS or X_windows.shape[3] != 3:
            raise ValueError(
                f"X_windows must have shape (N, T, {GaitAnalysisEngine.N_JOINTS}, 3), got {X_windows.shape}"
            )

        N = X_windows.shape[0]

        def _to_list(x):
            if x is None:
                return [None] * N
            x_list = list(x)
            if len(x_list) != N:
                raise ValueError(f"Expected metadata length {N}, got {len(x_list)}")
            return x_list

        gait_pattern_l = _to_list(gait_pattern)
        movement_type_l = _to_list(movement_type)
        side_l = _to_list(side)
        source_file_l = _to_list(source_file)

        rows: list[dict[str, any]] = []

        for i in range(N):
            feats = GaitAnalysisEngine.compute_window_features(X_windows[i], fps=fps)

            feats["label_fine"] = gait_pattern_l[i]
            feats["movement_type"] = movement_type_l[i]
            feats["side"] = side_l[i]
            feats["source_file"] = source_file_l[i]

            rows.append(feats)

        return pd.DataFrame(rows)
    
    @staticmethod
    def extract_features_from_csv(csv_path: Path, video_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Extract features from a CSV file containing MediaPipe landmarks using the new implementation.
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
            
            # Fill null values with "N" for movement_type if it exists
            if "movement_type" in df.columns:
                df = df.fillna({"movement_type": "N"})
            
            # Remove rows with slowmotion videos if movement_type exists
            if "movement_type" in df.columns:
                df = df[df["movement_type"] != "SLOWMOTION"]
            
            # Fill null values with "NA" for gait_markers if it exists
            if "gait_markers" in df.columns:
                df = df.fillna({"gait_markers": "NA"})
            
            # Check if video_id column exists, if not create a default one
            if "video_id" not in df.columns:
                df["video_id"] = "default_video"
            
            # Convert long format to pose column
            df_video = GaitAnalysisEngine.add_pose_column(df)
            
            if df_video.empty:
                st.error("No valid pose data found in CSV")
                return pd.DataFrame(), None
            
            # Preprocess gait dataframe using sliding windows
            X_windows, y_binary, y_multilabel, window_ids = GaitAnalysisEngine.preprocess_gait_sliding_windows(
                df_video,
                window_seconds=2.0,
                overlap=0.5,
                resample_frames=60
            )
            
            # Apply QC filtering to sliding windows
            X_clean, y_binary_clean, y_multilabel_clean, window_ids_clean, qc_df = GaitAnalysisEngine.apply_qc_windows(
                X_windows,
                y_binary,
                y_multilabel,
                window_ids,
                fps=60  # adjust if your videos/windows have a different FPS
            )
            
            # Extract features from windows
            df_features = GaitAnalysisEngine.extract_features_from_windows(
                X_clean,
                fps=60,
                gait_pattern=None,  # Can be filled if available in the CSV
                movement_type=None,  # Can be filled if available in the CSV
                side=None,  # Can be filled if available in the CSV
                source_file=[csv_path.name] * len(X_clean)
            )
            
            # Add binary and multilabel information
            df_features["binary_label"] = y_binary_clean
            for i, col in enumerate(GaitAnalysisEngine.ANOMALY_COLS):
                df_features[col] = y_multilabel_clean[:, i]
            
            # Convert windows to gait cycles for visualization
            # Keep all 33 joints for visualization
            gait_cycles = X_clean
            
            return df_features, gait_cycles
        
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


# ═══════════════════════════════════════════════════════════════════════════
# MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class ModelManager:
    """Manages loading and using trained models for prediction"""
    
    @staticmethod
    def load_baseline_model() -> Optional[xgb.XGBClassifier]:
        """Load the baseline XGBoost model"""
        model_path = MODELS_DIR / "baseline" / "xgboost_model.bin"
        
        if not model_path.exists():
            logger.error(f"Baseline model not found at {model_path}")
            return None
        
        try:
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            logger.info(f"Loaded baseline model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load baseline model: {e}")
            return None
    
    @staticmethod
    def load_binary_model() -> Optional[xgb.XGBClassifier]:
        """Load the binary classification model"""
        model_path = MODELS_DIR / "advance" / "binary_model_full.bin"
        
        if not model_path.exists():
            logger.error(f"Binary model not found at {model_path}")
            return None
        
        try:
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            logger.info(f"Loaded binary model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load binary model: {e}")
            return None
    
    @staticmethod
    def load_multilabel_model() -> Optional[xgb.XGBClassifier]:
        """Load the multi-label classification model"""
        model_path = MODELS_DIR / "advance" / "multi_label_model_full.bin"
        
        if not model_path.exists():
            logger.error(f"Multi-label model not found at {model_path}")
            return None
        
        try:
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            logger.info(f"Loaded multi-label model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load multi-label model: {e}")
            return None
    
    @staticmethod
    def prepare_features_for_prediction(df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model prediction by selecting only the columns
        that the model was trained on.
        """
        # Columns to exclude from features
        exclude_cols = [
            "label_fine", "label_class", "label_id",
            "movement_type", "side", "source_file",
            "binary_label"  # This is the target, not a feature
        ]
        
        # Add anomaly columns if they exist (these are also targets for multi-label)
        exclude_cols.extend(GaitAnalysisEngine.ANOMALY_COLS)
        
        # Select only numeric feature columns
        feature_cols = []
        for col in df_features.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_features[col]):
                feature_cols.append(col)
        
        X = df_features[feature_cols].copy()
        
        # Fill missing values with median (same as training)
        X = X.fillna(X.median(numeric_only=True))
        
        return X
    
    @staticmethod
    def predict_with_baseline(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the baseline model
        Returns: binary predictions (0=normal, 1=abnormal)
        """
        try:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.array([]), np.array([])
    
    @staticmethod
    def create_prediction_summary(predictions: np.ndarray, probabilities: np.ndarray) -> dict:
        """
        Create a summary of predictions
        """
        unique, counts = np.unique(predictions, return_counts=True)
        
        summary = {
            "total_windows": len(predictions),
            "normal_windows": int(counts[unique == 0][0]) if 0 in unique else 0,
            "abnormal_windows": int(counts[unique == 1][0]) if 1 in unique else 0,
            "abnormal_percentage": float(counts[unique == 1][0] / len(predictions) * 100) if 1 in unique else 0,
            "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
            "high_confidence_abnormal": int(np.sum((predictions == 1) & (np.max(probabilities, axis=1) > 0.8))),
            "low_confidence_normal": int(np.sum((predictions == 0) & (np.max(probabilities, axis=1) < 0.6)))
        }
        
        return summary

    @staticmethod
    def predict_from_pose_csv(csv_path, video_path=None, model=None):
        """
        Complete pipeline from raw pose CSV to predictions
        """
        # Extract features from raw pose data
        df_features, gait_cycles = GaitAnalysisEngine.extract_features_from_csv(csv_path, video_path)
        
        if df_features.empty:
            return None, None
        
        # Prepare features for prediction
        X = ModelManager.prepare_features_for_prediction(df_features)
        
        # Make predictions
        predictions, probabilities = ModelManager.predict_with_baseline(model, X)
        
        # Create results dataframe
        df_results = df_features.copy()
        df_results['prediction'] = predictions
        df_results['confidence'] = np.max(probabilities, axis=1)
        df_results['predicted_label'] = ['Normal' if p == 0 else 'Abnormal' for p in predictions]
        
        return df_results, probabilities

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

def create_prediction_visualization(predictions: np.ndarray, probabilities: np.ndarray, summary: dict):
    """Create visualizations for model predictions"""
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Pie chart of predictions
    ax1 = plt.subplot(2, 3, 1)
    labels = ['Normal', 'Abnormal']
    sizes = [summary['normal_windows'], summary['abnormal_windows']]
    colors = ['lightgreen', 'lightcoral']
    explode = (0, 0.1)  # explode abnormal slice
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Prediction Distribution')
    
    # 2. Confidence distribution
    ax2 = plt.subplot(2, 3, 2)
    confidences = np.max(probabilities, axis=1)
    ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Confidence Distribution')
    ax2.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    ax2.legend()
    
    # 3. Prediction timeline (if we have window IDs)
    ax3 = plt.subplot(2, 3, 3)
    window_indices = np.arange(len(predictions))
    colors_map = ['green' if p == 0 else 'red' for p in predictions]
    ax3.scatter(window_indices, predictions, c=colors_map, alpha=0.6)
    ax3.set_xlabel('Window Index')
    ax3.set_ylabel('Prediction (0=Normal, 1=Abnormal)')
    ax3.set_title('Predictions Timeline')
    ax3.grid(True, alpha=0.3)
    
    # 4. Confidence vs Prediction
    ax4 = plt.subplot(2, 3, 4)
    normal_mask = predictions == 0
    abnormal_mask = predictions == 1
    
    ax4.scatter(confidences[normal_mask], predictions[normal_mask], 
               c='green', alpha=0.6, label='Normal')
    ax4.scatter(confidences[abnormal_mask], predictions[abnormal_mask], 
               c='red', alpha=0.6, label='Abnormal')
    ax4.set_xlabel('Confidence Score')
    ax4.set_ylabel('Prediction')
    ax4.set_title('Confidence vs Prediction')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Summary metrics as text
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    summary_text = f"""
    Prediction Summary
    ═════════════════════════════
    
    Total Windows: {summary['total_windows']}
    Normal: {summary['normal_windows']} ({100-summary['abnormal_percentage']:.1f}%)
    Abnormal: {summary['abnormal_windows']} ({summary['abnormal_percentage']:.1f}%)
    
    Mean Confidence: {summary['mean_confidence']:.3f}
    High Confidence Abnormal: {summary['high_confidence_abnormal']}
    Low Confidence Normal: {summary['low_confidence_normal']}
    
    Recommendation: 
    {'Seek medical evaluation' if summary['abnormal_percentage'] > 20 else 'Normal gait pattern detected'}
    """
    
    ax5.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray'))
    
    # 6. ROC-like visualization (simplified)
    ax6 = plt.subplot(2, 3, 6)
    
    # Create a simple visualization of prediction quality
    normal_conf = confidences[normal_mask]
    abnormal_conf = confidences[abnormal_mask]
    
    if len(normal_conf) > 0 and len(abnormal_conf) > 0:
        ax6.hist(normal_conf, bins=10, alpha=0.5, label='Normal', color='green', density=True)
        ax6.hist(abnormal_conf, bins=10, alpha=0.5, label='Abnormal', color='red', density=True)
        ax6.set_xlabel('Confidence Score')
        ax6.set_ylabel('Density')
        ax6.set_title('Confidence Distribution by Class')
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'Insufficient data for\nclass-wise visualization', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Confidence Distribution by Class')
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_visualization(y_true: np.ndarray, y_pred: np.ndarray):
    """Create a confusion matrix visualization"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Add labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # Add tick marks
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Normal', 'Abnormal'])
    ax.set_yticklabels(['Normal', 'Abnormal'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
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
    if 'model_predictions' not in st.session_state:
        st.session_state.model_predictions = None
    if 'model_summary' not in st.session_state:
        st.session_state.model_summary = None
    
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
        
        # Check model availability
        baseline_model = ModelManager.load_baseline_model()
        if baseline_model:
            st.success("✅ Baseline model")
        else:
            st.error("❌ Baseline model missing")
        
        if VideoConverter.check_ffmpeg():
            st.success("✅ FFmpeg available")
        else:
            st.warning("⚠️ FFmpeg unavailable")
        
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
        
        if st.session_state.model_predictions is not None:
            st.success("✅ Model predictions available")
        
        st.markdown("---")
        
        if st.button("🔄 Reset", use_container_width=True):
            # Clear all session state
            keys_to_clear = [
                'uploaded_video_path', 'processing_complete', 'output_videos', 
                'last_tab', 'initialized', 'features_df', 'features_path',
                'gait_cycles', 'gait_cycles_path',
                'uploaded_csv_path', 'csv_features_df', 'csv_features_path',
                'model_predictions', 'model_summary'
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
            st.write(f"Models: {len(list(MODELS_DIR.glob('*')))}")
            st.write(f"Session keys: {len(st.session_state.keys())}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📤 Upload", "⚙️ Process", "🎬 Landmarker Videos", "📊 Feature Engineering", "🔬 Detailed Analysis", "🤖 Model Prediction"])

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
                        st.session_state.model_summary = None
                    
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
                                        st.session_state.model_predictions = None
                                        st.session_state.model_summary = None
                                        
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
    
    # with tab6:
    #     st.subheader("🤖 Model Prediction")
        
    #     # Check if baseline model is available
    #     baseline_model = ModelManager.load_baseline_model()
    #     if not baseline_model:
    #         st.error("❌ Baseline model not found. Please ensure the model file exists at `models/baseline/xgboost_model.bin`")
    #         st.info("To train a model, run the training script provided in the documentation.")
    #         return
        
    #     st.success("✅ Baseline model loaded successfully")
        
    #     # Create two sub-tabs
    #     subtab1, subtab2 = st.tabs(["From Feature Engineering", "From CSV Upload"])
        
    #     # Sub-tab 1: Use features from Feature Engineering
    #     with subtab1:
    #         if st.session_state.features_df is None or st.session_state.features_df.empty:
    #             st.warning("⚠️ No features available. Please extract features in the Feature Engineering tab first.")
    #             if st.button("Go to Feature Engineering Tab"):
    #                 st.info("Please navigate to the Feature Engineering tab to extract features first.")
    #         else:
    #             st.markdown("---")
    #             st.subheader("🔮 Run Model Prediction")
                
    #             col1, col2 = st.columns([2, 1])
                
    #             with col1:
    #                 st.info(f"**Features Available:** {len(st.session_state.features_df)} feature vectors")
    #                 st.info(f"**Model:** Baseline XGBoost Binary Classifier")
                
    #             with col2:
    #                 if st.button("🚀 Run Prediction", use_container_width=True, type="primary"):
    #                     with st.spinner("Preparing features and running prediction..."):
    #                         try:
    #                             # Prepare features for prediction
    #                             X = ModelManager.prepare_features_for_prediction(st.session_state.features_df)
                                
    #                             # Make predictions
    #                             predictions, probabilities = ModelManager.predict_with_baseline(baseline_model, X)
                                
    #                             if len(predictions) > 0:
    #                                 # Create prediction summary
    #                                 summary = ModelManager.create_prediction_summary(predictions, probabilities)
                                    
    #                                 # Store in session state
    #                                 st.session_state.model_predictions = predictions
    #                                 st.session_state.model_summary = summary
                                    
    #                                 st.success("✅ Prediction completed successfully!")
    #                             else:
    #                                 st.error("❌ Prediction failed. No valid predictions generated.")
    #                         except Exception as e:
    #                             st.error(f"❌ Error during prediction: {str(e)}")
    #                             logger.error(f"Prediction error: {e}")
                
    #             # Display results if available
    #             if st.session_state.model_predictions is not None:
    #                 st.markdown("---")
    #                 st.subheader("📊 Prediction Results")
                    
    #                 summary = st.session_state.model_summary
                    
    #                 # Display summary metrics
    #                 col1, col2, col3, col4 = st.columns(4)
    #                 with col1:
    #                     st.metric("Total Windows", summary['total_windows'])
    #                 with col2:
    #                     st.metric("Normal", summary['normal_windows'])
    #                 with col3:
    #                     st.metric("Abnormal", summary['abnormal_windows'])
    #                 with col4:
    #                     st.metric("Abnormal %", f"{summary['abnormal_percentage']:.1f}%")
                    
    #                 # Recommendation
    #                 st.markdown("---")
    #                 st.subheader("💡 Recommendation")
                    
    #                 if summary['abnormal_percentage'] > 20:
    #                     st.error("⚠️ **High percentage of abnormal gait patterns detected.**")
    #                     st.info("Recommendation: Seek medical evaluation for further assessment.")
    #                 elif summary['abnormal_percentage'] > 10:
    #                     st.warning("⚠️ **Some abnormal gait patterns detected.**")
    #                     st.info("Recommendation: Monitor closely and consider consultation if symptoms persist.")
    #                 else:
    #                     st.success("✅ **Normal gait pattern detected.**")
    #                     st.info("Recommendation: Continue regular monitoring.")
                    
    #                 # Visualizations
    #                 st.markdown("---")
    #                 st.subheader("📈 Prediction Visualizations")
                    
    #                 # Create prediction visualization
    #                 fig = create_prediction_visualization(
    #                     st.session_state.model_predictions,
    #                     probabilities,
    #                     summary
    #                 )
    #                 st.pyplot(fig)
    #                 plt.close()
                    
    #                 # Detailed results table
    #                 st.markdown("---")
    #                 st.subheader("📋 Detailed Results")
                    
    #                 # Create results dataframe
    #                 results_df = st.session_state.features_df.copy()
    #                 results_df['prediction'] = st.session_state.model_predictions
    #                 results_df['confidence'] = np.max(probabilities, axis=1)
    #                 results_df['predicted_label'] = ['Normal' if p == 0 else 'Abnormal' for p in st.session_state.model_predictions]
                    
    #                 # Show sample of results
    #                 with st.expander("👁️ View Detailed Predictions"):
    #                     st.dataframe(
    #                         results_df[['predicted_label', 'confidence', 'step_height_symmetry', 'step_length_symmetry', 
    #                                   'cadence_asym', 'knee_rom_asym', 'hip_rom_asym']].head(20),
    #                         use_container_width=True
    #                     )
                    
    #                 # Download predictions
    #                 st.markdown("---")
    #                 st.subheader("📥 Download Results")
                    
    #                 # Save predictions to CSV
    #                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #                 predictions_path = FEATURES_DIR / f"predictions_{timestamp}.csv"
                    
    #                 results_df.to_csv(predictions_path, index=False)
                    
    #                 col1, col2 = st.columns([3, 1])
                    
    #                 with col1:
    #                     st.info(f"**Predictions saved:** {predictions_path.name}")
    #                     st.caption(f"📏 Size: {predictions_path.stat().st_size / 1024:.1f} KB")
                    
    #                 with col2:
    #                     # Create download button
    #                     predictions_data_key = f"data_predictions_{timestamp}"
    #                     predictions_widget_key = f"widget_predictions_{timestamp}"
                        
    #                     if predictions_data_key not in st.session_state:
    #                         try:
    #                             with open(predictions_path, 'rb') as f:
    #                                 st.session_state[predictions_data_key] = f.read()
    #                         except Exception as e:
    #                             logger.error(f"Failed to read predictions for download: {e}")
    #                             st.session_state[predictions_data_key] = None
                        
    #                     if st.session_state[predictions_data_key]:
    #                         st.download_button(
    #                             "📥 Download Predictions",
    #                             data=st.session_state[predictions_data_key],
    #                             file_name=predictions_path.name,
    #                             mime="text/csv",
    #                             key=predictions_widget_key,
    #                             use_container_width=True
    #                         )
        
    #     # Sub-tab 2: Use features from uploaded CSV
    #     with subtab2:
    #         st.markdown("---")
    #         st.subheader("📁 Upload CSV for Model Prediction")
            
    #         col1, col2 = st.columns([2, 1])
            
    #         with col1:
    #             uploaded_csv = st.file_uploader(
    #                 "Upload a CSV file with extracted features",
    #                 type=['csv'],
    #                 help="Upload a CSV file containing the extracted features for prediction",
    #                 key="model_csv_upload"
    #             )
                
    #             if uploaded_csv:
    #                 st.info(f"📄 {uploaded_csv.name}")
    #                 st.info(f"📏 {uploaded_csv.size / (1024*1024):.2f} MB")
                    
    #                 # Save uploaded CSV
    #                 csv_path = FEATURES_DIR / f"model_{uploaded_csv.name}"
    #                 with open(csv_path, 'wb') as f:
    #                     f.write(uploaded_csv.getvalue())
                    
    #                 st.session_state.model_csv_path = csv_path
                    
    #                 # Preview CSV
    #                 with st.expander("👁️ Preview CSV Data"):
    #                     try:
    #                         df = pd.read_csv(csv_path)
    #                         st.dataframe(df.head(10), use_container_width=True)
    #                         st.caption(f"Showing first 10 rows of {len(df)} total rows")
    #                     except Exception as e:
    #                         st.error(f"Could not preview CSV: {e}")
            
    #         with col2:
    #             if 'model_csv_path' in st.session_state and st.session_state.model_csv_path:
    #                 st.metric("Status", "CSV uploaded")
    #                 st.metric("File", st.session_state.model_csv_path.name)
            
    #         # Run prediction on uploaded CSV
    #         if 'model_csv_path' in st.session_state and st.session_state.model_csv_path:
    #             st.markdown("---")
    #             st.subheader("🔮 Run Model Prediction")
                
    #             col1, col2 = st.columns([2, 1])
                
    #             with col1:
    #                 st.info(f"**CSV File:** {st.session_state.model_csv_path.name}")
    #                 st.caption(f"📏 Size: {st.session_state.model_csv_path.stat().st_size / 1024:.1f} KB")
                
    #             with col2:
    #                 if st.button("🚀 Run Prediction", use_container_width=True, type="primary", key="predict_model_csv"):
    #                     with st.spinner("Preparing features and running prediction..."):
    #                         try:
    #                             # Load features from CSV
    #                             df_features = pd.read_csv(st.session_state.model_csv_path)
                                
    #                             # Prepare features for prediction
    #                             X = ModelManager.prepare_features_for_prediction(df_features)
                                
    #                             # Make predictions
    #                             predictions, probabilities = ModelManager.predict_with_baseline(baseline_model, X)
                                
    #                             if len(predictions) > 0:
    #                                 # Create prediction summary
    #                                 summary = ModelManager.create_prediction_summary(predictions, probabilities)
                                    
    #                                 # Store in session state
    #                                 st.session_state.model_csv_predictions = predictions
    #                                 st.session_state.model_csv_summary = summary
                                    
    #                                 st.success("✅ Prediction completed successfully!")
    #                             else:
    #                                 st.error("❌ Prediction failed. No valid predictions generated.")
    #                         except Exception as e:
    #                             st.error(f"❌ Error during prediction: {str(e)}")
    #                             logger.error(f"Prediction error: {e}")
                
    #             # Display results if available
    #             if 'model_csv_predictions' in st.session_state and st.session_state.model_csv_predictions is not None:
    #                 st.markdown("---")
    #                 st.subheader("📊 Prediction Results")
                    
    #                 summary = st.session_state.model_csv_summary
                    
    #                 # Display summary metrics
    #                 col1, col2, col3, col4 = st.columns(4)
    #                 with col1:
    #                     st.metric("Total Windows", summary['total_windows'])
    #                 with col2:
    #                     st.metric("Normal", summary['normal_windows'])
    #                 with col3:
    #                     st.metric("Abnormal", summary['abnormal_windows'])
    #                 with col4:
    #                     st.metric("Abnormal %", f"{summary['abnormal_percentage']:.1f}%")
                    
    #                 # Recommendation
    #                 st.markdown("---")
    #                 st.subheader("💡 Recommendation")
                    
    #                 if summary['abnormal_percentage'] > 20:
    #                     st.error("⚠️ **High percentage of abnormal gait patterns detected.**")
    #                     st.info("Recommendation: Seek medical evaluation for further assessment.")
    #                 elif summary['abnormal_percentage'] > 10:
    #                     st.warning("⚠️ **Some abnormal gait patterns detected.**")
    #                     st.info("Recommendation: Monitor closely and consider consultation if symptoms persist.")
    #                 else:
    #                     st.success("✅ **Normal gait pattern detected.**")
    #                     st.info("Recommendation: Continue regular monitoring.")
                    
    #                 # Visualizations
    #                 st.markdown("---")
    #                 st.subheader("📈 Prediction Visualizations")
                    
    #                 # Create prediction visualization
    #                 fig = create_prediction_visualization(
    #                     st.session_state.model_csv_predictions,
    #                     probabilities,
    #                     summary
    #                 )
    #                 st.pyplot(fig)
    #                 plt.close()
                    
    #                 # Download predictions
    #                 st.markdown("---")
    #                 st.subheader("📥 Download Results")
                    
    #                 # Save predictions to CSV
    #                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #                 predictions_path = FEATURES_DIR / f"csv_predictions_{timestamp}.csv"
                    
    #                 # Create results dataframe
    #                 df_features = pd.read_csv(st.session_state.model_csv_path)
    #                 df_features['prediction'] = st.session_state.model_csv_predictions
    #                 df_features['confidence'] = np.max(probabilities, axis=1)
    #                 df_features['predicted_label'] = ['Normal' if p == 0 else 'Abnormal' for p in st.session_state.model_csv_predictions]
                    
    #                 df_features.to_csv(predictions_path, index=False)
                    
    #                 col1, col2 = st.columns([3, 1])
                    
    #                 with col1:
    #                     st.info(f"**Predictions saved:** {predictions_path.name}")
    #                     st.caption(f"📏 Size: {predictions_path.stat().st_size / 1024:.1f} KB")
                    
    #                 with col2:
    #                     # Create download button
    #                     csv_predictions_data_key = f"data_csv_predictions_{timestamp}"
    #                     csv_predictions_widget_key = f"widget_csv_predictions_{timestamp}"
                        
    #                     if csv_predictions_data_key not in st.session_state:
    #                         try:
    #                             with open(predictions_path, 'rb') as f:
    #                                 st.session_state[csv_predictions_data_key] = f.read()
    #                         except Exception as e:
    #                             logger.error(f"Failed to read predictions for download: {e}")
    #                             st.session_state[csv_predictions_data_key] = None
                        
    #                     if st.session_state[csv_predictions_data_key]:
    #                         st.download_button(
    #                             "📥 Download Predictions",
    #                             data=st.session_state[csv_predictions_data_key],
    #                             file_name=predictions_path.name,
    #                             mime="text/csv",
    #                             key=csv_predictions_widget_key,
    #                             use_container_width=True
    #                         )
    #             else:
    #                 st.info("👆 Click 'Run Prediction' to analyze the uploaded CSV data.")

    # with tab6:
    #     st.subheader("🤖 Model Prediction")
        
    #     # Check if baseline model is available with proper error handling
    #     try:
    #         baseline_model = ModelManager.load_baseline_model()
    #         model_loaded = baseline_model is not None
    #     except Exception as e:
    #         logger.error(f"Error loading baseline model: {e}")
    #         baseline_model = None
    #         model_loaded = False
        
    #     if not model_loaded:
    #         st.error("❌ Baseline model not found. Please ensure the model file exists at `models/baseline/xgboost_model.bin`")
    #         st.info("To train a model, run the training script provided in the documentation.")
            
    #         # Provide model troubleshooting information
    #         with st.expander("🔧 Model Troubleshooting"):
    #             st.markdown("""
    #             ### Possible Issues:
    #             1. Model file doesn't exist at the expected location
    #             2. Model file is corrupted
    #             3. Incompatible model version
                
    #             ### Solutions:
    #             1. Check if the model file exists: `models/baseline/xgboost_model.bin`
    #             2. Re-train the model using the provided training script
    #             3. Verify model compatibility with current application version
    #             """)
    #         return
        
    #     st.success("✅ Baseline model loaded successfully")
        
    #     # Add model information with error handling
    #     with st.expander("🔍 Model Information"):
    #         try:
    #             if hasattr(baseline_model, 'get_params'):
    #                 st.json(baseline_model.get_params())
                
    #             # Add model performance metrics if available
    #             model_metrics_path = Path("models/baseline/model_metrics.json")
    #             if model_metrics_path.exists():
    #                 with open(model_metrics_path, 'r') as f:
    #                     metrics = json.load(f)
    #                 st.write("**Model Performance Metrics:**")
    #                 st.json(metrics)
    #             else:
    #                 st.info("Model performance metrics not available")
    #         except Exception as e:
    #             st.error(f"Error retrieving model information: {e}")
    #             logger.error(f"Error retrieving model information: {e}")
        
    #     # Create three sub-tabs
    #     subtab1, subtab2, subtab3 = st.tabs(["From Feature Engineering", "From Feature CSV", "From Raw Pose Data"])
        
    #     # Sub-tab 1: Use features from Feature Engineering
    #     with subtab1:
    #         if st.session_state.features_df is None or st.session_state.features_df.empty:
    #             st.warning("⚠️ No features available. Please extract features in the Feature Engineering tab first.")
    #             if st.button("Go to Feature Engineering Tab"):
    #                 st.info("Please navigate to the Feature Engineering tab to extract features first.")
    #         else:
    #             st.markdown("---")
    #             st.subheader("🔮 Run Model Prediction")
                
    #             col1, col2 = st.columns([2, 1])
                
    #             with col1:
    #                 st.info(f"**Features Available:** {len(st.session_state.features_df)} feature vectors")
    #                 st.info(f"**Model:** Baseline XGBoost Binary Classifier")
                
    #             with col2:
    #                 if st.button("🚀 Run Prediction", use_container_width=True, type="primary"):
    #                     with st.spinner("Preparing features and running prediction..."):
    #                         try:
    #                             # Prepare features for prediction
    #                             X = ModelManager.prepare_features_for_prediction(st.session_state.features_df)
                                
    #                             # Make predictions
    #                             predictions, probabilities = ModelManager.predict_with_baseline(baseline_model, X)
                                
    #                             if len(predictions) > 0:
    #                                 # Create prediction summary
    #                                 summary = ModelManager.create_prediction_summary(predictions, probabilities)
                                    
    #                                 # Store in session state
    #                                 st.session_state.model_predictions = predictions
    #                                 st.session_state.model_summary = summary
    #                                 st.session_state.model_probabilities = probabilities
                                    
    #                                 st.success("✅ Prediction completed successfully!")
    #                             else:
    #                                 st.error("❌ Prediction failed. No valid predictions generated.")
    #                         except Exception as e:
    #                             st.error(f"❌ Error during prediction: {str(e)}")
    #                             logger.error(f"Prediction error: {e}")
                
    #             # Display results if available
    #             if st.session_state.model_predictions is not None:
    #                 display_prediction_results(
    #                     st.session_state.model_predictions,
    #                     st.session_state.model_probabilities,
    #                     st.session_state.model_summary,
    #                     st.session_state.features_df,
    #                     baseline_model  # Pass the model to avoid global variable reference
    #                 )
        
    #     # Sub-tab 2: Use features from uploaded CSV
    #     with subtab2:
    #         st.markdown("---")
    #         st.subheader("📁 Upload Feature CSV for Model Prediction")
            
    #         # Add instructions
    #         with st.expander("📖 CSV Format Instructions"):
    #             st.markdown("""
    #             The uploaded CSV should contain extracted gait features with the following columns:
    #             - step_height_symmetry
    #             - step_length_symmetry
    #             - cadence_asym
    #             - knee_rom_asym
    #             - hip_rom_asym
    #             - [Other gait features...]
                
    #             **Note**: If you have raw pose data (landmarks), please use the "From Raw Pose Data" tab.
    #             """)
            
    #         # Add template download
    #         template_path = Path("templates/feature_template.csv")
    #         if template_path.exists():
    #             with open(template_path, "rb") as f:
    #                 st.download_button(
    #                     "📥 Download Feature Template",
    #                     data=f.read(),
    #                     file_name="feature_template.csv",
    #                     mime="text/csv"
    #                 )
    #         else:
    #             st.warning("Feature template not available")
            
    #         col1, col2 = st.columns([2, 1])
            
    #         with col1:
    #             uploaded_csv = st.file_uploader(
    #                 "Upload a CSV file with extracted features",
    #                 type=['csv'],
    #                 help="Upload a CSV file containing the extracted features for prediction",
    #                 key="model_csv_upload"
    #             )
                
    #             if uploaded_csv:
    #                 st.info(f"📄 {uploaded_csv.name}")
    #                 st.info(f"📏 {uploaded_csv.size / (1024*1024):.2f} MB")
                    
    #                 # Save uploaded CSV
    #                 csv_path = FEATURES_DIR / f"model_{uploaded_csv.name}"
    #                 with open(csv_path, 'wb') as f:
    #                     f.write(uploaded_csv.getvalue())
                    
    #                 st.session_state.model_csv_path = csv_path
                    
    #                 # Preview CSV
    #                 with st.expander("👁️ Preview CSV Data"):
    #                     try:
    #                         df = pd.read_csv(csv_path)
    #                         st.dataframe(df.head(10), use_container_width=True)
    #                         st.caption(f"Showing first 10 rows of {len(df)} total rows")
    #                     except Exception as e:
    #                         st.error(f"Could not preview CSV: {e}")
            
    #         with col2:
    #             if 'model_csv_path' in st.session_state and st.session_state.model_csv_path:
    #                 st.metric("Status", "CSV uploaded")
    #                 st.metric("File", st.session_state.model_csv_path.name)
            
    #         # Run prediction on uploaded CSV
    #         if 'model_csv_path' in st.session_state and st.session_state.model_csv_path:
    #             st.markdown("---")
    #             st.subheader("🔮 Run Model Prediction")
                
    #             col1, col2 = st.columns([2, 1])
                
    #             with col1:
    #                 st.info(f"**CSV File:** {st.session_state.model_csv_path.name}")
    #                 st.caption(f"📏 Size: {st.session_state.model_csv_path.stat().st_size / 1024:.1f} KB")
                
    #             with col2:
    #                 if st.button("🚀 Run Prediction", use_container_width=True, type="primary", key="predict_model_csv"):
    #                     with st.spinner("Preparing features and running prediction..."):
    #                         try:
    #                             # Load features from CSV
    #                             df_features = pd.read_csv(st.session_state.model_csv_path)
                                
    #                             # Prepare features for prediction
    #                             X = ModelManager.prepare_features_for_prediction(df_features)
                                
    #                             # Make predictions
    #                             predictions, probabilities = ModelManager.predict_with_baseline(baseline_model, X)
                                
    #                             if len(predictions) > 0:
    #                                 # Create prediction summary
    #                                 summary = ModelManager.create_prediction_summary(predictions, probabilities)
                                    
    #                                 # Store in session state
    #                                 st.session_state.model_csv_predictions = predictions
    #                                 st.session_state.model_csv_summary = summary
    #                                 st.session_state.model_csv_probabilities = probabilities
                                    
    #                                 st.success("✅ Prediction completed successfully!")
    #                             else:
    #                                 st.error("❌ Prediction failed. No valid predictions generated.")
    #                         except Exception as e:
    #                             st.error(f"❌ Error during prediction: {str(e)}")
    #                             logger.error(f"Prediction error: {e}")
                
    #             # Display results if available
    #             if 'model_csv_predictions' in st.session_state and st.session_state.model_csv_predictions is not None:
    #                 df_features = pd.read_csv(st.session_state.model_csv_path)
    #                 display_prediction_results(
    #                     st.session_state.model_csv_predictions,
    #                     st.session_state.model_csv_probabilities,
    #                     st.session_state.model_csv_summary,
    #                     df_features,
    #                     baseline_model  # Pass the model to avoid global variable reference
    #                 )
    #         else:
    #             st.info("👆 Upload a feature CSV file to begin analysis.")
        
    #     # Sub-tab 3: Process raw pose data
    #     with subtab3:
    #         st.markdown("---")
    #         st.subheader("📁 Upload Raw Pose Data for Analysis")
            
    #         # Add instructions
    #         with st.expander("📖 Raw Pose Data Format"):
    #             st.markdown("""
    #             The uploaded CSV should contain MediaPipe pose landmarks with the following columns:
    #             - frame: Frame number
    #             - landmark_id: Landmark index (0-32)
    #             - x_norm: Normalized X coordinate
    #             - y_norm: Normalized Y coordinate
    #             - z_norm: Normalized Z coordinate
    #             - video_id: (Optional) Video identifier
                
    #             **Note**: This option processes raw pose data and extracts features automatically.
    #             """)
            
    #         uploaded_pose_csv = st.file_uploader(
    #             "Upload a CSV file with MediaPipe pose landmarks",
    #             type=['csv'],
    #             help="Upload a CSV file containing pose landmarks (x_norm, y_norm, z_norm) for each frame",
    #             key="pose_csv_upload"
    #         )
            
    #         if uploaded_pose_csv:
    #             # Check file size to prevent memory issues
    #             if uploaded_pose_csv.size > 50 * 1024 * 1024:  # 50MB limit
    #                 st.error("File too large. Please upload a smaller file or split it into multiple files.")
    #                 return
                
    #             # Save uploaded CSV
    #             pose_csv_path = FEATURES_DIR / f"pose_{uploaded_pose_csv.name}"
    #             with open(pose_csv_path, 'wb') as f:
    #                 f.write(uploaded_pose_csv.getvalue())
                
    #             # Preview CSV
    #             with st.expander("👁️ Preview Pose Data"):
    #                 try:
    #                     df = pd.read_csv(pose_csv_path)
    #                     st.dataframe(df.head(10), use_container_width=True)
    #                     st.caption(f"Showing first 10 rows of {len(df)} total rows")
    #                 except Exception as e:
    #                     st.error(f"Could not preview CSV: {e}")
                
    #             # Run analysis
    #             if st.button("🚀 Analyze Pose Data", use_container_width=True, type="primary"):
    #                 with st.spinner("Processing pose data and running prediction..."):
    #                     progress_bar = st.progress(0)
    #                     status_placeholder = st.empty()
                        
    #                     try:
    #                         # Update progress
    #                         progress_bar.progress(25)
    #                         status_placeholder.text("Extracting features from pose data...")
                            
    #                         # Check if predict_from_pose_csv method exists
    #                         if not hasattr(GaitAnalysisEngine, 'predict_from_pose_csv'):
    #                             st.error("The method predict_from_pose_csv is not available in GaitAnalysisEngine.")
    #                             logger.error("predict_from_pose_csv method not found")
    #                             return
                            
    #                         # Process pose data and make predictions
    #                         df_results, probabilities = GaitAnalysisEngine.predict_from_pose_csv(
    #                             pose_csv_path, 
    #                             model=baseline_model
    #                         )
                            
    #                         # Update progress
    #                         progress_bar.progress(75)
    #                         status_placeholder.text("Analyzing results...")
                            
    #                         if df_results is not None and len(df_results) > 0:
    #                             # Store in session state
    #                             st.session_state.pose_predictions = df_results['prediction'].values
    #                             st.session_state.pose_probabilities = probabilities
                                
    #                             # Create summary
    #                             predictions = df_results['prediction'].values
    #                             normal_count = np.sum(predictions == 0)
    #                             abnormal_count = np.sum(predictions == 1)
    #                             summary = {
    #                                 'total_windows': len(predictions),
    #                                 'normal_windows': int(normal_count),
    #                                 'abnormal_windows': int(abnormal_count),
    #                                 'abnormal_percentage': 100 * abnormal_count / len(predictions)
    #                             }
    #                             st.session_state.pose_summary = summary
    #                             st.session_state.pose_results_df = df_results
                                
    #                             # Complete progress
    #                             progress_bar.progress(100)
    #                             status_placeholder.empty()
                                
    #                             st.success("✅ Analysis completed successfully!")
    #                         else:
    #                             progress_bar.empty()
    #                             status_placeholder.empty()
    #                             st.error("❌ Analysis failed. No valid results generated.")
    #                     except Exception as e:
    #                         progress_bar.empty()
    #                         status_placeholder.empty()
    #                         st.error(f"❌ Error during analysis: {str(e)}")
    #                         logger.error(f"Pose analysis error: {e}")
                
    #             # Display results if available
    #             if 'pose_predictions' in st.session_state and st.session_state.pose_predictions is not None:
    #                 display_prediction_results(
    #                     st.session_state.pose_predictions,
    #                     st.session_state.pose_probabilities,
    #                     st.session_state.pose_summary,
    #                     st.session_state.pose_results_df,
    #                     baseline_model  # Pass the model to avoid global variable reference
    #                 )
    #         else:
    #             st.info("👆 Upload a pose CSV file to begin analysis.")

    with tab6:
        st.subheader("🤖 Model Prediction")
        
        # Initialize debug information in session state if not exists
        if 'debug_info' not in st.session_state:
            st.session_state.debug_info = {}
        
        # Debug section
        with st.expander("🐛 Debug Information"):
            st.write("Current debug information:")
            st.json(st.session_state.debug_info)
        
        try:
            # Check if baseline model is available with proper error handling
            st.session_state.debug_info['model_loading'] = "Starting model loading..."
            
            try:
                baseline_model = ModelManager.load_baseline_model()
                model_loaded = baseline_model is not None
                st.session_state.debug_info['model_loading'] = "Model loading completed"
                st.session_state.debug_info['model_loaded'] = model_loaded
            except Exception as e:
                st.session_state.debug_info['model_loading'] = f"Error: {str(e)}"
                logger.error(f"Error loading baseline model: {e}")
                baseline_model = None
                model_loaded = False
            
            if not model_loaded:
                st.error("❌ Baseline model not found. Please ensure the model file exists at `models/baseline/xgboost_model.bin`")
                st.info("To train a model, run the training script provided in the documentation.")
                
                # Provide model troubleshooting information
                with st.expander("🔧 Model Troubleshooting"):
                    st.markdown("""
                    ### Possible Issues:
                    1. Model file doesn't exist at the expected location
                    2. Model file is corrupted
                    3. Incompatible model version
                    
                    ### Solutions:
                    1. Check if the model file exists: `models/baseline/xgboost_model.bin`
                    2. Re-train the model using the provided training script
                    3. Verify model compatibility with current application version
                    """)
                return
            
            st.success("✅ Baseline model loaded successfully")
            
            # Add model information with error handling
            with st.expander("🔍 Model Information"):
                try:
                    if hasattr(baseline_model, 'get_params'):
                        st.json(baseline_model.get_params())
                    
                    # Add model performance metrics if available
                    model_metrics_path = Path("models/baseline/model_metrics.json")
                    if model_metrics_path.exists():
                        with open(model_metrics_path, 'r') as f:
                            metrics = json.load(f)
                        st.write("**Model Performance Metrics:**")
                        st.json(metrics)
                    else:
                        st.info("Model performance metrics not available")
                except Exception as e:
                    st.error(f"Error retrieving model information: {e}")
                    logger.error(f"Error retrieving model information: {e}")
            
            # Create three sub-tabs
            subtab1, subtab2, subtab3 = st.tabs(["From Feature Engineering", "From Feature CSV", "From Raw Pose Data"])
            
            # Sub-tab 1: Use features from Feature Engineering
            with subtab1:
                st.session_state.debug_info['subtab1'] = "Entered subtab1"
                
                if st.session_state.features_df is None or st.session_state.features_df.empty:
                    st.warning("⚠️ No features available. Please extract features in the Feature Engineering tab first.")
                    if st.button("Go to Feature Engineering Tab"):
                        st.info("Please navigate to the Feature Engineering tab to extract features first.")
                else:
                    st.markdown("---")
                    st.subheader("🔮 Run Model Prediction")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.info(f"**Features Available:** {len(st.session_state.features_df)} feature vectors")
                        st.info(f"**Model:** Baseline XGBoost Binary Classifier")
                    
                    with col2:
                        if st.button("🚀 Run Prediction", use_container_width=True, type="primary"):
                            st.session_state.debug_info['subtab1_prediction'] = "Starting prediction..."
                            
                            with st.spinner("Preparing features and running prediction..."):
                                try:
                                    # Prepare features for prediction
                                    st.session_state.debug_info['subtab1_prediction'] = "Preparing features..."
                                    X = ModelManager.prepare_features_for_prediction(st.session_state.features_df)
                                    
                                    # Make predictions
                                    st.session_state.debug_info['subtab1_prediction'] = "Making predictions..."
                                    predictions, probabilities = ModelManager.predict_with_baseline(baseline_model, X)
                                    
                                    if len(predictions) > 0:
                                        # Create prediction summary
                                        st.session_state.debug_info['subtab1_prediction'] = "Creating summary..."
                                        summary = ModelManager.create_prediction_summary(predictions, probabilities)
                                        
                                        # Store in session state
                                        st.session_state.model_predictions = predictions
                                        st.session_state.model_summary = summary
                                        st.session_state.model_probabilities = probabilities
                                        
                                        st.session_state.debug_info['subtab1_prediction'] = "Prediction completed successfully"
                                        st.success("✅ Prediction completed successfully!")
                                    else:
                                        st.error("❌ Prediction failed. No valid predictions generated.")
                                        st.session_state.debug_info['subtab1_prediction'] = "No valid predictions generated"
                                except Exception as e:
                                    st.error(f"❌ Error during prediction: {str(e)}")
                                    logger.error(f"Prediction error: {e}")
                                    st.session_state.debug_info['subtab1_prediction'] = f"Error: {str(e)}"
                    
                    # Display results if available
                    if st.session_state.model_predictions is not None:
                        st.session_state.debug_info['subtab1_display'] = "Starting display..."
                        
                        try:
                            display_prediction_results(
                                st.session_state.model_predictions,
                                st.session_state.model_probabilities,
                                st.session_state.model_summary,
                                st.session_state.features_df,
                                baseline_model  # Pass the model to avoid global variable reference
                            )
                            st.session_state.debug_info['subtab1_display'] = "Display completed"
                        except Exception as e:
                            st.error(f"❌ Error displaying results: {str(e)}")
                            logger.error(f"Error displaying results: {e}")
                            st.session_state.debug_info['subtab1_display'] = f"Error: {str(e)}"
            
            # Sub-tab 2: Use features from uploaded CSV
            with subtab2:
                st.session_state.debug_info['subtab2'] = "Entered subtab2"
                
                st.markdown("---")
                st.subheader("📁 Upload Feature CSV for Model Prediction")
                
                # Add instructions
                with st.expander("📖 CSV Format Instructions"):
                    st.markdown("""
                    The uploaded CSV should contain extracted gait features with the following columns:
                    - step_height_symmetry
                    - step_length_symmetry
                    - cadence_asym
                    - knee_rom_asym
                    - hip_rom_asym
                    - [Other gait features...]
                    
                    **Note**: If you have raw pose data (landmarks), please use the "From Raw Pose Data" tab.
                    """)
                
                # Add template download
                template_path = Path("templates/feature_template.csv")
                if template_path.exists():
                    with open(template_path, "rb") as f:
                        st.download_button(
                            "📥 Download Feature Template",
                            data=f.read(),
                            file_name="feature_template.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Feature template not available")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    uploaded_csv = st.file_uploader(
                        "Upload a CSV file with extracted features",
                        type=['csv'],
                        help="Upload a CSV file containing the extracted features for prediction",
                        key="model_csv_upload"
                    )
                    
                    if uploaded_csv:
                        st.session_state.debug_info['subtab2_upload'] = "File uploaded"
                        st.info(f"📄 {uploaded_csv.name}")
                        st.info(f"📏 {uploaded_csv.size / (1024*1024):.2f} MB")
                        
                        # Save uploaded CSV
                        csv_path = FEATURES_DIR / f"model_{uploaded_csv.name}"
                        with open(csv_path, 'wb') as f:
                            f.write(uploaded_csv.getvalue())
                        
                        st.session_state.model_csv_path = csv_path
                        
                        # Preview CSV
                        with st.expander("👁️ Preview CSV Data"):
                            try:
                                df = pd.read_csv(csv_path)
                                st.dataframe(df.head(10), use_container_width=True)
                                st.caption(f"Showing first 10 rows of {len(df)} total rows")
                            except Exception as e:
                                st.error(f"Could not preview CSV: {e}")
                                st.session_state.debug_info['subtab2_preview'] = f"Error: {str(e)}"
                
                with col2:
                    if 'model_csv_path' in st.session_state and st.session_state.model_csv_path:
                        st.metric("Status", "CSV uploaded")
                        st.metric("File", st.session_state.model_csv_path.name)
                
                # Run prediction on uploaded CSV
                if 'model_csv_path' in st.session_state and st.session_state.model_csv_path:
                    st.markdown("---")
                    st.subheader("🔮 Run Model Prediction")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.info(f"**CSV File:** {st.session_state.model_csv_path.name}")
                        st.caption(f"📏 Size: {st.session_state.model_csv_path.stat().st_size / 1024:.1f} KB")
                    
                    with col2:
                        if st.button("🚀 Run Prediction", use_container_width=True, type="primary", key="predict_model_csv"):
                            st.session_state.debug_info['subtab2_prediction'] = "Starting prediction..."
                            
                            with st.spinner("Preparing features and running prediction..."):
                                try:
                                    # Load features from CSV
                                    st.session_state.debug_info['subtab2_prediction'] = "Loading CSV..."
                                    df_features = pd.read_csv(st.session_state.model_csv_path)
                                    
                                    # Prepare features for prediction
                                    st.session_state.debug_info['subtab2_prediction'] = "Preparing features..."
                                    X = ModelManager.prepare_features_for_prediction(df_features)
                                    
                                    # Make predictions
                                    st.session_state.debug_info['subtab2_prediction'] = "Making predictions..."
                                    predictions, probabilities = ModelManager.predict_with_baseline(baseline_model, X)
                                    
                                    if len(predictions) > 0:
                                        # Create prediction summary
                                        st.session_state.debug_info['subtab2_prediction'] = "Creating summary..."
                                        summary = ModelManager.create_prediction_summary(predictions, probabilities)
                                        
                                        # Store in session state
                                        st.session_state.model_csv_predictions = predictions
                                        st.session_state.model_csv_summary = summary
                                        st.session_state.model_csv_probabilities = probabilities
                                        
                                        st.session_state.debug_info['subtab2_prediction'] = "Prediction completed successfully"
                                        st.success("✅ Prediction completed successfully!")
                                    else:
                                        st.error("❌ Prediction failed. No valid predictions generated.")
                                        st.session_state.debug_info['subtab2_prediction'] = "No valid predictions generated"
                                except Exception as e:
                                    st.error(f"❌ Error during prediction: {str(e)}")
                                    logger.error(f"Prediction error: {e}")
                                    st.session_state.debug_info['subtab2_prediction'] = f"Error: {str(e)}"
                    
                    # Display results if available
                    if 'model_csv_predictions' in st.session_state and st.session_state.model_csv_predictions is not None:
                        st.session_state.debug_info['subtab2_display'] = "Starting display..."
                        
                        try:
                            df_features = pd.read_csv(st.session_state.model_csv_path)
                            display_prediction_results(
                                st.session_state.model_csv_predictions,
                                st.session_state.model_csv_probabilities,
                                st.session_state.model_csv_summary,
                                df_features,
                                baseline_model  # Pass the model to avoid global variable reference
                            )
                            st.session_state.debug_info['subtab2_display'] = "Display completed"
                        except Exception as e:
                            st.error(f"❌ Error displaying results: {str(e)}")
                            logger.error(f"Error displaying results: {e}")
                            st.session_state.debug_info['subtab2_display'] = f"Error: {str(e)}"
                else:
                    st.info("👆 Upload a feature CSV file to begin analysis.")
            
            # Sub-tab 3: Process raw pose data
            with subtab3:
                st.session_state.debug_info['subtab3'] = "Entered subtab3"
                
                st.markdown("---")
                st.subheader("📁 Upload Raw Pose Data for Analysis")
                
                # Add instructions
                with st.expander("📖 Raw Pose Data Format"):
                    st.markdown("""
                    The uploaded CSV should contain MediaPipe pose landmarks with the following columns:
                    - frame: Frame number
                    - landmark_id: Landmark index (0-32)
                    - x_norm: Normalized X coordinate
                    - y_norm: Normalized Y coordinate
                    - z_norm: Normalized Z coordinate
                    - video_id: (Optional) Video identifier
                    
                    **Note**: This option processes raw pose data and extracts features automatically.
                    """)
                
                uploaded_pose_csv = st.file_uploader(
                    "Upload a CSV file with MediaPipe pose landmarks",
                    type=['csv'],
                    help="Upload a CSV file containing pose landmarks (x_norm, y_norm, z_norm) for each frame",
                    key="pose_csv_upload"
                )
                
                if uploaded_pose_csv:
                    st.session_state.debug_info['subtab3_upload'] = "File uploaded"
                    
                    # Check file size to prevent memory issues
                    if uploaded_pose_csv.size > 50 * 1024 * 1024:  # 50MB limit
                        st.error("File too large. Please upload a smaller file or split it into multiple files.")
                        st.session_state.debug_info['subtab3_upload'] = "File too large"
                        return
                    
                    # Save uploaded CSV
                    pose_csv_path = FEATURES_DIR / f"pose_{uploaded_pose_csv.name}"
                    with open(pose_csv_path, 'wb') as f:
                        f.write(uploaded_pose_csv.getvalue())
                    
                    # Preview CSV
                    with st.expander("👁️ Preview Pose Data"):
                        try:
                            df = pd.read_csv(pose_csv_path)
                            st.dataframe(df.head(10), use_container_width=True)
                            st.caption(f"Showing first 10 rows of {len(df)} total rows")
                        except Exception as e:
                            st.error(f"Could not preview CSV: {e}")
                            st.session_state.debug_info['subtab3_preview'] = f"Error: {str(e)}"
                    
                    # Run analysis
                    if st.button("🚀 Analyze Pose Data", use_container_width=True, type="primary"):
                        st.session_state.debug_info['subtab3_analysis'] = "Starting analysis..."
                        
                        with st.spinner("Processing pose data and running prediction..."):
                            progress_bar = st.progress(0)
                            status_placeholder = st.empty()
                            
                            try:
                                # Update progress
                                progress_bar.progress(25)
                                status_placeholder.text("Extracting features from pose data...")
                                st.session_state.debug_info['subtab3_analysis'] = "Extracting features..."
                                
                                # Check if predict_from_pose_csv method exists
                                if not hasattr(GaitAnalysisEngine, 'predict_from_pose_csv'):
                                    st.error("The method predict_from_pose_csv is not available in GaitAnalysisEngine.")
                                    logger.error("predict_from_pose_csv method not found")
                                    st.session_state.debug_info['subtab3_analysis'] = "Method not found"
                                    return
                                
                                # Process pose data and make predictions
                                progress_bar.progress(50)
                                status_placeholder.text("Running prediction...")
                                st.session_state.debug_info['subtab3_analysis'] = "Running prediction..."
                                
                                df_results, probabilities = GaitAnalysisEngine.predict_from_pose_csv(
                                    pose_csv_path, 
                                    model=baseline_model
                                )
                                
                                # Update progress
                                progress_bar.progress(75)
                                status_placeholder.text("Analyzing results...")
                                st.session_state.debug_info['subtab3_analysis'] = "Analyzing results..."
                                
                                if df_results is not None and len(df_results) > 0:
                                    # Store in session state
                                    st.session_state.pose_predictions = df_results['prediction'].values
                                    st.session_state.pose_probabilities = probabilities
                                    
                                    # Create summary
                                    predictions = df_results['prediction'].values
                                    normal_count = np.sum(predictions == 0)
                                    abnormal_count = np.sum(predictions == 1)
                                    summary = {
                                        'total_windows': len(predictions),
                                        'normal_windows': int(normal_count),
                                        'abnormal_windows': int(abnormal_count),
                                        'abnormal_percentage': 100 * abnormal_count / len(predictions)
                                    }
                                    st.session_state.pose_summary = summary
                                    st.session_state.pose_results_df = df_results
                                    
                                    # Complete progress
                                    progress_bar.progress(100)
                                    status_placeholder.empty()
                                    
                                    st.session_state.debug_info['subtab3_analysis'] = "Analysis completed successfully"
                                    st.success("✅ Analysis completed successfully!")
                                else:
                                    progress_bar.empty()
                                    status_placeholder.empty()
                                    st.error("❌ Analysis failed. No valid results generated.")
                                    st.session_state.debug_info['subtab3_analysis'] = "No valid results generated"
                            except Exception as e:
                                progress_bar.empty()
                                status_placeholder.empty()
                                st.error(f"❌ Error during analysis: {str(e)}")
                                logger.error(f"Pose analysis error: {e}")
                                st.session_state.debug_info['subtab3_analysis'] = f"Error: {str(e)}"
                    
                    # Display results if available
                    if 'pose_predictions' in st.session_state and st.session_state.pose_predictions is not None:
                        st.session_state.debug_info['subtab3_display'] = "Starting display..."
                        
                        try:
                            display_prediction_results(
                                st.session_state.pose_predictions,
                                st.session_state.pose_probabilities,
                                st.session_state.pose_summary,
                                st.session_state.pose_results_df,
                                baseline_model  # Pass the model to avoid global variable reference
                            )
                            st.session_state.debug_info['subtab3_display'] = "Display completed"
                        except Exception as e:
                            st.error(f"❌ Error displaying results: {str(e)}")
                            logger.error(f"Error displaying results: {e}")
                            st.session_state.debug_info['subtab3_display'] = f"Error: {str(e)}"
                else:
                    st.info("👆 Upload a pose CSV file to begin analysis.")
        
        except Exception as e:
            st.error(f"❌ Unexpected error in Model Prediction tab: {str(e)}")
            logger.error(f"Unexpected error in Model Prediction tab: {e}")
            st.session_state.debug_info['unexpected_error'] = str(e)
            st.code(traceback.format_exc())

def display_prediction_results(predictions, probabilities, summary, features_df, model=None):
    """Display prediction results with enhanced visualizations"""
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Windows", summary['total_windows'])
    with col2:
        st.metric("Normal", summary['normal_windows'])
    with col3:
        st.metric("Abnormal", summary['abnormal_windows'])
    with col4:
        st.metric("Abnormal %", f"{summary['abnormal_percentage']:.1f}%")
    
    # Recommendation
    st.markdown("---")
    st.subheader("💡 Recommendation")
    
    if summary['abnormal_percentage'] > 20:
        st.error("⚠️ **High percentage of abnormal gait patterns detected.**")
        st.info("Recommendation: Seek medical evaluation for further assessment.")
    elif summary['abnormal_percentage'] > 10:
        st.warning("⚠️ **Some abnormal gait patterns detected.**")
        st.info("Recommendation: Monitor closely and consider consultation if symptoms persist.")
    else:
        st.success("✅ **Normal gait pattern detected.**")
        st.info("Recommendation: Continue regular monitoring.")
    
    # Visualizations
    st.markdown("---")
    st.subheader("📈 Prediction Visualizations")
    
    # Create enhanced prediction visualization
    fig = create_enhanced_prediction_visualization(
        predictions,
        probabilities,
        summary,
        features_df,
        model  # Pass the model to avoid global variable reference
    )
    st.pyplot(fig)
    plt.close()
    
    # Detailed results table
    st.markdown("---")
    st.subheader("📋 Detailed Results")
    
    # Create results dataframe
    results_df = features_df.copy()
    results_df['prediction'] = predictions
    results_df['confidence'] = np.max(probabilities, axis=1)
    results_df['predicted_label'] = ['Normal' if p == 0 else 'Abnormal' for p in predictions]
    
    # Show sample of results
    with st.expander("👁️ View Detailed Predictions"):
        # Select relevant columns to display
        display_cols = ['predicted_label', 'confidence']
        
        # Add key gait metrics if they exist
        key_metrics = ['step_height_symmetry', 'step_length_symmetry', 'cadence_asym', 'knee_rom_asym', 'hip_rom_asym']
        for metric in key_metrics:
            if metric in results_df.columns:
                display_cols.append(metric)
        
        st.dataframe(
            results_df[display_cols].head(20),
            use_container_width=True
        )
    
    # Download predictions
    st.markdown("---")
    st.subheader("📥 Download Results")
    
    # Save predictions to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_path = FEATURES_DIR / f"predictions_{timestamp}.csv"
    
    results_df.to_csv(predictions_path, index=False)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"**Predictions saved:** {predictions_path.name}")
        st.caption(f"📏 Size: {predictions_path.stat().st_size / 1024:.1f} KB")
    
    with col2:
        # Create download button
        predictions_data_key = f"data_predictions_{timestamp}"
        predictions_widget_key = f"widget_predictions_{timestamp}"
        
        if predictions_data_key not in st.session_state:
            try:
                with open(predictions_path, 'rb') as f:
                    st.session_state[predictions_data_key] = f.read()
            except Exception as e:
                logger.error(f"Failed to read predictions for download: {e}")
                st.session_state[predictions_data_key] = None
        
        if st.session_state[predictions_data_key]:
            st.download_button(
                "📥 Download Predictions",
                data=st.session_state[predictions_data_key],
                file_name=predictions_path.name,
                mime="text/csv",
                key=predictions_widget_key,
                use_container_width=True
            )


def create_enhanced_prediction_visualization(predictions, probabilities, summary, features_df, model=None):
    """Create enhanced prediction visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Prediction distribution
    ax = axes[0, 0]
    labels = ['Normal', 'Abnormal']
    counts = [summary['normal_windows'], summary['abnormal_windows']]
    colors = ['#4CAF50', '#F44336']
    ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Prediction Distribution')
    
    # 2. Confidence distribution
    ax = axes[0, 1]
    confidences = np.max(probabilities, axis=1)
    ax.hist(confidences, bins=20, alpha=0.7, color='skyblue')
    ax.set_title('Prediction Confidence Distribution')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    
    # 3. Feature importance (if available)
    ax = axes[1, 0]
    if model and hasattr(model, 'feature_importances_'):
        # Get top 10 important features
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        feature_names = features_df.columns[indices]
        
        ax.barh(range(len(indices)), importances[indices], color='lightgreen')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(feature_names)
        ax.set_title('Top 10 Important Features')
        ax.set_xlabel('Importance')
    else:
        ax.text(0.5, 0.5, 'Feature importance not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Feature Importance')
    
    # 4. Key gait metrics comparison
    ax = axes[1, 1]
    key_metrics = ['step_height_symmetry', 'step_length_symmetry', 'cadence_asym', 'knee_rom_asym']
    available_metrics = [m for m in key_metrics if m in features_df.columns]
    
    if available_metrics:
        normal_data = features_df[features_df['prediction'] == 0][available_metrics]
        abnormal_data = features_df[features_df['prediction'] == 1][available_metrics]
        
        x = np.arange(len(available_metrics))
        width = 0.35
        
        ax.bar(x - width/2, normal_data.mean(), width, label='Normal', color='green', alpha=0.7)
        ax.bar(x + width/2, abnormal_data.mean(), width, label='Abnormal', color='red', alpha=0.7)
        
        ax.set_xlabel('Gait Metrics')
        ax.set_ylabel('Mean Value')
        ax.set_title('Key Gait Metrics by Prediction')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics], rotation=45, ha='right')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Key gait metrics not available in data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Key Gait Metrics')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    main()