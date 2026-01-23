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

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.absolute()
CONFIG_PATH = PROJECT_ROOT / "config.json"
MEDIAPIPE_SCRIPT = PROJECT_ROOT / "pre-processing-models" / "mediapipe" / "pre_mediapipe.py"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, FEATURES_DIR]:
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
        return True
    
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
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

class FeatureEngineering:
    """Feature extraction from MediaPipe landmarks"""
    
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
        pelvis = (pose[:, FeatureEngineering.LEFT_HIP] + pose[:, FeatureEngineering.RIGHT_HIP]) / 2
        pose_centered = pose - pelvis[:, None, :]

        torso = (pose_centered[:, FeatureEngineering.LEFT_SHOULDER] + pose_centered[:, FeatureEngineering.RIGHT_SHOULDER]) / 2
        scale = np.linalg.norm(torso, axis=1).mean()

        pose_scaled = pose_centered / scale
        return pose_scaled
    
    @staticmethod
    def joint_speed(pose_norm, joint_idx, fps, smooth_sigma=1.0):
        """
        Frame-to-frame 3D speed of one joint in a normalized clip.
        """
        joint_traj = pose_norm[:, joint_idx, :]  # (T, 3)

        if smooth_sigma and smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
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
        speed = FeatureEngineering.joint_speed(pose_norm, joint_idx, fps, smooth_sigma=smooth_sigma)

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
        ankle_y = np.asarray(ankle_y, dtype=float)
        if ankle_y.size < 3 or fps <= 0:
            return np.array([], dtype=int)

        from scipy.signal import find_peaks
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
        peaks = FeatureEngineering.detect_step_events_from_ankle(ankle_y, fps, min_step_time=min_step_time)

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
        if clip.ndim != 3 or clip.shape[1] != FeatureEngineering.N_JOINTS:
            raise ValueError(f"clip must be of shape (T, {FeatureEngineering.N_JOINTS}, 3), got {clip.shape}")

        # Normalize if needed (pelvis should be near 0)
        pelvis = (clip[:, FeatureEngineering.LEFT_HIP] + clip[:, FeatureEngineering.RIGHT_HIP]) / 2
        pelvis_mean_norm = np.linalg.norm(pelvis.mean(axis=0))

        if pelvis_mean_norm > 1e-2:
            pose_norm = FeatureEngineering.normalize_pose_3d(clip)
        else:
            pose_norm = clip

        feats = {}

        # Basic spatial features
        left_ankle_y = pose_norm[:, FeatureEngineering.LEFT_ANKLE, 1]
        right_ankle_y = pose_norm[:, FeatureEngineering.RIGHT_ANKLE, 1]

        feats["step_height_L"] = float(left_ankle_y.max() - left_ankle_y.min())
        feats["step_height_R"] = float(right_ankle_y.max() - right_ankle_y.min())

        left_ankle_x = pose_norm[:, FeatureEngineering.LEFT_ANKLE, 0]
        right_ankle_x = pose_norm[:, FeatureEngineering.RIGHT_ANKLE, 0]

        feats["step_length_L"] = float(left_ankle_x.max() - left_ankle_x.min())
        feats["step_length_R"] = float(right_ankle_x.max() - right_ankle_x.min())

        left_hip_y = pose_norm[:, FeatureEngineering.LEFT_HIP, 1]
        right_hip_y = pose_norm[:, FeatureEngineering.RIGHT_HIP, 1]
        pelvis_diff = left_hip_y - right_hip_y

        feats["pelvis_drop_mean"] = float(pelvis_diff.mean())
        feats["pelvis_drop_std"] = float(pelvis_diff.std())

        left_sh_x = pose_norm[:, FeatureEngineering.LEFT_SHOULDER, 0]
        right_sh_x = pose_norm[:, FeatureEngineering.RIGHT_SHOULDER, 0]
        trunk_lean = left_sh_x - right_sh_x

        feats["trunk_lean_mean"] = float(trunk_lean.mean())
        feats["trunk_lean_std"] = float(trunk_lean.std())

        left_heel_y = pose_norm[:, FeatureEngineering.LEFT_HEEL, 1]
        right_heel_y = pose_norm[:, FeatureEngineering.RIGHT_HEEL, 1]

        feats["heel_range_L"] = float(left_heel_y.max() - left_heel_y.min())
        feats["heel_range_R"] = float(right_heel_y.max() - right_heel_y.min())

        eps = 1e-6
        hL, hR = feats["step_height_L"], feats["step_height_R"]
        lL, lR = feats["step_length_L"], feats["step_length_R"]

        feats["step_height_symmetry"] = float((hL - hR) / (hL + hR + eps))
        feats["step_length_symmetry"] = float((lL - lR) / (lL + lR + eps))

        # Knee motion
        left_knee_move = FeatureEngineering.moving_and_still_times(pose_norm, FeatureEngineering.LEFT_KNEE, fps)
        right_knee_move = FeatureEngineering.moving_and_still_times(pose_norm, FeatureEngineering.RIGHT_KNEE, fps)

        for k, v in left_knee_move.items():
            feats[f"knee_L_{k}"] = v
        for k, v in right_knee_move.items():
            feats[f"knee_R_{k}"] = v

        feats["knee_L_rom_y"] = FeatureEngineering.range_of_motion(pose_norm, FeatureEngineering.LEFT_KNEE, axis="y")["rom_y"]
        feats["knee_R_rom_y"] = FeatureEngineering.range_of_motion(pose_norm, FeatureEngineering.RIGHT_KNEE, axis="y")["rom_y"]

        # Joint ROM
        hip_L_rom_y = FeatureEngineering.range_of_motion(pose_norm, FeatureEngineering.LEFT_HIP, axis="y")["rom_y"]
        hip_R_rom_y = FeatureEngineering.range_of_motion(pose_norm, FeatureEngineering.RIGHT_HIP, axis="y")["rom_y"]
        feats["hip_L_rom_y"] = hip_L_rom_y
        feats["hip_R_rom_y"] = hip_R_rom_y

        shoulder_L_rom_x = FeatureEngineering.range_of_motion(pose_norm, FeatureEngineering.LEFT_SHOULDER, axis="x")["rom_x"]
        shoulder_R_rom_x = FeatureEngineering.range_of_motion(pose_norm, FeatureEngineering.RIGHT_SHOULDER, axis="x")["rom_x"]
        feats["shoulder_L_rom_x"] = shoulder_L_rom_x
        feats["shoulder_R_rom_x"] = shoulder_R_rom_x

        ankle_L_rom_y = FeatureEngineering.range_of_motion(pose_norm, FeatureEngineering.LEFT_ANKLE, axis="y")["rom_y"]
        ankle_R_rom_y = FeatureEngineering.range_of_motion(pose_norm, FeatureEngineering.RIGHT_ANKLE, axis="y")["rom_y"]
        feats["ankle_L_rom_y"] = ankle_L_rom_y
        feats["ankle_R_rom_y"] = ankle_R_rom_y

        # ROM asymmetries
        feats["knee_rom_asym"] = FeatureEngineering.asymmetry(feats["knee_L_rom_y"], feats["knee_R_rom_y"])
        feats["hip_rom_asym"] = FeatureEngineering.asymmetry(hip_L_rom_y, hip_R_rom_y)
        feats["shoulder_rom_asym"] = FeatureEngineering.asymmetry(shoulder_L_rom_x, shoulder_R_rom_x)
        feats["ankle_rom_asym"] = FeatureEngineering.asymmetry(ankle_L_rom_y, ankle_R_rom_y)

        # Stance / swing ratio
        ankle_L_move = FeatureEngineering.moving_and_still_times(pose_norm, FeatureEngineering.LEFT_ANKLE, fps)
        ankle_R_move = FeatureEngineering.moving_and_still_times(pose_norm, FeatureEngineering.RIGHT_ANKLE, fps)

        feats["ankle_L_moving_fraction"] = ankle_L_move["moving_fraction"]
        feats["ankle_L_still_fraction"] = ankle_L_move["still_fraction"]
        feats["ankle_R_moving_fraction"] = ankle_R_move["moving_fraction"]
        feats["ankle_R_still_fraction"] = ankle_R_move["still_fraction"]

        stance_ratio_L = ankle_L_move["still_fraction"] / (ankle_L_move["moving_fraction"] + 1e-6)
        stance_ratio_R = ankle_R_move["still_fraction"] / (ankle_R_move["moving_fraction"] + 1e-6)

        feats["stance_ratio_L"] = float(stance_ratio_L)
        feats["stance_ratio_R"] = float(stance_ratio_R)
        feats["stance_ratio_asym"] = FeatureEngineering.asymmetry(stance_ratio_L, stance_ratio_R)

        # Joint angles
        knee_angle_L = FeatureEngineering.joint_angle(
            pose_norm[:, FeatureEngineering.LEFT_HIP, :],
            pose_norm[:, FeatureEngineering.LEFT_KNEE, :],
            pose_norm[:, FeatureEngineering.LEFT_ANKLE, :],
        )
        knee_angle_R = FeatureEngineering.joint_angle(
            pose_norm[:, FeatureEngineering.RIGHT_HIP, :],
            pose_norm[:, FeatureEngineering.RIGHT_KNEE, :],
            pose_norm[:, FeatureEngineering.RIGHT_ANKLE, :],
        )

        feats["knee_angle_L_mean"] = float(knee_angle_L.mean())
        feats["knee_angle_L_std"] = float(knee_angle_L.std())
        feats["knee_angle_L_rom"] = float(knee_angle_L.max() - knee_angle_L.min())

        feats["knee_angle_R_mean"] = float(knee_angle_R.mean())
        feats["knee_angle_R_std"] = float(knee_angle_R.std())
        feats["knee_angle_R_rom"] = float(knee_angle_R.max() - knee_angle_R.min())

        hip_angle_L = FeatureEngineering.joint_angle(
            pose_norm[:, FeatureEngineering.LEFT_SHOULDER, :],
            pose_norm[:, FeatureEngineering.LEFT_HIP, :],
            pose_norm[:, FeatureEngineering.LEFT_KNEE, :],
        )
        hip_angle_R = FeatureEngineering.joint_angle(
            pose_norm[:, FeatureEngineering.RIGHT_SHOULDER, :],
            pose_norm[:, FeatureEngineering.RIGHT_HIP, :],
            pose_norm[:, FeatureEngineering.RIGHT_KNEE, :],
        )

        feats["hip_angle_L_mean"] = float(hip_angle_L.mean())
        feats["hip_angle_L_std"] = float(hip_angle_L.std())
        feats["hip_angle_L_rom"] = float(hip_angle_L.max() - hip_angle_L.min())

        feats["hip_angle_R_mean"] = float(hip_angle_R.mean())
        feats["hip_angle_R_std"] = float(hip_angle_R.std())
        feats["hip_angle_R_rom"] = float(hip_angle_R.max() - hip_angle_R.min())

        ankle_angle_L = FeatureEngineering.joint_angle(
            pose_norm[:, FeatureEngineering.LEFT_KNEE, :],
            pose_norm[:, FeatureEngineering.LEFT_ANKLE, :],
            pose_norm[:, FeatureEngineering.LEFT_FOOT_INDEX, :],
        )
        ankle_angle_R = FeatureEngineering.joint_angle(
            pose_norm[:, FeatureEngineering.RIGHT_KNEE, :],
            pose_norm[:, FeatureEngineering.RIGHT_ANKLE, :],
            pose_norm[:, FeatureEngineering.RIGHT_FOOT_INDEX, :],
        )

        feats["ankle_angle_L_mean"] = float(ankle_angle_L.mean())
        feats["ankle_angle_L_std"] = float(ankle_angle_L.std())
        feats["ankle_angle_L_rom"] = float(ankle_angle_L.max() - ankle_angle_L.min())

        feats["ankle_angle_R_mean"] = float(ankle_angle_R.mean())
        feats["ankle_angle_R_std"] = float(ankle_angle_R.std())
        feats["ankle_angle_R_rom"] = float(ankle_angle_R.max() - ankle_angle_R.min())

        # Angle-based ROM asymmetries
        feats["knee_angle_rom_asym"] = FeatureEngineering.asymmetry(
            feats["knee_angle_L_rom"], feats["knee_angle_R_rom"]
        )
        feats["hip_angle_rom_asym"] = FeatureEngineering.asymmetry(
            feats["hip_angle_L_rom"], feats["hip_angle_R_rom"]
        )
        feats["ankle_angle_rom_asym"] = FeatureEngineering.asymmetry(
            feats["ankle_angle_L_rom"], feats["ankle_angle_R_rom"]
        )

        # Temporal gait features
        left_temporal = FeatureEngineering.step_temporal_features(left_ankle_y, fps)
        right_temporal = FeatureEngineering.step_temporal_features(right_ankle_y, fps)

        for k, v in left_temporal.items():
            feats[f"step_L_{k}"] = float(v) if v is not None else np.nan
        for k, v in right_temporal.items():
            feats[f"step_R_{k}"] = float(v) if v is not None else np.nan

        # Asymmetries from temporal features
        if not np.isnan(left_temporal["mean_step_time"]) and not np.isnan(right_temporal["mean_step_time"]):
            feats["step_time_asym"] = FeatureEngineering.asymmetry(
                left_temporal["mean_step_time"], right_temporal["mean_step_time"]
            )
        else:
            feats["step_time_asym"] = np.nan

        if not np.isnan(left_temporal["cadence"]) and not np.isnan(right_temporal["cadence"]):
            feats["cadence_asym"] = FeatureEngineering.asymmetry(
                left_temporal["cadence"], right_temporal["cadence"]
            )
        else:
            feats["cadence_asym"] = np.nan

        # Step width proxy
        ankle_L_x = pose_norm[:, FeatureEngineering.LEFT_ANKLE, 0]
        ankle_R_x = pose_norm[:, FeatureEngineering.RIGHT_ANKLE, 0]
        step_width_series = np.abs(ankle_L_x - ankle_R_x)

        feats["step_width_mean"] = float(step_width_series.mean())
        feats["step_width_std"] = float(step_width_series.std())

        return feats
    
    @staticmethod
    def extract_features_from_csv(csv_path: Path, video_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Extract features from a CSV file containing MediaPipe landmarks.
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Check if the CSV has the expected columns
            required_columns = ['frame', 'landmark_id', 'x_norm', 'y_norm', 'z_norm']
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV file missing required columns: {required_columns}")
                return pd.DataFrame()
            
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
                pose = np.zeros((T, FeatureEngineering.N_JOINTS, 3), dtype=np.float32)

                for _, r in group.iterrows():
                    f = int(r.frame)
                    j = int(r.landmark_id)
                    pose[f, j] = [r.x_norm, r.y_norm, r.z_norm]

                pose_rows.append(pose)
            
            if not pose_rows:
                st.error("No valid pose data found in CSV")
                return pd.DataFrame()
            
            # Extract features for each pose
            all_features = []
            for i, pose in enumerate(pose_rows):
                try:
                    features = FeatureEngineering.compute_clip_features(pose, fps)
                    features["clip_id"] = i
                    all_features.append(features)
                except Exception as e:
                    logger.error(f"Error extracting features from clip {i}: {e}")
            
            if not all_features:
                st.error("Failed to extract features from any clips")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_features = pd.DataFrame(all_features)
            return df_features
        
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            logger.error(f"Error processing CSV file: {e}")
            return pd.DataFrame()
    
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
    if 'uploaded_csv_path' not in st.session_state:
        st.session_state.uploaded_csv_path = None
    if 'csv_features_df' not in st.session_state:
        st.session_state.csv_features_df = None
    if 'csv_features_path' not in st.session_state:
        st.session_state.csv_features_path = None
    
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
        
        if st.session_state.csv_features_df is not None:
            st.success("✅ CSV features extracted")
        
        st.markdown("---")
        
        if st.button("🔄 Reset", use_container_width=True):
            # Clear all session state
            keys_to_clear = [
                'uploaded_video_path', 'processing_complete', 'output_videos', 
                'last_tab', 'initialized', 'features_df', 'features_path',
                'uploaded_csv_path', 'csv_features_df', 'csv_features_path'
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
            st.write(f"Session keys: {len(st.session_state.keys())}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "⚙️ Process", "🎬 Landmarker Videos", "📊 Feature Engineering"])
    
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
                                    # Extract features
                                    df_features = FeatureEngineering.extract_features_from_csv(
                                        csv_path, 
                                        st.session_state.uploaded_video_path
                                    )
                                    
                                    if not df_features.empty:
                                        # Save features
                                        features_path = FeatureEngineering.save_features(
                                            df_features, 
                                            st.session_state.uploaded_video_path
                                        )
                                        
                                        # Update session state
                                        st.session_state.features_df = df_features
                                        st.session_state.features_path = features_path
                                        
                                        st.success(f"✅ Features extracted successfully! Found {len(df_features)} features.")
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
                        
                        # Feature preview
                        with st.expander("👁️ Preview Features"):
                            st.dataframe(st.session_state.features_df, use_container_width=True)
                        
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
                        
                        # Feature visualization
                        st.markdown("---")
                        st.subheader("📈 Feature Visualization")
                        
                        # Select feature to visualize
                        numeric_features = st.session_state.features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        
                        if numeric_features:
                            selected_feature = st.selectbox("Select a feature to visualize:", numeric_features)
                            
                            if selected_feature:
                                import matplotlib.pyplot as plt
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
                                df_features = FeatureEngineering.extract_features_from_csv(
                                    st.session_state.uploaded_csv_path
                                )
                                
                                if not df_features.empty:
                                    # Save features
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    csv_stem = st.session_state.uploaded_csv_path.stem
                                    features_path = FEATURES_DIR / f"{csv_stem}_features_{timestamp}.csv"
                                    
                                    df_features.to_csv(features_path, index=False)
                                    
                                    # Update session state
                                    st.session_state.csv_features_df = df_features
                                    st.session_state.csv_features_path = features_path
                                    
                                    st.success(f"✅ Features extracted successfully! Found {len(df_features)} features.")
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
                    
                    # Feature preview
                    with st.expander("👁️ Preview Features"):
                        st.dataframe(st.session_state.csv_features_df, use_container_width=True)
                    
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
                            import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    main()