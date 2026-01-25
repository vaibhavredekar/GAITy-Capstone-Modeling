#!/usr/bin/env python3
"""
MEDIAPIPE POSE DETECTION PIPELINE - PRODUCTION GRADE APPLICATION
Complete implementation with robust video rendering, feature engineering, and baseline modeling.
"""

from dataclasses import dataclass
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
from typing import Optional, Dict, Tuple, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import polars as pl
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample

# Scikit-learn and XGBoost for modeling
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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
MODELS_DIR = PROJECT_ROOT / "models" # Centralized model directory
GAIT_CYCLES_DIR = PROJECT_ROOT / "data" / "gait_cycles"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, FEATURES_DIR, MODELS_DIR, GAIT_CYCLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# VIDEO CODEC DETECTION & CONVERSION (From your original app)
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
# FILE MANAGEMENT (From your original app)
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
        
        with open(video_path, 'wb') as f:
            f.write(file_bytes)
        
        logger.info(f"Saved: {video_path.name}")
        return video_path, False

    @staticmethod
    def find_output_videos(video_path: Path) -> Dict[str, Optional[Path]]:
        video_stem = video_path.stem
        results = {'annotated': None, 'skeleton': None, 'csv': None}
        
        candidates = {
            'annotated': [OUTPUT_DIR / f"{video_stem}_annotated.mp4"],
            'skeleton': [OUTPUT_DIR / f"{video_stem}_skeleton.mp4"],
            'csv': [OUTPUT_DIR / f"{video_stem}_landmarks.csv"],
        }
        
        for key, paths in candidates.items():
            for candidate in paths:
                if candidate.exists() and candidate.stat().st_size > 0:
                    results[key] = candidate
                    logger.info(f"Found {key}: {candidate.name}")
                    break
        
        return results

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE MANAGEMENT (From your original app)
# ═══════════════════════════════════════════════════════════════════════════

class PipelineManager:
    
    @staticmethod
    def load_config() -> Optional[dict]:
        if not CONFIG_PATH.exists():
            return {"input_paths": []}
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Load config failed: {e}")
            return {"input_paths": []}
    
    @staticmethod
    def save_config(config: dict) -> bool:
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Save config failed: {e}")
            return False

# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE GAIT ANALYSIS & MODELING PIPELINE (From our conversation)
# This section replaces the old GaitAnalysisEngine with the complete, working version.
# ═══════════════════════════════════════════════════════════════════════════

# --- Helper Dataclass for Feature Extraction ---
@dataclass(frozen=True)
class FeatureConfig:
    smooth_sigma: float = 1.0
    speed_thresh: float = 0.02
    min_step_time: float = 0.3
    auto_normalize_if_needed: bool = True

# --- Preprocessing Class ---
class Preprocessing:
    def __init__(self):
        self.LEFT_HIP, self.RIGHT_HIP = 23, 24
        self.LEFT_SHOULDER, self.RIGHT_SHOULDER = 11, 12
        self.LEFT_KNEE, self.RIGHT_KNEE = 25, 26
        self.LEFT_ANKLE, self.RIGHT_ANKLE = 27, 28
        self.NUM_JOINTS = 33
        self.GAIT_JOINTS = [2, 5, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        self.LABEL_MAP = {"normal": 0, "abnormal": 1}
        self.ANOMALY_COLS = [
            "gait_anomaly_knee_sagittal_plane_abnormality", "gait_anomaly_trunk_balance_abnormality",
            "gait_anomaly_spatiotemporal_asymmetry", "gait_anomaly_hip_pelvic_control_deficit",
            "gait_anomaly_distal_foot_control_deficit",
        ]

    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(pl.col("movement_type").fill_null("N"))
        df = df.filter(pl.col("movement_type") != "SLOWMOTION")
        df = df.with_columns(pl.col("gait_markers").fill_null("NA"))
        return df

    def missing_frames_summary_polars(self, df: pl.DataFrame, video_col="video_id", frame_col="frame"):
        unique_counts = df.group_by(video_col).agg(pl.col(frame_col).n_unique().alias("num_unique_frames"))
        frame_range = df.group_by(video_col).agg([pl.col(frame_col).min().alias("min_frame"), pl.col(frame_col).max().alias("max_frame")])
        frame_stats = unique_counts.join(frame_range, on=video_col)
        frame_stats = frame_stats.with_columns([
            (pl.col("max_frame") - pl.col("min_frame") + 1 - pl.col("num_unique_frames")).alias("num_missing_frames"),
            ((pl.col("max_frame") - pl.col("min_frame") + 1 - pl.col("num_unique_frames")) / (pl.col("max_frame") - pl.col("min_frame") + 1) * 100).alias("pct_missing")
        ])
        return frame_stats.sort("num_missing_frames", descending=True)

    def filter_videos_by_missing_frames(self, df: pl.DataFrame, video_col="video_id", frame_col="frame", threshold_pct=5.0):
        frame_stats = self.missing_frames_summary_polars(df, video_col=video_col, frame_col=frame_col)
        videos_to_keep = frame_stats.filter(pl.col("pct_missing") <= threshold_pct)[video_col].to_list()
        filtered_df = df.filter(pl.col(video_col).is_in(videos_to_keep))
        removed_videos = frame_stats.filter(pl.col("pct_missing") > threshold_pct)[video_col].to_list()
        return filtered_df, removed_videos

    def interpolate_pose(self, pose: np.ndarray) -> np.ndarray:
        T, J, C = pose.shape
        pose_interp = pose.copy()
        for j in range(J):
            for c in range(C):
                coord = pose[:, j, c]
                missing = coord == 0
                if missing.all(): continue
                valid_idx = np.where(~missing)[0]
                valid_values = coord[valid_idx]
                pose_interp[:, j, c] = np.interp(np.arange(T), valid_idx, valid_values)
        return pose_interp

    def add_pose_column(self, df: pl.DataFrame) -> pl.DataFrame:
        metadata_cols = ['video_id', 'patient_name', 'source_file', 'dataset', 'fps', 'movement_type', 'side', 'gait_markers'] + self.ANOMALY_COLS
        pose_rows = []
        for video_id in df.select("video_id").unique().to_series():
            group = df.filter(pl.col("video_id") == video_id).sort("frame")
            frames = np.sort(group["frame"].unique().to_numpy())
            frame_to_idx = {f: i for i, f in enumerate(frames)}
            T = len(frames)
            pose = np.zeros((T, self.NUM_JOINTS, 3), dtype=np.float32)
            for row in group.iter_rows(named=True):
                f_idx = frame_to_idx[row["frame"]]
                j = int(row["landmark_id"])
                if j >= self.NUM_JOINTS: continue
                pose[f_idx, j, :] = [row["x_norm"], row["y_norm"], row["z_norm"]]
            pose = self.interpolate_pose(pose)
            first_row_dict = group.row(0, named=True)
            new_row = {col: first_row_dict.get(col) for col in metadata_cols}
            new_row["pose"] = pose
            pose_rows.append(new_row)
        return pl.DataFrame(pose_rows)

    def normalize_pose_3d(self, pose):
        pelvis = (pose[:, self.LEFT_HIP] + pose[:, self.RIGHT_HIP]) / 2.0
        pose_centered = pose - pelvis[:, None, :]
        torso = (pose_centered[:, self.LEFT_SHOULDER] + pose_centered[:, self.RIGHT_SHOULDER]) / 2.0
        scale = np.linalg.norm(torso, axis=1).mean()
        if scale == 0 or not np.isfinite(scale):
            raise ValueError("Invalid torso scale during pose normalisation")
        return pose_centered / scale

    def extract_sliding_windows(self, pose: np.ndarray, fps: int = 30, window_seconds: float = 2.0, overlap: float = 0.5):
        T, J, C = pose.shape
        window_frames = int(window_seconds * fps)
        step_frames = int(window_frames * (1 - overlap))
        if window_frames > T: return [pose]
        windows = []
        start = 0
        while start + window_frames <= T:
            windows.append(pose[start:start + window_frames])
            start += step_frames
        if start < T and (T - start) >= int(0.5 * window_frames):
            windows.append(pose[T - window_frames : T])
        return windows

    def preprocess_gait_sliding_windows(self, df_video: pl.DataFrame, window_seconds: float = 2.0, overlap: float = 0.5, resample_frames: int = 60):
        all_windows, all_binary_labels, all_multilabels, all_window_ids = [], [], [], []
        for idx, row in enumerate(df_video.iter_rows(named=True)):
            pose = row["pose"]; pose = np.asarray(pose)
            if pose.size == 0 or pose.ndim != 3: continue
            fps = row.get("fps", 30) or 30
            label_str = str(row.get("dataset", "none")).strip().lower()
            binary_label = self.LABEL_MAP.get(label_str, 0)
            multilabel = np.array([int(row.get(col, 0)) for col in self.ANOMALY_COLS], dtype=np.int32)
            try: pose_norm = self.normalize_pose_3d(pose)
            except Exception as e: continue
            windows = self.extract_sliding_windows(pose_norm, fps=fps, window_seconds=window_seconds, overlap=overlap)
            windows = [w[:, self.GAIT_JOINTS, :] for w in windows]
            video_id = row.get("video_id", f"vid{idx}")
            for win_idx, w in enumerate(windows):
                if w.shape[0] < 2: continue
                w_resampled = resample(w, resample_frames, axis=0)
                start_frame = win_idx * int(window_seconds * fps * (1 - overlap))
                end_frame = start_frame + w_resampled.shape[0] - 1
                window_id = f"{video_id}_win{win_idx:03d}_f{start_frame}-{end_frame}"
                all_windows.append(w_resampled); all_binary_labels.append(binary_label)
                all_multilabels.append(multilabel); all_window_ids.append(window_id)
        return np.asarray(all_windows, dtype=np.float32), np.asarray(all_binary_labels, dtype=np.int32), np.asarray(all_multilabels, dtype=np.int32), all_window_ids

# --- Quality Control Class ---
class QualityControl:
    def __init__(self):
        self.GAIT_JOINTS = [2, 5, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        self.GAIT_JOINT_INDEX = {joint_id: i for i, joint_id in enumerate(self.GAIT_JOINTS)}
        self.LEFT_HIP, self.RIGHT_HIP = 23, 24; self.LEFT_SHOULDER = 11, 12
        self.LEFT_KNEE, self.RIGHT_KNEE = 25, 26; self.LEFT_ANKLE = 27, 28

    def qc_gait_window(self, window, fps, visualize=False, title=""):
        L_HIP = self.GAIT_JOINT_INDEX[23]; R_HIP = self.GAIT_JOINT_INDEX[24]
        L_SHOULDER = self.GAIT_JOINT_INDEX[11]; L_KNEE = self.GAIT_JOINT_INDEX[25]
        L_ANKLE = self.GAIT_JOINT_INDEX[27]
        qc = {}
        qc["n_frames"] = window.shape[0]; qc["duration_s"] = window.shape[0] / fps
        qc["flag_short"] = qc["duration_s"] < 1.0
        pelvis = (window[:, L_HIP] + window[:, R_HIP]) / 2
        qc["pelvis_offset"] = np.linalg.norm(pelvis.mean(axis=0)); qc["flag_off_center"] = qc["pelvis_offset"] > 0.1
        torso = window[:, L_SHOULDER] - pelvis; torso_len = np.linalg.norm(torso, axis=1)
        qc["torso_std_length"] = torso_len.std(); qc["flag_torso_unstable"] = qc["torso_std_length"] > 0.15
        ankle_y = window[:, L_ANKLE, 1]; qc["ankle_y_velocity_std"] = np.std(np.diff(ankle_y))
        qc["flag_jitter"] = qc["ankle_y_velocity_std"] > 0.2
        knee_y = window[:, L_KNEE, 1]; min_peak_dist = int(0.4 * fps)
        peaks, _ = find_peaks(knee_y, distance=min_peak_dist); qc["n_knee_peaks"] = len(peaks)
        qc["flag_no_periodicity"] = qc["n_knee_peaks"] < 1
        ankle_z = window[:, L_ANKLE, 2]; qc["ankle_z_range"] = ankle_z.max() - ankle_z.min()
        qc["flag_flat_depth"] = qc["ankle_z_range"] < 0.05
        qc["qc_fail"] = any([qc["flag_short"], qc["flag_off_center"], qc["flag_jitter"], qc["flag_no_periodicity"], qc["flag_flat_depth"], qc["flag_torso_unstable"]])
        return qc

    def apply_qc_windows(self, X_windows, y_binary, y_multilabel, window_ids, fps=60):
        qc_rows = [self.qc_gait_window(w, fps=fps) for w in X_windows]
        qc_df = pd.DataFrame(qc_rows); qc_df["binary_label"] = y_binary; qc_df["window_id"] = window_ids
        keep_mask = ~qc_df["qc_fail"].values; window_ids = np.array(window_ids); y_multilabel = np.asarray(y_multilabel)
        X_clean = X_windows[keep_mask]; y_binary_clean = y_binary[keep_mask]
        y_multilabel_clean = y_multilabel[keep_mask]; window_ids_clean = window_ids[keep_mask]
        print(f"QC-clean windows: {len(X_clean)} / {len(X_windows)} ({100*len(X_clean)/len(X_windows):.2f}%)")
        return X_clean, y_binary_clean, y_multilabel_clean, window_ids_clean, qc_df

# --- Feature Extraction Class ---
class FeatureExtraction:
    def __init__(self):
        self.N_JOINTS = 33; self.LEFT_SHOULDER, self.RIGHT_SHOULDER = 11, 12
        self.LEFT_HIP, self.RIGHT_HIP = 23, 24; self.LEFT_KNEE, self.RIGHT_KNEE = 25, 26
        self.LEFT_ANKLE, self.RIGHT_ANKLE = 27, 28; self.LEFT_HEEL, self.RIGHT_HEEL = 29, 30
        self.LEFT_FOOT_INDEX, self.RIGHT_FOOT_INDEX = 31, 32

    def normalize_pose_3d(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=float); pelvis = (pose[:, self.LEFT_HIP] + pose[:, self.RIGHT_HIP]) / 2.0
        pose_centered = pose - pelvis[:, None, :]
        torso = (pose_centered[:, self.LEFT_SHOULDER] + pose_centered[:, self.RIGHT_SHOULDER]) / 2.0
        scale = np.linalg.norm(torso, axis=1).mean()
        if scale == 0 or not np.isfinite(scale): raise ValueError("Invalid torso scale during pose normalisation")
        return pose_centered / scale

    def joint_speed(self, pose_norm: np.ndarray, joint_idx: int, fps: float, smooth_sigma: float = 1.0) -> np.ndarray:
        joint_traj = pose_norm[:, joint_idx, :]
        if smooth_sigma and smooth_sigma > 0: joint_traj = gaussian_filter1d(joint_traj, sigma=smooth_sigma, axis=0)
        diffs = np.diff(joint_traj, axis=0); disp = np.linalg.norm(diffs, axis=1); return disp * fps

    def moving_and_still_times(self, pose_norm: np.ndarray, joint_idx: int, fps: float, speed_thresh: float = 0.02, smooth_sigma: float = 1.0) -> dict:
        speed = self.joint_speed(pose_norm, joint_idx, fps, smooth_sigma=smooth_sigma)
        moving_mask = speed >= speed_thresh; still_mask = ~moving_mask
        return {"moving_fraction": float(moving_mask.mean()), "still_fraction": float(still_mask.mean())}

    def range_of_motion(self, pose_norm: np.ndarray, joint_idx: int, axis: str | None = None) -> dict:
        traj = pose_norm[:, joint_idx, :]
        if axis is None: mean_pos = traj.mean(axis=0); dist = np.linalg.norm(traj - mean_pos, axis=1); return {"rom_3d": float(dist.max() - dist.min())}
        axis_to_idx = {"x": 0, "y": 1, "z": 2}; idx = axis_to_idx[axis]; coord = traj[:, idx]
        return {f"rom_{axis}": float(coord.max() - coord.min())}

    def asymmetry(self, L: float, R: float, eps: float = 1e-6) -> float: return float((L - R) / (L + R + eps))

    def joint_angle(self, p_prox: np.ndarray, p_joint: np.ndarray, p_dist: np.ndarray) -> np.ndarray:
        v1 = p_prox - p_joint; v2 = p_dist - p_joint
        num = np.einsum("ij,ij->i", v1, v2); den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6
        cosang = np.clip(num / den, -1.0, 1.0); return np.degrees(np.arccos(cosang))

    def detect_step_events_from_ankle(self, ankle_y: np.ndarray, fps: float, min_step_time: float = 0.3) -> np.ndarray:
        ankle_y = np.asarray(ankle_y, dtype=float); 
        if ankle_y.size < 3 or fps <= 0: return np.array([], dtype=int)
        inv = -ankle_y; min_distance = max(1, int(min_step_time * fps)); peaks, _ = find_peaks(inv, distance=min_distance); return peaks

    def step_temporal_features(self, ankle_y: np.ndarray, fps: float, min_step_time: float = 0.3) -> dict:
        ankle_y = np.asarray(ankle_y, dtype=float); peaks = self.detect_step_events_from_ankle(ankle_y, fps, min_step_time=min_step_time)
        if peaks.size < 2 or fps <= 0: return {"mean_step_time": np.nan, "std_step_time": np.nan, "cadence": np.nan, "mean_stride_time": np.nan, "std_stride_time": np.nan, "step_time_cv": np.nan}
        times = peaks / fps; step_intervals = np.diff(times); mean_step = float(step_intervals.mean()); std_step = float(step_intervals.std())
        cadence = 60.0 / mean_step if mean_step > 0 else np.nan
        if times.size >= 3: stride_intervals = times[2:] - times[:-2]; mean_stride = float(stride_intervals.mean()); std_stride = float(stride_intervals.std())
        else: mean_stride, std_stride = np.nan, np.nan
        step_time_cv = (std_step / mean_step) if mean_step > 0 else np.nan
        return {"mean_step_time": mean_step, "std_step_time": std_step, "cadence": float(cadence), "mean_stride_time": mean_stride, "std_stride_time": std_stride, "step_time_cv": float(step_time_cv)}

    def compute_window_features(self, window: np.ndarray, fps: float, cfg: FeatureConfig | None = None) -> dict[str, float]:
        cfg = cfg or FeatureConfig(); window = np.asarray(window)
        if window.ndim != 3 or window.shape[1] != self.N_JOINTS or window.shape[2] != 3:
            raise ValueError(f"window must be of shape (T, {self.N_JOINTS}, 3), got {window.shape}")
        pelvis = (window[:, self.LEFT_HIP] + window[:, self.RIGHT_HIP]) / 2
        pelvis_mean_norm = np.linalg.norm(pelvis.mean(axis=0))
        pose_norm = self.normalize_pose_3d(window) if pelvis_mean_norm > 1e-2 else window
        feats: dict[str, float] = {}
        left_ankle_y = pose_norm[:, self.LEFT_ANKLE, 1]; right_ankle_y = pose_norm[:, self.RIGHT_ANKLE, 1]
        feats["step_height_L"] = float(left_ankle_y.max() - left_ankle_y.min()); feats["step_height_R"] = float(right_ankle_y.max() - right_ankle_y.min())
        left_ankle_x = pose_norm[:, self.LEFT_ANKLE, 0]; right_ankle_x = pose_norm[:, self.RIGHT_ANKLE, 0]
        feats["step_length_L"] = float(left_ankle_x.max() - left_ankle_x.min()); feats["step_length_R"] = float(right_ankle_x.max() - right_ankle_x.min())
        left_hip_y = pose_norm[:, self.LEFT_HIP, 1]; right_hip_y = pose_norm[:, self.RIGHT_HIP, 1]; pelvis_diff = left_hip_y - right_hip_y
        feats["pelvis_drop_mean"] = float(pelvis_diff.mean()); feats["pelvis_drop_std"] = float(pelvis_diff.std())
        left_sh_x = pose_norm[:, self.LEFT_SHOULDER, 0]; right_sh_x = pose_norm[:, self.RIGHT_SHOULDER, 0]; trunk_lean = left_sh_x - right_sh_x
        feats["trunk_lean_mean"] = float(trunk_lean.mean()); feats["trunk_lean_std"] = float(trunk_lean.std())
        left_heel_y = pose_norm[:, self.LEFT_HEEL, 1]; right_heel_y = pose_norm[:, self.RIGHT_HEEL, 1]
        feats["heel_range_L"] = float(left_heel_y.max() - left_heel_y.min()); feats["heel_range_R"] = float(right_heel_y.max() - right_heel_y.min())
        eps = 1e-6; hL, hR = feats["step_height_L"], feats["step_height_R"]; lL, lR = feats["step_length_L"], feats["step_length_R"]
        feats["step_height_symmetry"] = float((hL - hR) / (hL + hR + eps)); feats["step_length_symmetry"] = float((lL - lR) / (lL + lR + eps))
        left_knee_move = self.moving_and_still_times(pose_norm, self.LEFT_KNEE, fps, speed_thresh=cfg.speed_thresh, smooth_sigma=cfg.smooth_sigma)
        right_knee_move = self.moving_and_still_times(pose_norm, self.RIGHT_KNEE, fps, speed_thresh=cfg.speed_thresh, smooth_sigma=cfg.smooth_sigma)
        for k, v in left_knee_move.items(): feats[f"knee_L_{k}"] = v; 
        for k, v in right_knee_move.items(): feats[f"knee_R_{k}"] = v
        feats["knee_L_rom_y"] = self.range_of_motion(pose_norm, self.LEFT_KNEE, axis="y")["rom_y"]; feats["knee_R_rom_y"] = self.range_of_motion(pose_norm, self.RIGHT_KNEE, axis="y")["rom_y"]
        hip_L_rom_y = self.range_of_motion(pose_norm, self.LEFT_HIP, axis="y")["rom_y"]; hip_R_rom_y = self.range_of_motion(pose_norm, self.RIGHT_HIP, axis="y")["rom_y"]
        feats["hip_L_rom_y"] = hip_L_rom_y; feats["hip_R_rom_y"] = hip_R_rom_y
        shoulder_L_rom_x = self.range_of_motion(pose_norm, self.LEFT_SHOULDER, axis="x")["rom_x"]; shoulder_R_rom_x = self.range_of_motion(pose_norm, self.RIGHT_SHOULDER, axis="x")["rom_x"]
        feats["shoulder_L_rom_x"] = shoulder_L_rom_x; feats["shoulder_R_rom_x"] = shoulder_R_rom_x
        ankle_L_rom_y = self.range_of_motion(pose_norm, self.LEFT_ANKLE, axis="y")["rom_y"]; ankle_R_rom_y = self.range_of_motion(pose_norm, self.RIGHT_ANKLE, axis="y")["rom_y"]
        feats["ankle_L_rom_y"] = ankle_L_rom_y; feats["ankle_R_rom_y"] = ankle_R_rom_y
        feats["knee_rom_asym"] = self.asymmetry(feats["knee_L_rom_y"], feats["knee_R_rom_y"])
        feats["hip_rom_asym"] = self.asymmetry(hip_L_rom_y, hip_R_rom_y)
        feats["shoulder_rom_asym"] = self.asymmetry(shoulder_L_rom_x, shoulder_R_rom_x)
        feats["ankle_rom_asym"] = self.asymmetry(ankle_L_rom_y, ankle_R_rom_y)
        ankle_L_move = self.moving_and_still_times(pose_norm, self.LEFT_ANKLE, fps, speed_thresh=cfg.speed_thresh, smooth_sigma=cfg.smooth_sigma)
        ankle_R_move = self.moving_and_still_times(pose_norm, self.RIGHT_ANKLE, fps, speed_thresh=cfg.speed_thresh, smooth_sigma=cfg.smooth_sigma)
        feats["ankle_L_moving_fraction"] = ankle_L_move["moving_fraction"]; feats["ankle_L_still_fraction"] = ankle_L_move["still_fraction"]
        feats["ankle_R_moving_fraction"] = ankle_R_move["moving_fraction"]; feats["ankle_R_still_fraction"] = ankle_R_move["still_fraction"]
        stance_ratio_L = ankle_L_move["still_fraction"] / (ankle_L_move["moving_fraction"] + 1e-6)
        stance_ratio_R = ankle_R_move["still_fraction"] / (ankle_R_move["moving_fraction"] + 1e-6)
        feats["stance_ratio_L"] = float(stance_ratio_L); feats["stance_ratio_R"] = float(stance_ratio_R)
        feats["stance_ratio_asym"] = self.asymmetry(stance_ratio_L, stance_ratio_R)
        knee_angle_L = self.joint_angle(pose_norm[:, self.LEFT_HIP, :], pose_norm[:, self.LEFT_KNEE, :], pose_norm[:, self.LEFT_ANKLE, :])
        knee_angle_R = self.joint_angle(pose_norm[:, self.RIGHT_HIP, :], pose_norm[:, self.RIGHT_KNEE, :], pose_norm[:, self.RIGHT_ANKLE, :])
        feats["knee_angle_L_mean"] = float(knee_angle_L.mean()); feats["knee_angle_L_std"] = float(knee_angle_L.std())
        feats["knee_angle_L_rom"] = float(knee_angle_L.max() - knee_angle_L.min())
        feats["knee_angle_R_mean"] = float(knee_angle_R.mean()); feats["knee_angle_R_std"] = float(knee_angle_R.std())
        feats["knee_angle_R_rom"] = float(knee_angle_R.max() - knee_angle_R.min())
        hip_angle_L = self.joint_angle(pose_norm[:, self.LEFT_SHOULDER, :], pose_norm[:, self.LEFT_HIP, :], pose_norm[:, self.LEFT_KNEE, :])
        hip_angle_R = self.joint_angle(pose_norm[:, self.RIGHT_SHOULDER, :], pose_norm[:, self.RIGHT_HIP, :], pose_norm[:, self.RIGHT_KNEE, :])
        feats["hip_angle_L_mean"] = float(hip_angle_L.mean()); feats["hip_angle_L_std"] = float(hip_angle_L.std())
        feats["hip_angle_L_rom"] = float(hip_angle_L.max() - hip_angle_L.min())
        feats["hip_angle_R_mean"] = float(hip_angle_R.mean()); feats["hip_angle_R_std"] = float(hip_angle_R.std())
        feats["hip_angle_R_rom"] = float(hip_angle_R.max() - hip_angle_R.min())
        ankle_angle_L = self.joint_angle(pose_norm[:, self.LEFT_KNEE, :], pose_norm[:, self.LEFT_ANKLE, :], pose_norm[:, self.LEFT_FOOT_INDEX, :])
        ankle_angle_R = self.joint_angle(pose_norm[:, self.RIGHT_KNEE, :], pose_norm[:, self.RIGHT_ANKLE, :], pose_norm[:, self.RIGHT_FOOT_INDEX, :])
        feats["ankle_angle_L_mean"] = float(ankle_angle_L.mean()); feats["ankle_angle_L_std"] = float(ankle_angle_L.std())
        feats["ankle_angle_L_rom"] = float(ankle_angle_L.max() - ankle_angle_L.min())
        feats["ankle_angle_R_mean"] = float(ankle_angle_R.mean()); feats["ankle_angle_R_std"] = float(ankle_angle_R.std())
        feats["ankle_angle_R_rom"] = float(ankle_angle_R.max() - ankle_angle_R.min())
        feats["knee_angle_rom_asym"] = self.asymmetry(feats["knee_angle_L_rom"], feats["knee_angle_R_rom"])
        feats["hip_angle_rom_asym"] = self.asymmetry(feats["hip_angle_L_rom"], feats["hip_angle_R_rom"])
        feats["ankle_angle_rom_asym"] = self.asymmetry(feats["ankle_angle_L_rom"], feats["ankle_angle_R_rom"])
        left_temporal = self.step_temporal_features(left_ankle_y, fps, cfg.min_step_time)
        right_temporal = self.step_temporal_features(right_ankle_y, fps, cfg.min_step_time)
        for k, v in left_temporal.items(): feats[f"step_L_{k}"] = float(v) if v is not None else np.nan
        for k, v in right_temporal.items(): feats[f"step_R_{k}"] = float(v) if v is not None else np.nan
        if not np.isnan(left_temporal["mean_step_time"]) and not np.isnan(right_temporal["mean_step_time"]):
            feats["step_time_asym"] = self.asymmetry(left_temporal["mean_step_time"], right_temporal["mean_step_time"])
        else: feats["step_time_asym"] = np.nan
        if not np.isnan(left_temporal["cadence"]) and not np.isnan(right_temporal["cadence"]):
            feats["cadence_asym"] = self.asymmetry(left_temporal["cadence"], right_temporal["cadence"])
        else: feats["cadence_asym"] = np.nan
        ankle_L_x = pose_norm[:, self.LEFT_ANKLE, 0]; ankle_R_x = pose_norm[:, self.RIGHT_ANKLE, 0]
        step_width_series = np.abs(ankle_L_x - ankle_R_x); feats["step_width_mean"] = float(step_width_series.mean())
        feats["step_width_std"] = float(step_width_series.std())
        return feats

    def extract_features_from_windows(self, X_windows: np.ndarray, fps: float, cfg: FeatureConfig | None = None) -> pd.DataFrame:
        cfg = cfg or FeatureConfig(); X_windows = np.asarray(X_windows)
        if X_windows.ndim != 4 or X_windows.shape[2] != self.N_JOINTS or X_windows.shape[3] != 3:
            raise ValueError(f"X_windows must have shape (N, T, {self.N_JOINTS}, 3), got {X_windows.shape}")
        N = X_windows.shape[0]; rows = []
        for i in range(N): rows.append(self.compute_window_features(X_windows[i], fps=fps, cfg=cfg))
        return pd.DataFrame(rows)

# --- Main Orchestrator Class ---
class GaitAnalysis:
    def __init__(self, data_path=None, window_seconds=2.0, overlap=0.5, resample_frames=60):
        self.data_path = data_path; self.window_seconds = window_seconds; self.overlap = overlap
        self.resample_frames = resample_frames; self.preprocessing = Preprocessing()
        self.feature_extraction = FeatureExtraction(); self.quality_control = QualityControl()
        self.df_raw = None; self.df_clean = None; self.df_video = None; self.X_windows = None
        self.y_binary = None; self.y_multilabel = None; self.window_ids = None; self.X_clean = None
        self.y_binary_clean = None; self.y_multilabel_clean = None; self.window_ids_clean = None; self.qc_df = None; self.features_df = None

    def load_data(self, path=None):
        if path: self.data_path = path
        if not self.data_path: raise ValueError("No data path provided")
        self.df_raw = pl.read_parquet(self.data_path); return self.df_raw

    def run_preprocessing(self, threshold_pct=5.0):
        self.df_clean, removed_videos = self.preprocessing.filter_videos_by_missing_frames(self.df_raw, threshold_pct=threshold_pct)
        print(f"Removed {len(removed_videos)} videos due to missing frames")
        self.df_video = self.preprocessing.add_pose_column(self.df_clean)
        print(f"Remaining dataframe has {self.df_clean['video_id'].n_unique()} videos")

    def run_feature_extraction(self, fps=60):
        X_windows, y_binary, y_multilabel, window_ids = self.preprocessing.preprocess_gait_sliding_windows(
            self.df_video, window_seconds=self.window_seconds, overlap=self.overlap, resample_frames=self.resample_frames
        )
        self.X_windows, self.y_binary, self.y_multilabel, self.window_ids = X_windows, y_binary, y_multilabel, window_ids
        self.X_clean, self.y_binary_clean, self.y_multilabel_clean, self.window_ids_clean, self.qc_df = self.quality_control.apply_qc_windows(
            self.X_windows, self.y_binary, self.y_multilabel, self.window_ids, fps=fps
        )
        N, T, Jg, C = self.X_clean.shape; X_full = np.full((N, T, 33, 3), np.nan, dtype=np.float32)
        X_full[:, :, self.preprocessing.GAIT_JOINTS, :] = self.X_clean
        cfg = FeatureConfig()
        self.features_df = self.feature_extraction.extract_features_from_windows(X_full, fps=fps, cfg=cfg)
        return self.features_df

    def run_full_pipeline(self, data_path=None, threshold_pct=5.0):
        if data_path: self.load_data(data_path)
        elif not self.df_raw: self.load_data()
        self.run_preprocessing(threshold_pct=threshold_pct)
        self.run_feature_extraction()
        return self.features_df

    def _extract_patient_names(self):
        if not hasattr(self, 'df_video') or 'patient_name' not in self.df_video.columns:
            return [wid.split("_win")[0] for wid in self.window_ids_clean]
        video_to_patient = dict(self.df_video.select(["video_id", "patient_name"]).unique().iter_rows())
        return [video_to_patient.get(wid.split("_win")[0], wid.split("_win")[0]) for wid in self.window_ids_clean]

# --- Baseline Model Class ---
class BaselineModel:
    def __init__(self, gait_analyzer=None):
        self.gait_analyzer = gait_analyzer; self.model = None; self.X_train = None; self.y_train = None
        self.X_test = None; self.y_test = None; self.feature_names = None; self.train_patients = None; self.test_patients = None

    def prepare_data(self, data_path=None, window_seconds=2.0, overlap=0.5, resample_frames=60, threshold_pct=5.0):
        if self.gait_analyzer is None: self.gait_analyzer = GaitAnalysis(data_path, window_seconds, overlap, resample_frames)
        features_df = self.gait_analyzer.run_full_pipeline(data_path=data_path, threshold_pct=threshold_pct)
        patient_names = self.gait_analyzer._extract_patient_names()
        non_feature_cols = ["label_fine", "label_class", "label_id", "binary_label", "movement_type", "side", "source_file"]
        feature_cols = [c for c in features_df.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(features_df[c])]
        X = features_df[feature_cols].copy().fillna(features_df[feature_cols].median())
        y = pd.Series(self.gait_analyzer.y_binary_clean.astype(int))
        self.feature_names = feature_cols; return X, y, patient_names

    def split_data(self, X, y, groups, test_size=0.2, random_state=42):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        self.X_train, self.X_test = X.iloc[train_idx], X.iloc[test_idx]
        self.y_train, self.y_test = y.iloc[train_idx], y.iloc[test_idx]
        self.train_patients = np.array(groups)[train_idx]; self.test_patients = np.array(groups)[test_idx]

    def train_model(self, n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42):
        if self.X_train is None: raise ValueError("Training data not available")
        ratio = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        self.model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=ratio, random_state=random_state, n_jobs=1, use_label_encoder=False)
        self.model.fit(self.X_train, self.y_train); return self.model

    def evaluate_model(self):
        if self.model is None: raise ValueError("Model not trained")
        y_pred = self.model.predict(self.X_test); y_proba = self.model.predict_proba(self.X_test)[:, 1]
        report = classification_report(self.y_test, y_pred, target_names=["normal", "abnormal"], output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred); roc_auc = roc_auc_score(self.y_test, y_proba)
        return {"classification_report": report, "confusion_matrix": cm, "roc_auc": roc_auc, "data_leakage": len(set(self.train_patients) & set(self.test_patients))}

    def save_model(self, output_dir="models"):
        out_dir = Path(output_dir); out_dir.mkdir(exist_ok=True)
        model_path = out_dir / "xgboost_model.bin"; self.model.save_model(model_path)
        with open(out_dir / "feature_names.json", 'w') as f: json.dump(self.feature_names, f, indent=2)
        return model_path

# --- Prediction Class ---
class GaitPredictor:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or MODELS_DIR; self.model = None; self.feature_names = None; self.is_loaded = False
        self.model_path = self.model_dir / "xgboost_model.bin"; self.features_path = self.model_dir / "feature_names.json"

    def load_model(self) -> bool:
        try:
            if not self.model_path.exists() or not self.features_path.exists(): return False
            self.model = xgb.XGBClassifier(); self.model.load_model(self.model_path)
            with open(self.features_path, 'r') as f: self.feature_names = json.load(f)
            self.is_loaded = True; return True
        except Exception as e: logger.error(f"Error loading model: {e}"); return False

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_loaded: raise ValueError("Model not loaded")
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features: raise ValueError(f"Missing required features: {missing_features}")
        X_input = input_data[self.feature_names].copy().fillna(input_data[self.feature_names].median())
        predictions = self.model.predict(X_input); probabilities = self.model.predict_proba(X_input)
        results_df = input_data.copy()
        results_df['prediction'] = ['abnormal' if p == 1 else 'normal' for p in predictions]
        results_df['probability_normal'] = probabilities[:, 0]; results_df['probability_abnormal'] = probabilities[:, 1]
        results_df['confidence'] = np.max(probabilities, axis=1); return results_df


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Gait Analysis Pipeline", page_icon="🚶", layout="wide", initial_sidebar_state="expanded")
    st.markdown("<style>.stAlert > div { padding: 1rem; }</style>", unsafe_allow_html=True)
    st.title("🚶 Production-Grade Gait Analysis Pipeline")
    st.markdown("**Complete pipeline from video to prediction.**")
    st.markdown("---")

    # Initialize session state
    if 'gait_analyzer' not in st.session_state: st.session_state.gait_analyzer = None
    if 'baseline_model' not in st.session_state: st.session_state.baseline_model = None
    if 'model_trained' not in st.session_state: st.session_state.model_trained = False
    if 'model_metrics' not in st.session_state: st.session_state.model_metrics = None
    if 'prediction_results' not in st.session_state: st.session_state.prediction_results = None
    if 'features_df' not in st.session_state: st.session_state.features_df = None
    if 'gait_predictor' not in st.session_state: st.session_state.gait_predictor = GaitPredictor()

    # Sidebar
    with st.sidebar:
        st.header("📋 System Status")
        st.success("✅ App Initialized")
        if st.session_state.gait_predictor.load_model(): st.success("✅ Model Loaded")
        else: st.warning("⚠️ No Model Found")
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🤖 Modeling", "🔮 Predict"])

    with tab1:
        st.subheader("Upload Data and Run Full Pipeline")
        DATA_PATH = PROJECT_ROOT / "filled_gait_data_encoded.parquet"
        if not DATA_PATH.exists():
            st.error(f"❌ Training data not found at `{DATA_PATH}`. Please ensure the file is in the project root.")
            st.stop()
        st.info(f"📄 Using data: `{DATA_PATH.name}`")

        if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running full pipeline... this may take a few minutes."):
                try:
                    st.session_state.gait_analyzer = GaitAnalysis(data_path=DATA_PATH)
                    st.session_state.features_df = st.session_state.gait_analyzer.run_full_pipeline()
                    st.success(f"✅ Pipeline complete! Extracted {len(st.session_state.features_df)} feature rows.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
                    st.exception(e)

        if st.session_state.features_df is not None:
            st.markdown("---")
            st.subheader("📊 Extracted Features")
            st.dataframe(st.session_state.features_df.head())
            csv = st.session_state.features_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Features CSV", data=csv, file_name="gait_features.csv", mime="text/csv")

    with tab2:
        st.subheader("Train a New Baseline Model")
        if st.session_state.features_df is None:
            st.warning("⚠️ No features available. Please run the pipeline in the 'Upload & Process' tab first.")
            st.stop()

        with st.sidebar:
            st.header("⚙️ Model Config")
            n_estimators = st.slider("Trees", 50, 500, 300)
            max_depth = st.slider("Depth", 3, 10, 4)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05)

        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model..."):
                try:
                    st.session_state.baseline_model = BaselineModel(gait_analyzer=st.session_state.gait_analyzer)
                    X, y, groups = st.session_state.baseline_model.prepare_data()
                    st.session_state.baseline_model.split_data(X, y, groups)
                    st.session_state.baseline_model.train_model(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
                    st.session_state.model_metrics = st.session_state.baseline_model.evaluate_model()
                    st.session_state.baseline_model.save_model()
                    st.session_state.model_trained = True
                    st.success("🎉 Model trained and saved successfully!")
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
                    st.exception(e)

        if st.session_state.model_trained:
            st.markdown("---")
            st.subheader("📈 Model Evaluation")
            metrics = st.session_state.model_metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            col2.metric("Accuracy", f"{metrics['classification_report']['accuracy']:.4f}")
            col3.metric("Data Leakage", "No" if metrics['data_leakage'] == 0 else "Yes")
            with st.expander("📋 Detailed Report"):
                st.text(classification_report([0, 1], [0, 1], target_names=["normal", "abnormal"]))

    with tab3:
        st.subheader("Make Predictions")
        if not st.session_state.gait_predictor.load_model():
            st.warning("⚠️ No trained model found. Please train a model in the 'Modeling' tab first.")
            st.stop()

        prediction_option = st.radio("Select data source:", ("Use features from 'Upload & Process' tab", "Upload a features CSV file"))
        input_df = None
        if prediction_option == "Use features from 'Upload & Process' tab":
            if st.session_state.features_df is not None:
                st.success(f"✅ Using {len(st.session_state.features_df)} feature rows.")
                input_df = st.session_state.features_df
            else: st.warning("⚠️ No features found. Run the pipeline first.")
        else:
            uploaded_file = st.file_uploader("Upload a CSV file with features", type=["csv"])
            if uploaded_file:
                try: input_df = pd.read_csv(uploaded_file); st.success(f"✅ Uploaded CSV with {len(input_df)} rows.")
                except Exception as e: st.error(f"❌ Error reading CSV: {e}")

        if input_df is not None and st.button("🔮 Make Predictions", type="primary", use_container_width=True):
            with st.spinner("Predicting..."):
                try:
                    results_df = st.session_state.gait_predictor.predict(input_df)
                    st.session_state.prediction_results = results_df
                    st.success("✅ Predictions complete!")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

        if st.session_state.prediction_results is not None:
            st.markdown("---")
            st.subheader("🔮 Prediction Results")
            results_df = st.session_state.prediction_results
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Abnormal", (results_df['prediction'] == 'abnormal').sum())
            col2.metric("Predicted Normal", (results_df['prediction'] == 'normal').sum())
            col3.metric("Avg Confidence", f"{results_df['confidence'].mean():.2%}")
            st.dataframe(results_df)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

if __name__ == "__main__":
    main()