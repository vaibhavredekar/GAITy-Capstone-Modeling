"""
Feature extraction for gait clips based on gait_preprocessing_pipeline.py

This module assumes:
- MediaPipe Pose with 33 joints
- Coordinates have been normalized by normalize_pose_3d
- Gait clips come from preprocess_gait_dataframe or similar
"""

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d  #for smoothing speeds

import gait_preprocessing_pipeline as gait
from gait_preprocessing_pipeline import (
    add_pose_column,
    normalize_pose_3d,
    preprocess_gait_dataframe,
    N_JOINTS,
    LEFT_HIP, RIGHT_HIP,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HEEL, RIGHT_HEEL,
)

# Extra joint indices (MediaPipe Pose)
LEFT_KNEE, RIGHT_KNEE = 25, 26   # NEW
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# ---------------------------------------------------------------------
# High-level 5-class label scheme
# ---------------------------------------------------------------------

CLASS_MAP = {
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

# fine-grained label -> 5-class name
PATTERN_TO_CLASS: dict[str, str] = {}
for class_name, patterns in CLASS_MAP.items():
    for p in patterns:
        PATTERN_TO_CLASS[p] = class_name

# 5-class name -> integer id (0..4) for XGBoost
CLASS_NAME_TO_ID: dict[str, int] = {
    name: idx for idx, name in enumerate(CLASS_MAP.keys())
}

# ---------------------------------------------------------------------
# Helper: build df_video from raw long-form DataFrame
# ---------------------------------------------------------------------

def build_df_video(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper around gait.add_pose_column.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Long-format MediaPipe dataframe with at least:
        ['source_file', 'frame', 'landmark_id', 'x_norm', 'y_norm', 'z_norm']

    Returns
    -------
    pd.DataFrame
        One row per video, with a 'pose' column of shape (T, 33, 3),
        plus any metadata columns from the first row of each group.
    """
    return add_pose_column(df_raw)


# ---------------------------------------------------------------------
# 1. Features for a single normalized clip
# ---------------------------------------------------------------------
def compute_clip_features(clip: np.ndarray, fps: float) -> dict:
    """
    Compute basic gait features from a single clip.

    Parameters
    ----------
    clip : np.ndarray
        Raw or already-normalized pose for one clip, shape (T, 33, 3).
    fps : float
        Frames per second of this clip.

    Returns
    -------
    dict
        A dict of scalar features for this clip.
    """
    clip = np.asarray(clip)
    if clip.ndim != 3 or clip.shape[1] != N_JOINTS:
        raise ValueError(f"clip must be of shape (T, {N_JOINTS}, 3), got {clip.shape}")

    # 1) Normalize if needed (pelvis should be ~0 after normalization)
    pelvis = (clip[:, LEFT_HIP] + clip[:, RIGHT_HIP]) / 2
    pelvis_mean_norm = np.linalg.norm(pelvis.mean(axis=0))

    if pelvis_mean_norm > 1e-2:
        pose_norm = normalize_pose_3d(clip)
    else:
        pose_norm = clip

    feats: dict[str, float] = {}

    # -------------------
    # Step height features (ankle vertical range)
    # -------------------
    left_ankle_y = pose_norm[:, LEFT_ANKLE, 1]
    right_ankle_y = pose_norm[:, RIGHT_ANKLE, 1]

    feats["step_height_L"] = float(left_ankle_y.max() - left_ankle_y.min())
    feats["step_height_R"] = float(right_ankle_y.max() - right_ankle_y.min())

    # -------------------
    # Step length features (ankle horizontal range)
    # -------------------
    left_ankle_x = pose_norm[:, LEFT_ANKLE, 0]
    right_ankle_x = pose_norm[:, RIGHT_ANKLE, 0]

    feats["step_length_L"] = float(left_ankle_x.max() - left_ankle_x.min())
    feats["step_length_R"] = float(right_ankle_x.max() - right_ankle_x.min())

    # -------------------
    # Pelvic drop (hip height asymmetry)
    # -------------------
    left_hip_y = pose_norm[:, LEFT_HIP, 1]
    right_hip_y = pose_norm[:, RIGHT_HIP, 1]
    pelvis_diff = left_hip_y - right_hip_y

    feats["pelvis_drop_mean"] = float(pelvis_diff.mean())
    feats["pelvis_drop_std"] = float(pelvis_diff.std())

    # -------------------
    # Trunk lean (shoulder horizontal asymmetry)
    # -------------------
    left_sh_x = pose_norm[:, LEFT_SHOULDER, 0]
    right_sh_x = pose_norm[:, RIGHT_SHOULDER, 0]
    trunk_lean = left_sh_x - right_sh_x

    feats["trunk_lean_mean"] = float(trunk_lean.mean())
    feats["trunk_lean_std"] = float(trunk_lean.std())

    # -------------------
    # Heel clearance (vertical heel range)
    # -------------------
    left_heel_y = pose_norm[:, LEFT_HEEL, 1]
    right_heel_y = pose_norm[:, RIGHT_HEEL, 1]

    feats["heel_range_L"] = float(left_heel_y.max() - left_heel_y.min())
    feats["heel_range_R"] = float(right_heel_y.max() - right_heel_y.min())

    # -------------------
    # Simple symmetry indices
    # -------------------
    eps = 1e-6
    hL, hR = feats["step_height_L"], feats["step_height_R"]
    lL, lR = feats["step_length_L"], feats["step_length_R"]

    feats["step_height_symmetry"] = float((hL - hR) / (hL + hR + eps))
    feats["step_length_symmetry"] = float((lL - lR) / (lL + lR + eps))

    # -------------------
    # NEW: Knee motion features using your 3 ideas
    # -------------------
    left_knee_move = moving_and_still_times(pose_norm, LEFT_KNEE, fps)
    right_knee_move = moving_and_still_times(pose_norm, RIGHT_KNEE, fps)

    for k, v in left_knee_move.items():
        feats[f"knee_L_{k}"] = v
    for k, v in right_knee_move.items():
        feats[f"knee_R_{k}"] = v

    # Range of motion for knees in vertical axis (y)
    feats["knee_L_rom_y"] = range_of_motion(pose_norm, LEFT_KNEE, axis="y")["rom_y"]
    feats["knee_R_rom_y"] = range_of_motion(pose_norm, RIGHT_KNEE, axis="y")["rom_y"]

    return feats



# ---------------------------------------------------------------------
# preprocess dataframe → clips → features
# ---------------------------------------------------------------------
def extract_features_from_df_video(
    df_video: pd.DataFrame,
    cycles: int = 1,
    resample_frames: int = 60,
) -> pd.DataFrame:
    """
    Vereinfachte Pipeline:
        df_video (mit pose, fps, gait_pattern, ...) →
        pro Zeile Features aus dem gesamten Clip →
        tabellarisches Feature-DataFrame mit 5-Klassen-Labels.
    """

    feature_rows = []

    for _, row in df_video.iterrows():
        pose = row["pose"]  # (T, 33, 3)

        # ---- fps bestimmen ----
        fps_val = row.get("fps", np.nan)
        if isinstance(fps_val, (int, float, np.floating)) and not np.isnan(fps_val):
            fps = float(fps_val)
        else:
            duration = row.get("duration", np.nan)
            if isinstance(duration, (int, float, np.floating)) and duration > 0:
                fps = pose.shape[0] / float(duration)
            else:
                fps = 30.0  # Fallback

        # ---- erst Features berechnen ----
        feats = compute_clip_features(pose, fps)

        # ---- dann Labels bestimmen (5 Klassen) ----
        fine_label = row.get("gait_pattern", None)
        class_name = PATTERN_TO_CLASS.get(fine_label, None)   # z.B. "gait_anomaly_distal_foot_control_deficit"
        class_id = CLASS_NAME_TO_ID.get(class_name, None)     # 0..4

        feats["label_fine"] = fine_label
        feats["label_class"] = class_name
        feats["label_id"] = class_id

        # optionale Meta-Infos
        feats["movement_type"] = row.get("movement_type", None)
        feats["side"] = row.get("side", None)
        feats["source_file"] = row.get("source_file", None)

        feature_rows.append(feats)

    df_features = pd.DataFrame(feature_rows)
    return df_features

# -----------------------------------------------------------------------
# Adding helper to define the base funcitions for feature calculation
# -----------------------------------------------------------------------
def joint_speed(
    pose_norm: np.ndarray,
    joint_idx: int,
    fps: float,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Compute frame-to-frame speed (3D) of one joint in a normalized clip.

    pose_norm : (T, 33, 3) normalized pose
    joint_idx : index of the joint (e.g. LEFT_KNEE)
    fps       : frames per second
    smooth_sigma : Gaussian smoothing in frames (0 = no smoothing)

    Returns
    -------
    speed : (T-1,) array of speed in normalized units per second
    """
    joint_traj = pose_norm[:, joint_idx, :]  # (T, 3)

    if smooth_sigma and smooth_sigma > 0:
        # smooth along time for each coord
        joint_traj = gaussian_filter1d(joint_traj, sigma=smooth_sigma, axis=0)

    # frame-to-frame displacement
    diffs = np.diff(joint_traj, axis=0)        # (T-1, 3)
    disp = np.linalg.norm(diffs, axis=1)       # (T-1,)
    speed = disp * fps                         # per-second speed
    return speed


def moving_and_still_times(
    pose_norm: np.ndarray,
    joint_idx: int,
    fps: float,
    speed_thresh: float = 0.02,
    smooth_sigma: float = 1.0,
) -> dict:
    """
    How long a joint is moving vs not moving.

    Moving = speed >= speed_thresh
    Not moving = speed < speed_thresh

    Returns times in seconds and fractions of the clip.
    """
    speed = joint_speed(pose_norm, joint_idx, fps, smooth_sigma=smooth_sigma)

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


def range_of_motion(
    pose_norm: np.ndarray,
    joint_idx: int,
    axis: str | None = None,
) -> dict:
    """
    Range of motion (ROM) of a joint.

    axis:
        None  -> 3D ROM (max distance from mean)
        'x'   -> max(x) - min(x)
        'y'   -> max(y) - min(y)
        'z'   -> max(z) - min(z)
    """
    traj = pose_norm[:, joint_idx, :]  # (T, 3)

    if axis is None:
        # 3D ROM relative to mean position
        mean_pos = traj.mean(axis=0)
        dist = np.linalg.norm(traj - mean_pos, axis=1)
        rom_3d = dist.max() - dist.min()
        return {"rom_3d": float(rom_3d)}

    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    idx = axis_to_idx[axis]
    coord = traj[:, idx]
    rom_axis = coord.max() - coord.min()
    return {f"rom_{axis}": float(rom_axis)}


