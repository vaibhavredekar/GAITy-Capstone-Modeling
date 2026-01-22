"""
Important
---------
This module still relies on `gait_preprocessing_pipeline` for:
- `normalize_pose_3d`, `N_JOINTS`, and MediaPipe joint indices.

`Pose_Preprocessing_Pipeline_2.py` should be run first to produce windowed
arrays (N, T, 33, 3). Those windows are then passed to
`extract_features_from_windows`.
"""


import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any, Iterable

from scipy.ndimage import gaussian_filter1d  # smoothing speeds
from scipy.signal import find_peaks          # step event detection

# ---------------------------------------------------------------------
# MediaPipe Pose constants
# ---------------------------------------------------------------------
N_JOINTS = 33

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature extraction.

    Defaults are chosen to match the previous behavior.
    """

    # Smoothing (applied in joint_speed)
    smooth_sigma: float = 1.0

    # Moving vs still threshold (used in moving_and_still_times)
    speed_thresh: float = 0.02

    # Step detection constraint
    min_step_time: float = 0.3

    # Normalize if pose doesn't look normalized
    auto_normalize_if_needed: bool = True

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
# Normalization (copied from Pose_Preprocessing_Pipeline_2.py to keep behavior consistent)
# ---------------------------------------------------------------------

def normalize_pose_3d(pose: np.ndarray) -> np.ndarray:
    """Normalize pose sequence by pelvis centering and torso-length scaling.

    pose: (T, 33, 3)

    Returns
    -------
    (T, 33, 3) normalized pose
    """
    pose = np.asarray(pose, dtype=float)

    pelvis = (pose[:, LEFT_HIP] + pose[:, RIGHT_HIP]) / 2.0               # (T, 3)
    pose_centered = pose - pelvis[:, None, :]                             # (T, 33, 3)
    torso = (pose_centered[:, LEFT_SHOULDER] + pose_centered[:, RIGHT_SHOULDER]) / 2.0

    scale = np.linalg.norm(torso, axis=1).mean()
    if scale == 0 or not np.isfinite(scale):
        raise ValueError("Invalid torso scale during pose normalisation")

    return pose_centered / scale

# ---------------------------------------------------------------------
# Low-level helpers: speed, ROM, asymmetry, angles, temporal features
# ---------------------------------------------------------------------

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
    joint_traj = pose_norm[:, joint_idx, :]  # (T, 3)

    if smooth_sigma and smooth_sigma > 0:
        joint_traj = gaussian_filter1d(joint_traj, sigma=smooth_sigma, axis=0)
    diffs = np.diff(joint_traj, axis=0)
    disp = np.linalg.norm(diffs, axis=1)
    return disp * fps


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


def asymmetry(L: float, R: float, eps: float = 1e-6) -> float:
    """
    Generic left-right asymmetry index: (L - R) / (L + R + eps)
    """
    return float((L - R) / (L + R + eps))


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


def detect_step_events_from_ankle(
    ankle_y: np.ndarray,
    fps: float,
    min_step_time: float = 0.3,
) -> np.ndarray:
    """
    - Use local minima (peaks on -ankle_y) as heel strike proxies
    - min_step_time limits unrealistically fast steps
    """
    ankle_y = np.asarray(ankle_y, dtype=float)
    if ankle_y.size < 3 or fps <= 0:
        return np.array([], dtype=int)
    inv = -ankle_y
    min_distance = max(1, int(min_step_time * fps))
    peaks, _ = find_peaks(inv, distance=min_distance)
    return peaks


def step_temporal_features(
    ankle_y: np.ndarray,
    fps: float,
    min_step_time: float = 0.3,
) -> dict:
    """
    Temporal gait features from one ankle trajectory.
    """
    ankle_y = np.asarray(ankle_y, dtype=float)
    peaks = detect_step_events_from_ankle(ankle_y, fps, min_step_time=min_step_time)

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

# ---------------------------------------------------------------------
# High-level helpers: build df_video from raw long-form DataFrame
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
# Features for a single window
# ---------------------------------------------------------------------

def compute_window_features(window: np.ndarray, fps: float, cfg: FeatureConfig | None = None) -> dict[str, float]:
    """Compute gait features from a single window.

    Parameters
    ----------
    window : np.ndarray
        Pose window, shape (T, 33, 3). Can be raw or already-normalized.
    fps : float
        Effective FPS for this window time axis (after any resampling).
    cfg : FeatureConfig | None
        Feature extraction parameters (defaults preserve old behavior).

    Returns
    -------
    dict
        Scalar feature dictionary for this window.
    """

    cfg = cfg or FeatureConfig()

    window = np.asarray(window)
    if window.ndim != 3 or window.shape[1] != N_JOINTS or window.shape[2] != 3:
        raise ValueError(f"window must be of shape (T, {N_JOINTS}, 3), got {window.shape}")

    # Auto-normalize if the pelvis isn't near 0 (same heuristic as before)
    if cfg.auto_normalize_if_needed:
        pelvis = (window[:, LEFT_HIP] + window[:, RIGHT_HIP]) / 2
        pelvis_mean_norm = np.linalg.norm(pelvis.mean(axis=0))
        pose_norm = normalize_pose_3d(window) if pelvis_mean_norm > 1e-2 else window
    else:
        pose_norm = window

    feats: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Basic spatial features
    # ------------------------------------------------------------------
    # Step height (ankle vertical range)
    left_ankle_y = pose_norm[:, LEFT_ANKLE, 1]
    right_ankle_y = pose_norm[:, RIGHT_ANKLE, 1]

    feats["step_height_L"] = float(left_ankle_y.max() - left_ankle_y.min())
    feats["step_height_R"] = float(right_ankle_y.max() - right_ankle_y.min())

    # Step length (ankle horizontal range)
    left_ankle_x = pose_norm[:, LEFT_ANKLE, 0]
    right_ankle_x = pose_norm[:, RIGHT_ANKLE, 0]

    feats["step_length_L"] = float(left_ankle_x.max() - left_ankle_x.min())
    feats["step_length_R"] = float(right_ankle_x.max() - right_ankle_x.min())

    # Pelvic drop (hip height asymmetry over time)
    left_hip_y = pose_norm[:, LEFT_HIP, 1]
    right_hip_y = pose_norm[:, RIGHT_HIP, 1]
    pelvis_diff = left_hip_y - right_hip_y

    feats["pelvis_drop_mean"] = float(pelvis_diff.mean())
    feats["pelvis_drop_std"] = float(pelvis_diff.std())

    # Trunk lean (horizontal shoulder asymmetry)
    left_sh_x = pose_norm[:, LEFT_SHOULDER, 0]
    right_sh_x = pose_norm[:, RIGHT_SHOULDER, 0]
    trunk_lean = left_sh_x - right_sh_x

    feats["trunk_lean_mean"] = float(trunk_lean.mean())
    feats["trunk_lean_std"] = float(trunk_lean.std())

    # Heel clearance (vertical heel range)
    left_heel_y = pose_norm[:, LEFT_HEEL, 1]
    right_heel_y = pose_norm[:, RIGHT_HEEL, 1]

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
    left_knee_move = moving_and_still_times(pose_norm, LEFT_KNEE, fps)
    right_knee_move = moving_and_still_times(pose_norm, RIGHT_KNEE, fps)

    for k, v in left_knee_move.items():
        feats[f"knee_L_{k}"] = v
    for k, v in right_knee_move.items():
        feats[f"knee_R_{k}"] = v

    feats["knee_L_rom_y"] = range_of_motion(pose_norm, LEFT_KNEE, axis="y")["rom_y"]
    feats["knee_R_rom_y"] = range_of_motion(pose_norm, RIGHT_KNEE, axis="y")["rom_y"]

    # ------------------------------------------------------------------
    # Joint ROM (hip / shoulder / ankle) + asymmetries + stance/swing
    
    # Hip ROM (vertical axis)
    hip_L_rom_y = range_of_motion(pose_norm, LEFT_HIP, axis="y")["rom_y"]
    hip_R_rom_y = range_of_motion(pose_norm, RIGHT_HIP, axis="y")["rom_y"]
    feats["hip_L_rom_y"] = hip_L_rom_y
    feats["hip_R_rom_y"] = hip_R_rom_y

    # Shoulder ROM (horizontal axis, trunk sway / arm swing proxy)
    shoulder_L_rom_x = range_of_motion(pose_norm, LEFT_SHOULDER, axis="x")["rom_x"]
    shoulder_R_rom_x = range_of_motion(pose_norm, RIGHT_SHOULDER, axis="x")["rom_x"]
    feats["shoulder_L_rom_x"] = shoulder_L_rom_x
    feats["shoulder_R_rom_x"] = shoulder_R_rom_x

    # Ankle ROM (vertical axis, dorsiflexion / clearance proxy)
    ankle_L_rom_y = range_of_motion(pose_norm, LEFT_ANKLE, axis="y")["rom_y"]
    ankle_R_rom_y = range_of_motion(pose_norm, RIGHT_ANKLE, axis="y")["rom_y"]
    feats["ankle_L_rom_y"] = ankle_L_rom_y
    feats["ankle_R_rom_y"] = ankle_R_rom_y

    # ROM asymmetries
    feats["knee_rom_asym"] = asymmetry(feats["knee_L_rom_y"], feats["knee_R_rom_y"])
    feats["hip_rom_asym"] = asymmetry(hip_L_rom_y, hip_R_rom_y)
    feats["shoulder_rom_asym"] = asymmetry(shoulder_L_rom_x, shoulder_R_rom_x)
    feats["ankle_rom_asym"] = asymmetry(ankle_L_rom_y, ankle_R_rom_y)

    # Stance / swing ratio (ankle-based proxy)
    ankle_L_move = moving_and_still_times(pose_norm, LEFT_ANKLE, fps)
    ankle_R_move = moving_and_still_times(pose_norm, RIGHT_ANKLE, fps)

    feats["ankle_L_moving_fraction"] = ankle_L_move["moving_fraction"]
    feats["ankle_L_still_fraction"] = ankle_L_move["still_fraction"]
    feats["ankle_R_moving_fraction"] = ankle_R_move["moving_fraction"]
    feats["ankle_R_still_fraction"] = ankle_R_move["still_fraction"]

    stance_ratio_L = ankle_L_move["still_fraction"] / (ankle_L_move["moving_fraction"] + 1e-6)
    stance_ratio_R = ankle_R_move["still_fraction"] / (ankle_R_move["moving_fraction"] + 1e-6)

    feats["stance_ratio_L"] = float(stance_ratio_L)
    feats["stance_ratio_R"] = float(stance_ratio_R)
    feats["stance_ratio_asym"] = asymmetry(stance_ratio_L, stance_ratio_R)

    # ------------------------------------------------------------------
    # Joint angles (hip / knee / ankle) + angle-based ROM/asym
    
    # Knee angles (hip–knee–ankle)
    knee_angle_L = joint_angle(
        pose_norm[:, LEFT_HIP, :],
        pose_norm[:, LEFT_KNEE, :],
        pose_norm[:, LEFT_ANKLE, :],
    )
    knee_angle_R = joint_angle(
        pose_norm[:, RIGHT_HIP, :],
        pose_norm[:, RIGHT_KNEE, :],
        pose_norm[:, RIGHT_ANKLE, :],
    )

    feats["knee_angle_L_mean"] = float(knee_angle_L.mean())
    feats["knee_angle_L_std"] = float(knee_angle_L.std())
    feats["knee_angle_L_rom"] = float(knee_angle_L.max() - knee_angle_L.min())

    feats["knee_angle_R_mean"] = float(knee_angle_R.mean())
    feats["knee_angle_R_std"] = float(knee_angle_R.std())
    feats["knee_angle_R_rom"] = float(knee_angle_R.max() - knee_angle_R.min())

    # Hip angles (shoulder–hip–knee)
    hip_angle_L = joint_angle(
        pose_norm[:, LEFT_SHOULDER, :],
        pose_norm[:, LEFT_HIP, :],
        pose_norm[:, LEFT_KNEE, :],
    )
    hip_angle_R = joint_angle(
        pose_norm[:, RIGHT_SHOULDER, :],
        pose_norm[:, RIGHT_HIP, :],
        pose_norm[:, RIGHT_KNEE, :],
    )

    feats["hip_angle_L_mean"] = float(hip_angle_L.mean())
    feats["hip_angle_L_std"] = float(hip_angle_L.std())
    feats["hip_angle_L_rom"] = float(hip_angle_L.max() - hip_angle_L.min())

    feats["hip_angle_R_mean"] = float(hip_angle_R.mean())
    feats["hip_angle_R_std"] = float(hip_angle_R.std())
    feats["hip_angle_R_rom"] = float(hip_angle_R.max() - hip_angle_R.min())

    # Ankle angles (knee–ankle–foot index)
    ankle_angle_L = joint_angle(
        pose_norm[:, LEFT_KNEE, :],
        pose_norm[:, LEFT_ANKLE, :],
        pose_norm[:, LEFT_FOOT_INDEX, :],
    )
    ankle_angle_R = joint_angle(
        pose_norm[:, RIGHT_KNEE, :],
        pose_norm[:, RIGHT_ANKLE, :],
        pose_norm[:, RIGHT_FOOT_INDEX, :],
    )

    feats["ankle_angle_L_mean"] = float(ankle_angle_L.mean())
    feats["ankle_angle_L_std"] = float(ankle_angle_L.std())
    feats["ankle_angle_L_rom"] = float(ankle_angle_L.max() - ankle_angle_L.min())

    feats["ankle_angle_R_mean"] = float(ankle_angle_R.mean())
    feats["ankle_angle_R_std"] = float(ankle_angle_R.std())
    feats["ankle_angle_R_rom"] = float(ankle_angle_R.max() - ankle_angle_R.min())

    # Angle-based ROM asymmetries
    feats["knee_angle_rom_asym"] = asymmetry(
        feats["knee_angle_L_rom"], feats["knee_angle_R_rom"]
    )
    feats["hip_angle_rom_asym"] = asymmetry(
        feats["hip_angle_L_rom"], feats["hip_angle_R_rom"]
    )
    feats["ankle_angle_rom_asym"] = asymmetry(
        feats["ankle_angle_L_rom"], feats["ankle_angle_R_rom"]
    )

    # ------------------------------------------------------------------
    # Temporal gait features & step width proxy
    
    left_temporal = step_temporal_features(left_ankle_y, fps)
    right_temporal = step_temporal_features(right_ankle_y, fps)

    for k, v in left_temporal.items():
        feats[f"step_L_{k}"] = float(v) if v is not None else np.nan
    for k, v in right_temporal.items():
        feats[f"step_R_{k}"] = float(v) if v is not None else np.nan

    # Asymmetries from temporal features
    if not np.isnan(left_temporal["mean_step_time"]) and not np.isnan(right_temporal["mean_step_time"]):
        feats["step_time_asym"] = asymmetry(
            left_temporal["mean_step_time"], right_temporal["mean_step_time"]
        )
    else:
        feats["step_time_asym"] = np.nan

    if not np.isnan(left_temporal["cadence"]) and not np.isnan(right_temporal["cadence"]):
        feats["cadence_asym"] = asymmetry(
            left_temporal["cadence"], right_temporal["cadence"]
        )
    else:
        feats["cadence_asym"] = np.nan

    # Step width proxy (mediolateral ankle distance)
    ankle_L_x = pose_norm[:, LEFT_ANKLE, 0]
    ankle_R_x = pose_norm[:, RIGHT_ANKLE, 0]
    step_width_series = np.abs(ankle_L_x - ankle_R_x)

    feats["step_width_mean"] = float(step_width_series.mean())
    feats["step_width_std"] = float(step_width_series.std())

    return feats

#---------------------------------------------------------------------
# Batch extraction: windows -> DataFrame (schema identical to old output)
# ---------------------------------------------------------------------


def extract_features_from_windows(
    X_windows: np.ndarray,
    fps: float,
    gait_pattern: Iterable[str] | None = None,
    movement_type: Iterable[str] | None = None,
    side: Iterable[str] | None = None,
    source_file: Iterable[str] | None = None,
    cfg: FeatureConfig | None = None,
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
    cfg : FeatureConfig | None
        Feature extraction configuration.

    Returns
    -------
    pd.DataFrame
        Same columns as the old extract_features_from_df_video output.
    """

    cfg = cfg or FeatureConfig()

    X_windows = np.asarray(X_windows)
    if X_windows.ndim != 4 or X_windows.shape[2] != N_JOINTS or X_windows.shape[3] != 3:
        raise ValueError(
            f"X_windows must have shape (N, T, {N_JOINTS}, 3), got {X_windows.shape}"
        )

    N = X_windows.shape[0]

    def _to_list(x: Iterable[Any] | None) -> list[Any]:
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

    rows: list[dict[str, Any]] = []

    for i in range(N):
        feats = compute_window_features(X_windows[i], fps=fps, cfg=cfg)

        fine_label = gait_pattern_l[i]
        class_name = PATTERN_TO_CLASS.get(fine_label, None)
        class_id = CLASS_NAME_TO_ID.get(class_name, None)

        feats["label_fine"] = fine_label
        feats["label_class"] = class_name
        feats["label_id"] = class_id

        feats["movement_type"] = movement_type_l[i]
        feats["side"] = side_l[i]
        feats["source_file"] = source_file_l[i]

        rows.append(feats)

    return pd.DataFrame(rows)
