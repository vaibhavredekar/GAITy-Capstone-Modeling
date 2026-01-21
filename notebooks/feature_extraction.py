"""
Feature extraction for gait clips based on gait_preprocessing_pipeline.py.

Assumptions:
- MediaPipe Pose with 33 joints
- Coordinates can be normalized with normalize_pose_3d
- Gait clips are time series of shape (T, 33, 3)
"""

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d  # smoothing speeds
from scipy.signal import find_peaks          # step event detection

import gait_preprocessing_pipeline as gait
from gait_preprocessing_pipeline import (
    add_pose_column,
    normalize_pose_3d,
    N_JOINTS,
    LEFT_HIP, RIGHT_HIP,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HEEL, RIGHT_HEEL,
)

# ---------------------------------------------------------------------
# Joint indices (MediaPipe Pose)
# ---------------------------------------------------------------------

LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# ---------------------------------------------------------------------
# High-level 5-class label scheme (Pierre's categories)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ALSO IN PIERRES CODE !!!
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ALSO IN PIERRES CODE !!!
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ALSO IN PIERRES CODE !!!
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

    pose_norm : (T, 33, 3) normalized pose
    joint_idx : joint index (e.g. LEFT_KNEE)
    fps       : frames per second
    smooth_sigma : Gaussian smoothing in frames (0 = no smoothing)

    Returns
    -------
    speed : (T-1,) array in 'normalized units per second'
    """
    joint_traj = pose_norm[:, joint_idx, :]  # (T, 3)

    if smooth_sigma and smooth_sigma > 0:
        joint_traj = gaussian_filter1d(joint_traj, sigma=smooth_sigma, axis=0)

    diffs = np.diff(joint_traj, axis=0)      # (T-1, 3)
    disp = np.linalg.norm(diffs, axis=1)     # (T-1,)
    speed = disp * fps
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

    Moving    = speed >= speed_thresh
    Not moving = speed < speed_thresh

    Returns
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

    axis:
        None  -> 3D ROM (max distance from mean)
        'x'   -> max(x) - min(x)
        'y'   -> max(y) - min(y)
        'z'   -> max(z) - min(z)
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
    Generic left-right asymmetry index:
        (L - R) / (L + R + eps)
    """
    return float((L - R) / (L + R + eps))


def joint_angle(
    p_prox: np.ndarray,
    p_joint: np.ndarray,
    p_dist: np.ndarray,
) -> np.ndarray:
    """
    Joint angle in degrees over time.

    p_prox  : (T, 3) proximal point (e.g. hip for knee angle)
    p_joint : (T, 3) joint point (e.g. knee)
    p_dist  : (T, 3) distal point (e.g. ankle)

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
    Coarse step detection from an ankle vertical trajectory.

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

    Returns
    -------
    dict with:
      - mean_step_time
      - std_step_time
      - cadence (steps/min)
      - mean_stride_time
      - std_stride_time
      - step_time_cv (Coeff. of Variation)
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
# Features for a single clip
# ---------------------------------------------------------------------


def compute_clip_features(clip: np.ndarray, fps: float) -> dict:
    """
    Compute gait features from a single clip.

    Parameters
    ----------
    clip : np.ndarray
        Raw or already-normalized pose for one clip, shape (T, 33, 3).
    fps : float
        Frames per second of this clip.

    Returns
    -------
    dict
        Scalar feature dictionary for this clip.
    """
    clip = np.asarray(clip)
    if clip.ndim != 3 or clip.shape[1] != N_JOINTS:
        raise ValueError(f"clip must be of shape (T, {N_JOINTS}, 3), got {clip.shape}")

    # Normalize if needed (pelvis should be near 0)
    pelvis = (clip[:, LEFT_HIP] + clip[:, RIGHT_HIP]) / 2
    pelvis_mean_norm = np.linalg.norm(pelvis.mean(axis=0))

    if pelvis_mean_norm > 1e-2:
        pose_norm = normalize_pose_3d(clip)
    else:
        pose_norm = clip

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
    # Knee motion: moving vs still, ROM (your original idea)
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Extract features from df_video -ALSO IN PIERRES CODE !!!
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Extract features from df_video -ALSO IN PIERRES CODE !!!
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Extract features from df_video -ALSO IN PIERRES CODE !!!
# ---------------------------------------------------------------------
def extract_features_from_df_video(
    df_video: pd.DataFrame,
    cycles: int = 1,
    resample_frames: int = 60,
) -> pd.DataFrame:
    """
    Simplified pipeline:

        df_video (with pose, fps, gait_pattern, ...) →
        one feature row per video/clip →
        tabular DataFrame with 5-class labels.
    """
    feature_rows = []

    for _, row in df_video.iterrows():
        pose = row["pose"]  # (T, 33, 3)

        # fps: use per-row fps if available, otherwise estimate or fallback
        fps_val = row.get("fps", np.nan)
        if isinstance(fps_val, (int, float, np.floating)) and not np.isnan(fps_val):
            fps = float(fps_val)
        else:
            duration = row.get("duration", np.nan)
            if isinstance(duration, (int, float, np.floating)) and duration > 0:
                fps = pose.shape[0] / float(duration)
            else:
                fps = 30.0  # fallback

        # 1) clip-level features
        feats = compute_clip_features(pose, fps)

        # 2) label mapping: fine label -> 5-class -> id
        fine_label = row.get("gait_pattern", None)
        class_name = PATTERN_TO_CLASS.get(fine_label, None)
        class_id = CLASS_NAME_TO_ID.get(class_name, None)

        feats["label_fine"] = fine_label
        feats["label_class"] = class_name
        feats["label_id"] = class_id

        # 3) optional metadata
        feats["movement_type"] = row.get("movement_type", None)
        feats["side"] = row.get("side", None)
        feats["source_file"] = row.get("source_file", None)

        feature_rows.append(feats)

    df_features = pd.DataFrame(feature_rows)
    return df_features
