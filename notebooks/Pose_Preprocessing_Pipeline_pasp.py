# Pose processing Pipeline:
    # Input: Reads parquet datafile
    # Converts landmarker/joint  coordinates into 3D pose column
    # Runs data cleaning steps
    # Normalises the pose for better analysis and comparability
    # Creates sliding windows per per source_video to increase data
    # Filters Preprocessing output against Quality Control Screen
# %%
# Imports 
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample
import sqlite3

# %%
# Constants

# MediaPipe joint indices (33 joints)
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

NUM_JOINTS = 33 # All Mediapipe landmarkers

GAIT_JOINTS = [
    2, 5,     # eyes (head orientation)
    11, 12,   # shoulders
    23, 24,   # hips
    25, 26,   # knees
    27, 28,   # ankles
    29, 30,   # heels
    31, 32    # foot index
]

GAIT_JOINT_INDEX = {joint_id: i for i, joint_id in enumerate(GAIT_JOINTS)}

#Target labels
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


# %%
# Load data from MediaPipe Pose stored in Parquet file

df = pl.read_parquet('../data/clean_gait_data.parquet')

#%%
# replace the patient seq with the subject number in the patient_name column to match each seq to the video snippet for train/test split
# Load data
summary_df = pl.read_csv(
    "../data/csv/merged_summary_enriched_full_final.csv",
    separator=";",
    truncate_ragged_lines=True
)

# Ensure both join keys are strings
df = df.with_columns(pl.col("patient_name").cast(pl.Utf8))
summary_df = summary_df.with_columns(pl.col("seq").cast(pl.Utf8))

# Join seq → subject
df = (
    df.join(
        summary_df.select(["seq", "subject"]),
        left_on="patient_name",
        right_on="seq",
        how="left"
    )
    .with_columns(
        pl.coalesce([pl.col("subject"), pl.col("patient_name")])
        .alias("patient_name")
    )
    .drop(["subject"])
)

# %%
# 0 Data cleaning: 

#Fill null values with "N"
df = df.with_columns(
    pl.col("movement_type").fill_null("N")
)
# Remove columns with slowmotion videos and fill null values in movement type
df = df.filter(
    pl.col("movement_type") != "SLOWMOTION"
)

#Fill null values with "N" for gait:markers
df = df.with_columns(
    pl.col("gait_markers").fill_null("NA")
)

# %%
# 1. Data cleaning: Remove videos with high percentage of missing frames 

def missing_frames_summary_polars(df: pl.DataFrame, video_col="video_id", frame_col="frame"):
    """
    Compute number and % of missing frames per video.
    
    Returns:
        summary: dict with overall stats
        frame_stats: Polars DataFrame with per-video missing frames info
    """
    # Count unique frames per video
    unique_counts = (
        df.group_by(video_col)
          .agg(pl.col(frame_col).n_unique().alias("num_unique_frames"))
    )

    # Min/max frame per video
    frame_range = (
        df.group_by(video_col)
          .agg([
              pl.col(frame_col).min().alias("min_frame"),
              pl.col(frame_col).max().alias("max_frame")
          ])
    )

    # Join to compute missing frames
    frame_stats = unique_counts.join(frame_range, on=video_col)

    # Compute missing frames and percentage
    frame_stats = frame_stats.with_columns([
        (pl.col("max_frame") - pl.col("min_frame") + 1 - pl.col("num_unique_frames")).alias("num_missing_frames"),
        ((pl.col("max_frame") - pl.col("min_frame") + 1 - pl.col("num_unique_frames")) /
         (pl.col("max_frame") - pl.col("min_frame") + 1) * 100).alias("pct_missing")
    ])

    # Overall summary stats
    summary = {
        "mean_missing": frame_stats["num_missing_frames"].mean(),
        "std_missing": frame_stats["num_missing_frames"].std(),
        "max_missing": frame_stats["num_missing_frames"].max(),
        "mean_pct_missing": frame_stats["pct_missing"].mean()
    }

    # Order by descending number of missing frames
    frame_stats = frame_stats.sort("num_missing_frames", descending=True)

    return summary, frame_stats



def filter_videos_by_missing_frames(df: pl.DataFrame, video_col="video_id", frame_col="frame", threshold_pct=5.0):
    """
    Filter out videos where the percentage of missing frames exceeds threshold_pct.
    
    Args:
        df: Polars DataFrame with per-frame landmarks
        video_col: column identifying videos
        frame_col: column with frame numbers
        threshold_pct: maximum allowed percentage of missing frames (e.g., 5.0)
    
    Returns:
        filtered_df: DataFrame with videos below the missing frame threshold
        removed_videos: list of video_ids that were removed
    """
    
    # Compute missing frames per video
    _, frame_stats = missing_frames_summary_polars(df, video_col=video_col, frame_col=frame_col)
    
    # Find videos to keep
    videos_to_keep = frame_stats.filter(pl.col("pct_missing") <= threshold_pct)[video_col].to_list()
    
    # Filter original dataframe
    filtered_df = df.filter(pl.col(video_col).is_in(videos_to_keep))
    
    # List of removed videos for logging
    removed_videos = frame_stats.filter(pl.col("pct_missing") > threshold_pct)[video_col].to_list()
    
    return filtered_df, removed_videos


# %%
# 2. Data Interpolation Function: for missing frames in order to keep temporal integrity of the video sequence:

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


# %%
# 3. Convert pose coordinates into pose column AND Group rows(frames & landmarkers) by each video
# Output: A new DataFrame where each row corresponds to one video, and includes a pose column.

def add_pose_column(df: pl.DataFrame) -> pl.DataFrame:
    pose_rows = []

    for video_id in df.select("video_id").unique().to_series():
        group = df.filter(pl.col("video_id") == video_id).sort("frame")

        frames = np.sort(group["frame"].unique().to_numpy())
        frame_to_idx = {f: i for i, f in enumerate(frames)}

        T = len(frames)
        pose = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)

        for row in group.iter_rows(named=True):
            f_idx = frame_to_idx[row["frame"]]
            j = int(row["landmark_id"])
            if j >= NUM_JOINTS:
                continue  # skip any bad IDs
            pose[f_idx, j, :] = [row["x_norm"], row["y_norm"], row["z_norm"]]

        # Interpolation across all 33 joints
        pose = interpolate_pose(pose)

        base_row = group.row(0, named=True)
        base_row["pose"] = pose
        pose_rows.append(base_row)

    return pl.DataFrame(pose_rows)

# %%
# 4. Normalize the Pose

def normalize_pose_3d(pose):  
    #pose: (T,J,3)
    
    # Joint indices
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    # pelvis per frame (midpoint of hips)
    pelvis = (pose[:, LEFT_HIP] + pose[:, RIGHT_HIP]) / 2.0   # (T, 3)

    # center all joints around pelvis
    pose_centered = pose - pelvis[:, None, :]           # (T, J, 3)

    # torso vector: midpoint of shoulders (already pelvis-centered)
    torso = (pose_centered[:, LEFT_SHOULDER] +
             pose_centered[:, RIGHT_SHOULDER]) / 2.0          # (T, 3)

    # single scale per sequence: avg torso length
    scale = np.linalg.norm(torso, axis=1).mean() 

    if scale == 0 or not np.isfinite(scale):
        raise ValueError("Invalid torso scale during pose normalisation")


    # All coordinates are now expressed in units of torso_length, making them more comparable across subjects
    pose_scaled = pose_centered / scale 

    return pose_scaled

"""
#Removed:
✔ Body size differences
✔ Camera distance effects
Preserved:
✔ Relative joint motion
✔ Asymmetry
✔ Trunk lean & pelvic drop
✔ Temporal dynamics
"""


# %%
# 5. Extract sliding windows

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

# %%
# 6. Gait Preprocessing Pipeline
# ---------------------------
from scipy.signal import resample
import numpy as np

def preprocess_gait_sliding_windows(
    df_video: pl.DataFrame,
    window_seconds: float = 2.0,
    overlap: float = 0.5,
    resample_frames: int = 60
):
    """
    Preprocess gait data using sliding windows, resampling each window to a fixed frame count.

    Returns:
        X_windows:        (N_windows, resample_frames, len(GAIT_JOINTS), 3)
        y_binary:         (N_windows,)
        y_multilabel:     (N_windows, 5)
        window_ids:       list[str]
    """
    all_windows = []
    all_binary_labels = []
    all_multilabels = []
    all_window_ids = []

    for idx, row in enumerate(df_video.iter_rows(named=True)):
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
        binary_label = LABEL_MAP.get(label_str, 0)

        # Label mapping Multi-label
        multilabel = np.array(
            [int(row.get(col, 0)) for col in ANOMALY_COLS],
            dtype=np.int32
        )

        # Normalize pose
        try:
            pose_norm = normalize_pose_3d(pose)
        except Exception as e:
            print(f"Skipping row {idx} due to normalization error: {e}")
            continue

        # Extract sliding windows
        windows = extract_sliding_windows(
            pose_norm,
            fps=fps,
            window_seconds=window_seconds,
            overlap=overlap
        )

        # Select gait joints only
        windows = [w[:, GAIT_JOINTS, :] for w in windows]

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



# %%
# 7. Quality Controls

def qc_gait_window(window, fps, visualize=False, title=""):
    """
    QC for a single sliding window.
    
    window: (T, J, 3) where J == len(GAIT_JOINTS)
    fps: frames per second for this window
    """

    # ---- Joint index mapping (LOCAL, reduced joints) ----
    GAIT_JOINT_INDEX = {jid: i for i, jid in enumerate(GAIT_JOINTS)}

    L_HIP = GAIT_JOINT_INDEX[23]
    R_HIP = GAIT_JOINT_INDEX[24]
    L_SHOULDER = GAIT_JOINT_INDEX[11]
    L_KNEE = GAIT_JOINT_INDEX[25]
    L_ANKLE = GAIT_JOINT_INDEX[27]

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


# %%
# 8. Apply QC filters on df to remove windows that fail the QC checks
def apply_qc_windows(X_windows, y_binary, y_multilabel, window_ids, fps=60):
    """
    Apply QC to sliding windows and filter out windows that fail QC.
    
    Returns:
        X_clean, y_binary_clean, y_multilabel_clean, window_ids_clean, qc_df
    """
    
    qc_rows = []
    for i, window in enumerate(X_windows):
        qc = qc_gait_window(window, fps=fps, visualize=False, title=f"Window {i}")
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


# %%
# 9. Apply Gait Preprocessing Pipeline

# ---------------------------
# Step 0: Filter videos with too many missing frames
# ---------------------------
df_clean, removed_videos = filter_videos_by_missing_frames(
    df, video_col="video_id", frame_col="frame", threshold_pct=5.0
)

print(f"Removed {len(removed_videos)} videos due to missing frames:")
print(removed_videos)

print(f"Remaining dataframe has {df_clean['video_id'].n_unique()} videos")

# %%
# ---------------------------
# Step 1: Convert pose coordinates into pose column with interpolation
# ---------------------------
df_video = add_pose_column(df_clean)
print(f"Pose column added. Example shape: {df_video['pose'][0].shape}")

# ---------------------------
# Step 2: Preprocess gait dataframe using sliding windows
# ---------------------------
X_windows, y_binary, y_multilabel, window_ids = preprocess_gait_sliding_windows(
    df_video,
    window_seconds=2.0,
    overlap=0.5,
    resample_frames=60
)

print(f"X_windows shape: {X_windows.shape}")
print(f"y_binary shape: {y_binary.shape}")
print(f"y_multilabel shape: {y_multilabel.shape}")  # should be (N_windows, 5)
print(f"First 5 window IDs: {window_ids[:5]}")

# Output: All processed windows


# %%
# Step 3: Apply QC filtering to sliding windows
X_clean, y_binary_clean, y_multilabel_clean, window_ids_clean, qc_df = apply_qc_windows(
    X_windows,
    y_binary,
    y_multilabel,
    window_ids,
    fps=60  # adjust if your videos/windows have a different FPS
)

#Output: Only windows that passed the QC criteria

# Quick sanity print
print(f"QC-clean windows: {X_clean.shape[0]}")
print(f"QC-clean binary_labels: {y_binary_clean.shape[0]}")
print(f"QC-clean multi_labels: {y_multilabel_clean.shape[0]}") # should be (N_clean, 5)
print(f"First 5 clean window IDs: {window_ids_clean[:5]}")

# Optional: check QC summary
print("Total QC-fail windows:", qc_df["qc_fail"].sum())
print("QC fail count by binary_label:\n", qc_df.groupby("binary_label")["qc_fail"].sum())




