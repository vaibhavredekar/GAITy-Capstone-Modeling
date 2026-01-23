# ============================================================
# GAIT PREPROCESSING PIPELINE (MediaPipe Pose)
# ============================================================

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample
from scipy.ndimage import gaussian_filter1d

# ============================================================
# CONSTANTS
# ============================================================

N_JOINTS = 33

LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HEEL, RIGHT_HEEL = 29, 30

LABEL_MAP = {
    "normal gait": 0,
    "abnormal gait": 1
}

# ============================================================
# 1. BUILD POSE TENSOR FROM LONG-FORM DATAFRAME
# ============================================================

def add_pose_column(df):
    """
    Converts long-format MediaPipe dataframe into video-level dataframe
    with a pose column of shape (T, 33, 3)
    """

    pose_rows = []

    for source_file, group in df.groupby("source_file"):
        T = group["frame"].nunique()
        pose = np.zeros((T, N_JOINTS, 3), dtype=np.float32)

        for _, r in group.iterrows():
            f = int(r.frame)
            j = int(r.landmark_id)
            pose[f, j] = [r.x_norm, r.y_norm, r.z_norm]

        row = group.iloc[0].copy()
        row["pose"] = pose
        pose_rows.append(row)

    return pd.DataFrame(pose_rows)

# ============================================================
# 2. POSE NORMALIZATION
# ============================================================

def normalize_pose_3d(pose):
    """
    Pelvis-centered, torso-scaled normalization
    pose: (T, 33, 3)
    """
    pelvis = (pose[:, LEFT_HIP] + pose[:, RIGHT_HIP]) / 2
    pose_centered = pose - pelvis[:, None, :]

    torso = (pose_centered[:, LEFT_SHOULDER] + pose_centered[:, RIGHT_SHOULDER]) / 2
    scale = np.linalg.norm(torso, axis=1).mean()

    pose_scaled = pose_centered / scale
    return pose_scaled

# ============================================================
# 3. Joint Selection 
# ============================================================

GAIT_JOINTS = [
    2, 5,     # eyes
    11, 12,   # shoulders
    23, 24,   # hips
    25, 26,   # knees
    27, 28,   # ankles
    29, 30,   # heels
    31, 32    # foot index
]

def select_joints(pose, joint_indices):
    pose = np.asarray(pose)
    assert pose.ndim == 3 and pose.shape[1] >= max(joint_indices) + 1
    return pose[:, joint_indices, :]


# ============================================================
# 4. HEEL STRIKE DETECTION (FPS-AWARE)
# ============================================================

def find_heel_strikes(
    pose,
    foot="left",
    fps=30,
    min_time_between_steps=0.5,
    smooth_sigma=1
):
    heel_idx = LEFT_HEEL if foot == "left" else RIGHT_HEEL
    y = pose[:, heel_idx, 1]

    y_smooth = gaussian_filter1d(y, sigma=smooth_sigma)
    min_distance_frames = int(min_time_between_steps * fps)

    peaks, _ = find_peaks(-y_smooth, distance=min_distance_frames)
    return peaks

# ============================================================
# 5. GAIT-CYCLE CLIP EXTRACTION
# ============================================================

def extract_gait_cycle_clips(
    pose,
    fps,
    cycles=1,
    min_time_between_steps=0.5,
    min_frames=40,
    max_frames=150,
    resample_frames=60
):
    """
    Returns a list of clips shaped (resample_frames, 33, 3)
    """

    clips = []

    left_events = find_heel_strikes(
        pose, "left", fps, min_time_between_steps
    )
    right_events = find_heel_strikes(
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

# ============================================================
# 6. FULL PREPROCESSING PIPELINE
# ============================================================

def preprocess_gait_dataframe(
    df_video,
    cycles=1,
    resample_frames=60
):
    """
    Input dataframe must contain:
      - pose: (T,33,3)
      - fps
      - dataset (label)
    """

    all_clips = []
    all_labels = []

    for _, row in df_video.iterrows():
        pose = row["pose"]
        fps = row["fps"]
        label = LABEL_MAP[row["dataset"]]

        pose_norm = normalize_pose_3d(pose)

        pose = select_joints(pose, GAIT_JOINTS)

        clips = extract_gait_cycle_clips(
            pose_norm,
            fps=fps,
            cycles=cycles,
            resample_frames=resample_frames
        )

        all_clips.extend(clips)
        all_labels.extend([label] * len(clips))

    X = np.array(all_clips)    # (N_clips, T, 33, 3)
    y = np.array(all_labels)   # (N_clips,)

    return X, y

# ============================================================
# 7. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    # Load raw dataframe (example)
    # df_raw = pd.read_csv("your_raw_mediapipe_data.csv")

    # Step 1: Build pose tensors
    # df_video = add_pose_column(df_raw)

    # Step 2: Preprocess into gait clips
    # X_clips, y_labels = preprocess_gait_dataframe(
    #     df_video,
    #     cycles=1,
    #     resample_frames=60
    # )

    # print("Clips shape:", X_clips.shape)
    # print("Labels shape:", y_labels.shape)

    pass

