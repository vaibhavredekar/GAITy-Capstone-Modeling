# ------------------------------------------------------------------------------
# # XGBoost from feature extraction
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# The script builds a binary XGBoost classifier that predicts normal vs. abnormal gait using hand-engineered numerical features extracted from video-based gait data.
# Raw parquet gait data are converted into per-video feature vectors, cleaned, and split into train/test sets using grouped splitting by patient to avoid data leakage, after which an XGBoost model is trained with class-imbalance handling and evaluated via precision/recall, confusion matrix, and ROC-AUC.
# The model is trained on extracted gait features (not raw video) derived from filled_gait_data_encoded.parquet, with the target being the binary_label indicating normal or abnormal movement.
# ------------------------------------------------------------------------------


# 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb
from feature_extraction_cleaned import extract_features_from_windows, FeatureConfig
from Pose_Preprocessing_Pipeline_2 import add_pose_column, preprocess_gait_sliding_windows, apply_qc_windows, GAIT_JOINTS


# 

df_video = pd.read_parquet("../data/clean_gait_data.parquet")
df_video.head()


#
import sys
from pathlib import Path

# Make sure project root is on path (adjust if needed)
sys.path.append(str(Path("..").resolve()))

import pandas as pd
import polars as pl


# 
df_raw = pd.read_parquet("../data/clean_gait_data.parquet")
df_raw.head()


# 
# mappen file_path to source_file
if "source_file" not in df_raw.columns and "file_path" in df_raw.columns:
    df_raw = df_raw.rename(columns={"file_path": "source_file"})


# 
df_raw = pd.read_parquet("../data/clean_gait_data.parquet")

# rename file_path -> source_file 
if "source_file" not in df_raw.columns and "file_path" in df_raw.columns:
    df_raw = df_raw.rename(columns={"file_path": "source_file"})

# Frames integer 
df_raw["frame"] = df_raw["frame"].astype(int)

# FRAMES PRO VIDEO START AT 0
# add_pose_column assumes to start frames at 0
df_raw["frame"] = df_raw["frame"] - df_raw.groupby("video_id")["frame"].transform("min")


# 
window_patient_name = []


# 
# Build df_video using Pose_Preprocessing_Pipeline_2 (polars)

# a) Load parquet 
df_raw_pd = pd.read_parquet("../data/filled_gait_data_encoded.parquet")

# file_path -> source_file if needed 
if "source_file" not in df_raw_pd.columns and "file_path" in df_raw_pd.columns:
    df_raw_pd = df_raw_pd.rename(columns={"file_path": "source_file"})

df_raw_pl = pl.from_pandas(df_raw_pd)

# IMPORTANT: pipeline2 expects these columns to exist:
# ['video_id', 'frame', 'landmark_id', 'x_norm', 'y_norm', 'z_norm']
# If your parquet groups by 'source_file' but doesn't have video_id, create one:
if "video_id" not in df_raw_pl.columns and "source_file" in df_raw_pl.columns:
    df_raw_pl = df_raw_pl.with_columns(pl.col("source_file").alias("video_id"))

# b) Build one-row-per-video table with a 'pose' column (T, 33, 3)
df_video_pl = add_pose_column(df_raw_pl)
df_video_pl.shape


# 
video_to_patient = dict(
    df_video_pl.select(["video_id", "patient_name"]).unique().iter_rows()
)

video_to_source = dict(
    df_video_pl.select(["video_id", "source_file"]).unique().iter_rows()
)


# 
def video_id_from_window_id(wid: str) -> str:
    return wid.split("_win")[0]


# 
import os
import polars as pl

# stable per-video id (file stem)
df_video_pl = df_video_pl.with_columns(
    pl.col("source_file")
      .map_elements(lambda s: os.path.splitext(os.path.basename(s))[0] if s is not None else None,
                    return_dtype=pl.Utf8)
      .alias("video_id")
)


# 
#Sliding windows + QC + feature extraction (window-based)

# Parameters (match Pose_Preprocessing_Pipeline_2)
window_seconds = 2.0
overlap = 0.5
resample_frames = 60

# 1) Windowing (returns reduced-joint windows: (N, resample_frames, len(GAIT_JOINTS), 3))
X_windows, y_binary, y_multilabel, window_ids = preprocess_gait_sliding_windows(
    df_video_pl,
    window_seconds=window_seconds,
    overlap=overlap,
    resample_frames=resample_frames,
)

# 2) QC (use the *effective* fps of the resampled time axis)
fps_effective = resample_frames / window_seconds  # e.g. 60/2 = 30

X_clean, y_binary_clean, y_multilabel_clean, window_ids_clean, qc_df = apply_qc_windows(
    X_windows, y_binary, y_multilabel, window_ids, fps=fps_effective
)

print("Windows before QC:", X_windows.shape[0])
print("Windows after QC:", X_clean.shape[0])

window_patient_name = []
window_source_file = []

for wid in window_ids_clean:
    vid = video_id_from_window_id(wid)
    window_patient_name.append(video_to_patient.get(vid))
    window_source_file.append(video_to_source.get(vid))

def video_id_from_window_id(wid: str) -> str:
    return wid.split("_win")[0]

window_patient_name = [video_to_patient.get(video_id_from_window_id(w)) for w in window_ids_clean]
window_source_file  = [video_to_source.get(video_id_from_window_id(w))  for w in window_ids_clean]


# 3) Convert reduced-joint windows back to full 33 joints (so feature code stays unchanged)
#    (All joints not in GAIT_JOINTS are NaN; features only use hips/shoulders/knees/ankles/heels/foot_index)
N, T, Jg, C = X_clean.shape
X_full = np.full((N, T, 33, 3), np.nan, dtype=np.float32)
X_full[:, :, GAIT_JOINTS, :] = X_clean

# 4) Extract features
cfg = FeatureConfig()  # defaults match prior behavior
df_features = extract_features_from_windows(
    X_windows=X_full,
    fps=fps_effective,
    gait_pattern=None,     # optional: per-window list of fine labels
    movement_type=None,
    side=None,
    source_file=None,
    cfg=cfg,
)

df_features.shape, df_features.head()


# 
def video_id_from_window_id(wid: str) -> str:
    return wid.split("_win")[0]

example_vids = [video_id_from_window_id(w) for w in window_ids_clean[:10]]
print("example vids:", example_vids)

# how many match mapping?
matches = sum(v in video_to_source for v in example_vids)
print("matches among first 10:", matches)

# show one miss + a few mapping keys
print("first mapping keys:", list(video_to_source.keys())[:10])


# 
df_video.columns


# 
# Sanity checks (labels + class distribution)
print(df_features.shape)
print(df_features.columns)

print(df_features[["label_fine", "label_class", "label_id"]].head())
print(df_features["label_id"].value_counts(dropna=False))


# 
df_features.columns


# ------------------------------------------------------------------------------
# ---
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# # first binary (abnormal or normal gait) baseline
# ------------------------------------------------------------------------------


# 
# df_features rows must align with y_binary_clean and window_patient_name
assert len(df_features) == len(y_binary_clean) == len(window_patient_name)

df_bin = df_features.copy()

# Use the pipeline's binary labels (window-based)
df_bin["binary_label"] = y_binary_clean.astype(int)

# If you still want patient_name for splits/inspection, keep it separate...
groups = np.array(window_patient_name)

print("Spalten:", df_bin.columns.tolist())
print("Anzahl Samples:", df_bin.shape[0])
print("Binary-Label-Verteilung:\n", pd.Series(df_bin["binary_label"]).value_counts())
print("Anzahl Patienten:", len(np.unique(groups)))


# 
df_debug = df_bin.copy()
df_debug["patient_name"] = window_patient_name
df_debug[["patient_name", "binary_label"]].head()


# 
from sklearn.model_selection import GroupShuffleSplit

# Build X/y ONCE (your preferred way: numeric-only + drop known non-features)
non_feature_cols = [
    "label_fine", "label_class", "label_id",
    "binary_label",
    "movement_type", "side", "source_file"
]
feature_cols = [
    c for c in df_bin.columns
    if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df_bin[c])
]

X = df_bin[feature_cols].copy()
X = X.fillna(X.median(numeric_only=True))  # median impute
y = df_bin["binary_label"].astype(int)

assert len(X) == len(y) == len(groups)

print("X, y shape:", X.shape, y.shape)


# 
# Group split (no leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


# 
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

ratio = (y_train == 0).sum() / (y_train == 1).sum()

xgb_bin = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    random_state=42,
    n_jobs=-1,
)

xgb_bin.fit(X_train, y_train)

y_pred = xgb_bin.predict(X_test)
y_proba = xgb_bin.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["normal", "abnormal"]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# ------------------------------------------------------------------------------
# ## Train/Test leakage check
# ------------------------------------------------------------------------------


# 
# Leakage check
train_pats = set(groups[train_idx])
test_pats  = set(groups[test_idx])
print("gemeinsame Patienten in Train & Test:", len(train_pats & test_pats))


# 
df_leak = pd.DataFrame({
    "binary_label": df_bin["binary_label"].values,
    "patient_name": groups,   # <-- no .values
})


# unique patients per label
print(df_leak.groupby("binary_label")["patient_name"].nunique())

# patients that have BOTH labels (important to know)
patient_label_counts = df_leak.groupby("patient_name")["binary_label"].nunique()
print("Patients with both labels:", (patient_label_counts > 1).sum())


# 
assert "window_source_file" in globals(), "window_source_file not defined — run metadata cell first"


# 
sources = pd.Series(window_source_file, index=df_bin.index, name="source_file")
train_src = set(sources.iloc[train_idx])
test_src  = set(sources.iloc[test_idx])

print("Overlap source_file:", len(train_src & test_src))


# ------------------------------------------------------------------------------
# ---
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# model output
# ------------------------------------------------------------------------------


# 
from pathlib import Path

out_dir = Path("models")
out_dir.mkdir(exist_ok=True)

# Option A (simple): native XGBoost serialization
xgb_bin.save_model(out_dir / "xgboost_model.bin")

print("Saved to:", (out_dir / "xgboost_model.bin").resolve())


#
from pathlib import Path

out_dir = Path("models")
out_dir.mkdir(exist_ok=True)

booster = xgb_bin.get_booster()
raw_bytes = booster.save_raw()

out_path = (out_dir / "xgboost_model.ubj").resolve()
out_path.write_bytes(raw_bytes)

print("Saved raw binary to:", out_path)


# ------------------------------------------------------------------------------
# ---
# ## graphics
# ------------------------------------------------------------------------------


# 
# ---confusion matrix copied from above ---
# rows = actual [normal, abnormal], cols = predicted [normal, abnormal]
cm = np.array([[3067,  19],
               [   6, 314]])

class_names = ["Normal", "Abnormal"]


def plot_confusion_matrix_stakeholder(cm, class_names=("Normal", "Abnormal"), title="Model decisions (test set)"):
    """
    Stakeholder-friendly confusion matrix:
    - shows counts
    - shows row-wise percentages (i.e., within each actual class)
    - adds plain English labels inside cells
    """
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = cm / np.maximum(row_sums, 1)  # avoid divide-by-zero

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(cm)  # uses matplotlib default colormap (no explicit colors)

    # Axes labels
    ax.set_xticks([0, 1], labels=class_names)
    ax.set_yticks([0, 1], labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    # Add a colorbar for visual weight (optional but nice)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of samples", rotation=90)

    # Cell annotations
    # Choose text color based on background intensity
    thresh = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = row_pct[i, j] * 100

            if i == 0 and j == 0:
                label = "Correct normal"
            elif i == 0 and j == 1:
                label = "False alarm\n(normal→abnormal)"
            elif i == 1 and j == 0:
                label = "Missed abnormal\n(abnormal→normal)"
            else:
                label = "Correct abnormal"

            text_color = "white" if count > thresh else "black"
            ax.text(
                j, i,
                f"{label}\n{count} ({pct:.1f}%)",
                ha="center", va="center",
                color=text_color,
                fontsize=10
            )

    # Add subtle gridlines
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    return fig, ax


def plot_out_of_100(cm, title="What happens out of 100 cases?"):
    """
    Converts performance into "out of 100" for each actual class.
    Stakeholders love this because it's instantly interpretable.
    """
    cm = np.asarray(cm)
    # Row-wise rates: actual normal row -> predicted normal/abnormal; actual abnormal row -> predicted normal/abnormal
    row_sums = cm.sum(axis=1)
    normal_row = cm[0] / max(row_sums[0], 1)
    abnormal_row = cm[1] / max(row_sums[1], 1)

    # Convert to "out of 100"
    out100_normal_correct = normal_row[0] * 100
    out100_normal_false_alarm = normal_row[1] * 100

    out100_abnormal_missed = abnormal_row[0] * 100
    out100_abnormal_correct = abnormal_row[1] * 100

    labels = ["Actual Normal (per 100)", "Actual Abnormal (per 100)"]
    correct = [out100_normal_correct, out100_abnormal_correct]
    errors = [out100_normal_false_alarm, out100_abnormal_missed]

    fig, ax = plt.subplots(figsize=(8, 4.8))

    x = np.arange(len(labels))
    ax.bar(x, correct, label="Correct")         # default colors
    ax.bar(x, errors, bottom=correct, label="Incorrect")

    ax.set_title(title)
    ax.set_ylabel("Cases (out of 100)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")

    # Add text on bars
    for i in range(len(labels)):
        ax.text(x[i], correct[i] / 2, f"{correct[i]:.1f}", ha="center", va="center", fontsize=11)
        ax.text(x[i], correct[i] + errors[i] / 2, f"{errors[i]:.1f}", ha="center", va="center", fontsize=11)

    plt.tight_layout()
    return fig, ax


# --- Example usage ---
plot_confusion_matrix_stakeholder(cm, class_names=class_names, title="Gait classifier outcomes")
plt.show()

plot_out_of_100(cm, title="Gait classifier: results per 100 cases")
plt.show()


# 
# Confusion matrix
# rows = actual [normal, abnormal]
# cols = predicted [normal, abnormal]
cm = np.array([[3067,  19],
               [   6, 314]])

def plot_out_of_100_pies(cm, title="Gait classifier performance (out of 100 cases)"):
    cm = np.asarray(cm)

    # Row-wise normalization
    normal_row = cm[0] / cm[0].sum()
    abnormal_row = cm[1] / cm[1].sum()

    # Convert to percentages
    normal_correct = normal_row[0] * 100
    normal_false_alarm = normal_row[1] * 100

    abnormal_missed = abnormal_row[0] * 100
    abnormal_correct = abnormal_row[1] * 100

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle(title, fontsize=14)

    # --- Pie 1: Actual Normal ---
    axes[0].pie(
        [normal_correct, normal_false_alarm],
        labels=[
            f"Correctly identified\n{normal_correct:.1f}%",
            f"False alarm\n{normal_false_alarm:.1f}%"
        ],
        autopct=None,
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"}
    )
    axes[0].set_title("Actual normal gait")

    # --- Pie 2: Actual Abnormal ---
    axes[1].pie(
        [abnormal_correct, abnormal_missed],
        labels=[
            f"Correctly detected\n{abnormal_correct:.1f}%",
            f"Missed cases\n{abnormal_missed:.1f}%"
        ],
        autopct=None,
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"}
    )
    axes[1].set_title("Actual abnormal gait")

    plt.tight_layout()
    plt.show()


# --- Example usage ---
plot_out_of_100_pies(cm)
