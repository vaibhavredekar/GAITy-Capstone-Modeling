# %% [markdown]
# # XGBoost: 5 Gait-Anomaly Classes (Clean Notebook)
# 
# Dieses Notebook ist eine **aufgeräumte** Version, die zuverlässig bis Training/Evaluation läuft.
# 
# ## Kerngedanken
# - Labels kommen als **Multihot-Vektor (5)** aus `y_multilabel_clean`.
# - Diese 5 Dimensionen entsprechen bereits den **5 High-Level Klassen** (`gait_anomaly_*`).
# - Wir bauen daraus ein **Single-Label Multiclass** Target (Majority/Tie-Break per Priorität).
# - Split ist **patient-wise** via `GroupShuffleSplit` und wird mit Guards geprüft.
# 

# %%
import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

from modeling.feature_extraction_cleaned import extract_features_from_windows, FeatureConfig
from modeling.Pose_Preprocessing_Pipeline_2 import (
    add_pose_column,
    preprocess_gait_sliding_windows,
    apply_qc_windows,
    GAIT_JOINTS,
)

pd.set_option("display.max_columns", 200)


# %% [markdown]
# ## 1) Load data (Polars)

# %%
df_raw = pl.read_parquet("../data/clean_gait_data.parquet")
print("df_raw:", df_raw.shape)


# %% [markdown]
# ## 2) Build video-level structure + patient mapping
# `add_pose_column` gibt (in deiner Version) **nur** `df_video_pl` zurück.

# %%
df_video_pl = add_pose_column(df_raw)
print("df_video_pl:", df_video_pl.shape)
print("Columns:", df_video_pl.columns)

# Patient mapping MUST be based on the same key used in window_ids.
# In your data, window_ids look like: 'semantic_segmentation_..._DensePose_landmarks_win000_f0-59'
# and df_video_pl contains a matching 'video_id' string column.
video_to_patient = (
    df_video_pl
    .select(["video_id", "patient_name"])
    .unique()
    .to_pandas()
    .set_index("video_id")["patient_name"]
    .to_dict()
)

print("n videos in mapping:", len(video_to_patient))


# %% [markdown]
# ## 3) Sliding windows + QC + feature extraction

# %%
# window params (match your pipeline)
window_seconds = 2.0
overlap = 0.5
resample_frames = 60

X_windows, y_binary, y_multilabel, window_ids = preprocess_gait_sliding_windows(
    df_video_pl,
    window_seconds=window_seconds,
    overlap=overlap,
    resample_frames=resample_frames,
)

fps_effective = resample_frames / window_seconds  # e.g. 30 Hz
X_clean, y_binary_clean, y_multilabel_clean, window_ids_clean, qc_df = apply_qc_windows(
    X_windows, y_binary, y_multilabel, window_ids, fps=fps_effective
)

print("Windows before QC:", X_windows.shape[0])
print("Windows after QC :", X_clean.shape[0])
print("y_multilabel_clean shape:", y_multilabel_clean.shape)
print("Positive windows:", int((y_multilabel_clean.sum(axis=1) > 0).sum()))
print("Positives per label index:", y_multilabel_clean.sum(axis=0))
print("Example window_ids:", window_ids_clean[:3])


# %%
# Map window_id -> patient_name via video_id prefix (before '_win...')
def video_key_from_window_id(wid: str) -> str:
    return wid.split("_win")[0]

window_patient_name = [video_to_patient.get(video_key_from_window_id(w)) for w in window_ids_clean]

import pandas as pd
print("patient_name null count:", pd.isna(window_patient_name).sum(), "/", len(window_patient_name))
print("example patients:", window_patient_name[:5])


# %%
# Convert reduced joints back to full 33 for feature extractor
N, T, Jg, C = X_clean.shape
X_full = np.full((N, T, 33, 3), np.nan, dtype=np.float32)
X_full[:, :, GAIT_JOINTS, :] = X_clean

cfg = FeatureConfig()
df_features = extract_features_from_windows(
    X_windows=X_full,
    fps=fps_effective,
    gait_pattern=None,
    movement_type=None,
    side=None,
    source_file=None,
    cfg=cfg,
)
print("df_features:", df_features.shape)
df_features.head()


# %% [markdown]
# ## 4) Define 5 classes (index order!)
# ⚠️ `y_multilabel_clean` ist ein Multihot-Vektor der Länge 5. **Die Reihenfolge der 5 Indizes muss stimmen.**
# 
# Bei dir existieren Spalten in `df_raw` wie `gait_anomaly_knee_sagittal_plane_abnormality`, etc.
# Falls die Reihenfolge unten nicht passt, ändere einfach die Liste `INDEX_TO_CLASS`.
# 

# %%
# IMPORTANT: Ensure this order matches the multi-hot vector order in y_multilabel_clean.
# If you are unsure, compare with df_raw label columns or your pipeline definition.
INDEX_TO_CLASS = [
    "gait_anomaly_knee_sagittal_plane_abnormality",
    "gait_anomaly_trunk_balance_abnormality",
    "gait_anomaly_spatiotemporal_asymmetry",
    "gait_anomaly_hip_pelvic_control_deficit",
    "gait_anomaly_distal_foot_control_deficit",
]

NORMAL_LABEL = "normal"
CLASS_TO_ID = {c: i for i, c in enumerate(INDEX_TO_CLASS)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}

print("Classes:")
for i, c in enumerate(INDEX_TO_CLASS):
    print(i, c)


# %% [markdown]
# ## 5) Multihot → Single-label (Multiclass)
# Policy:
# - Keine 1 im Multihot → `normal`
# - Eine oder mehrere 1 → wähle per Priorität (Index-Reihenfolge).
# 

# %%
def multihot_to_single_label(row: np.ndarray) -> str:
    idx = np.flatnonzero(row)
    if len(idx) == 0:
        return NORMAL_LABEL
    # tie-break by priority order = first index
    return INDEX_TO_CLASS[int(idx[0])]

coarse_labels = [multihot_to_single_label(r) for r in y_multilabel_clean]

df_mc = df_features.copy()
df_mc["label_coarse"] = coarse_labels
df_mc["patient_name"] = window_patient_name

print("label_coarse counts:")
print(df_mc["label_coarse"].value_counts(dropna=False))


# %% [markdown]
# ## 6) Build training set (drop normal) + X/y/groups
# Wir trainieren das **5-class anomaly-type** Modell, daher droppen wir `normal`.
# 

# %%
# keep only anomaly windows + valid patient
df_train = df_mc[(df_mc["label_coarse"] != NORMAL_LABEL) & (df_mc["patient_name"].notna())].copy()
print("df_train:", df_train.shape)
print("unique patients:", df_train["patient_name"].nunique())
print("label counts:", df_train["label_coarse"].value_counts().to_dict())

# label_id
df_train["label_id"] = df_train["label_coarse"].map(CLASS_TO_ID)
before = len(df_train)
df_train = df_train[df_train["label_id"].notna()].copy()
df_train["label_id"] = df_train["label_id"].astype(int)
print("dropped unmapped:", before - len(df_train))

# feature cols (numeric only)
non_feature_cols = {"patient_name", "label_coarse", "label_id"}
feature_cols = [c for c in df_train.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df_train[c])]
print("n feature cols:", len(feature_cols))

X = df_train[feature_cols].copy()
X = X.fillna(X.median(numeric_only=True))

y = df_train["label_id"].to_numpy()
groups = df_train["patient_name"].to_numpy()

print("len(X):", len(X), "len(y):", len(y), "len(groups):", len(groups))


# %% [markdown]
# ## 7) Patient-wise split (GroupShuffleSplit) + Guards

# %%
import pandas as pd
assert len(X) == len(y) == len(groups), "X/y/groups length mismatch"
assert pd.isna(groups).sum() == 0, "groups contains NaNs"
assert pd.Series(groups).nunique() >= 2, "Need at least 2 patients"

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train patients:", pd.Series(groups[train_idx]).nunique())
print("Test patients :", pd.Series(groups[test_idx]).nunique())
print("Overlap:", set(groups[train_idx]) & set(groups[test_idx]))

print("Train label counts:", pd.Series(y_train).value_counts().sort_index().to_dict())
print("Test label counts :", pd.Series(y_test).value_counts().sort_index().to_dict())


# %% [markdown]
# ## 8) Train XGBoost multiclass + evaluation

# %%
num_class = len(INDEX_TO_CLASS)

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=num_class,
    eval_metric="mlogloss",
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
target_names = [ID_TO_CLASS[i] for i in range(num_class)]

print(classification_report(y_test, y_pred, target_names=target_names))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


# %% [markdown]
# ## 9) Save model + metadata

# %%
from pathlib import Path
import json

out_dir = Path("models")
out_dir.mkdir(exist_ok=True)

model_path = out_dir / "xgboost_gait_5class.bin"
model.save_model(model_path)

meta = {
    "classes": INDEX_TO_CLASS,
    "class_to_id": CLASS_TO_ID,
    "id_to_class": ID_TO_CLASS,
    "feature_cols": feature_cols,
    "normal_label": NORMAL_LABEL,
    "label_policy": "multihot->single via first positive index (priority order)",
}
meta_path = out_dir / "xgboost_gait_5class_metadata.json"
meta_path.write_text(json.dumps(meta, indent=2))

print("Saved model to:", model_path.resolve())
print("Saved metadata to:", meta_path.resolve())



