"""
multiclass_modeling.py

This script trains a 5-class XGBoost classifier for specific gait anomaly types.
It follows the same robust structure as baseline_modeling.py but is adapted for multi-class tasks.

Usage:
    python multiclass_modeling.py
"""

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json
import sys

# --- Import our custom pipeline components ---
# This assumes preprocessing_n_feature_engineering.py is in the same directory or PYTHONPATH
try:
    from preprocessing_n_feature_engineering import (
        GaitAnalysis, 
        FeatureExtraction, 
        FeatureConfig,
        Preprocessing,
        QualityControl
    )
except ImportError as e:
    print("Error: Could not import pipeline components.")
    print("Please ensure 'preprocessing_n_feature_engineering.py' is in the same directory or your PYTHONPATH.")
    print(f"Details: {e}")
    sys.exit(1)


# --- Class Definitions for Multi-Class Modeling ---

class MulticlassModel:
    """
    Class for training and evaluating a multi-class XGBoost model for gait anomaly types.
    """
    
    def __init__(self, gait_analyzer=None):
        """
        Initialize the multi-class model.
        
        Args:
            gait_analyzer: Optional GaitAnalysis instance with preprocessed data.
        """
        self.gait_analyzer = gait_analyzer
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.train_patients = None
        self.test_patients = None
        
        # Multi-class specific attributes
        self.index_to_class = None
        self.class_to_id = None
        self.id_to_class = None
        self.normal_label = "normal"

    def prepare_data(self, data_path=None, window_seconds=2.0, overlap=0.5, 
                     resample_frames=60, threshold_pct=5.0):
        """
        Prepare data for multi-class modeling.
        
        This includes loading data, preprocessing, feature extraction, and the crucial
        step of converting multi-hot labels to single labels.
        """
        # --- Step 1: Run the full pipeline to get features and multi-hot labels ---
        if self.gait_analyzer is None:
            self.gait_analyzer = GaitAnalysis(
                data_path=data_path,
                window_seconds=window_seconds,
                overlap=overlap,
                resample_frames=resample_frames
            )
        
        features_df = self.gait_analyzer.run_full_pipeline(
            data_path=data_path,
            threshold_pct=threshold_pct
        )
        
        # --- Step 2: Define class mappings (from notebook) ---
        # IMPORTANT: This order MUST match the multi-hot vector order in y_multilabel_clean
        self.index_to_class = [
            "gait_anomaly_knee_sagittal_plane_abnormality",
            "gait_anomaly_trunk_balance_abnormality",
            "gait_anomaly_spatiotemporal_asymmetry",
            "gait_anomaly_hip_pelvic_control_deficit",
            "gait_anomaly_distal_foot_control_deficit",
        ]
        self.class_to_id = {c: i for i, c in enumerate(self.index_to_class)}
        self.id_to_class = {i: c for c, i in self.class_to_id.items()}
        
        # --- Step 3: Map window IDs to patient names ---
        # Create a mapping from video_id to patient_name from the preprocessed video dataframe
        if not hasattr(self.gait_analyzer, 'df_video') or self.gait_analyzer.df_video is None:
            raise ValueError("df_video not found in gait_analyzer. Cannot map patients.")
        
        video_to_patient = (
            self.gait_analyzer.df_video
            .select(["video_id", "patient_name"])
            .unique()
            .to_pandas()
            .set_index("video_id")["patient_name"]
            .to_dict()
        )
        
        def video_key_from_window_id(wid: str) -> str:
            """Extracts the video key from a window_id string."""
            return wid.split("_win")[0]

        window_patient_name = [
            video_to_patient.get(video_key_from_window_id(w)) 
            for w in self.gait_analyzer.window_ids_clean
        ]
        
        # --- Step 4: Convert multi-hot labels to single labels ---
        def multihot_to_single_label(row: np.ndarray) -> str:
            """Converts a multi-hot label row to a single label string."""
            idx = np.flatnonzero(row)
            if len(idx) == 0:
                return self.normal_label
            # Tie-break by priority order (the first index)
            return self.index_to_class[int(idx[0])]

        y_multilabel_clean = self.gait_analyzer.y_multilabel_clean
        coarse_labels = [multihot_to_single_label(r) for r in y_multilabel_clean]

        # --- Step 5: Create the final DataFrame for modeling ---
        df_mc = features_df.copy()
        df_mc["label_coarse"] = coarse_labels
        df_mc["patient_name"] = window_patient_name

        # --- Step 6: Filter for training and create X, y, groups ---
        # We train the 5-class model on anomaly types only, so we drop 'normal' labels.
        df_train = df_mc[(df_mc["label_coarse"] != self.normal_label) & (df_mc["patient_name"].notna())].copy()
        
        if df_train.empty:
            raise ValueError("No data available for training after filtering. Check data and labels.")
            
        df_train["label_id"] = df_train["label_coarse"].map(self.class_to_id)
        df_train = df_train[df_train["label_id"].notna()].copy()
        df_train["label_id"] = df_train["label_id"].astype(int)

        # Define feature columns
        non_feature_cols = {"patient_name", "label_coarse", "label_id"}
        feature_cols = [
            c for c in df_train.columns 
            if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df_train[c])
        ]
        
        X = df_train[feature_cols].copy()
        X = X.fillna(X.median(numeric_only=True))
        y = df_train["label_id"].to_numpy()
        groups = df_train["patient_name"].to_numpy()
        
        self.feature_names = feature_cols
        
        print(f"Prepared {len(X)} samples for {len(self.index_to_class)} classes.")
        print("Class distribution:")
        print(pd.Series(y).value_counts().sort_index().to_dict())
        
        return X, y, groups

    def split_data(self, X, y, groups, test_size=0.2, random_state=42):
        """
        Split data into train and test sets using group-based splitting.
        """
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        self.X_train, self.X_test = X.iloc[train_idx], X.iloc[test_idx]
        self.y_train, self.y_test = y[train_idx], y[test_idx]
        self.train_patients = groups[train_idx]
        self.test_patients = groups[test_idx]
        
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Data leakage check: {len(set(self.train_patients) & set(self.test_patients))} shared patients.")

    def train_model(self, n_estimators=500, max_depth=4, learning_rate=0.05, 
                   subsample=0.8, colsample_bytree=0.8, random_state=42):
        """
        Train the multi-class XGBoost model.
        """
        if self.X_train is None:
            raise ValueError("Training data not available. Run prepare_data() first.")
        
        num_class = len(self.index_to_class)
        
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_class,
            eval_metric="mlogloss",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("Multi-class model training complete.")
        return self.model

    # def evaluate_model(self):
    #     """
    #     Evaluate the trained multi-class model.
    #     """
    #     if self.model is None:
    #         raise ValueError("Model not trained. Run train_model() first.")
        
    #     y_pred = self.model.predict(self.X_test)
    #     target_names = [self.id_to_class[i] for i in range(len(self.id_to_class))]
        
    #     report = classification_report(self.y_test, y_pred, target_names=target_names)
    #     cm = confusion_matrix(self.y_test, y_pred)
        
    #     print("--- Classification Report ---")
    #     print(report)
    #     print("--- Confusion Matrix ---")
    #     print(cm)
        
    #     return {
    #         "classification_report": report,
    #         "confusion_matrix": cm.tolist()
    #     }


    # In multiclass_modeling.py, inside the MulticlassModel class

    def evaluate_model(self):
        """
        Evaluate the trained multi-class model.
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train_model() first.")
        
        y_pred = self.model.predict(self.X_test)
        
        # --- FIX IS HERE ---
        # Get the number of classes the model was actually trained on
        num_model_classes = self.model.n_classes_
        
        # Create a list of target names that matches the model's classes
        # We use the id_to_class mapping which is already defined in the class
        target_names = [self.id_to_class[i] for i in range(num_model_classes)]
        
        print(f"Evaluating model with {num_model_classes} classes: {target_names}")
        
        report = classification_report(self.y_test, y_pred, target_names=target_names)
        cm = confusion_matrix(self.y_test, y_pred)
        
        print("--- Classification Report ---")
        print(report)
        print("--- Confusion Matrix ---")
        print(cm)
        
        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "target_names": target_names # Also return the names used for reference
        }

    def save_model(self, output_dir="models"):
        """
        Save the trained model and its metadata to disk.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)
        
        # Save the model
        model_path = out_dir / "xgboost_gait_5class.bin"
        self.model.save_model(model_path)
        print(f"Model saved to: {model_path.resolve()}")
        
        # Save the metadata
        metadata = {
            "classes": self.index_to_class,
            "class_to_id": self.class_to_id,
            "id_to_class": self.id_to_class,
            "feature_cols": self.feature_names,
            "normal_label": self.normal_label,
            "label_policy": "multihot->single via first positive index (priority order)",
        }
        
        meta_path = out_dir / "xgboost_gait_5class_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {meta_path.resolve()}")
        
        return model_path, meta_path


def run_multiclass_training_pipeline(data_path="../data/clean_gait_data.parquet"):
    """
    Run the complete multi-class training pipeline.
    
    Args:
        data_path: Path to the raw parquet data file.
    """
    # --- Determine the project root and data path robustly ---
    PROJECT_ROOT = Path(__file__).resolve().parent
    if not Path(data_path).is_absolute():
        data_path = PROJECT_ROOT / data_path
        
    print(f"Project root determined as: {PROJECT_ROOT}")
    print(f"Using data at: {data_path}")
    
    # --- Initialize and run the pipeline ---
    model = MulticlassModel()
    
    # 1. Prepare data
    print("Preparing data for multi-class modeling...")
    X, y, groups = model.prepare_data(
        data_path=data_path,
        window_seconds=2.0,
        overlap=0.5,
        resample_frames=60
    )
    
    # 2. Split data
    print("Splitting data...")
    model.split_data(X, y, groups, test_size=0.2, random_state=42)
    
    # 3. Train model
    print("Training multi-class model...")
    model.train_model(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05
    )
    
    # 4. Evaluate model
    print("Evaluating model...")
    results = model.evaluate_model()
    
    # 5. Save model and metadata
    print("Saving model and metadata...")
    model.save_model()
    
    print("--- Multi-class training pipeline finished successfully ---")
    return model


if __name__ == "__main__":
    run_multiclass_training_pipeline()