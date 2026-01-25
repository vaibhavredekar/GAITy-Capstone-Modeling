"""
Baseline Modeling Script

This script builds a binary XGBoost classifier that predicts normal vs. abnormal gait
using hand-engineered numerical features extracted from video-based gait data.

The script follows the same preprocessing and feature extraction steps as shown in the
reference implementation, ensuring consistency in data handling before model training.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# Import our custom modules
from preprocessing_n_feature_engineering import GaitAnalysis, FeatureConfig


class BaselineModel:
    """
    Class for building and evaluating baseline XGBoost models for gait classification.
    """
    
    def __init__(self, gait_analyzer=None):
        """
        Initialize the baseline model.
        
        Args:
            gait_analyzer: Optional GaitAnalysis instance with preprocessed data
        """
        self.gait_analyzer = gait_analyzer
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.train_patients = None
        self.test_patients = None
    
    # def prepare_data(self, data_path=None, window_seconds=2.0, overlap=0.5, 
    #                 resample_frames=60, threshold_pct=5.0):
    #     """
    #     Prepare data for modeling by running the full preprocessing pipeline
    #     and extracting features.
        
    #     Args:
    #         data_path: Path to the raw parquet data file
    #         window_seconds: Length of sliding window in seconds
    #         overlap: Overlap fraction between windows
    #         resample_frames: Number of frames to resample each window to
    #         threshold_pct: Maximum allowed percentage of missing frames
            
    #     Returns:
    #         Tuple of (X, y, groups) where groups are patient identifiers
    #     """
    #     # Initialize gait analyzer if not provided
    #     if self.gait_analyzer is None:
    #         self.gait_analyzer = GaitAnalysis(
    #             data_path=data_path,
    #             window_seconds=window_seconds,
    #             overlap=overlap,
    #             resample_frames=resample_frames
    #         )
        
    #     # Run the full pipeline
    #     features_df = self.gait_analyzer.run_full_pipeline(
    #         data_path=data_path,
    #         threshold_pct=threshold_pct
    #     )
        
    #     # Extract patient names from window IDs
    #     patient_names = self._extract_patient_names()
        
    #     # Prepare features and labels
    #     non_feature_cols = [
    #         "label_fine", "label_class", "label_id",
    #         "binary_label",
    #         "movement_type", "side", "source_file"
    #     ]
    #     feature_cols = [
    #         c for c in features_df.columns
    #         if c not in non_feature_cols and pd.api.types.is_numeric_dtype(features_df[c])
    #     ]
        
    #     X = features_df[feature_cols].copy()
    #     X = X.fillna(X.median(numeric_only=True))  # median impute
    #     y = self.gait_analyzer.y_binary_clean.astype(int)
        
    #     self.feature_names = feature_cols
        
    #     return X, y, patient_names
    
    
    # def prepare_data(self, data_path=None, window_seconds=2.0, overlap=0.5, 
    #             resample_frames=60, threshold_pct=5.0):
    #     """
    #     Prepare data for modeling by running the full preprocessing pipeline
    #     and extracting features.
    #     """
    #     # --- FIX IS HERE ---
    #     # Ensure data_path is an absolute Path object
    #     if data_path is None:
    #         raise ValueError("A data_path must be provided to prepare_data.")
    #     data_path = Path(data_path).resolve()

    #     # Initialize gait analyzer if not provided
    #     if self.gait_analyzer is None:
    #         self.gait_analyzer = GaitAnalysis(
    #             data_path=data_path,  # Pass the absolute path here
    #             window_seconds=window_seconds,
    #             overlap=overlap,
    #             resample_frames=resample_frames
    #         )
    #     else:
    #         # If it already exists, update its data_path
    #         self.gait_analyzer.data_path = data_path

    #     # Run the full pipeline. It will now use the correct path set above.
    #     features_df = self.gait_analyzer.run_full_pipeline(
    #         # No need to pass data_path here anymore
    #         threshold_pct=threshold_pct
    #     )

    #     # Extract patient names from window IDs
    #     patient_names = self._extract_patient_names()
        
    #     # Prepare features and labels
    #     non_feature_cols = [
    #         "label_fine", "label_class", "label_id",
    #         "binary_label",
    #         "movement_type", "side", "source_file"
    #     ]
    #     feature_cols = [
    #         c for c in features_df.columns
    #         if c not in non_feature_cols and pd.api.types.is_numeric_dtype(features_df[c])
    #     ]
        
    #     X = features_df[feature_cols].copy()
    #     X = X.fillna(X.median(numeric_only=True))  # median impute
    #     y = self.gait_analyzer.y_binary_clean.astype(int)
        
    #     self.feature_names = feature_cols
        
    #     return X, y, patient_names

    def prepare_data(self, data_path=None, window_seconds=2.0, overlap=0.5, 
                resample_frames=60, threshold_pct=5.0):
        """
        Prepare data for modeling by running the full preprocessing pipeline
        and extracting features.
        """
        # --- FIX IS HERE ---
        # Ensure data_path is an absolute Path object
        if data_path is None:
            raise ValueError("A data_path must be provided to prepare_data.")
        data_path = Path(data_path).resolve()

        # Initialize gait analyzer if not provided
        if self.gait_analyzer is None:
            self.gait_analyzer = GaitAnalysis(
                data_path=data_path,  # Pass the absolute path here
                window_seconds=window_seconds,
                overlap=overlap,
                resample_frames=resample_frames
            )
        else:
            # If it already exists, update its data_path
            self.gait_analyzer.data_path = data_path

        # Run the full pipeline. It will now use the correct path set above.
        features_df = self.gait_analyzer.run_full_pipeline(
            # No need to pass data_path here anymore
            threshold_pct=threshold_pct
        )
        
        # Extract patient names from window IDs
        patient_names = self._extract_patient_names()
        
        # Prepare features and labels
        non_feature_cols = [
            "label_fine", "label_class", "label_id",
            "binary_label",
            "movement_type", "side", "source_file"
        ]
        feature_cols = [
            c for c in features_df.columns
            if c not in non_feature_cols and pd.api.types.is_numeric_dtype(features_df[c])
        ]
        
        X = features_df[feature_cols].copy()
        X = X.fillna(X.median(numeric_only=True))  # median impute
        
        # --- THIS IS THE CRITICAL LINE ---
        # The source self.gait_analyzer.y_binary_clean is a NUMPY ARRAY.
        # We MUST convert it to a PANDAS SERIES to be able to use .iloc later.
        y = pd.Series(self.gait_analyzer.y_binary_clean.astype(int))

        # --- DEBUGGING PRINT ---
        # This line will confirm the type of 'y' right before we return it.
        # If the fix is correct, this will print "<class 'pandas.core.series.Series'>"
        print(f"DEBUG: The type of 'y' in prepare_data is: {type(y)}")
        
        self.feature_names = feature_cols
        
        return X, y, patient_names


    def _extract_patient_names(self):
        """
        Extract patient names from window IDs.
        
        Returns:
            List of patient names corresponding to each window
        """
        if not hasattr(self.gait_analyzer, 'window_ids_clean'):
            raise ValueError("Window IDs not available. Run preprocessing first.")
        
        # Create mapping from video_id to patient_name
        if hasattr(self.gait_analyzer, 'df_video') and 'patient_name' in self.gait_analyzer.df_video.columns:
            video_to_patient = dict(
                self.gait_analyzer.df_video.select(["video_id", "patient_name"]).unique().iter_rows()
            )
        else:
            # If patient_name is not available, use video_id as a proxy
            video_to_patient = {}
            for wid in self.gait_analyzer.window_ids_clean:
                vid = wid.split("_win")[0]
                video_to_patient[vid] = vid
        
        # Extract patient names for each window
        patient_names = []
        for wid in self.gait_analyzer.window_ids_clean:
            vid = wid.split("_win")[0]
            patient_names.append(video_to_patient.get(vid, vid))
        
        return patient_names
    
    def split_data(self, X, y, groups, test_size=0.2, random_state=42):
        """
        Split data into train and test sets using group-based splitting to avoid data leakage.
        
        Args:
            X: Feature matrix
            y: Target labels
            groups: Group identifiers (patient names)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, train_groups, test_groups)
        """
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        self.X_train, self.X_test = X.iloc[train_idx], X.iloc[test_idx]
        self.y_train, self.y_test = y.iloc[train_idx], y.iloc[test_idx]
        self.train_patients = np.array(groups)[train_idx]
        self.test_patients = np.array(groups)[test_idx]
        
        return (self.X_train, self.X_test, self.y_train, self.y_test, 
                self.train_patients, self.test_patients)
    
    # def train_model(self, n_estimators=300, max_depth=4, learning_rate=0.05, 
    #                subsample=0.8, colsample_bytree=0.8, random_state=42):
    #     """
    #     Train an XGBoost classifier with class imbalance handling.
        
    #     Args:
    #         n_estimators: Number of trees in the ensemble
    #         max_depth: Maximum tree depth
    #         learning_rate: Learning rate
    #         subsample: Subsample ratio of rows
    #         colsample_bytree: Subsample ratio of columns
    #         random_state: Random seed for reproducibility
            
    #     Returns:
    #         Trained XGBoost model
    #     """
    #     if self.X_train is None or self.y_train is None:
    #         raise ValueError("Training data not available. Run split_data first.")
        
    #     # Calculate class weight ratio for imbalance handling
    #     ratio = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
    #     # Initialize and train the model
    #     self.model = xgb.XGBClassifier(
    #         objective="binary:logistic",
    #         eval_metric="logloss",
    #         n_estimators=n_estimators,
    #         max_depth=max_depth,
    #         learning_rate=learning_rate,
    #         subsample=subsample,
    #         colsample_bytree=colsample_bytree,
    #         scale_pos_weight=ratio,
    #         random_state=random_state,
    #         n_jobs=-1,
    #     )
        
    #     self.model.fit(self.X_train, self.y_train)
        
    #     return self.model
    

    # def train_model(self, n_estimators=50, max_depth=4, learning_rate=0.1, 
    #             subsample=0.8, colsample_bytree=0.8, random_state=42):
    #     """
    #     Train an XGBoost classifier with class imbalance handling.
    #     Includes diagnostics for common training issues.
        
    #     Args:
    #         n_estimators: Number of trees in the ensemble (reduced for faster testing)
    #         max_depth: Maximum tree depth
    #         learning_rate: Learning rate
    #         subsample: Subsample ratio of rows
    #         colsample_bytree: Subsample ratio of columns
    #         random_state: Random seed for reproducibility
            
    #     Returns:
    #         Trained XGBoost model
    #     """
    #     if self.X_train is None or self.y_train is None:
    #         raise ValueError("Training data not available. Run split_data first.")
        
    #     print("--- Diagnosing data before training ---")
        
    #     # 1. Check for NaN or infinite values
    #     if self.X_train.isnull().values.any():
    #         print("WARNING: X_train contains NaN values. Attempting to fill again.")
    #         self.X_train = self.X_train.fillna(self.X_train.median())
    #     if np.isinf(self.X_train.values).any():
    #         raise ValueError("X_train contains infinite values, cannot train.")
            
    #     if self.y_train.isnull().any():
    #         raise ValueError("y_train contains NaN values, cannot train.")

    #     # 2. Check class balance and calculate scale_pos_weight safely
    #     class_counts = self.y_train.value_counts()
    #     print(f"Class distribution in y_train:\n{class_counts}")
        
    #     if 0 not in class_counts or 1 not in class_counts:
    #         print("WARNING: Training data has only one class. Setting scale_pos_weight to 1.")
    #         ratio = 1.0
    #     else:
    #         ratio = class_counts[0] / class_counts[1]
    #         print(f"Calculated scale_pos_weight ratio: {ratio:.4f}")
        
    #     print("--- Starting model training ---")
    #     print(f"Using {n_estimators} estimators for faster training.")

    #     # Initialize the model
    #     self.model = xgb.XGBClassifier(
    #         objective="binary:logistic",
    #         eval_metric="logloss",
    #         n_estimators=n_estimators,
    #         max_depth=max_depth,
    #         learning_rate=learning_rate,
    #         subsample=subsample,
    #         colsample_bytree=colsample_bytree,
    #         scale_pos_weight=ratio,
    #         random_state=random_state,
    #         n_jobs=-1, # Use all available CPU cores
    #         use_label_encoder=False # Suppresses a future warning
    #     )
        
    #     try:
    #         # Train the model
    #         self.model.fit(self.X_train, self.y_train)
    #         print("--- Model training completed successfully ---")
    #     except Exception as e:
    #         print(f"!!! An error occurred during model.fit(): {e} !!!")
    #         raise # Re-raise the exception after printing it
            
    #     return self.model 


    # def train_model(self, n_estimators=50, max_depth=4, learning_rate=0.1, 
    #             subsample=0.8, colsample_bytree=0.8, random_state=42):
    #     """
    #     Train an XGBoost classifier with class imbalance handling.
    #     Includes progress callbacks to show training is active.
        
    #     Args:
    #         n_estimators: Number of trees in the ensemble
    #         max_depth: Maximum tree depth
    #         learning_rate: Learning rate
    #         subsample: Subsample ratio of rows
    #         colsample_bytree: Subsample ratio of columns
    #         random_state: Random seed for reproducibility
            
    #     Returns:
    #         Trained XGBoost model
    #     """
    #     import time
        
    #     if self.X_train is None or self.y_train is None:
    #         raise ValueError("Training data not available. Run split_data first.")
        
    #     print("--- Diagnosing data before training ---")
        
    #     # Check for NaN or infinite values
    #     if self.X_train.isnull().values.any():
    #         print("WARNING: X_train contains NaN values. Attempting to fill again.")
    #         self.X_train = self.X_train.fillna(self.X_train.median())
    #     if np.isinf(self.X_train.values).any():
    #         raise ValueError("X_train contains infinite values, cannot train.")
            
    #     if self.y_train.isnull().any():
    #         raise ValueError("y_train contains NaN values, cannot train.")

    #     # Check class balance and calculate scale_pos_weight safely
    #     class_counts = self.y_train.value_counts()
    #     print(f"Class distribution in y_train:\n{class_counts}")
        
    #     if 0 not in class_counts or 1 not in class_counts:
    #         print("WARNING: Training data has only one class. Setting scale_pos_weight to 1.")
    #         ratio = 1.0
    #     else:
    #         ratio = class_counts[0] / class_counts[1]
    #         print(f"Calculated scale_pos_weight ratio: {ratio:.4f}")
        
    #     print(f"--- Starting model training with {n_estimators} estimators ---")
    #     start_time = time.time()

    #     # --- FIX IS HERE ---
    #     # Create a callback to show progress
    #     class ProgressCallback(xgb.callback.TrainingCallback):
    #         def __init__(self, period=1):
    #             self.period = period
    #         def after_iteration(self, model, epoch, evals_log):
    #             if (epoch + 1) % self.period == 0:
    #                 print(f"  ... finished training tree {epoch + 1}/{n_estimators}")
    #             return False # Return False to continue training

    #     # Initialize the model
    #     self.model = xgb.XGBClassifier(
    #         objective="binary:logistic",
    #         eval_metric="logloss",
    #         n_estimators=n_estimators,
    #         max_depth=max_depth,
    #         learning_rate=learning_rate,
    #         subsample=subsample,
    #         colsample_bytree=colsample_bytree,
    #         scale_pos_weight=ratio,
    #         random_state=random_state,
    #         n_jobs=-1, # Use all available CPU cores
    #         use_label_encoder=False, # Suppresses the warning
    #         callbacks=[ProgressCallback(period=5)] # Print progress every 5 trees
    #     )
        
    #     try:
    #         # Train the model
    #         self.model.fit(self.X_train, self.y_train)
            
    #         end_time = time.time()
    #         duration = end_time - start_time
    #         print(f"--- Model training completed successfully in {duration:.2f} seconds ---")
            
    #     except Exception as e:
    #         print(f"!!! An error occurred during model.fit(): {e} !!!")
    #         raise # Re-raise the exception after printing it
            
    #     return self.model

    def train_model(self, n_estimators=50, max_depth=4, learning_rate=0.1, 
                subsample=0.8, colsample_bytree=0.8, random_state=42):
        """
        Train an XGBoost classifier with class imbalance handling.
        Includes progress callbacks to show training is active.
        """
        import time
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available. Run split_data first.")
        
        print("--- Diagnosing data before training ---")
        
        # Check for NaN or infinite values
        if self.X_train.isnull().values.any():
            print("WARNING: X_train contains NaN values. Attempting to fill again.")
            self.X_train = self.X_train.fillna(self.X_train.median())
        if np.isinf(self.X_train.values).any():
            raise ValueError("X_train contains infinite values, cannot train.")
            
        if self.y_train.isnull().any():
            raise ValueError("y_train contains NaN values, cannot train.")

        # Check class balance and calculate scale_pos_weight safely
        class_counts = self.y_train.value_counts()
        print(f"Class distribution in y_train:\n{class_counts}")
        
        if 0 not in class_counts or 1 not in class_counts:
            print("WARNING: Training data has only one class. Setting scale_pos_weight to 1.")
            ratio = 1.0
        else:
            ratio = class_counts[0] / class_counts[1]
            print(f"Calculated scale_pos_weight ratio: {ratio:.4f}")
        
        print(f"--- Starting model training with {n_estimators} estimators ---")
        start_time = time.time()

        # Create a callback to show progress
        class ProgressCallback(xgb.callback.TrainingCallback):
            def __init__(self, period=1):
                self.period = period
            def after_iteration(self, model, epoch, evals_log):
                if (epoch + 1) % self.period == 0:
                    print(f"  ... finished training tree {epoch + 1}/{n_estimators}")
                return False # Return False to continue training

        # Initialize the model
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=ratio,
            random_state=random_state,
            n_jobs=-1, # Use all available CPU cores
            use_label_encoder=False, # Suppresses the warning
            callbacks=[ProgressCallback(period=5)] # <--- THIS IS THE LINE TO CHECK FOR
        )
        
        try:
            # Train the model
            self.model.fit(self.X_train, self.y_train)
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"--- Model training completed successfully in {duration:.2f} seconds ---")
            
        except Exception as e:
            print(f"!!! An error occurred during model.fit(): {e} !!!")
            raise # Re-raise the exception after printing it
            
        return self.model


    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train_model first.")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(self.y_test, y_pred, target_names=["normal", "abnormal"], output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        # Check for data leakage
        train_pats = set(self.train_patients)
        test_pats = set(self.test_patients)
        leakage = len(train_pats & test_pats)
        
        return {
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc": roc_auc,
            "data_leakage": leakage
        }
    
    def save_model(self, output_dir="models"):
        """
        Save the trained model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train_model first.")
        
        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)
        
        # Save the model
        model_path = out_dir / "xgboost_model.bin"
        self.model.save_model(model_path)
        
        print(f"Model saved to: {model_path.resolve()}")
        
        return model_path
    
    def plot_confusion_matrix(self, cm, class_names=("Normal", "Abnormal"), 
                             title="Model decisions (test set)"):
        """
        Plot a stakeholder-friendly confusion matrix.
        
        Args:
            cm: Confusion matrix array
            class_names: Names of the classes
            title: Plot title
            
        Returns:
            Matplotlib figure and axis objects
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
    
    def plot_out_of_100(self, cm, title="What happens out of 100 cases?"):
        """
        Convert performance into "out of 100" for each actual class.
        
        Args:
            cm: Confusion matrix array
            title: Plot title
            
        Returns:
            Matplotlib figure and axis objects
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
    
    def plot_out_of_100_pies(self, cm, title="Gait classifier performance (out of 100 cases)"):
        """
        Plot pie charts showing performance out of 100 cases for each class.
        
        Args:
            cm: Confusion matrix array
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
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
        return fig


# def run_baseline_modeling(data_path="../filled_gait_data_encoded.parquet"):
#     """
#     Run the complete baseline modeling pipeline.
    
#     Args:
#         data_path: Path to the raw parquet data file
        
#     Returns:
#         Dictionary containing model, evaluation metrics, and visualizations
#     """
#     # Determine the project root directory relative to this script
#     # This assumes the script is in 'm_implementation' and data is in the project root.
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent
#     data_path = PROJECT_ROOT / "filled_gait_data_encoded.parquet"
    
#     print(f"Project root determined as: {PROJECT_ROOT}")
#     print(f"Looking for data at: {data_path}")

#     # Initialize the baseline model
#     baseline = BaselineModel()
    
#     # Prepare data
#     print("Preparing data...")
#     X, y, groups = baseline.prepare_data(
#         data_path=data_path,
#         window_seconds=2.0,
#         overlap=0.5,
#         resample_frames=60,
#         threshold_pct=5.0
#     )
    
#     # Split data
#     print("Splitting data...")
#     baseline.split_data(X, y, groups, test_size=0.2, random_state=42)
    
#     # Train model
#     print("Training model...")
#     baseline.train_model(
#         n_estimators=300,
#         max_depth=4,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )
    
#     # Evaluate model
#     print("Evaluating model...")
#     results = baseline.evaluate_model()
    
#     # Print evaluation results
#     print("\nClassification Report:")
#     print(classification_report(baseline.y_test, baseline.model.predict(baseline.X_test), 
#                                target_names=["normal", "abnormal"]))
#     print(f"ROC-AUC: {results['roc_auc']:.4f}")
#     print(f"Data leakage (shared patients): {results['data_leakage']}")
    
#     # Save model
#     print("\nSaving model...")
#     baseline.save_model()
    
#     # Create visualizations
#     print("Creating visualizations...")
#     cm = results['confusion_matrix']
    
#     # Confusion matrix
#     fig1, ax1 = baseline.plot_confusion_matrix(
#         cm, 
#         class_names=("Normal", "Abnormal"), 
#         title="Gait classifier outcomes"
#     )
    
#     # Out of 100 bar chart
#     fig2, ax2 = baseline.plot_out_of_100(
#         cm, 
#         title="Gait classifier: results per 100 cases"
#     )
    
#     # Out of 100 pie charts
#     fig3 = baseline.plot_out_of_100_pies(
#         cm, 
#         title="Gait classifier performance (out of 100 cases)"
#     )
    
#     # Save visualizations
#     out_dir = Path("visualizations")
#     out_dir.mkdir(exist_ok=True)
    
#     fig1.savefig(out_dir / "confusion_matrix.png")
#     fig2.savefig(out_dir / "out_of_100.png")
#     fig3.savefig(out_dir / "out_of_100_pies.png")
    
#     print(f"Visualizations saved to {out_dir.resolve()}")
    
#     return {
#         "model": baseline.model,
#         "baseline": baseline,
#         "results": results,
#         "confusion_matrix_fig": fig1,
#         "out_of_100_fig": fig2,
#         "out_of_100_pies_fig": fig3
#     }


# def run_baseline_modeling():
#     """
#     Run the complete baseline modeling pipeline.
#     This version calculates the data path robustly.
        
#     Returns:
#         Dictionary containing model, evaluation metrics, and visualizations
#     """
#     # --- FIX IS HERE ---
#     # Determine the project root directory relative to this script
#     # This assumes the script is in 'm_implementation' and data is in the project root.
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent
#     data_path = PROJECT_ROOT / "filled_gait_data_encoded.parquet"
    
#     print(f"Project root determined as: {PROJECT_ROOT}")
#     print(f"Looking for data at: {data_path}")
    
#     # Initialize the baseline model
#     baseline = BaselineModel()
    
#     # Prepare data
#     print("Preparing data...")
#     X, y, groups = baseline.prepare_data(
#         data_path=data_path,  # Pass the absolute path
#         window_seconds=2.0,
#         overlap=0.5,
#         resample_frames=60,
#         threshold_pct=5.0
#     )
    
#     # Split data
#     print("Splitting data...")
#     baseline.split_data(X, y, groups, test_size=0.2, random_state=42)
    
#     # Train model with FAST parameters for a quick test run
#     print("Training model with FAST parameters for a quick test...")
#     baseline.train_model(
#         n_estimators=50,      # Reduced from 300 to 50
#         learning_rate=0.1,    # Increased from 0.05 to 0.1
#         max_depth=4,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )
    
#     # Evaluate model
#     print("Evaluating model...")
#     results = baseline.evaluate_model()
    
#     # Print evaluation results
#     print("\nClassification Report:")
#     print(classification_report(baseline.y_test, baseline.model.predict(baseline.X_test), 
#                                target_names=["normal", "abnormal"]))
#     print(f"ROC-AUC: {results['roc_auc']:.4f}")
#     print(f"Data leakage (shared patients): {results['data_leakage']}")
    
#     # Save model
#     print("\nSaving model...")
#     baseline.save_model()
    
#     # Save the feature names for the prediction script
#     import json
#     out_dir = Path("models")
#     feature_names_path = out_dir / "feature_names.json"
#     with open(feature_names_path, 'w') as f:
#         json.dump(baseline.feature_names, f, indent=2)
#     print(f"Feature names saved to: {feature_names_path.resolve()}")
    
#     # Create visualizations
#     print("Creating visualizations...")
#     cm = results['confusion_matrix']
    
#     # Confusion matrix
#     fig1, ax1 = baseline.plot_confusion_matrix(
#         cm, 
#         class_names=("Normal", "Abnormal"), 
#         title="Gait classifier outcomes"
#     )
    
#     # Out of 100 bar chart
#     fig2, ax2 = baseline.plot_out_of_100(
#         cm, 
#         title="Gait classifier: results per 100 cases"
#     )
    
#     # Out of 100 pie charts
#     fig3 = baseline.plot_out_of_100_pies(
#         cm, 
#         title="Gait classifier performance (out of 100 cases)"
#     )
    
#     # Save visualizations
#     out_dir = Path("visualizations")
#     out_dir.mkdir(exist_ok=True)
    
#     fig1.savefig(out_dir / "confusion_matrix.png")
#     fig2.savefig(out_dir / "out_of_100.png")
#     fig3.savefig(out_dir / "out_of_100_pies.png")
    
#     print(f"Visualizations saved to {out_dir.resolve()}")
    
#     return {
#         "model": baseline.model,
#         "baseline": baseline,
#         "results": results,
#         "confusion_matrix_fig": fig1,
#         "out_of_100_fig": fig2,
#         "out_of_100_pies_fig": fig3
#     }


def run_baseline_modeling():
    """
    Run the complete baseline modeling pipeline.
    This version calculates the data path robustly.
        
    Returns:
        Dictionary containing model, evaluation metrics, and visualizations
    """
    # Determine the project root directory relative to this script
    # This assumes the script is in 'm_implementation' and data is in the project root.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_path = PROJECT_ROOT / "filled_gait_data_encoded.parquet"
    
    print(f"Project root determined as: {PROJECT_ROOT}")
    print(f"Looking for data at: {data_path}")
    
    # Initialize the baseline model
    baseline = BaselineModel()
    
    # Prepare data
    print("Preparing data...")
    X, y, groups = baseline.prepare_data(
        data_path=data_path,  # Pass the absolute path
        window_seconds=2.0,
        overlap=0.5,
        resample_frames=60,
        threshold_pct=5.0
    )
    
    # Split data
    print("Splitting data...")
    baseline.split_data(X, y, groups, test_size=0.2, random_state=42)
    
    # --- CHANGE IS HERE ---
    # Train model with the original, full parameters for the final model
    print("Training model with full parameters for the final model...")
    baseline.train_model(
        n_estimators=300,     # Reverted to original value
        learning_rate=0.05,   # Reverted to original value
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = baseline.evaluate_model()
    
    # Print evaluation results
    print("\nClassification Report:")
    print(classification_report(baseline.y_test, baseline.model.predict(baseline.X_test), 
                               target_names=["normal", "abnormal"]))
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"Data leakage (shared patients): {results['data_leakage']}")
    
    # Save model
    print("\nSaving model...")
    baseline.save_model()
    
    # Save the feature names for the prediction script
    import json
    out_dir = Path("models")
    feature_names_path = out_dir / "feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(baseline.feature_names, f, indent=2)
    print(f"Feature names saved to: {feature_names_path.resolve()}")
    
    # Create visualizations
    print("Creating visualizations...")
    cm = results['confusion_matrix']
    
    # Confusion matrix
    fig1, ax1 = baseline.plot_confusion_matrix(
        cm, 
        class_names=("Normal", "Abnormal"), 
        title="Gait classifier outcomes"
    )
    
    # Out of 100 bar chart
    fig2, ax2 = baseline.plot_out_of_100(
        cm, 
        title="Gait classifier: results per 100 cases"
    )
    
    # Out of 100 pie charts
    fig3 = baseline.plot_out_of_100_pies(
        cm, 
        title="Gait classifier performance (out of 100 cases)"
    )
    
    # Save visualizations
    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)
    
    fig1.savefig(out_dir / "confusion_matrix.png")
    fig2.savefig(out_dir / "out_of_100.png")
    fig3.savefig(out_dir / "out_of_100_pies.png")
    
    print(f"Visualizations saved to {out_dir.resolve()}")
    
    return {
        "model": baseline.model,
        "baseline": baseline,
        "results": results,
        "confusion_matrix_fig": fig1,
        "out_of_100_fig": fig2,
        "out_of_100_pies_fig": fig3
    }

if __name__ == "__main__":
    # Run the baseline modeling pipeline
    results = run_baseline_modeling()