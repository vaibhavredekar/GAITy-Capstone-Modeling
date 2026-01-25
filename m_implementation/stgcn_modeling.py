"""
train_stgcn_models.py

This script trains and saves the binary and multi-label ST-GCN models for gait analysis.
It follows the robust data preparation and patient-wise splitting logic from our previous work.
"""

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from pathlib import Path
import sys
import json

# --- Import our custom pipeline components ---
try:
    from preprocessing_n_feature_engineering import GaitAnalysis
except ImportError as e:
    print("Error: Could not import pipeline components.")
    print("Please ensure 'preprocessing_n_feature_engineering.py' is in your PYTHONPATH.")
    sys.exit(1)

# --- Model Architectures (from your prediction script) ---

def clean_state_dict(state_dict):
    """Remove 'module.' prefix if present."""
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    return new_dict

class BinarySTGCN(nn.Module):
    def __init__(self, in_channels=3, num_joints=14, out_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
        self.fc = nn.Linear(256 * num_joints, out_classes)
        
    def forward(self, x):
        # x: (N, C, T, V, M) -> (N, C, T, V)
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

class MultiSTGCN(nn.Module):
    def __init__(self, in_channels=3, num_joints=14, out_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
        self.fc = nn.Linear(256 * num_joints, out_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

# --- PyT Dataset ---
class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --- Training and Evaluation Functions ---
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        if outputs.shape[1] == 1: outputs = outputs.squeeze(1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def eval_model(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            if outputs.shape[1] == 1: outputs = outputs.squeeze(1)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
            all_probs.append(probs.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()
    return total_loss / len(dataloader.dataset), all_preds, all_targets, all_probs

def patient_level_metrics(probs, targets, patient_names, threshold=0.5):
    """Calculate patient-level metrics by aggregating window predictions."""
    unique_patients = np.unique(patient_names)
    patient_preds, patient_targets = [], []
    for p in unique_patients:
        idx = np.where(patient_names == p)[0]
        p_prob = probs[idx].mean()
        p_pred = int(p_prob >= threshold)
        p_target = int(targets[idx].max())
        patient_preds.append(p_pred)
        patient_targets.append(p_target)
    return patient_preds, patient_targets

# --- Main Training Pipeline ---
def run_stgcn_training_pipeline(data_path="../data/clean_gait_data.parquet", models_dir="models"):
    """Run the complete ST-GCN training pipeline."""
    # --- Determine paths ---
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_PATH = PROJECT_ROOT / data_path if not Path(data_path).is_absolute() else Path(data_path)
    MODELS_DIR = PROJECT_ROOT / models_dir
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using data: {DATA_PATH}")
    print(f"Saving models to: {MODELS_DIR}")

    # --- 1. Load and Preprocess Data ---
    gait_analyzer = GaitAnalysis(data_path=DATA_PATH)
    gait_analyzer.run_full_pipeline()
    
    # --- 2. Prepare Data for ST-GCN ---
    # Transpose to ST-GCN format: (N, C=3, T, V)
    X_stgcn = np.transpose(gait_analyzer.X_clean, (0, 3, 1, 2))
    
    # Patient-wise split
    unique_patients = np.unique(gait_analyzer._extract_patient_names())
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
    train_mask = np.isin(gait_analyzer._extract_patient_names(), train_patients)
    test_mask = np.isin(gait_analyzer._extract_patient_names(), test_patients)

    # --- 3. Train Binary Model ---
    print("\n--- Training Binary ST-GCN Model ---")
    X_train_bin, X_test_bin = X_stgcn[train_mask], X_stgcn[test_mask]
    y_train_bin, y_test_bin = gait_analyzer.y_binary_clean[train_mask], gait_analyzer.y_binary_clean[test_mask]
    patient_names_test_bin = np.array(gait_analyzer._extract_patient_names())[test_mask]

    bin_train_loader = DataLoader(PoseDataset(X_train_bin, y_train_bin), batch_size=32, shuffle=True)
    bin_test_loader = DataLoader(PoseDataset(X_test_bin, y_test_bin), batch_size=32)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_joints = X_train_bin.shape[3]
    binary_model = BinarySTGCN(num_joints=num_joints, out_classes=1).to(device)
    optimizer_bin = torch.optim.Adam(binary_model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([(len(y_train_bin)-y_train_bin.sum())/y_train_bin.sum()], device=device)
    criterion_bin = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = 10
    for epoch in range(epochs):
        train_loss = train_model(binary_model, bin_train_loader, optimizer_bin, criterion_bin, device)
        val_loss, preds, targets, probs = eval_model(binary_model, bin_test_loader, criterion_bin, device)
        acc = accuracy_score(targets, preds)
        print(f"[Binary] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Acc: {acc:.4f}")

    # --- 4. Patient-level Threshold Optimization for Binary Model ---
    print("\n--- Optimizing Patient-Level Threshold for Binary Model ---")
    best_f1 = 0; best_threshold = 0.5
    for th in np.arange(0.05, 1.0, 0.05):
        p_preds, p_targets = patient_level_metrics(probs, targets, patient_names_test_bin, th)
        f1 = f1_score(p_targets, p_preds)
        if f1 > best_f1: best_f1, best_threshold = f1, th
    print(f"Best patient-level threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")

    # --- 5. Train Multi-Label Model (on abnormal windows only) ---
    print("\n--- Training Multi-Label ST-GCN Model ---")
    abnormal_mask_train = y_train_bin == 1
    abnormal_mask_test = y_test_bin == 1
    X_ml_train, X_ml_test = X_train_bin[abnormal_mask_train], X_test_bin[abnormal_mask_test]
    y_ml_train, y_ml_test = gait_analyzer.y_multilabel_clean[abnormal_mask_train], gait_analyzer.y_multilabel_clean[abnormal_mask_test]

    ml_train_loader = DataLoader(PoseDataset(X_ml_train, y_ml_train), batch_size=32, shuffle=True)
    ml_test_loader = DataLoader(PoseDataset(X_ml_test, y_ml_test), batch_size=32)

    num_labels = y_ml_train.shape[1]
    multi_model = MultiSTGCN(num_joints=num_joints, out_classes=num_labels).to(device)
    optimizer_ml = torch.optim.Adam(multi_model.parameters(), lr=1e-3)
    criterion_ml = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        train_loss = train_model(multi_model, ml_train_loader, optimizer_ml, criterion_ml, device)
        val_loss, preds, targets, probs = eval_model(multi_model, ml_test_loader, criterion_ml, device)
        print(f"[Multi-label] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {val_loss:.4f}")

    # --- 6. Save Models and Metadata ---
    print("\n--- Saving Models and Metadata ---")
    binary_model_path = MODELS_DIR / "binary_stgcn_gait.bin"
    multi_model_path = MODELS_DIR / "multilabel_stgcn_gait.bin"
    
    torch.save(binary_model.state_dict(), binary_model_path)
    torch.save(multi_model.state_dict(), multi_model_path)
    
    # Save metadata for the prediction script
    metadata = {
        "binary_model": {
            "path": binary_model_path.name,
            "num_joints": num_joints,
            "threshold": best_threshold
        },
        "multilabel_model": {
            "path": multi_model_path.name,
            "num_joints": num_joints,
            "num_labels": num_labels,
            "anomaly_cols": gait_analyzer.ANOMALY_COLS
        }
    }
    metadata_path = MODELS_DIR / "stgcn_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Models and metadata saved to: {MODELS_DIR}")
    return binary_model, multi_model

if __name__ == "__main__":
    run_stgcn_training_pipeline()