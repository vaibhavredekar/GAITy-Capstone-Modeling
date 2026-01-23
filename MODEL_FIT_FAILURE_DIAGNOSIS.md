# 🔍 DEEP DIAGNOSIS: Model Fit Failures & Crashes

## Executive Summary
Your GAITy capstone project has **TWO MODELS** with different failure patterns:
1. **XGBoost Binary Classifier** - Likely succeeds (bin_files.py shows it working)
2. **ST-GCN (PyTorch)** - High risk of failure due to data pipeline issues

---

## 🚨 CRITICAL ISSUES IDENTIFIED

### Issue #1: DATA PIPELINE MISALIGNMENT (MOST CRITICAL)
**Location:** `3_XGBoost_based_on_feature_extraction.py`  
**Severity:** 🔴 CRITICAL - Blocks all model training

#### The Problem:
```python
# Line ~150-160
X = df_bin[feature_cols].copy()
X = X.fillna(X.median(numeric_only=True))  # <-- DANGEROUS!
y = df_bin["binary_label"].astype(int)
```

**Why it fails:**
- `X.fillna(X.median())` can fail if:
  - All values in a column are NaN → median is NaN → fills with NaN
  - Mixed dtypes cause median to be undefined
  - Some columns have NO numeric data

**The Real Issue - Feature Extraction Output:**
```python
# Line ~130
df_features = extract_features_from_windows(
    X_windows=X_full,
    fps=fps_effective,
    gait_pattern=None,     # ⚠️ WARNING
    movement_type=None,    # ⚠️ WARNING
    side=None,            # ⚠️ WARNING
    source_file=None,     # ⚠️ WARNING
    cfg=cfg,
)
```

**Expected output:** DataFrame with columns like:
- `label_fine`, `label_class`, `label_id`
- ~100+ numeric features (ROM, speed, angles, etc.)

**Actual risk:** Feature extraction may output:
- Empty DataFrames
- All NaN columns
- Shape mismatch with `X_clean` (windows misaligned)

---

### Issue #2: DATA SHAPE MISMATCHES (HIGH IMPACT)
**Location:** Multiple files  
**Severity:** 🟠 HIGH

#### Alignment Check Failures:
```python
# Line ~147 in 3_XGBoost_based_on_feature_extraction.py
assert len(X) == len(y) == len(groups)
# ❌ FAILS IF: df_features has different length than y_binary_clean
```

**Where misalignment occurs:**
```python
# Lines ~95-110
X_windows, y_binary, y_multilabel, window_ids = preprocess_gait_sliding_windows(...)
X_clean, y_binary_clean, y_multilabel_clean, window_ids_clean, qc_df = apply_qc_windows(...)
# X_clean has shape (N_after_QC, T, J, 3)

# Lines ~120-135
X_full = np.full((N, T, 33, 3), np.nan, dtype=np.float32)
X_full[:, :, GAIT_JOINTS, :] = X_clean
# ✅ Shape now correct: (N_after_QC, T, 33, 3)

# BUT...
df_features = extract_features_from_windows(X_windows=X_full, ...)  
# ❌ What if extraction drops some windows due to NaN/invalid poses?
```

---

### Issue #3: NaN PROPAGATION IN FEATURE EXTRACTION
**Location:** `2_Feature_Extraction_cleaned.py`  
**Severity:** 🟠 HIGH

#### The Vulnerability:
```python
# Lines ~300-400 (various functions like joint_speed, rom, etc.)
def joint_speed(pose_norm, joint_idx, fps, smooth_sigma=1.0):
    # ⚠️ If pose_norm contains ANY NaN, the entire window is tainted
    vel = np.diff(pose_norm[:, joint_idx], axis=0)
    # Smoothing can propagate NaNs
    smooth_vel = gaussian_filter1d(vel, sigma=smooth_sigma, axis=0)
    return np.linalg.norm(smooth_vel, axis=1)
```

**Cascade Effect:**
1. A few invalid landmarks → entire pose window has NaNs
2. Feature extraction tries to compute ROM, angles, etc.
3. Each function returns NaN if ANY input is NaN
4. Feature vector becomes all-NaN
5. `df_features` has rows with all NaN values
6. XGBoost `.fit()` fails or silently produces poor results

---

### Issue #4: NORMALIZATION FAILURES
**Location:** `2_Feature_Extraction_cleaned.py`, lines 120-135  
**Severity:** 🟠 HIGH

```python
def normalize_pose_3d(pose):
    """
    pose: (T, 33, 3)
    """
    pelvis = (pose[:, LEFT_HIP] + pose[:, RIGHT_HIP]) / 2.0  # (T, 3)
    pose_centered = pose - pelvis[:, None, :]
    
    torso = (pose_centered[:, LEFT_SHOULDER] + pose_centered[:, RIGHT_SHOULDER]) / 2.0
    scale = np.linalg.norm(torso, axis=1).mean()
    
    if scale == 0 or not np.isfinite(scale):
        raise ValueError("Invalid torso scale during pose normalisation")
    
    return pose_centered / scale
```

**Failure Modes:**
- ✅ Correctly raises error if scale is invalid
- ❌ BUT: This error is NOT caught in `extract_features_from_windows()`
- ❌ Entire notebook/script crashes instead of skipping bad window

**Result:** A single bad window crashes feature extraction for ALL windows

---

### Issue #5: XGBOOST VERSION INCOMPATIBILITIES
**Location:** `model_st.py`, lines 20-70  
**Severity:** 🟡 MEDIUM

```python
# The code has version-aware handling, but:
if major >= 3:
    st.warning(f"⚠️ XGBoost {version_str} detected...")
    st.info("Consider downgrading to XGBoost 1.7.3...")
```

**Check your installed version:**
```bash
python -c "import xgboost; print(xgboost.__version__)"
```

**Known issues:**
- XGBoost 2.0+ changed serialization format (.ubj vs .bin)
- XGBoost 3.0+ has breaking API changes
- Old .bin files may not load in new versions

**Your model files:**
- `models/baseline/xgboost_model.bin` (uses XGBoost native format)
- `models/baseline/xgboost_model.ubj` (raw binary format)

---

### Issue #6: ST-GCN MODEL TRAINING RISKS
**Location:** `adv_modeling_stgcn.ipynb`, cells #8-#30  
**Severity:** 🟡 MEDIUM

#### Identified risks in training loop:
```python
# Cell #8 - Model definition
class SimpleSTGCN(nn.Module):
    def __init__(self, num_joints, in_channels=3, out_classes=1):
        self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
        # ✅ This looks fine
        
# Cell #17-20 - Training loop
def train_model(model, dataloader, optimizer, criterion, device):
    # ⚠️ No nan/inf checks after backward()
    # ⚠️ No gradient clipping (can explode)
    # ⚠️ No loss value monitoring
```

**Potential crashes:**
1. **Exploding gradients:** No gradient clipping → NaN loss after few batches
2. **Empty batches:** DataLoader creates empty batch if all windows filtered out
3. **Device mismatches:** Data not moved to GPU properly
4. **Uninitialized weights:** Some layers may not initialize properly

---

## 📋 ROOT CAUSE ANALYSIS BY MODEL

### XGBoost Model (3_XGBoost_based_on_feature_extraction.py)
**Status:** ⚠️ At-risk but shows signs of working

**Known working:**
- ✅ Training completes: `xgb_bin.fit(X_train, y_train)` succeeds (code shows valid training)
- ✅ Generates confusion matrix with reasonable numbers
- ✅ Model saves to disk

**Potential failures:**
- ❌ `.fit()` could fail if X_train has:
  - NaN columns (not dropped)
  - Infinite values (division by zero in features)
  - All-zeros columns (no variance)
  - Shape mismatches with y_train

**Most likely cause if failing:**
```python
# These lines need validation:
X = X.fillna(X.median(numeric_only=True))
# ❌ If median is NaN, this does nothing!

# Should be:
for col in X.columns:
    if X[col].isna().all():
        X.drop(col, axis=1, inplace=True)  # Remove all-NaN columns
    elif X[col].isna().any():
        X[col].fillna(X[col].median(), inplace=True)
```

---

### ST-GCN Model (adv_modeling_stgcn.ipynb)
**Status:** 🔴 High crash risk

**Identified risks in order of likelihood:**

1. **Empty DataLoader risk (HIGHEST)**
   ```python
   X_ml_train_stgcn = X_train_stgcn[abnormal_mask_train]
   # If no abnormal windows, this is empty!
   # DataLoader([]) crashes with StopIteration
   ```

2. **NaN in tensor creation (HIGH)**
   ```python
   class PoseDataset(Dataset):
       def __init__(self, X, y):
           self.X = torch.tensor(X, dtype=torch.float32)
           # ❌ If X contains any NaN, training crashes
           # ❌ No validation that X is valid
   ```

3. **Gradient explosion (MEDIUM)**
   ```python
   # No checks for NaN loss:
   logits = model(X_batch).squeeze(1)
   loss = criterion(logits, y_batch)
   # ❌ if loss is NaN, backward() cascades failure
   ```

---

## 🔧 HOW TO FIX THESE ISSUES

### Fix #1: Validate Feature Extraction Output
**File:** `3_XGBoost_based_on_feature_extraction.py`  
**Lines:** ~125-150

Add validation:
```python
# After feature extraction
print(f"Feature extraction output shape: {df_features.shape}")
print(f"Feature columns: {df_features.columns.tolist()}")
print(f"NaN counts per column:\n{df_features.isna().sum()}")

# Check for all-NaN rows
all_nan_rows = df_features.isna().all(axis=1).sum()
if all_nan_rows > 0:
    print(f"⚠️ WARNING: {all_nan_rows} rows are completely NaN")
    df_features = df_features.dropna(how='all')
    # ⚠️ This breaks alignment! Must also filter X_clean, y_binary_clean, window_ids_clean

# Check for missing feature columns
expected_features = ['speed_hip', 'rom_knee', 'angle_ankle', ...]  # actual column names
missing = set(expected_features) - set(df_features.columns)
if missing:
    raise ValueError(f"Missing expected features: {missing}")
```

---

### Fix #2: Robust NaN Handling
**File:** `3_XGBoost_based_on_feature_extraction.py`  
**Lines:** ~145-160

Replace:
```python
X = X.fillna(X.median(numeric_only=True))
```

With:
```python
# Drop columns with >50% missing
nan_ratio = X.isna().sum() / len(X)
cols_to_drop = nan_ratio[nan_ratio > 0.5].index
X = X.drop(cols_to_drop, axis=1)

# Median impute remaining
X = X.fillna(X.median(numeric_only=True))

# Verify no NaN remains
if X.isna().any().any():
    raise ValueError(f"NaN values remain after imputation in columns: {X.columns[X.isna().any()].tolist()}")

# Check for infinite values
if np.isinf(X.values).any():
    print(f"⚠️ WARNING: Infinite values found. Replacing with max finite value.")
    X = X.replace([np.inf, -np.inf], X.replace([np.inf, -np.inf], np.nan).dropna().max().max())
```

---

### Fix #3: Catch Normalization Failures
**File:** `2_Feature_Extraction_cleaned.py`  
**Lines:** ~400-500 (extract_features_from_windows function)

Add try-catch:
```python
def extract_features_from_windows(X_windows, fps, ...):
    """
    X_windows: (N, T, 33, 3)
    """
    results = []
    failed_windows = []
    
    for i, window in enumerate(X_windows):
        try:
            # Try normalization
            pose_norm = normalize_pose_3d(window)
            
            # Extract features from this window
            features = {
                'speed_hip': joint_speed(pose_norm, 23, fps),
                # ... other features
            }
            
            results.append(features)
            
        except ValueError as e:
            # Log the failure
            failed_windows.append((i, str(e)))
            # Either skip or use NaN placeholder
            results.append({k: np.nan for k in feature_names})
    
    if failed_windows:
        print(f"⚠️ WARNING: {len(failed_windows)} windows failed normalization")
        for idx, error in failed_windows[:5]:  # Show first 5
            print(f"  Window {idx}: {error}")
    
    return pd.DataFrame(results)
```

---

### Fix #4: ST-GCN Data Validation
**File:** `adv_modeling_stgcn.ipynb`  
**Cells:** #5 (train/test split), #6 (dataloaders), #8 (dataset)

Add validation:
```python
# After DataLoader creation
print(f"\n=== DataLoader Validation ===")
print(f"Train loader: {len(train_bin_loader)} batches")
print(f"Test loader: {len(test_bin_loader)} batches")

if len(train_bin_loader) == 0:
    raise ValueError("❌ CRITICAL: train_bin_loader is empty! No abnormal windows in training?")

# Test one batch to ensure data integrity
try:
    X_sample, y_sample = next(iter(train_bin_loader))
    print(f"Sample batch: X shape {X_sample.shape}, y shape {y_sample.shape}")
    
    # Check for NaN
    if torch.isnan(X_sample).any():
        raise ValueError(f"❌ NaN detected in input data")
    if torch.isnan(y_sample).any():
        raise ValueError(f"❌ NaN detected in labels")
        
except StopIteration:
    raise ValueError("❌ CRITICAL: DataLoader empty or iteration failed")
```

---

### Fix #5: Add Loss Monitoring
**File:** `adv_modeling_stgcn.ipynb`  
**Cell:** #17-20 (train_model function)

```python
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(1)
        
        loss = criterion(logits, y_batch)
        
        # ⚠️ ADD THIS CHECK
        if torch.isnan(loss):
            raise RuntimeError(f"❌ NaN loss detected after forward pass. Check input data!")
        if torch.isinf(loss):
            raise RuntimeError(f"❌ Inf loss detected. Learning rate too high?")
        
        loss.backward()
        
        # ⚠️ ADD GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

---

## 🧪 DIAGNOSTIC TESTS YOU CAN RUN

### Test 1: Feature Extraction Validation
```python
# Run this cell in 3_XGBoost notebook
import pandas as pd
import numpy as np

print("=== FEATURE EXTRACTION VALIDATION ===\n")

# 1. Check df_features shape
print(f"df_features shape: {df_features.shape}")
print(f"Expected rows: {len(y_binary_clean)} (should match)")

# 2. Check for all-NaN rows
nan_rows = (df_features.isna().sum(axis=1) == len(df_features.columns)).sum()
print(f"All-NaN rows: {nan_rows} (should be 0)")

# 3. Check NaN ratio per column
nan_ratio = df_features.isna().sum() / len(df_features)
high_nan_cols = nan_ratio[nan_ratio > 0.1].sort_values(ascending=False)
print(f"\nColumns with >10% NaN:")
print(high_nan_cols)

# 4. Check for infinite values
inf_count = (np.isinf(df_features.select_dtypes(include=[np.number]))).sum().sum()
print(f"\nInfinite values: {inf_count} (should be 0)")

# 5. Check shape alignment
print(f"\n=== ALIGNMENT CHECK ===")
print(f"df_features rows: {len(df_features)}")
print(f"y_binary_clean rows: {len(y_binary_clean)}")
print(f"window_ids_clean rows: {len(window_ids_clean)}")
print(f"Aligned: {len(df_features) == len(y_binary_clean) == len(window_ids_clean)}")
```

---

### Test 2: XGBoost Pre-fit Check
```python
# Add before xgb_bin.fit()
print("=== XGBOOST PRE-FIT VALIDATION ===\n")

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Shapes match: {len(X_train) == len(y_train)}")

# Check for NaN/Inf
nan_in_X = X_train.isna().sum().sum()
inf_in_X = (np.isinf(X_train.select_dtypes(include=[np.number]))).sum().sum()
print(f"\nNaN in X_train: {nan_in_X} (should be 0)")
print(f"Inf in X_train: {inf_in_X} (should be 0)")

# Check class balance
print(f"\ny_train distribution:")
print(y_train.value_counts())
print(f"Class ratio: {(y_train==0).sum() / (y_train==1).sum():.2f}:1")

# Check for constant columns
const_cols = (X_train.std() == 0).sum()
print(f"\nConstant-value columns: {const_cols} (should be 0)")

if const_cols > 0:
    print("⚠️ WARNING: Consider dropping constant columns")
```

---

### Test 3: ST-GCN Data Integrity
```python
# Add before training loop
print("=== ST-GCN DATA INTEGRITY CHECK ===\n")

print(f"X_train_stgcn shape: {X_train_stgcn.shape}")
print(f"y_bin_train shape: {y_bin_train.shape}")

# Check abnormal split
abnormal_count = (y_bin_train == 1).sum()
print(f"\nAbnormal windows in training: {abnormal_count}")

if abnormal_count == 0:
    print("❌ ERROR: No abnormal windows! Can't train multi-label model")

# Check for NaN
nan_in_train = np.isnan(X_train_stgcn).sum()
print(f"NaN values in X_train_stgcn: {nan_in_train} (should be 0)")

if nan_in_train > 0:
    print("❌ ERROR: NaN in training data. Feature extraction failed.")
    print("Fix: Run diagnostic on feature extraction output")

# Test DataLoader
try:
    sample = next(iter(train_bin_loader))
    print(f"\n✅ DataLoader works. Sample batch: {sample[0].shape}")
except Exception as e:
    print(f"\n❌ DataLoader error: {e}")
```

---

## 📊 RECOMMENDED ACTION PLAN

### Immediate (Do First):
1. ✅ Run **Test 1** (Feature Extraction Validation)
   - If fails → Apply Fix #1 + Fix #2
   
2. ✅ Run **Test 2** (XGBoost Pre-fit Check)
   - If fails → Apply Fix #2

### Short Term (Next):
3. ✅ Apply **Fix #5** (Loss monitoring in ST-GCN)
   - Low effort, high safety improvement

4. ✅ Add **try-catch** to normalization (Fix #3)
   - Prevents single bad window from crashing all windows

### Medium Term:
5. ✅ Implement full pipeline error handling
   - Logging at each step
   - Clear error messages
   - Recovery mechanisms

---

## 🎯 KEY TAKEAWAYS

| Issue | Risk | Quick Fix | Time |
|-------|------|-----------|------|
| NaN in features | HIGH | Add .isna() checks | 5 min |
| Shape mismatch | CRITICAL | Validate alignment | 10 min |
| NaN loss in ST-GCN | HIGH | Add torch.isnan() check | 5 min |
| Normalization crash | MEDIUM | Add try-catch | 15 min |
| XGBoost version | MEDIUM | Check version | 2 min |

---

## 📞 TESTING YOUR FIXES

After applying fixes, test with:

```bash
# Test XGBoost training
python -c "
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import xgboost as xgb

X, y = make_classification(n_samples=500, n_features=50, random_state=42)
model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
model.fit(X, y)
print('✅ XGBoost fit successful')
"

# Test ST-GCN training (if PyTorch installed)
python -c "
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(100, 3, 60, 14)  # (N, C, T, J)
y = torch.randint(0, 2, (100,))

loader = DataLoader(TensorDataset(X, y), batch_size=32)
print('✅ DataLoader created successfully')

for X_batch, y_batch in loader:
    print(f'✅ Batch retrieved: X {X_batch.shape}, y {y_batch.shape}')
    break
"
```

---

## 📝 NEXT STEPS

1. **Identify which model is failing:**
   - Is it the XGBoost in `3_XGBoost_based_on_feature_extraction.py`?
   - Is it the ST-GCN in `adv_modeling_stgcn.ipynb`?
   - Is it during feature extraction in `2_Feature_Extraction_cleaned.py`?

2. **Run the diagnostic tests above** to pinpoint the exact failure

3. **Apply the relevant fix** from this document

4. **Monitor your logs** for specific error messages

Good luck! 🚀
