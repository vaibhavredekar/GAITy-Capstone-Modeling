# create_model_lgbm.py
import lightgbm as lgb
import numpy as np
from pathlib import Path

print(f"LightGBM version: {lgb.__version__}")

Path("models/baseline").mkdir(parents=True, exist_ok=True)

X = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.6],
    [0.3, 0.4, 0.5, 0.6, 0.7],
    [0.4, 0.5, 0.6, 0.7, 0.8],
    [0.9, 0.8, 0.7, 0.6, 0.5],
    [0.8, 0.7, 0.6, 0.5, 0.4],
    [0.7, 0.6, 0.5, 0.4, 0.3],
    [0.6, 0.5, 0.4, 0.3, 0.2],
], dtype=np.float32)

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

print("Training LightGBM model...")
train_data = lgb.Dataset(X, label=y)

params = {
    'objective': 'binary',
    'max_depth': 3,
    'learning_rate': 0.1,
    'verbose': -1
}

bst = lgb.train(params, train_data, num_boost_round=10)
print("✅ Training completed!")

# Save
model_path = "models/baseline/xgboost_model.bin"
bst.save_model(model_path)
print(f"✅ Model saved to: {model_path}")

# Load
bst_loaded = lgb.Booster(model_file=model_path)
print("✅ Model loaded!")

# Test prediction
preds = bst_loaded.predict(X[:2])
print(f"Predictions: {preds}")

print(f"\nFile size: {Path(model_path).stat().st_size} bytes")
print("🎉 Success!")