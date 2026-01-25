# # test_xgboost.py

# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import time

# print("--- Starting XGBoost Diagnostic Test ---")

# # 1. Create a small, simple dataset
# print("Creating a small dummy dataset...")
# X_dummy = pd.DataFrame(np.random.rand(100, 10))  # 100 samples, 10 features
# y_dummy = pd.Series(np.random.randint(0, 2, 100)) # 100 binary labels

# print(f"Dummy X shape: {X_dummy.shape}")
# print(f"Dummy y shape: {y_dummy.shape}")
# print("Data created successfully.")

# # 2. Initialize and train a very small XGBoost model
# print("\nInitializing a tiny XGBoost model...")
# model = xgb.XGBClassifier(
#     n_estimators=5,      # Only 5 trees!
#     max_depth=2,         # Very shallow trees
#     learning_rate=0.5,
#     use_label_encoder=False,
#     eval_metric='logloss'
# )
# print("Model initialized.")

# print("\nStarting training... (this should be very fast)")
# start_time = time.time()

# try:
#     model.fit(X_dummy, y_dummy)
    
#     end_time = time.time()
#     duration = end_time - start_time
    
#     print(f"SUCCESS: Training completed in {duration:.2f} seconds.")
#     print("Your XGBoost installation is working correctly.")
    
# except Exception as e:
#     print(f"FAILURE: An error occurred during training.")
#     print(f"The error was: {e}")
#     print("There may be an issue with your XGBoost environment or hardware.")

# print("\n--- Diagnostic Test Finished ---")


# test_xgboost_single_thread.py

import pandas as pd
import numpy as np
import xgboost as xgb
import time

print("--- Starting XGBoost Single-Thread Diagnostic Test ---")

# 1. Create a small, simple dataset
print("Creating a small dummy dataset...")
X_dummy = pd.DataFrame(np.random.rand(100, 10))  # 100 samples, 10 features
y_dummy = pd.Series(np.random.randint(0, 2, 100)) # 100 binary labels

print(f"Dummy X shape: {X_dummy.shape}")
print(f"Dummy y shape: {y_dummy.shape}")
print("Data created successfully.")

# 2. Initialize and train a very small XGBoost model, forcing it to use only one thread
print("\nInitializing a tiny XGBoost model (n_jobs=1)...")
model = xgb.XGBClassifier(
    n_estimators=5,      # Only 5 trees!
    max_depth=2,         # Very shallow trees
    learning_rate=0.5,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=1             # <--- THIS IS THE FIX
)
print("Model initialized.")

print("\nStarting training with a single thread... (this should be very fast)")
start_time = time.time()

try:
    model.fit(X_dummy, y_dummy)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"SUCCESS: Training completed in {duration:.2f} seconds.")
    print("The issue was parallelism. You must use n_jobs=1.")
    
except Exception as e:
    print(f"FAILURE: An error occurred during training.")
    print(f"The error was: {e}")
    print("The issue is not parallelism. We need to try reinstalling XGBoost.")

print("\n--- Diagnostic Test Finished ---")