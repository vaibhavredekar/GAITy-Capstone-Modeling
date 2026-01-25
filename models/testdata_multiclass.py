"""
testdata_multiclass.py

Generates a test_data_multiclass.csv file with the correct features for the 5-class model.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def create_test_csv():
    """
    Generates a test_data_multiclass.csv file with the correct features.
    """
    # --- FIX IS HERE ---
    # Determine the script's directory and find the model metadata
    script_dir = Path(__file__).parent.resolve()
    metadata_path = script_dir / "xgboost_gait_5class_metadata.json"
    
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found at: {metadata_path}")
        print("Please run the multiclass_modeling.py script first to generate the model and metadata.")
        return

    # Load the feature names from the model's metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata["feature_cols"]
    class_names = metadata["classes"]
    
    print(f"Loaded {len(feature_names)} feature names from metadata.")
    print(f"Model classes: {class_names}")
    
    # --- Generate realistic-looking data ---
    np.random.seed(42)  # For reproducibility
    n_samples = 10
    test_data = {}

    for feature in feature_names:
        # Generate different distributions based on feature name patterns
        if any(pattern in feature.lower() for pattern in ['mean', 'average', 'median']):
            test_data[feature] = np.random.normal(0.5, 0.2, n_samples)
        elif any(pattern in feature.lower() for pattern in ['std', 'variance', 'deviation']):
            test_data[feature] = np.random.uniform(0, 0.5, n_samples)
        elif any(pattern in feature.lower() for pattern in ['min', 'max', 'range']):
            test_data[feature] = np.random.uniform(0, 1, n_samples)
        else:
            # Default uniform distribution
            test_data[feature] = np.random.uniform(0, 1, n_samples)

    df = pd.DataFrame(test_data)
    
    # Add a few NaN values to test imputation
    if n_samples > 1:
        mask = np.random.random(df.shape) > 0.95
        df = df.mask(mask)

    # Save to CSV
    output_path = Path("testdata_multiclass.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Test data saved to: {output_path.resolve()}")
    print(f"Shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())

if __name__ == "__main__":
    create_test_csv()