# create_test_data.py

import json
import pandas as pd
import numpy as np
from pathlib import Path

def create_test_csv():
    """
    Generates a test_data.csv file with the correct features for the model.
    """
    # Path to the feature names file
    features_path = Path(__file__).parent / "feature_names.json"
    # features_path = "C:\\Vab\\Data-Science\\spiced-academy\\Projects\\capstone\\GAITy-Capstone-Modeling\\models\\feature_names.json"
    print(features_path)

    if not features_path.exists():
        print(f"ERROR: Feature names file not found at {features_path}")
        print("Please run the baseline_modeling.py script first to generate it.")
        return

    # Load the feature names
    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    print(f"Loaded {len(feature_names)} feature names. Generating test data...")

    # Generate realistic-looking data
    np.random.seed(42)  # For reproducibility
    n_samples = 20
    test_data = {}

    for feature in feature_names:
        # Generate different distributions based on feature name patterns
        if any(pattern in feature.lower() for pattern in ['mean', 'average', 'median']):
            # Central tendency measures
            test_data[feature] = np.random.normal(0.5, 0.2, n_samples)
        elif any(pattern in feature.lower() for pattern in ['std', 'variance', 'deviation']):
            # Variability measures
            test_data[feature] = np.random.uniform(0, 0.5, n_samples)
        elif any(pattern in feature.lower() for pattern in ['min', 'max', 'range']):
            # Range measures
            test_data[feature] = np.random.uniform(0, 1, n_samples)
        else:
            # Default uniform distribution
            test_data[feature] = np.random.uniform(0, 1, n_samples)

    df = pd.DataFrame(test_data)

    # Add a few NaN values to test imputation
    mask = np.random.random(df.shape) > 0.95
    df = df.mask(mask)

    # Save to CSV
    output_path = Path(__file__).parent / "test_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Test data saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())

if __name__ == "__main__":
    create_test_csv()