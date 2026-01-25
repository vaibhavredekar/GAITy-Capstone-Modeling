"""
predict_gait.py

Production-grade script for loading a pre-trained XGBoost model and making predictions on gait data.

Features:
- Comprehensive error handling and logging
- Input validation and sanitization
- Configuration management
- Support for different input formats
- Batch prediction capabilities
- Performance monitoring

Usage:
    python predict_gait.py --input data.csv --output predictions.csv
    python predict_gait.py --input data.csv  # Prints results to console
    python predict_gait.py --help  # Show help
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass
import time
from datetime import datetime
import joblib  # Alternative for saving/loading if needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gait_predictions.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Data class to hold prediction results"""
    sample_indices: List[int]
    predictions: List[str]
    probabilities_normal: List[float]
    probabilities_abnormal: List[float]
    confidence_scores: List[float]  # Confidence in prediction (max probability)
    timestamps: List[str]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        return pd.DataFrame({
            'sample_index': self.sample_indices,
            'prediction': self.predictions,
            'probability_normal': self.probabilities_normal,
            'probability_abnormal': self.probabilities_abnormal,
            'confidence': self.confidence_scores,
            'prediction_timestamp': self.timestamps
        })
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert results to list of dictionaries"""
        return [
            {
                'sample_index': idx,
                'prediction': pred,
                'probability_normal': prob_norm,
                'probability_abnormal': prob_abnorm,
                'confidence': conf,
                'prediction_timestamp': ts
            }
            for idx, pred, prob_norm, prob_abnorm, conf, ts in zip(
                self.sample_indices,
                self.predictions,
                self.probabilities_normal,
                self.probabilities_abnormal,
                self.confidence_scores,
                self.timestamps
            )
        ]


class GaitPredictor:
    """Production-grade gait prediction model handler"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the gait predictor.
        
        Args:
            model_dir: Directory containing model files. If None, uses current directory.
        """
        self.model_dir = model_dir or Path(__file__).parent
        self.model = None
        self.feature_names = None
        self.feature_importances = None
        self.model_metadata = None
        self.is_loaded = False
        
        # Model file paths
        self.model_path = self.model_dir / "xgboost_model.bin"
        self.features_path = self.model_dir / "feature_names.json"
        self.metadata_path = self.model_dir / "model_metadata.json"
        
        # Statistics for monitoring
        self.predictions_count = 0
        self.total_prediction_time = 0
        
    def load_model(self) -> bool:
        """
        Load the XGBoost model and associated files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Check if model files exist
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            if not self.features_path.exists():
                logger.error(f"Feature names file not found: {self.features_path}")
                return False
            
            # Load the model
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            
            # Load feature names
            with open(self.features_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load metadata if exists
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded model metadata: {self.model_metadata.get('model_info', 'Unknown')}")
            
            # Extract feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = dict(zip(
                    self.feature_names, 
                    self.model.feature_importances_
                ))
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in feature names file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False
    
    def validate_input(self, input_data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate input data format and content.
        
        Args:
            input_data: DataFrame containing input features
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.is_loaded:
            return False, "Model not loaded"
        
        if input_data.empty:
            return False, "Input data is empty"
        
        # Check for required features
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features:
            return False, f"Missing required features: {missing_features}"
        
        # Check for extra features
        extra_features = set(input_data.columns) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Input contains extra features that will be ignored: {extra_features}")
        
        # Check for NaN values
        if input_data.isna().any().any():
            nan_count = input_data.isna().sum().sum()
            logger.warning(f"Input contains {nan_count} NaN values. Will apply imputation.")
        
        # Check data types - ensure numeric
        non_numeric_cols = input_data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            return False, f"Non-numeric columns found: {non_numeric_cols}"
        
        return True, "Input data is valid"
    
    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data to match model requirements.
        
        Args:
            input_data: Raw input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Preprocessing input data with shape: {input_data.shape}")
        
        # Create a copy to avoid modifying original
        processed_data = input_data.copy()
        
        # Select only required features
        processed_data = processed_data[self.feature_names]
        
        # Handle missing values - use median imputation
        # Note: In production, you should use the same imputation strategy as during training
        if processed_data.isna().any().any():
            nan_before = processed_data.isna().sum().sum()
            # Use column-wise median for imputation
            processed_data = processed_data.fillna(processed_data.median())
            nan_after = processed_data.isna().sum().sum()
            logger.info(f"Imputed {nan_before - nan_after} NaN values")
        
        # Ensure correct data types
        processed_data = processed_data.astype(np.float32)
        
        # Optionally: Apply same scaling/transformation as training data
        # if self.scaler is not None:
        #     processed_data = self.scaler.transform(processed_data)
        
        logger.info(f"Preprocessed data shape: {processed_data.shape}")
        return processed_data
    
    def predict(self, input_data: pd.DataFrame) -> PredictionResult:
        """
        Make predictions on input data.
        
        Args:
            input_data: DataFrame containing input features
            
        Returns:
            PredictionResult object with predictions and metadata
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate input
        is_valid, error_msg = self.validate_input(input_data)
        if not is_valid:
            raise ValueError(f"Invalid input data: {error_msg}")
        
        # Preprocess input
        start_time = time.time()
        processed_data = self.preprocess_input(input_data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        self.predictions_count += len(input_data)
        self.total_prediction_time += prediction_time
        
        # Prepare results
        timestamp = datetime.now().isoformat()
        results = PredictionResult(
            sample_indices=list(range(len(input_data))),
            predictions=["abnormal" if pred == 1 else "normal" for pred in predictions],
            probabilities_normal=probabilities[:, 0].tolist(),
            probabilities_abnormal=probabilities[:, 1].tolist(),
            confidence_scores=np.max(probabilities, axis=1).tolist(),
            timestamps=[timestamp] * len(input_data)
        )
        
        logger.info(f"Made {len(input_data)} predictions in {prediction_time:.3f} seconds "
                   f"({prediction_time/len(input_data):.4f} sec per sample)")
        
        return results
    
    def predict_batch(self, input_file: Path, output_file: Optional[Path] = None) -> bool:
        """
        Process a batch of predictions from file.
        
        Args:
            input_file: Path to input CSV file
            output_file: Optional path to save predictions
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing batch predictions from: {input_file}")
            
            # Load input data
            if input_file.suffix.lower() == '.csv':
                input_data = pd.read_csv(input_file)
            elif input_file.suffix.lower() in ['.xlsx', '.xls']:
                input_data = pd.read_excel(input_file)
            else:
                logger.error(f"Unsupported file format: {input_file.suffix}")
                return False
            
            # Make predictions
            results = self.predict(input_data)
            
            # Convert to DataFrame
            results_df = results.to_dataframe()
            
            # Add original data (excluding predictions)
            for col in input_data.columns:
                if col not in results_df.columns:
                    results_df[col] = input_data[col].values
            
            # Save or display results
            if output_file:
                if output_file.suffix.lower() == '.csv':
                    results_df.to_csv(output_file, index=False)
                elif output_file.suffix.lower() in ['.xlsx', '.xls']:
                    results_df.to_excel(output_file, index=False)
                else:
                    # Default to CSV
                    output_file = output_file.with_suffix('.csv')
                    results_df.to_csv(output_file, index=False)
                logger.info(f"Predictions saved to: {output_file}")
            else:
                # Display results
                print("\n" + "="*60)
                print("PREDICTION RESULTS")
                print("="*60)
                print(results_df.to_string(index=False))
                
                # Summary statistics
                abnormal_count = sum(1 for p in results.predictions if p == "abnormal")
                normal_count = len(results.predictions) - abnormal_count
                avg_confidence = np.mean(results.confidence_scores)
                
                print("\n" + "="*60)
                print("SUMMARY STATISTICS")
                print("="*60)
                print(f"Total Samples: {len(results.predictions)}")
                print(f"Normal Predictions: {normal_count}")
                print(f"Abnormal Predictions: {abnormal_count}")
                print(f"Average Confidence: {avg_confidence:.2%}")
                print(f"Prediction Time: {self.total_prediction_time/len(results.predictions):.4f} sec/sample")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}", exc_info=True)
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"status": "Model not loaded"}
        
        info = {
            "status": "Model loaded successfully",
            "model_type": "XGBoost Classifier",
            "feature_count": len(self.feature_names),
            "predictions_made": self.predictions_count,
            "average_prediction_time": self.total_prediction_time / max(self.predictions_count, 1)
        }
        
        if self.model_metadata:
            info.update(self.model_metadata)
        
        return info
    
    def create_dummy_input(self, n_samples: int = 5) -> pd.DataFrame:
        """
        Create dummy input data for testing.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with dummy data
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Cannot create dummy input without feature names.")
        
        logger.info(f"Creating dummy input with {n_samples} samples and {len(self.feature_names)} features")
        
        # Generate realistic-looking data based on typical ranges
        np.random.seed(42)  # For reproducibility
        
        # Create DataFrame with realistic ranges for gait features
        dummy_data = {}
        for feature in self.feature_names:
            # Generate different distributions based on feature name patterns
            if any(pattern in feature.lower() for pattern in ['mean', 'average', 'median']):
                # Central tendency measures
                dummy_data[feature] = np.random.normal(0.5, 0.2, n_samples)
            elif any(pattern in feature.lower() for pattern in ['std', 'variance', 'deviation']):
                # Variability measures
                dummy_data[feature] = np.random.uniform(0, 0.5, n_samples)
            elif any(pattern in feature.lower() for pattern in ['min', 'max', 'range']):
                # Range measures
                dummy_data[feature] = np.random.uniform(0, 1, n_samples)
            else:
                # Default uniform distribution
                dummy_data[feature] = np.random.uniform(0, 1, n_samples)
        
        df = pd.DataFrame(dummy_data)
        
        # Add some NaN values to test imputation
        if n_samples > 1:
            mask = np.random.random(df.shape) > 0.95
            df = df.mask(mask)
        
        logger.info(f"Dummy input created with shape: {df.shape}")
        return df


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Gait Analysis Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data.csv --output predictions.csv
  %(prog)s --input data.csv  # Display results in console
  %(prog)s --test  # Run with dummy data
  %(prog)s --info  # Show model information
        """
    )
    
    parser.add_argument('--input', '-i', type=Path, 
                       help='Input CSV/Excel file containing features')
    parser.add_argument('--output', '-o', type=Path, 
                       help='Output file for predictions (CSV/Excel)')
    parser.add_argument('--model-dir', '-m', type=Path, default=Path.cwd(),
                       help='Directory containing model files (default: current directory)')
    parser.add_argument('--test', action='store_true',
                       help='Test prediction with dummy data')
    parser.add_argument('--info', action='store_true',
                       help='Show model information')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Initialize predictor
    predictor = GaitPredictor(args.model_dir)
    
    # Load model
    if not predictor.load_model():
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Handle different modes
    if args.info:
        # Show model information
        info = predictor.get_model_info()
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        for key, value in info.items():
            print(f"{key}: {value}")
        sys.exit(0)
    
    elif args.test:
        # Test with dummy data
        dummy_data = predictor.create_dummy_input(n_samples=10)
        results = predictor.predict(dummy_data)
        
        print("\n" + "="*60)
        print("TEST PREDICTIONS (Dummy Data)")
        print("="*60)
        print(results.to_dataframe().to_string(index=False))
        sys.exit(0)
    
    elif args.input:
        # Process input file
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        success = predictor.predict_batch(args.input, args.output)
        if not success:
            logger.error("Batch prediction failed")
            sys.exit(1)
    
    else:
        # No arguments provided
        parser.print_help()
        print("\n" + "="*60)
        print("ERROR: No action specified.")
        print("Please provide --input, --test, or --info")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Prediction cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)