"""
predict_multiclass.py

Production-grade script for loading a pre-trained 5-class XGBoost model and making predictions.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multiclass_predictions.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Data class to hold multi-class prediction results"""
    sample_indices: List[int]
    predicted_class_names: List[str]
    predicted_class_ids: List[int]
    probabilities: List[np.ndarray]  # Array of probabilities for each class
    confidence_scores: List[float]  # Confidence in prediction (max probability)
    timestamps: List[str]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        return pd.DataFrame({
            'sample_index': self.sample_indices,
            'predicted_class_name': self.predicted_class_names,
            'predicted_class_id': self.predicted_class_ids,
            'confidence': self.confidence_scores,
            'prediction_timestamp': self.timestamps
        })
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert results to list of dictionaries"""
        return [
            {
                'sample_index': idx,
                'predicted_class_name': pred_name,
                'predicted_class_id': pred_id,
                'probabilities': prob.tolist(), # Convert numpy array to list for JSON serialization
                'confidence': conf,
                'prediction_timestamp': ts
            }
            for idx, pred_name, pred_id, prob, conf, ts in zip(
                self.sample_indices,
                self.predicted_class_names,
                self.predicted_class_ids,
                self.probabilities,
                self.confidence_scores,
                self.timestamps
            )
        ]


class MulticlassGaitPredictor:
    """Production-grade 5-class gait prediction model handler"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the multi-class gait predictor.
        
        Args:
            model_dir: Directory containing model files. If None, uses current directory.
        """
        self.model_dir = model_dir or Path(__file__).parent
        self.model = None
        self.metadata = None
        self.class_names = []
        self.id_to_class = {}
        self.is_loaded = False
        
        # Model file paths for the 5-class model
        self.model_path = self.model_dir / "xgboost_gait_5class.bin"
        self.metadata_path = self.model_dir / "xgboost_gait_5class_metadata.json"
        
    def load_model(self) -> bool:
        """
        Load the XGBoost model and its associated metadata.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            if not self.metadata_path.exists():
                logger.error(f"Metadata file not found: {self.metadata_path}")
                return False
            
            # Load the model
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            
            # Load the metadata
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Extract class information from metadata
            self.id_to_class = {int(k): v for k, v in self.metadata["id_to_class"].items()}
            self.class_names = [self.id_to_class[i] for i in range(len(self.id_to_class))]
            
            logger.info(f"Loaded {len(self.class_names)} classes: {self.class_names}")
            self.is_loaded = True
            logger.info("Model and metadata loaded successfully")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in metadata file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False
    
    def validate_input(self, input_data: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate input data format and content.
        """
        if not self.is_loaded:
            return False, "Model not loaded"
        
        if input_data.empty:
            return False, "Input data is empty"
        
        # Check for required features
        missing_features = set(self.metadata["feature_cols"]) - set(input_data.columns)
        if missing_features:
            return False, f"Missing required features: {missing_features}"
        
        # Check for NaN values
        if input_data.isna().any().any():
            nan_count = input_data.isna().sum().sum()
            logger.warning(f"Input contains {nan_count} NaN values. Will apply imputation.")
        
        # Check data types
        non_numeric_cols = input_data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            return False, f"Non-numeric columns found: {non_numeric_cols}"
        
        return True, "Input data is valid"
    
    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data to match model requirements.
        """
        logger.info(f"Preprocessing input data with shape: {input_data.shape}")
        
        processed_data = input_data.copy()
        
        # Select only required features
        processed_data = processed_data[self.metadata["feature_cols"]]
        
        # Handle missing values (median imputation)
        if processed_data.isna().any().any():
            nan_before = processed_data.isna().sum().sum()
            processed_data = processed_data.fillna(processed_data.median())
            nan_after = processed_data.isna().sum().sum()
            logger.info(f"Imputed {nan_before - nan_after} NaN values")
        
        logger.info(f"Preprocessed data shape: {processed_data.shape}")
        return processed_data
    
    def predict(self, input_data: pd.DataFrame) -> PredictionResult:
        """
        Make predictions on input data.
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate and preprocess input
        is_valid, error_msg = self.validate_input(input_data)
        if not is_valid:
            raise ValueError(f"Invalid input data: {error_msg}")
        
        processed_data = self.preprocess_input(input_data)
        
        # Make predictions
        start_time = time.time()
        y_pred_indices = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        
        # Prepare results
        timestamp = datetime.now().isoformat()
        
        # Map predicted indices to class names
        predicted_class_names = [self.id_to_class[idx] for idx in y_pred_indices]
        
        # Get the probability of the predicted class for each sample
        predicted_probabilities = probabilities[np.arange(len(y_pred_indices)), y_pred_indices]
        
        results = PredictionResult(
            sample_indices=list(range(len(input_data))),
            predicted_class_names=predicted_class_names,
            predicted_class_ids=y_pred_indices.tolist(),
            probabilities=probabilities,
            confidence_scores=predicted_probabilities.tolist(),
            timestamps=[timestamp] * len(input_data)
        )
        
        logger.info(f"Made {len(input_data)} predictions in {prediction_time:.3f} seconds")
        
        return results

    def predict_batch(self, input_file: Path, output_file: Optional[Path] = None) -> bool:
        """
        Process a batch of predictions from a CSV file.
        """
        try:
            logger.info(f"Processing batch predictions from: {input_file}")
            
            if input_file.suffix.lower() != '.csv':
                logger.error(f"Unsupported file format: {input_file.suffix}. Please provide a CSV file.")
                return False
            
            input_data = pd.read_csv(input_file)
            
            # Make predictions
            results = self.predict(input_data)
            results_df = results.to_dataframe()
            
            # Save or display results
            if output_file:
                results_df.to_csv(output_file, index=False)
                logger.info(f"Predictions saved to: {output_file}")
            else:
                # Display results
                print("\n" + "="*60)
                print("MULTI-CLASS PREDICTION RESULTS")
                print("="*60)
                print(results_df.to_string(index=False))
                
                # Summary statistics
                print("\n" + "="*60)
                print("SUMMARY STATISTICS")
                print("="*60)
                print(f"Total Samples: {len(results_df)}")
                print(results_df['predicted_class_name'].value_counts())
                print(f"Average Confidence: {results_df['confidence'].mean():.2%}")
            
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
            "model_type": "XGBoost Multi-class Classifier",
            "class_count": len(self.class_names),
            "classes": self.class_names,
            "feature_count": len(self.metadata["feature_cols"]),
        }
        
        return info


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='5-Class Gait Analysis Prediction System',
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
                       help='Input CSV file containing features')
    parser.add_argument('--output', '-o', type=Path, 
                       help='Output file for predictions (CSV)')
    parser.add_argument('--model-dir', '-m', type=Path, default=Path(__file__).parent,
                       help='Directory containing the script and model files')
    parser.add_argument('--test', action='store_true',
                       help='Test prediction with dummy data')
    parser.add_argument('--info', action='store_true',
                       help='Show model information')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MulticlassGaitPredictor(args.model_dir)
    
    # Load model
    if not predictor.load_model():
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Handle different modes
    if args.info:
        info = predictor.get_model_info()
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        for key, value in info.items():
            print(f"{key}: {value}")
        sys.exit(0)
    
    elif args.test:
        # Test with dummy data
        # This would require a helper function to create valid dummy data
        print("Test mode requires a pre-generated test data file.")
        print("Please run 'testdata_multiclass.py' first to create 'testdata_multiclass.csv'.")
        test_data_path = Path("testdata_multiclass.csv")
        if not test_data_path.exists():
            print(f"Test data file not found at: {test_data_path}")
            sys.exit(1)
        success = predictor.predict_batch(test_data_path)
        if not success:
            logger.error("Test prediction failed")
            sys.exit(1)
    
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