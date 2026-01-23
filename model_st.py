# version_aware_app.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Set page config
st.set_page_config(page_title="GAITy", layout="wide")

# Check XGBoost version and handle accordingly
def get_xgboost_version():
    try:
        import xgboost as xgb
        version = xgb.__version__
        major, minor, patch = map(int, version.split('.'))
        return (major, minor, patch), version
    except:
        return None, "Not installed"

def load_model_safely():
    """Load model with version-aware error handling."""
    model_path = Path("models/baseline/xgboost_model.bin")
    
    if not model_path.exists():
        return None, "Model file not found", None
    
    version_info, version_str = get_xgboost_version()
    
    if version_info is None:
        return None, "XGBoost not installed", None
    
    major, minor, patch = version_info
    
    # Handle different XGBoost versions
    try:
        import xgboost as xgb
        
        # Version-specific handling
        if major >= 3:
            # XGBoost 3.x might need different approach
            st.warning(f"⚠️ XGBoost {version_str} detected. This version may have compatibility issues.")
            st.info("Consider downgrading to XGBoost 1.7.3 for best compatibility.")
            
            # Try to load with error handling
            try:
                model = xgb.XGBClassifier()
                model.load_model(str(model_path))
                return model, f"Model loaded with XGBoost {version_str}", version_str
            except Exception as e:
                return None, f"Failed to load with XGBoost {version_str}: {str(e)}", version_str
        
        elif major == 1 and minor >= 7:
            # XGBoost 1.7+ should work
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            return model, f"Model loaded with XGBoost {version_str}", version_str
        
        else:
            # Older versions
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            return model, f"Model loaded with XGBoost {version_str}", version_str
            
    except Exception as e:
        return None, f"Failed to load model: {str(e)}", version_str

def create_mock_prediction():
    """Create a mock prediction for testing when model loading fails."""
    import random
    prediction = random.choice([0, 1])
    confidence = random.uniform(0.7, 0.95)
    
    return {
        'prediction': prediction,
        'label': 'Normal' if prediction == 0 else 'Abnormal',
        'confidence': confidence,
        'probabilities': {
            'Normal': 1 - confidence if prediction == 1 else confidence,
            'Abnormal': confidence if prediction == 1 else 1 - confidence
        },
        'mock': True
    }

def main():
    st.title("🚶 GAITy - Version-Aware Gait Analysis")
    
    # Check XGBoost version
    version_info, version_str = get_xgboost_version()
    
    if version_info:
        major, minor, patch = version_info
        st.write(f"XGBoost version: {version_str}")
        
        if major >= 3:
            st.warning("⚠️ XGBoost 3.x detected. This version may have compatibility issues.")
            st.info("Recommended: `pip install xgboost==1.7.3`")
        elif major == 1 and minor >= 7:
            st.success("✅ XGBoost version is compatible")
        else:
            st.warning("⚠️ XGBoost version might be too old")
    else:
        st.error("❌ XGBoost not installed")
        st.info("Install with: `pip install xgboost==1.7.3`")
        st.stop()
    
    # Try to load model
    model, status, model_version = load_model_safely()
    
    if model:
        st.success(f"✅ {status}")
        use_mock = st.checkbox("Use mock prediction (for testing)", value=False)
    else:
        st.error(f"❌ {status}")
        st.info("Using mock prediction for demonstration")
        use_mock = True
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV with pose landmarks", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Check required columns
            required_cols = ['frame', 'landmark_id', 'x_norm', 'y_norm', 'z_norm']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                # Display preview
                with st.expander("Data Preview"):
                    st.dataframe(df.head(10))
                
                # Extract simple features
                if st.button("Extract Features and Predict"):
                    with st.spinner("Processing..."):
                        features = {}
                        
                        # Basic statistics for each coordinate
                        for coord in ['x_norm', 'y_norm', 'z_norm']:
                            if coord in df.columns:
                                features[f"{coord}_mean"] = df[coord].mean()
                                features[f"{coord}_std"] = df[coord].std()
                                features[f"{coord}_min"] = df[coord].min()
                                features[f"{coord}_max"] = df[coord].max()
                        
                        # Display features
                        st.write("Extracted Features:")
                        st.json(features)
                        
                        # Make prediction
                        if use_mock or not model:
                            result = create_mock_prediction()
                            st.warning("⚠️ Using mock prediction (model not available)")
                        else:
                            try:
                                feature_df = pd.DataFrame([features]).fillna(0)
                                prediction = model.predict(feature_df)[0]
                                probabilities = model.predict_proba(feature_df)[0]
                                
                                result = {
                                    'prediction': int(prediction),
                                    'label': 'Normal' if prediction == 0 else 'Abnormal',
                                    'confidence': float(max(probabilities)),
                                    'probabilities': {
                                        'Normal': float(probabilities[0]),
                                        'Abnormal': float(probabilities[1])
                                    },
                                    'mock': False
                                }
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
                                result = create_mock_prediction()
                                st.warning("Falling back to mock prediction")
                        
                        # Display results
                        st.write("---")
                        st.header("Prediction Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediction", result['label'])
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                        with col3:
                            st.metric("Mock", "Yes" if result['mock'] else "No")
                        
                        # Probability chart
                        st.write("Prediction Probabilities:")
                        probs = result['probabilities']
                        st.bar_chart(probs)
                        
                        # Download results
                        results_df = pd.DataFrame([{
                            'prediction': result['prediction'],
                            'label': result['label'],
                            'confidence': result['confidence'],
                            'normal_probability': result['probabilities']['Normal'],
                            'abnormal_probability': result['probabilities']['Abnormal'],
                            'mock': result['mock']
                        }])
                        
                        # Add features
                        for key, value in features.items():
                            results_df[key] = value
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "gait_analysis_results.csv",
                            "text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()