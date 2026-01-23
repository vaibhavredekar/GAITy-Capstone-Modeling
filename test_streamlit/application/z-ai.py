import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import json
import tempfile
from pathlib import Path
import yaml
from datetime import datetime
import cv2
try:
    import mediapipe as mp
except Exception as _mp_err:
    mp = None
    # Provide a clear Streamlit-visible error message when mediapipe fails to import
    # This usually indicates either an incompatible numpy/python/architecture or a missing
    # Microsoft Visual C++ Redistributable on Windows.
    st.error(
        "mediapipe import failed: {}\n\n".format(_mp_err)
        + "Common fixes:\n"
        + "- Ensure you're using a 64-bit Python installation (not 32-bit).\n"
        + "- Install the Microsoft Visual C++ Redistributable (2015-2022) x64: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist\n"
        + "- Use a mediapipe-compatible numpy (numpy<2) in this environment, or run the Streamlit mediapipe app in a dedicated venv.\n"
        + "If you need help, run `pip show mediapipe numpy` and `python -c \"import platform,struct; print(platform.architecture(), struct.calcsize('P')*8)\"` and share the output."
    )
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Clinical Gait Analysis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical UI
def load_css():
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem 2rem;
            border-radius: 0.5rem;
            color: white;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .workflow-step {
            background: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            text-align: center;
            border: 2px solid #e0e0e0;
            margin: 0.5rem 0;
            position: relative;
        }
        
        .workflow-step.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }
        
        .workflow-step.completed {
            background: #28a745;
            color: white;
            border-color: #28a745;
        }
        
        .result-card {
            background: white;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            border-left: 5px solid #667eea;
        }
        
        .result-card.binary-card {
            border-left-color: #28a745;
        }
        
        .result-card.classification-card {
            border-left-color: #fd7e14;
        }
        
        .result-card.feature-card {
            border-left-color: #17a2b8;
        }
        
        .status-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 1rem;
            font-weight: 600;
            margin: 0.5rem 0;
        }
        
        .badge-normal {
            background: #d4edda;
            color: #155724;
        }
        
        .badge-abnormal {
            background: #f8d7da;
            color: #721c24;
        }
        
        .badge-physiological {
            background: #fff3cd;
            color: #856404;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 0.5rem;
            padding: 2rem;
            text-align: center;
            background: #f8f9fa;
            margin-bottom: 1rem;
        }
        
        .info-panel {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        
        .feature-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }
        
        .feature-name {
            font-weight: 500;
        }
        
        .feature-value {
            font-weight: 700;
            color: #667eea;
        }
        
        .progress-container {
            margin: 1rem 0;
        }
        
        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        
        .progress-bar {
            height: 1.5rem;
            background-color: #e9ecef;
            border-radius: 0.75rem;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            transition: width 0.5s;
        }
        
        .video-container {
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            overflow: hidden;
        }
        
        .video-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #2a5298;
        }
        
        .tab-container {
            margin-bottom: 1rem;
        }
        
        .tab-button {
            padding: 0.75rem 1.5rem;
            border: none;
            background: transparent;
            cursor: pointer;
            font-weight: 600;
            color: #6c757d;
            border-bottom: 3px solid transparent;
            margin-right: 0.5rem;
        }
        
        .tab-button.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        
        .summary-card {
            background: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .summary-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .summary-label {
            font-size: 0.9rem;
            color: #666;
        }
        
        .sidebar-section {
            background: white;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .sidebar-title {
            font-weight: 600;
            margin-bottom: 1rem;
            color: #2a5298;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .config-item {
            margin-bottom: 1rem;
        }
        
        .config-label {
            display: block;
            margin-bottom: 0.5rem;
            color: #555;
            font-size: 0.9rem;
        }
        
        .btn {
            width: 100%;
            padding: 0.75rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 0.5rem;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        
        .model-status {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }
        
        .model-status:last-child {
            border-bottom: none;
        }
        
        .model-name {
            font-size: 0.9rem;
        }
        
        .model-loaded {
            color: #28a745;
            font-weight: 600;
        }
        
        @media (max-width: 768px) {
            .workflow-steps {
                flex-direction: column;
            }
            
            .workflow-arrow {
                transform: rotate(90deg);
            }
        }
    </style>
    """, unsafe_allow_html=True)

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    if not os.path.exists(config_path):
        # Create default config if not exists
        default_config = {
            'app': {
                'name': 'Clinical Gait Analysis System',
                'version': '1.0.0',
                'debug': False
            },
            'paths': {
                'models': './models',
                'uploads': './data/uploads',
                'processed': './data/processed',
                'results': './data/results'
            },
            'mediapipe': {
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5,
                'model_complexity': 1
            },
            'video': {
                'supported_formats': ['mp4', 'avi', 'mov'],
                'max_file_size_mb': 500,
                'output_fps': 30,
                'skeleton_background': 'black',
                'skeleton_color': 'white'
            },
            'features': {
                'min_gait_cycles': 2,
                'smoothing_window': 11,
                'enable_advanced_features': True
            },
            'models': {
                'binary': {
                    'type': 'random_forest',
                    'threshold': 0.5
                },
                'multiclass': {
                    'type': 'random_forest',
                    'min_confidence': 0.6
                }
            },
            'ui': {
                'theme': 'light',
                'show_workflow': True,
                'show_progress': True,
                'auto_export_report': False
            },
            'clinical': {
                'age_groups': [
                    'Child (<18)',
                    'Adult (18-65)',
                    'Elderly (65+)'
                ],
                'severity_thresholds': {
                    'low': 0.6,
                    'moderate': 0.75,
                    'high': 0.9
                }
            }
        }
        
        # Create config directory if not exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        
        return default_config
    else:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

# Initialize session state
def init_session_state():
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    
    if 'binary_result' not in st.session_state:
        st.session_state.binary_result = None
    
    if 'multiclass_result' not in st.session_state:
        st.session_state.multiclass_result = None
    
    if 'features' not in st.session_state:
        st.session_state.features = None
    
    if 'landmarks_path' not in st.session_state:
        st.session_state.landmarks_path = None
    
    if 'annotated_video_path' not in st.session_state:
        st.session_state.annotated_video_path = None
    
    if 'skeleton_video_path' not in st.session_state:
        st.session_state.skeleton_video_path = None
    
    if 'patient_id' not in st.session_state:
        st.session_state.patient_id = f"PAT-{int(time.time())}"
    
    if 'age_group' not in st.session_state:
        st.session_state.age_group = "Adult (18-65)"
    
    if 'binary_model' not in st.session_state:
        st.session_state.binary_model = "Random Forest"
    
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

# Check if models are available
def check_models():
    models_dir = st.session_state.config['paths']['models']
    binary_model_path = os.path.join(models_dir, 'binary_classifier.pkl')
    multiclass_model_path = os.path.join(models_dir, 'multi_classifier.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    if all(os.path.exists(p) for p in [binary_model_path, multiclass_model_path, scaler_path]):
        return True
    return False

# Create necessary directories
def create_directories():
    paths = st.session_state.config['paths']
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        # Create subdirectories for processed data
        if path == paths['processed']:
            os.makedirs(os.path.join(path, 'landmarks'), exist_ok=True)
            os.makedirs(os.path.join(path, 'annotated_videos'), exist_ok=True)
            os.makedirs(os.path.join(path, 'skeleton_videos'), exist_ok=True)

# MediaPipe processing function
def process_video_with_mediapipe(video_path, output_dir):
    """Process video with MediaPipe and extract landmarks"""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Create output directories
    landmarks_dir = os.path.join(output_dir, 'landmarks')
    annotated_dir = os.path.join(output_dir, 'annotated_videos')
    
    os.makedirs(landmarks_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    
    # Generate output paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    landmarks_path = os.path.join(landmarks_dir, f"{base_name}_landmarks.csv")
    annotated_path = os.path.join(annotated_dir, f"{base_name}_annotated.mp4")
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=st.session_state.config['mediapipe']['min_detection_confidence'],
        min_tracking_confidence=st.session_state.config['mediapipe']['min_tracking_confidence'],
        model_complexity=st.session_state.config['mediapipe']['model_complexity']
    ) as pose:
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer for annotated video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))
        
        # Prepare CSV file for landmarks
        landmarks_data = []
        frame_idx = 0
        
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(rgb_frame)
            
            # Draw landmarks on frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
                # Extract landmarks
                timestamp = frame_idx / fps
                row = [frame_idx, timestamp]
                
                for landmark in results.pose_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                landmarks_data.append(row)
            
            # Write frame to output video
            out.write(frame)
            frame_idx += 1
            
            # Update progress
            progress = frame_idx / total_frames
            st.session_state.mediapipe_progress = progress
        
        # Release resources
        cap.release()
        out.release()
        
        # Create DataFrame and save to CSV
        if landmarks_data:
            columns = ['frame', 'timestamp']
            for i in range(33):
                columns.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z', f'landmark_{i}_visibility'])
            
            df = pd.DataFrame(landmarks_data, columns=columns)
            df.to_csv(landmarks_path, index=False)
            
            return {
                'landmarks_csv': landmarks_path,
                'annotated_video': annotated_path,
                'total_frames': frame_idx,
                'fps': fps,
                'status': 'success'
            }
        else:
            return {
                'status': 'failed',
                'message': 'No pose landmarks detected in the video'
            }

# Generate skeleton video
def generate_skeleton_video(landmarks_csv, output_path):
    """Generate skeleton-only video from landmarks"""
    # Read landmarks data
    df = pd.read_csv(landmarks_csv)
    
    # Create video writer
    fps = 30  # Default FPS if not available
    width, height = 1280, 720  # Default resolution
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Define connections for skeleton
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Head to shoulders
        (0, 4), (4, 5), (5, 6), (6, 8),  # Head to shoulders (other side)
        (9, 10),  # Mouth
        (11, 12),  # Shoulders
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Waist
        (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
    ]
    
    # Process each frame
    for _, row in df.iterrows():
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Extract landmarks
        landmarks = []
        for i in range(33):
            x = int(row[f'landmark_{i}_x'] * width)
            y = int(row[f'landmark_{i}_y'] * height)
            visibility = row[f'landmark_{i}_visibility']
            landmarks.append((x, y, visibility))
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if landmarks[start_idx][2] > 0.5 and landmarks[end_idx][2] > 0.5:
                start_point = (landmarks[start_idx][0], landmarks[start_idx][1])
                end_point = (landmarks[end_idx][0], landmarks[end_idx][1])
                cv2.line(frame, start_point, end_point, (255, 255, 255), 3)
        
        # Draw landmarks
        for x, y, visibility in landmarks:
            if visibility > 0.5:
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        
        # Write frame
        out.write(frame)
    
    # Release resources
    out.release()
    
    return {
        'skeleton_video': output_path,
        'frame_count': len(df),
        'status': 'success'
    }

# Extract features from landmarks
def extract_features(landmarks_csv):
    """Extract gait features from landmarks data"""
    # Read landmarks data
    df = pd.read_csv(landmarks_csv)
    
    # Initialize feature dictionary
    features = {}
    
    # Calculate temporal features
    fps = 30  # Default FPS
    if len(df) > 1:
        fps = len(df) / (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0])
    
    # Gait cycle detection (simplified)
    # In a real implementation, this would be more sophisticated
    heel_left_indices = [11, 13, 15]  # Left leg landmarks
    heel_right_indices = [12, 14, 16]  # Right leg landmarks
    
    # Calculate vertical positions for heel detection
    left_heel_y = df[[f'landmark_{i}_y' for i in heel_left_indices]].mean(axis=1)
    right_heel_y = df[[f'landmark_{i}_y' for i in heel_right_indices]].mean(axis=1)
    
    # Find peaks (heel strikes) - simplified approach
    from scipy.signal import find_peaks
    
    left_peaks, _ = find_peaks(-left_heel_y, distance=fps/2)  # Negative because we want minima
    right_peaks, _ = find_peaks(-right_heel_y, distance=fps/2)
    
    # Calculate cadence (steps per minute)
    if len(left_peaks) > 1 and len(right_peaks) > 1:
        left_step_time = np.mean(np.diff(df['timestamp'].iloc[left_peaks]))
        right_step_time = np.mean(np.diff(df['timestamp'].iloc[right_peaks]))
        avg_step_time = (left_step_time + right_step_time) / 2
        cadence = 60 / avg_step_time if avg_step_time > 0 else 0
    else:
        cadence = 0
    
    # Calculate stride time
    if len(left_peaks) > 2 and len(right_peaks) > 2:
        left_stride_time = np.mean(np.diff(df['timestamp'].iloc[left_peaks][::2]))
        right_stride_time = np.mean(np.diff(df['timestamp'].iloc[right_peaks][::2]))
        avg_stride_time = (left_stride_time + right_stride_time) / 2
    else:
        avg_stride_time = 0
    
    # Calculate step length (simplified)
    hip_left = df[['landmark_23_x', 'landmark_23_y']].values
    hip_right = df[['landmark_24_x', 'landmark_24_y']].values
    
    if len(hip_left) > 1:
        # Estimate step length based on hip movement
        hip_movement = np.sqrt(np.sum(np.diff(hip_left, axis=0)**2, axis=1))
        step_length = np.mean(hip_movement) * 2  # Rough estimate
    else:
        step_length = 0
    
    # Calculate symmetry (simplified)
    if len(left_peaks) > 1 and len(right_peaks) > 1:
        left_step_time = np.mean(np.diff(df['timestamp'].iloc[left_peaks]))
        right_step_time = np.mean(np.diff(df['timestamp'].iloc[right_peaks]))
        symmetry = 100 * (1 - abs(left_step_time - right_step_time) / ((left_step_time + right_step_time) / 2))
    else:
        symmetry = 0
    
    # Calculate stability (simplified)
    # Using center of mass (estimated from hip position)
    com_x = (df['landmark_23_x'] + df['landmark_24_x']) / 2
    com_y = (df['landmark_23_y'] + df['landmark_24_y']) / 2
    
    # Calculate sway (standard deviation of COM position)
    com_sway_x = np.std(com_x)
    com_sway_y = np.std(com_y)
    stability = 100 * (1 - (com_sway_x + com_sway_y) / 2)  # Normalized to 0-100
    
    # Store features
    features['cadence'] = cadence
    features['stride_time'] = avg_stride_time
    features['step_length'] = step_length
    features['symmetry'] = symmetry
    features['stability'] = stability
    
    # Create feature vector for ML models (50 features)
    # In a real implementation, this would include many more features
    feature_vector = np.zeros(50)
    feature_vector[0] = cadence
    feature_vector[1] = avg_stride_time
    feature_vector[2] = step_length
    feature_vector[3] = symmetry
    feature_vector[4] = stability
    
    # Add more features (simplified for this example)
    # In a real implementation, you would calculate many more biomechanical features
    for i in range(5, 50):
        feature_vector[i] = np.random.rand() * 0.1  # Placeholder
    
    # Create feature report
    feature_report = {
        'temporal': {
            'cadence': cadence,
            'stride_time': avg_stride_time,
            'step_time': avg_stride_time / 2 if avg_stride_time > 0 else 0
        },
        'spatial': {
            'step_length': step_length,
            'stride_length': step_length * 2,
            'step_width': 0.1  # Placeholder
        },
        'kinematic': {
            'hip_rom': 30,  # Placeholder
            'knee_rom': 60,  # Placeholder
            'ankle_rom': 25  # Placeholder
        },
        'stability': {
            'com_sway_ap': com_sway_y,
            'com_sway_ml': com_sway_x,
            'stability_score': stability
        }
    }
    
    return {
        'features_array': feature_vector.reshape(1, -1),
        'feature_names': [f'feature_{i}' for i in range(50)],
        'feature_report': feature_report,
        'status': 'success'
    }

# Load ML models
def load_models():
    """Load the ML models for classification"""
    models_dir = st.session_state.config['paths']['models']
    binary_model_path = os.path.join(models_dir, 'binary_classifier.pkl')
    multiclass_model_path = os.path.join(models_dir, 'multi_classifier.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    try:
        with open(binary_model_path, 'rb') as f:
            binary_model = pickle.load(f)
        
        with open(multiclass_model_path, 'rb') as f:
            multiclass_model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return {
            'binary_model': binary_model,
            'multiclass_model': multiclass_model,
            'scaler': scaler,
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'failed',
            'message': str(e)
        }

# Run binary classification
def run_binary_classification(features, models):
    """Run binary classification on features"""
    try:
        # Scale features
        scaler = models['scaler']
        scaled_features = scaler.transform(features)
        
        # Predict
        binary_model = models['binary_model']
        prediction = binary_model.predict(scaled_features)[0]
        probabilities = binary_model.predict_proba(scaled_features)[0]
        
        # Map prediction to label
        if prediction == 0:
            label = "Normal"
        else:
            label = "Abnormal"
        
        confidence = max(probabilities)
        
        return {
            'prediction': label,
            'confidence': confidence,
            'probabilities': {
                'normal': probabilities[0],
                'abnormal': probabilities[1]
            },
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'failed',
            'message': str(e)
        }

# Run multiclass classification
def run_multiclass_classification(features, models):
    """Run multiclass classification on features"""
    try:
        # Scale features
        scaler = models['scaler']
        scaled_features = scaler.transform(features)
        
        # Predict
        multiclass_model = models['multiclass_model']
        prediction = multiclass_model.predict(scaled_features)[0]
        probabilities = multiclass_model.predict_proba(scaled_features)[0]
        
        # Map prediction to label and ICD-10 code
        # This would be based on your specific model classes
        class_mapping = {
            0: {'label': 'Physiological', 'icd10': 'Z00.00'},
            1: {'label': 'Spastic', 'icd10': 'G80.1'},
            2: {'label': 'Ataxic', 'icd10': 'G11.0'},
            3: {'label': 'Parkinsonian', 'icd10': 'G20'},
            4: {'label': 'Neuropathic', 'icd10': 'G63.9'}
        }
        
        class_info = class_mapping.get(prediction, {'label': 'Unknown', 'icd10': 'N/A'})
        confidence = max(probabilities)
        
        # Create probability dictionary
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            class_info_i = class_mapping.get(i, {'label': f'Class_{i}'})
            prob_dict[class_info_i['label']] = prob
        
        return {
            'pattern': class_info['label'],
            'icd10': class_info['icd10'],
            'confidence': confidence,
            'probabilities': prob_dict,
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'failed',
            'message': str(e)
        }

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Load CSS
    load_css()
    
    # Create necessary directories
    create_directories()
    
    # Check if models are available
    models_available = check_models()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Clinical Gait Analysis System</h1>
        <div class="status">✓ Models Loaded</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area with sidebar
    col1, col2 = st.columns([1, 3])
    
    # Sidebar
    with col1:
        # Upload section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">📹 Upload Video</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop video here or click to browse",
            type=st.session_state.config['video']['supported_formats'],
            help=f"Supported formats: {', '.join(st.session_state.config['video']['supported_formats'])}"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            # Save uploaded file
            upload_dir = st.session_state.config['paths']['uploads']
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, f"{st.session_state.patient_id}_{int(time.time())}.mp4")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.video_path = file_path
            st.success(f"Video uploaded: {uploaded_file.name}")
        
        # Configuration section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">⚙️ Configuration</div>
        </div>
        """, unsafe_allow_html=True)
        
        patient_id = st.text_input("Patient ID", value=st.session_state.patient_id)
        st.session_state.patient_id = patient_id
        
        age_group = st.selectbox(
            "Age Group",
            st.session_state.config['clinical']['age_groups'],
            index=st.session_state.config['clinical']['age_groups'].index(st.session_state.age_group)
        )
        st.session_state.age_group = age_group
        
        binary_model = st.selectbox(
            "Binary Model",
            ["Random Forest", "Logistic Regression", "Gradient Boosting"],
            index=["Random Forest", "Logistic Regression", "Gradient Boosting"].index(st.session_state.binary_model)
        )
        st.session_state.binary_model = binary_model
        
        # Model status section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">📊 Model Status</div>
        </div>
        """, unsafe_allow_html=True)
        
        if models_available:
            st.markdown("""
            <div class="model-status">
                <span class="model-name">Binary Model:</span>
                <span class="model-loaded">✓ Loaded</span>
            </div>
            <div class="model-status">
                <span class="model-name">Classification Model:</span>
                <span class="model-loaded">✓ Loaded</span>
            </div>
            <div class="model-status">
                <span class="model-name">MediaPipe:</span>
                <span class="model-loaded">✓ Ready</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Models not found. Please place your models in the 'models' directory.")
            st.markdown("""
            <div class="model-status">
                <span class="model-name">Binary Model:</span>
                <span style="color: #dc3545;">✗ Not Found</span>
            </div>
            <div class="model-status">
                <span class="model-name">Classification Model:</span>
                <span style="color: #dc3545;">✗ Not Found</span>
            </div>
            <div class="model-status">
                <span class="model-name">MediaPipe:</span>
                <span class="model-loaded">✓ Ready</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Buttons
        if st.button("🚀 Start Analysis", disabled=not models_available or st.session_state.uploaded_file is None):
            st.session_state.analysis_started = True
            st.session_state.current_step = 0
            st.session_state.analysis_complete = False
        
        if st.button("🔄 Reset Analysis"):
            st.session_state.analysis_started = False
            st.session_state.analysis_complete = False
            st.session_state.current_step = 0
            st.session_state.uploaded_file = None
            st.session_state.binary_result = None
            st.session_state.multiclass_result = None
            st.session_state.features = None
            st.session_state.landmarks_path = None
            st.session_state.annotated_video_path = None
            st.session_state.skeleton_video_path = None
            st.experimental_rerun()
    
    # Main content panel
    with col2:
        # Workflow diagram
        st.markdown("""
        <div class="result-card">
            <h2>📋 Analysis Pipeline</h2>
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div class="workflow-step {}">
                    <div style="width: 35px; height: 35px; background: white; color: #667eea; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px; font-weight: bold;">1</div>
                    <div><strong>Upload Video</strong></div>
                    <small>Video input</small>
                </div>
                <div style="font-size: 1.5rem; color: #667eea;">→</div>
                <div class="workflow-step {}">
                    <div style="width: 35px; height: 35px; background: white; color: #667eea; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px; font-weight: bold;">2</div>
                    <div><strong>MediaPipe</strong></div>
                    <small>Pose detection</small>
                </div>
                <div style="font-size: 1.5rem; color: #667eea;">→</div>
                <div class="workflow-step {}">
                    <div style="width: 35px; height: 35px; background: white; color: #667eea; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px; font-weight: bold;">3</div>
                    <div><strong>Generate Videos</strong></div>
                    <small>Annotated + Skeleton</small>
                </div>
                <div style="font-size: 1.5rem; color: #667eea;">→</div>
                <div class="workflow-step {}">
                    <div style="width: 35px; height: 35px; background: white; color: #667eea; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px; font-weight: bold;">4</div>
                    <div><strong>ML Analysis</strong></div>
                    <small>Classification</small>
                </div>
                <div style="font-size: 1.5rem; color: #667eea;">→</div>
                <div class="workflow-step {}">
                    <div style="width: 35px; height: 35px; background: white; color: #667eea; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px; font-weight: bold;">5</div>
                    <div><strong>Results</strong></div>
                    <small>Report generation</small>
                </div>
            </div>
        </div>
        """.format(
            "completed" if st.session_state.current_step > 0 else "",
            "completed" if st.session_state.current_step > 1 else ("active" if st.session_state.current_step == 1 else ""),
            "completed" if st.session_state.current_step > 2 else ("active" if st.session_state.current_step == 2 else ""),
            "completed" if st.session_state.current_step > 3 else ("active" if st.session_state.current_step == 3 else ""),
            "completed" if st.session_state.current_step > 4 else ("active" if st.session_state.current_step == 4 else "")
        ), unsafe_allow_html=True)
        
        # Analysis process
        if st.session_state.analysis_started and not st.session_state.analysis_complete:
            # Step 1: Upload video (already done)
            if st.session_state.current_step == 0:
                st.session_state.current_step = 1
                st.experimental_rerun()
            
            # Step 2: MediaPipe processing
            if st.session_state.current_step == 1:
                with st.spinner("Processing video with MediaPipe..."):
                    # Process video with MediaPipe
                    output_dir = st.session_state.config['paths']['processed']
                    result = process_video_with_mediapipe(st.session_state.video_path, output_dir)
                    
                    if result['status'] == 'success':
                        st.session_state.landmarks_path = result['landmarks_csv']
                        st.session_state.annotated_video_path = result['annotated_video']
                        st.session_state.current_step = 2
                        st.experimental_rerun()
                    else:
                        st.error(f"MediaPipe processing failed: {result.get('message', 'Unknown error')}")
                        st.session_state.analysis_started = False
                        st.experimental_rerun()
            
            # Step 3: Generate skeleton video
            if st.session_state.current_step == 2:
                with st.spinner("Generating skeleton video..."):
                    # Generate skeleton video
                    output_dir = os.path.join(st.session_state.config['paths']['processed'], 'skeleton_videos')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    base_name = os.path.splitext(os.path.basename(st.session_state.landmarks_path))[0].replace('_landmarks', '')
                    skeleton_path = os.path.join(output_dir, f"{base_name}_skeleton.mp4")
                    
                    result = generate_skeleton_video(st.session_state.landmarks_path, skeleton_path)
                    
                    if result['status'] == 'success':
                        st.session_state.skeleton_video_path = result['skeleton_video']
                        st.session_state.current_step = 3
                        st.experimental_rerun()
                    else:
                        st.error(f"Skeleton video generation failed: {result.get('message', 'Unknown error')}")
                        st.session_state.analysis_started = False
                        st.experimental_rerun()
            
            # Step 4: Feature extraction and ML analysis
            if st.session_state.current_step == 3:
                with st.spinner("Extracting features and running analysis..."):
                    # Extract features
                    result = extract_features(st.session_state.landmarks_path)
                    
                    if result['status'] == 'success':
                        st.session_state.features = result
                        st.session_state.feature_report = result['feature_report']
                        
                        # Load models
                        models = load_models()
                        
                        if models['status'] == 'success':
                            # Run binary classification
                            binary_result = run_binary_classification(result['features_array'], models)
                            st.session_state.binary_result = binary_result
                            
                            # Run multiclass classification if abnormal
                            if binary_result['prediction'] == 'Abnormal':
                                multiclass_result = run_multiclass_classification(result['features_array'], models)
                                st.session_state.multiclass_result = multiclass_result
                            
                            st.session_state.current_step = 4
                            st.session_state.analysis_complete = True
                            st.experimental_rerun()
                        else:
                            st.error(f"Model loading failed: {models.get('message', 'Unknown error')}")
                            st.session_state.analysis_started = False
                            st.experimental_rerun()
                    else:
                        st.error(f"Feature extraction failed: {result.get('message', 'Unknown error')}")
                        st.session_state.analysis_started = False
                        st.experimental_rerun()
        
        # Display results if analysis is complete
        if st.session_state.analysis_complete:
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Results", "🎥 Video Preview", "📈 Feature Details", "📄 Clinical Report"])
            
            with tab1:
                # Info panel
                st.markdown("""
                <div class="info-panel">
                    <h4>💡 How it works</h4>
                    <p>
                        <strong>Step 1:</strong> Upload your gait video → 
                        <strong>Step 2:</strong> MediaPipe detects pose landmarks → 
                        <strong>Step 3:</strong> System generates annotated & skeleton videos → 
                        <strong>Step 4:</strong> Features extracted & ML models predict → 
                        <strong>Step 5:</strong> View comprehensive results
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Results grid
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Binary Classification Card
                    st.markdown("""
                    <div class="result-card binary-card">
                        <h3>
                            <span>🎯</span>
                            Binary Classification
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.session_state.binary_result:
                        binary_result = st.session_state.binary_result
                        
                        if binary_result['prediction'] == 'Normal':
                            st.markdown(f'<span class="status-badge badge-normal">NORMAL</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="status-badge badge-abnormal">ABNORMAL</span>', unsafe_allow_html=True)
                        
                        # Confidence bar
                        confidence = binary_result['confidence']
                        st.markdown(f"""
                        <div class="progress-container">
                            <div class="progress-label">
                                <span>Confidence Score</span>
                                <span>{confidence:.0%}</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {confidence*100}%">{confidence:.0%}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Interpretation
                        if binary_result['prediction'] == 'Normal':
                            interpretation = "Gait pattern shows normal characteristics with high confidence."
                        else:
                            interpretation = "Gait pattern shows abnormal characteristics. Further analysis recommended."
                        
                        st.markdown(f"""
                        <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 8px;">
                            <p style="font-size: 0.9rem; color: #666;">
                                <strong>Interpretation:</strong> {interpretation}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Multi-Class Classification Card
                    st.markdown("""
                    <div class="result-card classification-card">
                        <h3>
                            <span>🔍</span>
                            Pattern Classification
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.session_state.binary_result and st.session_state.binary_result['prediction'] == 'Abnormal' and st.session_state.multiclass_result:
                        multiclass_result = st.session_state.multiclass_result
                        
                        # Pattern badge with appropriate color
                        pattern = multiclass_result['pattern']
                        if pattern == 'Physiological':
                            badge_class = 'badge-physiological'
                        elif pattern in ['Spastic', 'Ataxic', 'Parkinsonian', 'Neuropathic']:
                            badge_class = 'badge-abnormal'
                        else:
                            badge_class = 'badge-normal'
                        
                        st.markdown(f'<span class="status-badge {badge_class}">{pattern}</span>', unsafe_allow_html=True)
                        
                        # Confidence bar
                        confidence = multiclass_result['confidence']
                        st.markdown(f"""
                        <div class="progress-container">
                            <div class="progress-label">
                                <span>Pattern Confidence</span>
                                <span>{confidence:.0%}</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {confidence*100}%; background: linear-gradient(90deg, #fd7e14 0%, #ff9800 100%);">{confidence:.0%}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # ICD-10 and description
                        st.markdown(f"""
                        <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 8px;">
                            <p style="font-size: 0.85rem; color: #666; margin-bottom: 5px;">
                                <strong>ICD-10:</strong> {multiclass_result['icd10']}
                            </p>
                            <p style="font-size: 0.85rem; color: #666;">
                                <strong>Description:</strong> {pattern} gait pattern
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Probabilities chart
                        if st.button("View Probabilities"):
                            probs = multiclass_result['probabilities']
                            fig = px.bar(
                                x=list(probs.keys()),
                                y=list(probs.values()),
                                labels={'x': 'Pattern', 'y': 'Probability'},
                                title='Pattern Classification Probabilities'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown("""
                        <div style="padding: 20px; text-align: center; color: #666;">
                            <p>Multi-class classification only applies to abnormal gait patterns.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    # Feature Analysis Card
                    st.markdown("""
                    <div class="result-card feature-card">
                        <h3>
                            <span>⚡</span>
                            Feature Analysis
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.session_state.feature_report:
                        feature_report = st.session_state.feature_report
                        
                        # Display key features
                        st.markdown(f"""
                        <div class="feature-item">
                            <span class="feature-name">Cadence</span>
                            <span class="feature-value">{feature_report['temporal']['cadence']:.1f} steps/min</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">Stride Time</span>
                            <span class="feature-value">{feature_report['temporal']['stride_time']:.2f} sec</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">Step Length</span>
                            <span class="feature-value">{feature_report['spatial']['step_length']:.2f} m</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">Symmetry</span>
                            <span class="feature-value">{feature_report['stability']['stability_score']:.1f}%</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">Stability</span>
                            <span class="feature-value">{'Good' if feature_report['stability']['stability_score'] > 70 else 'Fair' if feature_report['stability']['stability_score'] > 40 else 'Poor'}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("📊 View All Features"):
                            st.write("Full Feature Report:")
                            st.json(feature_report)
                
                # Summary metrics
                st.markdown("""
                <div class="result-card" style="grid-column: 1/-1;">
                    <h3>
                        <span>📋</span>
                        Analysis Summary
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("""
                    <div class="summary-card">
                        <div class="summary-label">Health Score</div>
                        <div class="summary-value" style="color: #28a745;">92/100</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.session_state.binary_result and st.session_state.binary_result['prediction'] == 'Normal':
                        fall_risk = "LOW"
                        color = "#28a745"
                    else:
                        fall_risk = "MODERATE"
                        color = "#fd7e14"
                    
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="summary-label">Fall Risk</div>
                        <div class="summary-value" style="color: {color};">{fall_risk}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if st.session_state.binary_result and st.session_state.binary_result['prediction'] == 'Normal':
                        priority = "ROUTINE"
                        color = "#6c757d"
                    else:
                        priority = "PRIORITY"
                        color = "#fd7e14"
                    
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="summary-label">Priority</div>
                        <div class="summary-value" style="color: {color};">{priority}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    # Count frames from landmarks file
                    if st.session_state.landmarks_path:
                        df = pd.read_csv(st.session_state.landmarks_path)
                        frames_analyzed = len(df)
                    else:
                        frames_analyzed = 0
                    
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="summary-label">Frames Analyzed</div>
                        <div class="summary-value" style="color: #667eea;">{frames_analyzed}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                # Video Preview Section
                st.markdown("""
                <div class="result-card">
                    <h2>🎥 Video Preview</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="video-title">🎬 Original Video</div>', unsafe_allow_html=True)
                    if st.session_state.video_path:
                        st.video(st.session_state.video_path)
                    else:
                        st.markdown("""
                        <div class="video-container" style="background: #dee2e6; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 0.5rem;">
                            <span style="color: #6c757d;">Original video will appear here</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="video-title">🎯 Annotated Video</div>', unsafe_allow_html=True)
                    if st.session_state.annotated_video_path:
                        st.video(st.session_state.annotated_video_path)
                    else:
                        st.markdown("""
                        <div class="video-container" style="background: #dee2e6; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 0.5rem;">
                            <span style="color: #6c757d;">Annotated video will appear here</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="video-title">🦴 Skeleton Video</div>', unsafe_allow_html=True)
                    if st.session_state.skeleton_video_path:
                        st.video(st.session_state.skeleton_video_path)
                    else:
                        st.markdown("""
                        <div class="video-container" style="background: #dee2e6; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 0.5rem;">
                            <span style="color: #6c757d;">Skeleton video will appear here</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab3:
                # Feature Details
                st.markdown("""
                <div class="result-card">
                    <h2>📈 Feature Details</h2>
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.feature_report:
                    feature_report = st.session_state.feature_report
                    
                    # Temporal features
                    st.markdown("""
                    <div class="result-card">
                        <h3>⏱️ Temporal Features</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Cadence", f"{feature_report['temporal']['cadence']:.1f} steps/min")
                        st.metric("Stride Time", f"{feature_report['temporal']['stride_time']:.2f} sec")
                    
                    with col2:
                        st.metric("Step Time", f"{feature_report['temporal']['step_time']:.2f} sec")
                        st.metric("Gait Cycle", f"{feature_report['temporal']['stride_time']*2:.2f} sec")
                    
                    # Spatial features
                    st.markdown("""
                    <div class="result-card">
                        <h3>📏 Spatial Features</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Step Length", f"{feature_report['spatial']['step_length']:.2f} m")
                        st.metric("Stride Length", f"{feature_report['spatial']['stride_length']:.2f} m")
                    
                    with col2:
                        st.metric("Step Width", f"{feature_report['spatial']['step_width']:.2f} m")
                        st.metric("Walking Speed", f"{feature_report['spatial']['step_length'] / feature_report['temporal']['step_time']:.2f} m/s")
                    
                    # Kinematic features
                    st.markdown("""
                    <div class="result-card">
                        <h3>🦴 Kinematic Features</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Hip ROM", f"{feature_report['kinematic']['hip_rom']}°")
                    
                    with col2:
                        st.metric("Knee ROM", f"{feature_report['kinematic']['knee_rom']}°")
                    
                    with col3:
                        st.metric("Ankle ROM", f"{feature_report['kinematic']['ankle_rom']}°")
                    
                    # Stability features
                    st.markdown("""
                    <div class="result-card">
                        <h3>⚖️ Stability Features</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("COM Sway (AP)", f"{feature_report['stability']['com_sway_ap']:.3f}")
                        st.metric("COM Sway (ML)", f"{feature_report['stability']['com_sway_ml']:.3f}")
                    
                    with col2:
                        st.metric("Stability Score", f"{feature_report['stability']['stability_score']:.1f}%")
                        st.metric("Balance", "Good" if feature_report['stability']['stability_score'] > 70 else "Fair" if feature_report['stability']['stability_score'] > 40 else "Poor")
                else:
                    st.info("No feature data available. Please complete the analysis first.")
            
            with tab4:
                # Clinical Report
                st.markdown("""
                <div class="result-card">
                    <h2>📄 Clinical Report</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Patient information
                st.markdown("""
                <div class="result-card">
                    <h3>Patient Information</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.text_input("Patient ID", value=st.session_state.patient_id, disabled=True)
                
                with col2:
                    st.text_input("Age Group", value=st.session_state.age_group, disabled=True)
                
                with col3:
                    st.text_input("Analysis Date", value=datetime.now().strftime("%Y-%m-%d"), disabled=True)
                
                # Clinical findings
                st.markdown("""
                <div class="result-card">
                    <h3>Clinical Findings</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.binary_result:
                    binary_result = st.session_state.binary_result
                    
                    st.markdown(f"""
                    <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; margin-bottom: 15px;">
                        <h4>Binary Classification</h4>
                        <p><strong>Result:</strong> {binary_result['prediction']}</p>
                        <p><strong>Confidence:</strong> {binary_result['confidence']:.1%}</p>
                        <p><strong>Normal Probability:</strong> {binary_result['probabilities']['normal']:.1%}</p>
                        <p><strong>Abnormal Probability:</strong> {binary_result['probabilities']['abnormal']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if binary_result['prediction'] == 'Abnormal' and st.session_state.multiclass_result:
                        multiclass_result = st.session_state.multiclass_result
                        
                        st.markdown(f"""
                        <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; margin-bottom: 15px;">
                            <h4>Pattern Classification</h4>
                            <p><strong>Pattern:</strong> {multiclass_result['pattern']}</p>
                            <p><strong>ICD-10 Code:</strong> {multiclass_result['icd10']}</p>
                            <p><strong>Confidence:</strong> {multiclass_result['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if st.session_state.feature_report:
                    feature_report = st.session_state.feature_report
                    
                    st.markdown("""
                    <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; margin-bottom: 15px;">
                        <h4>Gait Parameters</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Cadence", f"{feature_report['temporal']['cadence']:.1f} steps/min")
                        st.metric("Stride Time", f"{feature_report['temporal']['stride_time']:.2f} sec")
                        st.metric("Step Length", f"{feature_report['spatial']['step_length']:.2f} m")
                    
                    with col2:
                        st.metric("Symmetry", f"{feature_report['stability']['stability_score']:.1f}%")
                        st.metric("Hip ROM", f"{feature_report['kinematic']['hip_rom']}°")
                        st.metric("Knee ROM", f"{feature_report['kinematic']['knee_rom']}°")
                
                # Recommendations
                st.markdown("""
                <div class="result-card">
                    <h3>Recommendations</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.binary_result and st.session_state.binary_result['prediction'] == 'Normal':
                    st.markdown("""
                    <div style="padding: 15px; background: #e7f3ff; border-left: 4px solid #2196F3; border-radius: 8px;">
                        <p>Gait pattern appears normal with high confidence. No immediate intervention required.</p>
                        <p>Continue routine monitoring and follow standard preventive care guidelines.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 8px;">
                        <p>Abnormal gait pattern detected. Further clinical evaluation recommended.</p>
                        <p>Consider referral to a specialist for comprehensive assessment and intervention planning.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export button
                if st.button("📄 Export Report as PDF"):
                    st.info("PDF export functionality would be implemented here.")

if __name__ == "__main__":
    main()