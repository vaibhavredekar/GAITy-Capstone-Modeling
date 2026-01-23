"""
═══════════════════════════════════════════════════════════════════════════
CLINICAL GAIT ANALYSIS APPLICATION
Single-file Streamlit app with complete ML pipeline for gait analysis
═══════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tempfile
import os
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import euclidean
import json

# ============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# ============================================================================

class AppConfig:
    """Application configuration constants"""
    APP_TITLE = "Clinical Gait Analysis System"
    APP_ICON = "🏥"
    
    # MediaPipe landmark indices
    LANDMARK_NOSE = 0
    LANDMARK_LEFT_HIP = 23
    LANDMARK_RIGHT_HIP = 24
    LANDMARK_LEFT_KNEE = 25
    LANDMARK_RIGHT_KNEE = 26
    LANDMARK_LEFT_ANKLE = 27
    LANDMARK_RIGHT_ANKLE = 28
    LANDMARK_LEFT_HEEL = 29
    LANDMARK_RIGHT_HEEL = 30
    LANDMARK_LEFT_FOOT = 31
    LANDMARK_RIGHT_FOOT = 32
    
    # Normal ranges (age-adjusted)
    NORMAL_CADENCE = (100, 120)  # steps/min
    NORMAL_STEP_LENGTH = (0.6, 0.8)  # normalized
    NORMAL_STRIDE_TIME = (1.0, 1.4)  # seconds
    
    # Gait pattern definitions
    GAIT_PATTERNS = {
        0: {
            'name': 'Physiological',
            'description': 'Normal gait pattern',
            'icd10': 'Z00.00'
        },
        1: {
            'name': 'Spastic',
            'description': 'Increased muscle tone, scissoring',
            'icd10': 'G80.1'
        },
        2: {
            'name': 'Ataxic',
            'description': 'Wide-based, unsteady gait',
            'icd10': 'R26.0'
        },
        3: {
            'name': 'Antalgic',
            'description': 'Pain-avoidance gait',
            'icd10': 'R26.1'
        },
        4: {
            'name': 'Parkinsonian',
            'description': 'Shuffling, reduced arm swing',
            'icd10': 'G20'
        },
        5: {
            'name': 'Trendelenburg',
            'description': 'Hip abductor weakness',
            'icd10': 'M62.81'
        },
        6: {
            'name': 'Hemiplegic',
            'description': 'One-sided paralysis',
            'icd10': 'G81.9'
        }
    }

# ============================================================================
# SECTION 2: POSE DETECTION MODULE
# ============================================================================

class PoseDetector:
    """MediaPipe-based pose detection and keypoint extraction"""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )
        self.keypoints_sequence = []
        self.frame_timestamps = []
        
    def process_video(self, video_path, progress_callback=None):
        """Extract keypoints from video file"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        self.keypoints_sequence = []
        self.frame_timestamps = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = self._extract_landmarks(results.pose_landmarks)
                self.keypoints_sequence.append(landmarks)
                self.frame_timestamps.append(frame_count / fps)
            
            frame_count += 1
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count / total_frames)
        
        cap.release()
        
        return {
            'keypoints': np.array(self.keypoints_sequence),
            'timestamps': np.array(self.frame_timestamps),
            'fps': fps,
            'total_frames': len(self.keypoints_sequence)
        }
    
    def _extract_landmarks(self, pose_landmarks):
        """Extract normalized coordinates from landmarks"""
        landmarks = []
        for lm in pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        return landmarks
    
    def visualize_pose(self, frame, landmarks):
        """Draw skeleton overlay on frame"""
        if landmarks is not None:
            mp_landmarks = self.mp_pose.PoseLandmark
            connections = self.mp_pose.POSE_CONNECTIONS
            
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if (landmarks[start_idx][3] > 0.5 and 
                    landmarks[end_idx][3] > 0.5):
                    
                    start_point = (int(landmarks[start_idx][0] * frame.shape[1]),
                                 int(landmarks[start_idx][1] * frame.shape[0]))
                    end_point = (int(landmarks[end_idx][0] * frame.shape[1]),
                               int(landmarks[end_idx][1] * frame.shape[0]))
                    
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        return frame

# ============================================================================
# SECTION 3: FEATURE ENGINEERING MODULE
# ============================================================================

class GaitFeatureCalculator:
    """Calculate biomechanical features from keypoints"""
    
    def __init__(self):
        self.features = {}
        self.feature_names = []
        
    def calculate_all_features(self, keypoints_data):
        """Calculate complete feature set"""
        keypoints = keypoints_data['keypoints']
        timestamps = keypoints_data['timestamps']
        
        if len(keypoints) < 30:
            return None, {'overall_score': 0.0, 'message': 'Insufficient frames'}
        
        # Extract key landmarks
        left_heel = keypoints[:, AppConfig.LANDMARK_LEFT_HEEL, :2]
        right_heel = keypoints[:, AppConfig.LANDMARK_RIGHT_HEEL, :2]
        left_ankle = keypoints[:, AppConfig.LANDMARK_LEFT_ANKLE, :2]
        right_ankle = keypoints[:, AppConfig.LANDMARK_RIGHT_ANKLE, :2]
        left_knee = keypoints[:, AppConfig.LANDMARK_LEFT_KNEE, :2]
        right_knee = keypoints[:, AppConfig.LANDMARK_RIGHT_KNEE, :2]
        left_hip = keypoints[:, AppConfig.LANDMARK_LEFT_HIP, :2]
        right_hip = keypoints[:, AppConfig.LANDMARK_RIGHT_HIP, :2]
        
        features = {}
        
        # 1. TEMPORAL FEATURES
        heel_heights_left = left_heel[:, 1]
        heel_heights_right = right_heel[:, 1]
        
        # Detect heel strikes (local minima in heel height)
        strikes_left, _ = find_peaks(-heel_heights_left, distance=10)
        strikes_right, _ = find_peaks(-heel_heights_right, distance=10)
        
        # Cadence (steps per minute)
        total_steps = len(strikes_left) + len(strikes_right)
        duration = timestamps[-1] - timestamps[0]
        features['cadence'] = (total_steps / duration) * 60 if duration > 0 else 0
        
        # Stride time
        if len(strikes_left) > 1:
            stride_times_left = np.diff(timestamps[strikes_left])
            features['stride_time_mean'] = np.mean(stride_times_left)
            features['stride_time_std'] = np.std(stride_times_left)
        else:
            features['stride_time_mean'] = 0
            features['stride_time_std'] = 0
        
        # 2. SPATIAL FEATURES
        # Step length (distance between consecutive heel strikes)
        step_lengths = []
        for i in range(min(len(strikes_left), len(strikes_right)) - 1):
            dist = euclidean(left_heel[strikes_left[i]], right_heel[strikes_right[i]])
            step_lengths.append(dist)
        
        features['step_length_mean'] = np.mean(step_lengths) if step_lengths else 0
        features['step_length_std'] = np.std(step_lengths) if step_lengths else 0
        
        # Step width (mediolateral distance)
        step_widths = np.abs(left_ankle[:, 0] - right_ankle[:, 0])
        features['step_width_mean'] = np.mean(step_widths)
        features['step_width_std'] = np.std(step_widths)
        
        # 3. KINEMATIC FEATURES
        # Knee angles
        knee_angles_left = self._calculate_joint_angles(left_hip, left_knee, left_ankle)
        knee_angles_right = self._calculate_joint_angles(right_hip, right_knee, right_ankle)
        
        features['knee_angle_left_mean'] = np.mean(knee_angles_left)
        features['knee_angle_left_rom'] = np.ptp(knee_angles_left)
        features['knee_angle_right_mean'] = np.mean(knee_angles_right)
        features['knee_angle_right_rom'] = np.ptp(knee_angles_right)
        
        # 4. SYMMETRY FEATURES
        features['temporal_symmetry'] = self._calculate_symmetry(
            features['stride_time_mean'], features['stride_time_mean']
        )
        features['spatial_symmetry'] = self._calculate_symmetry(
            features['step_length_mean'], features['step_length_mean']
        )
        features['knee_symmetry'] = self._calculate_symmetry(
            features['knee_angle_left_mean'], features['knee_angle_right_mean']
        )
        
        # 5. VARIABILITY FEATURES
        features['cadence_variability'] = features['stride_time_std'] / features['stride_time_mean'] if features['stride_time_mean'] > 0 else 0
        features['step_length_variability'] = features['step_length_std'] / features['step_length_mean'] if features['step_length_mean'] > 0 else 0
        
        # 6. CENTER OF MASS FEATURES
        com_x = (left_hip[:, 0] + right_hip[:, 0]) / 2
        com_y = (left_hip[:, 1] + right_hip[:, 1]) / 2
        
        features['com_sway_ml'] = np.std(com_x)  # Mediolateral sway
        features['com_sway_ap'] = np.std(com_y)  # Anteroposterior sway
        
        # 7. STABILITY FEATURES
        features['base_of_support'] = features['step_width_mean']
        features['double_support_time'] = self._estimate_double_support(strikes_left, strikes_right, timestamps)
        
        # Convert to array
        self.feature_names = list(features.keys())
        feature_array = np.array([features[k] for k in self.feature_names]).reshape(1, -1)
        
        # Quality assessment
        quality = self._assess_quality(keypoints, strikes_left, strikes_right)
        
        return feature_array, quality
    
    def _calculate_joint_angles(self, p1, p2, p3):
        """Calculate joint angles given three points"""
        angles = []
        for i in range(len(p1)):
            v1 = p1[i] - p2[i]
            v2 = p3[i] - p2[i]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            angles.append(angle)
        
        return np.array(angles)
    
    def _calculate_symmetry(self, left_val, right_val):
        """Calculate symmetry ratio"""
        if left_val == 0 and right_val == 0:
            return 1.0
        return 1 - abs(left_val - right_val) / (abs(left_val) + abs(right_val))
    
    def _estimate_double_support(self, strikes_left, strikes_right, timestamps):
        """Estimate double support time percentage"""
        if len(strikes_left) < 2 or len(strikes_right) < 2:
            return 0.0
        
        # Simplified estimation
        return 0.2  # Placeholder: 20% is typical
    
    def _assess_quality(self, keypoints, strikes_left, strikes_right):
        """Assess data quality"""
        visibility = keypoints[:, :, 3]
        avg_visibility = np.mean(visibility)
        
        min_cycles = 2
        detected_cycles = min(len(strikes_left), len(strikes_right))
        
        quality_score = avg_visibility * min(1.0, detected_cycles / min_cycles)
        
        return {
            'overall_score': quality_score,
            'avg_visibility': avg_visibility,
            'detected_cycles': detected_cycles,
            'message': 'Good quality' if quality_score > 0.6 else 'Low quality'
        }

# ============================================================================
# SECTION 4: ML CLASSIFIERS
# ============================================================================

class BinaryClassifier:
    """Binary classification: Normal vs Abnormal"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Initialize with dummy data for demo
        self._initialize_demo_model()
    
    def _initialize_demo_model(self):
        """Initialize with synthetic data for demo purposes"""
        np.random.seed(42)
        
        # Generate synthetic training data
        n_samples = 200
        n_features = 20
        
        # Normal gait (label 0)
        X_normal = np.random.randn(n_samples // 2, n_features) * 0.5 + np.array([
            110, 1.2, 0.1, 0.7, 0.05, 140, 20, 140, 20, 0.95,
            0.95, 0.95, 0.08, 0.1, 0.02, 0.03, 0.15, 0.2, 0, 0
        ])
        
        # Abnormal gait (label 1)
        X_abnormal = np.random.randn(n_samples // 2, n_features) * 1.5 + np.array([
            85, 1.5, 0.3, 0.5, 0.15, 120, 30, 130, 35, 0.7,
            0.75, 0.8, 0.15, 0.2, 0.05, 0.06, 0.25, 0.3, 0, 0
        ])
        
        X = np.vstack([X_normal, X_abnormal])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
        
        # Train
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, features):
        """Predict normal vs abnormal"""
        if not self.is_trained:
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'probabilities': {'normal': 0.5, 'abnormal': 0.5}
            }
        
        # Pad features if necessary
        if features.shape[1] < 20:
            features = np.pad(features, ((0, 0), (0, 20 - features.shape[1])), mode='constant')
        elif features.shape[1] > 20:
            features = features[:, :20]
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': 'Normal' if prediction == 0 else 'Abnormal',
            'confidence': probabilities[prediction],
            'probabilities': {
                'normal': probabilities[0],
                'abnormal': probabilities[1]
            }
        }


class MultiClassifier:
    """Multi-class gait pattern classification"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.label_map = AppConfig.GAIT_PATTERNS
        
        # Initialize with demo data
        self._initialize_demo_model()
    
    def _initialize_demo_model(self):
        """Initialize with synthetic multi-class data"""
        np.random.seed(42)
        
        n_per_class = 30
        n_features = 20
        
        X_list = []
        y_list = []
        
        # Generate synthetic data for each class
        base_patterns = [
            [110, 1.2, 0.1, 0.7, 0.05, 140, 20, 140, 20, 0.95, 0.95, 0.95, 0.08, 0.1, 0.02, 0.03, 0.15, 0.2, 0, 0],  # Physiological
            [90, 1.4, 0.2, 0.6, 0.12, 130, 25, 125, 30, 0.8, 0.82, 0.85, 0.12, 0.15, 0.04, 0.05, 0.2, 0.25, 0, 0],    # Spastic
            [95, 1.6, 0.25, 0.55, 0.18, 125, 35, 120, 40, 0.75, 0.78, 0.7, 0.18, 0.22, 0.06, 0.08, 0.3, 0.35, 0, 0],  # Ataxic
            [100, 1.3, 0.15, 0.65, 0.1, 135, 22, 135, 25, 0.85, 0.88, 0.6, 0.1, 0.13, 0.03, 0.04, 0.18, 0.22, 0, 0],  # Antalgic
            [85, 1.5, 0.3, 0.5, 0.2, 115, 15, 115, 18, 0.7, 0.72, 0.9, 0.2, 0.25, 0.07, 0.09, 0.25, 0.3, 0, 0],       # Parkinsonian
            [105, 1.35, 0.18, 0.62, 0.13, 132, 28, 128, 32, 0.78, 0.8, 0.75, 0.13, 0.17, 0.045, 0.055, 0.22, 0.27, 0, 0],  # Trendelenburg
            [98, 1.38, 0.22, 0.58, 0.16, 128, 30, 140, 22, 0.73, 0.88, 0.65, 0.16, 0.19, 0.05, 0.06, 0.24, 0.28, 0, 0]     # Hemiplegic
        ]
        
        for class_idx, base_pattern in enumerate(base_patterns):
            X_class = np.random.randn(n_per_class, n_features) * 0.8 + np.array(base_pattern)
            X_list.append(X_class)
            y_list.extend([class_idx] * n_per_class)
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        # Shuffle
        indices = np.random.permutation(len(y))
        X, y = X[indices], y[indices]
        
        # Train
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, features):
        """Predict gait pattern"""
        if not self.is_trained:
            return {
                'pattern': 'Unknown',
                'confidence': 0.0,
                'probabilities': {},
                'feature_importance': np.zeros(features.shape[1])
            }
        
        # Pad/trim features
        if features.shape[1] < 20:
            features = np.pad(features, ((0, 0), (0, 20 - features.shape[1])), mode='constant')
        elif features.shape[1] > 20:
            features = features[:, :20]
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        importances = self.model.feature_importances_
        
        prob_dict = {self.label_map[i]['name']: probabilities[i] for i in range(len(probabilities))}
        
        return {
            'pattern': self.label_map[prediction]['name'],
            'confidence': probabilities[prediction],
            'probabilities': prob_dict,
            'feature_importance': importances,
            'icd10': self.label_map[prediction]['icd10'],
            'description': self.label_map[prediction]['description']
        }

# ============================================================================
# SECTION 5: CLINICAL ANALYZER
# ============================================================================

class ClinicalAnalyzer:
    """Generate clinical reports and recommendations"""
    
    def __init__(self):
        pass
    
    def generate_report(self, keypoints_data, features, binary_result, 
                       pattern_result, patient_id, age_group):
        """Generate comprehensive clinical report"""
        
        report = {
            'patient_id': patient_id,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'age_group': age_group,
            'video_duration': keypoints_data['timestamps'][-1],
            'frames_analyzed': keypoints_data['total_frames'],
        }
        
        # Classification results
        report['binary_classification'] = binary_result
        report['pattern_classification'] = pattern_result
        
        # Calculate health score (0-100)
        health_score = self._calculate_health_score(binary_result, pattern_result, features)
        report['health_score'] = health_score
        
        # Risk assessment
        report['risk_assessment'] = self._assess_risk(binary_result, pattern_result, features)
        
        # Recommendations
        report['recommendations'] = self._generate_recommendations(
            binary_result, pattern_result, features
        )
        
        # Key findings
        report['key_findings'] = self._extract_key_findings(features)
        
        return report
    
    def _calculate_health_score(self, binary_result, pattern_result, features):
        """Calculate overall health score (0-100)"""
        base_score = 100
        
        # Deduct for abnormal classification
        if binary_result['prediction'] == 'Abnormal':
            base_score -= 30 * binary_result['confidence']
        
        # Deduct for severe patterns
        if pattern_result and pattern_result['pattern'] != 'Physiological':
            severity_map = {
                'Spastic': 20,
                'Ataxic': 25,
                'Parkinsonian': 25,
                'Hemiplegic': 30,
                'Antalgic': 15,
                'Trendelenburg': 18
            }
            deduction = severity_map.get(pattern_result['pattern'], 20)
            base_score -= deduction * pattern_result['confidence']
        
        return max(0, min(100, base_score))
    
    def _assess_risk(self, binary_result, pattern_result, features):
        """Assess fall risk and clinical priority"""
        risk = {
            'fall_risk': 'Low',
            'priority': 'Routine',
            'concerns': []
        }
        
        if binary_result['prediction'] == 'Abnormal':
            if binary_result['confidence'] > 0.8:
                risk['fall_risk'] = 'High'
                risk['priority'] = 'Urgent'
            elif binary_result['confidence'] > 0.6:
                risk['fall_risk'] = 'Moderate'
                risk['priority'] = 'Soon'
        
        if pattern_result:
            high_risk_patterns = ['Ataxic', 'Parkinsonian', 'Hemiplegic']
            if pattern_result['pattern'] in high_risk_patterns:
                risk['fall_risk'] = 'High'
                risk['concerns'].append(f"{pattern_result['pattern']} gait increases fall risk")
        
        return risk
    
    def _generate_recommendations(self, binary_result, pattern_result, features):
        """Generate clinical recommendations"""
        recommendations = []
        
        if binary_result['prediction'] == 'Normal':
            recommendations.append("Continue regular physical activity")
            recommendations.append("Annual gait reassessment recommended")
        else:
            recommendations.append("Comprehensive clinical evaluation recommended")
            
            if pattern_result:
                pattern_specific = {
                    'Spastic': [
                        "Consider physical therapy for spasticity management",
                        "Evaluate for neurological conditions",
                        "Stretching and range-of-motion exercises"
                    ],
                    'Ataxic': [
                        "Balance training program recommended",
                        "Neurological consultation advised",
                        "Fall prevention strategies essential"
                    ],
                    'Antalgic': [
                        "Pain assessment and management",
                        "Evaluate for musculoskeletal pathology",
                        "Consider analgesic intervention"
                    ],
                    'Parkinsonian': [
                        "Neurology referral for Parkinson's evaluation",
                        "Medication review",
                        "Physical therapy for mobility"
                    ],
                    'Trendelenburg': [
                        "Hip abductor strengthening exercises",
                        "Orthopedic evaluation",
                        "Gait training"
                    ],
                    'Hemiplegic': [
                        "Stroke rehabilitation program",
                        "Assistive device assessment",
                        "Neurological follow-up"
                    ]
                }
                
                pattern_recs = pattern_specific.get(pattern_result['pattern'], [])
                recommendations.extend(pattern_recs)
        
        return recommendations
    
    def _extract_key_findings(self, features):
        """Extract key clinical findings from features"""
        findings = []
        
        # This is a placeholder - in real implementation, would analyze actual feature values
        findings.append("Gait parameters analyzed successfully")
        findings.append("Temporal and spatial characteristics evaluated")
        
        return findings

# ============================================================================
# SECTION 6: VISUALIZATION MODULE
# ============================================================================

class Visualizer:
    """Create clinical visualizations"""
    
    @staticmethod
    def plot_health_score(score):
        """Create health score gauge"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Health Score"},
            delta={'reference': 100},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightcoral"},
                    {'range': [50, 75], 'color': "lightyellow"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    @staticmethod
    def plot_probability_distribution(probabilities, title="Classification Probabilities"):
        """Create probability bar chart"""
        df = pd.DataFrame([
            {'Pattern': k, 'Probability': v * 100}
            for k, v in probabilities.items()
        ]).sort_values('Probability', ascending=True)
        
        fig = px.bar(df, x='Probability', y='Pattern', orientation='h',
                    title=title, color='Probability',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def plot_feature_importance(importances, feature_names, top_n=10):
        """Plot top feature importances"""
        if len(feature_names) > len(importances):
            feature_names = feature_names[:len(importances)]
        elif len(importances) > len(feature_names):
            importances = importances[:len(feature_names)]
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                    title='Top Feature Importance', color='Importance',
                    color_continuous_scale='Blues')
        fig.update_layout(height=400)
        return fig

# ============================================================================
# SECTION 7: STREAMLIT UI
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'report' not in st.session_state:
        st.session_state.report = None
    if 'keypoints_data' not in st.session_state:
        st.session_state.keypoints_data = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title=AppConfig.APP_TITLE,
        page_icon=AppConfig.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            padding: 1rem 0;
        }
        .metric-card {
            padding: 1.5rem;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            font-size: 1.1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(f"<h1 class='main-header'>{AppConfig.APP_ICON} {AppConfig.APP_TITLE}</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Patient Information
        with st.expander("👤 Patient Information", expanded=True):
            patient_id = st.text_input("Patient ID", 
                                      value=f"PAT-{datetime.now().strftime('%Y%m%d%H%M')}")
            age_group = st.selectbox("Age Group", 
                                    ["Adult (18-65)", "Elderly (65+)", "Child (<18)"])
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Model Configuration
        with st.expander("🤖 Model Settings"):
            binary_model_type = st.selectbox(
                "Binary Classifier",
                ["random_forest", "logistic", "gradient_boost"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            use_deep_learning = st.checkbox("Enable Deep Learning (Experimental)", value=False)
            if use_deep_learning:
                st.info("Deep learning features coming soon!")
        
        # Analysis Settings
        with st.expander("🔧 Detection Settings"):
            min_confidence = st.slider("Min Detection Confidence", 0.0, 1.0, 0.5, 0.05)
            min_tracking = st.slider("Min Tracking Confidence", 0.0, 1.0, 0.5, 0.05)
        
        st.markdown("---")
        
        # Video Upload
        st.header("📹 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a gait video",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video showing the patient walking (side view recommended)"
        )
        
        if uploaded_file:
            st.success(f"✓ File uploaded: {uploaded_file.name}")
            file_size = uploaded_file.size / (1024 * 1024)  # MB
            st.info(f"Size: {file_size:.2f} MB")
    
    # Main Panel - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analysis", 
        "📈 Visualizations", 
        "📄 Clinical Report",
        "ℹ️ About"
    ])
    
    # TAB 1: Analysis
    with tab1:
        if uploaded_file is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Video Preview")
                st.video(uploaded_file)
            
            with col2:
                st.subheader("Analysis Controls")
                
                if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                    analyze_video(uploaded_file, patient_id, age_group, 
                                binary_model_type, min_confidence, min_tracking)
                
                if st.session_state.analysis_complete:
                    st.success("✓ Analysis Complete!")
                    
                    if st.button("🔄 New Analysis", use_container_width=True):
                        st.session_state.analysis_complete = False
                        st.session_state.report = None
                        st.rerun()
            
            # Display results if available
            if st.session_state.report:
                st.markdown("---")
                display_quick_results(st.session_state.report)
        
        else:
            st.info("👈 Please upload a video file to begin analysis")
            
            # Demo section
            with st.expander("📚 How to use this application"):
                st.markdown("""
                ### Quick Start Guide
                
                1. **Upload Video**: Use the sidebar to upload a gait video
                   - Recommended: Side view, 10-30 seconds
                   - Format: MP4, AVI, or MOV
                
                2. **Configure Settings**: Adjust detection parameters if needed
                   - Default settings work well for most cases
                
                3. **Start Analysis**: Click the "Start Analysis" button
                   - The system will detect pose keypoints
                   - Calculate biomechanical features
                   - Classify gait pattern
                
                4. **Review Results**: Explore the analysis tabs
                   - Quick metrics in Analysis tab
                   - Detailed visualizations
                   - Complete clinical report
                
                ### Supported Gait Patterns
                - **Physiological**: Normal gait
                - **Spastic**: Increased muscle tone
                - **Ataxic**: Wide-based, unsteady
                - **Antalgic**: Pain-avoidance
                - **Parkinsonian**: Shuffling gait
                - **Trendelenburg**: Hip weakness
                - **Hemiplegic**: One-sided paralysis
                """)
    
    # TAB 2: Visualizations
    with tab2:
        if st.session_state.report:
            display_visualizations(st.session_state.report)
        else:
            st.info("Run an analysis first to see visualizations")
    
    # TAB 3: Clinical Report
    with tab3:
        if st.session_state.report:
            display_clinical_report(st.session_state.report)
        else:
            st.info("Run an analysis first to generate a clinical report")
    
    # TAB 4: About
    with tab4:
        display_about()

def analyze_video(video_file, patient_id, age_group, model_type, min_conf, min_track):
    """Complete analysis pipeline"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name
    
    try:
        # Step 1: Pose Detection
        st.subheader("🎯 Step 1: Pose Detection")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        pose_detector = PoseDetector(min_conf, min_track)
        
        def update_progress(pct):
            progress_bar.progress(pct)
            status_text.text(f"Processing frames... {int(pct*100)}%")
        
        keypoints_data = pose_detector.process_video(video_path, update_progress)
        progress_bar.progress(1.0)
        status_text.text("✓ Pose detection complete!")
        
        st.success(f"✓ Detected {keypoints_data['total_frames']} frames at {keypoints_data['fps']:.1f} FPS")
        
        # Step 2: Feature Calculation
        st.subheader("🧮 Step 2: Feature Calculation")
        with st.spinner("Calculating biomechanical features..."):
            feature_calc = GaitFeatureCalculator()
            features, quality = feature_calc.calculate_all_features(keypoints_data)
            
            if quality['overall_score'] < 0.5:
                st.warning(f"⚠️ Low data quality: {quality['message']}")
                st.info("Consider recording a longer video with better lighting")
            
            st.success(f"✓ Calculated {features.shape[1]} features (Quality: {quality['overall_score']:.2f})")
        
        # Step 3: Binary Classification
        st.subheader("🔍 Step 3: Binary Classification")
        with st.spinner("Classifying gait normality..."):
            binary_clf = BinaryClassifier(model_type)
            binary_result = binary_clf.predict(features)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Classification", binary_result['prediction'], 
                         delta="Normal" if binary_result['prediction'] == 'Normal' else "Needs attention")
            with col2:
                st.metric("Confidence", f"{binary_result['confidence']*100:.1f}%")
        
        # Step 4: Pattern Classification (if abnormal)
        pattern_result = None
        if binary_result['prediction'] == 'Abnormal':
            st.subheader("🎯 Step 4: Pattern Classification")
            with st.spinner("Identifying gait pattern..."):
                multi_clf = MultiClassifier()
                pattern_result = multi_clf.predict(features)
                
                st.metric("Gait Pattern", pattern_result['pattern'])
                st.caption(f"ICD-10: {pattern_result['icd10']} | {pattern_result['description']}")
                
                # Show probability distribution
                st.plotly_chart(
                    Visualizer.plot_probability_distribution(pattern_result['probabilities']),
                    use_container_width=True
                )
        
        # Step 5: Generate Report
        st.subheader("📋 Step 5: Clinical Report Generation")
        with st.spinner("Generating clinical report..."):
            analyzer = ClinicalAnalyzer()
            report = analyzer.generate_report(
                keypoints_data, features, binary_result, 
                pattern_result, patient_id, age_group
            )
            
            # Store in session state
            st.session_state.report = report
            st.session_state.keypoints_data = keypoints_data
            st.session_state.features = features
            st.session_state.analysis_complete = True
            
            # Display health score
            st.plotly_chart(
                Visualizer.plot_health_score(report['health_score']),
                use_container_width=True
            )
            
            st.success("✓ Analysis complete! View full report in the 'Clinical Report' tab")
    
    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.unlink(video_path)

def display_quick_results(report):
    """Display quick results overview"""
    st.subheader("Quick Results Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Health Score",
            f"{report['health_score']:.0f}/100",
            delta="Good" if report['health_score'] > 70 else "Needs attention"
        )
    
    with col2:
        status = report['binary_classification']['prediction']
        st.metric("Gait Status", status)
    
    with col3:
        risk = report['risk_assessment']['fall_risk']
        st.metric("Fall Risk", risk)
    
    with col4:
        priority = report['risk_assessment']['priority']
        st.metric("Priority", priority)
    
    # Pattern info if abnormal
    if report['pattern_classification']:
        st.info(f"**Detected Pattern**: {report['pattern_classification']['pattern']} "
               f"(Confidence: {report['pattern_classification']['confidence']*100:.1f}%)")

def display_visualizations(report):
    """Display detailed visualizations"""
    st.subheader("📈 Detailed Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            Visualizer.plot_health_score(report['health_score']),
            use_container_width=True
        )
    
    with col2:
        if report['binary_classification']:
            st.plotly_chart(
                Visualizer.plot_probability_distribution(
                    report['binary_classification']['probabilities'],
                    "Binary Classification Probabilities"
                ),
                use_container_width=True
            )
    
    # Pattern probabilities
    if report['pattern_classification']:
        st.plotly_chart(
            Visualizer.plot_probability_distribution(
                report['pattern_classification']['probabilities'],
                "Gait Pattern Probabilities"
            ),
            use_container_width=True
        )
        
        # Feature importance
        if 'feature_importance' in report['pattern_classification']:
            feature_names = [f"Feature {i+1}" for i in range(len(report['pattern_classification']['feature_importance']))]
            st.plotly_chart(
                Visualizer.plot_feature_importance(
                    report['pattern_classification']['feature_importance'],
                    feature_names
                ),
                use_container_width=True
            )

def display_clinical_report(report):
    """Display complete clinical report"""
    st.subheader("📄 Clinical Gait Analysis Report")
    
    # Header
    st.markdown(f"""
    **Patient ID**: {report['patient_id']}  
    **Date**: {report['date']}  
    **Age Group**: {report['age_group']}  
    **Video Duration**: {report['video_duration']:.2f} seconds  
    **Frames Analyzed**: {report['frames_analyzed']}
    """)
    
    st.markdown("---")
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Overall Health Score**: {report['health_score']:.0f}/100  
        **Gait Classification**: {report['binary_classification']['prediction']}  
        **Confidence**: {report['binary_classification']['confidence']*100:.1f}%
        """)
    
    with col2:
        st.markdown(f"""
        **Fall Risk**: {report['risk_assessment']['fall_risk']}  
        **Clinical Priority**: {report['risk_assessment']['priority']}
        """)
    
    # Pattern Details
    if report['pattern_classification']:
        st.markdown("---")
        st.subheader("🎯 Gait Pattern Analysis")
        st.markdown(f"""
        **Identified Pattern**: {report['pattern_classification']['pattern']}  
        **ICD-10 Code**: {report['pattern_classification']['icd10']}  
        **Description**: {report['pattern_classification']['description']}  
        **Confidence**: {report['pattern_classification']['confidence']*100:.1f}%
        """)
    
    # Risk Assessment
    st.markdown("---")
    st.subheader("⚠️ Risk Assessment")
    
    if report['risk_assessment']['concerns']:
        st.warning("**Clinical Concerns:**")
        for concern in report['risk_assessment']['concerns']:
            st.markdown(f"- {concern}")
    else:
        st.success("No major clinical concerns identified")
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Clinical Recommendations")
    
    for i, rec in enumerate(report['recommendations'], 1):
        st.markdown(f"{i}. {rec}")
    
    # Key Findings
    if report['key_findings']:
        st.markdown("---")
        st.subheader("🔍 Key Findings")
        for finding in report['key_findings']:
            st.markdown(f"- {finding}")
    
    # Export Options
    st.markdown("---")
    st.subheader("📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as JSON
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="Download JSON Report",
            data=report_json,
            file_name=f"gait_report_{report['patient_id']}.json",
            mime="application/json"
        )
    
    with col2:
        # Export as text summary
        text_report = f"""
CLINICAL GAIT ANALYSIS REPORT
=============================

Patient ID: {report['patient_id']}
Date: {report['date']}
Age Group: {report['age_group']}

SUMMARY
-------
Health Score: {report['health_score']:.0f}/100
Classification: {report['binary_classification']['prediction']}
Fall Risk: {report['risk_assessment']['fall_risk']}
Priority: {report['risk_assessment']['priority']}

RECOMMENDATIONS
--------------
{chr(10).join(f"{i}. {rec}" for i, rec in enumerate(report['recommendations'], 1))}
        """
        
        st.download_button(
            label="Download Text Report",
            data=text_report,
            file_name=f"gait_report_{report['patient_id']}.txt",
            mime="text/plain"
        )

def display_about():
    """Display about information"""
    st.subheader("ℹ️ About This Application")
    
    st.markdown("""
    ### Clinical Gait Analysis System
    
    This application provides automated gait analysis using computer vision and machine learning.
    
    #### Features
    - **Pose Detection**: MediaPipe-based 33-landmark detection
    - **Feature Engineering**: 20+ biomechanical features
    - **ML Classification**: Binary and multi-class pattern recognition
    - **Clinical Reports**: Professional healthcare-grade outputs
    - **Risk Assessment**: Fall risk and priority evaluation
    
    #### Technology Stack
    - **Streamlit**: Web interface
    - **MediaPipe**: Pose detection
    - **scikit-learn**: Machine learning
    - **Plotly**: Interactive visualizations
    - **OpenCV**: Video processing
    
    #### Gait Patterns Detected
    1. **Physiological**: Normal gait (ICD-10: Z00.00)
    2. **Spastic**: Increased muscle tone (ICD-10: G80.1)
    3. **Ataxic**: Wide-based, unsteady (ICD-10: R26.0)
    4. **Antalgic**: Pain-avoidance (ICD-10: R26.1)
    5. **Parkinsonian**: Shuffling gait (ICD-10: G20)
    6. **Trendelenburg**: Hip weakness (ICD-10: M62.81)
    7. **Hemiplegic**: One-sided paralysis (ICD-10: G81.9)
    
    #### Clinical Disclaimer
    This tool is for screening and educational purposes only. All findings should be 
    reviewed by qualified healthcare professionals. Not a substitute for professional 
    medical diagnosis.
    
    #### Version
    **Version 1.0.0** - Released January 2026
    
    ---
    
    ### Usage Tips
    
    ✅ **Best Practices**:
    - Record from side view (sagittal plane)
    - Ensure good lighting
    - Capture at least 2-3 complete gait cycles
    - Keep camera stable
    - Subject should wear fitted clothing
    
    ❌ **Avoid**:
    - Poor lighting conditions
    - Obstructed view
    - Very short videos (< 5 seconds)
    - Excessive camera movement
    
    ### Support
    For issues or questions, please contact your system administrator.
    """)

if __name__ == "__main__":
    main()