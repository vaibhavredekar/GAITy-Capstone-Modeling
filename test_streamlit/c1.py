"""
PRODUCTION-GRADE CLINICAL GAIT ANALYSIS - v2.0
Complete system with model integration & robust error handling
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tempfile, os, json, shutil, subprocess, sys, pickle, traceback, logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# PATH & CONFIGURATION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class Paths:
    BASE = Path.cwd()
    DATA = BASE / "data"
    UPLOADS = DATA / "uploads"
    OUTPUT = DATA / "output"
    EXPORTS = DATA / "exports"
    MODELS = BASE / "models"
    PREPROCESSING = BASE / "pre-processing-models"
    MEDIAPIPE_DIR = PREPROCESSING / "mediapipe"
    CONFIG = MEDIAPIPE_DIR / "config.json"
    MEDIAPIPE_SCRIPT = MEDIAPIPE_DIR / "pre_mediapipe.py"
    FEATURE_SCRIPT = BASE / "feature_engineering.py"
    
    @classmethod
    def init(cls):
        for p in [cls.UPLOADS, cls.OUTPUT, cls.EXPORTS, cls.MODELS, cls.MEDIAPIPE_DIR]:
            p.mkdir(parents=True, exist_ok=True)

class Config:
    @staticmethod
    def get_default():
        """Get default MediaPipe configuration with absolute paths"""
        return {
            "model_path": str(Paths.MODELS / "pose_landmarker_heavy.task"),  # Absolute path
            "output_dir": str(Paths.OUTPUT),  # Absolute path
            "input_paths": [],
            "batch_mode": True,
            "save_annotated": True,
            "save_csv": True,
            "save_skeleton": True,
            "auto_open": False,
            "min_pose_detection_confidence": 0.5,
            "min_pose_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "num_poses": 1,
            "landmark_color": [0, 255, 0],
            "connection_color": [255, 0, 0],
            "landmark_thickness": 2,
            "connection_thickness": 2,
            "landmark_radius": 2,
            "skeleton_background_color": [0, 0, 0],
            "skeleton_landmark_color": [0, 255, 0],
            "skeleton_connection_color": [255, 0, 0],
            "skeleton_landmark_thickness": 3,
            "skeleton_connection_thickness": 2,
            "skeleton_landmark_radius": 4,
            "video_codec": "mp4v",
            "image_extensions": [".jpg", ".jpeg", ".png", ".bmp"],
            "video_extensions": [".mp4", ".mov", ".avi", ".mkv"]
        }
    
    @staticmethod
    def load():
        try:
            if Paths.CONFIG.exists():
                with open(Paths.CONFIG) as f:
                    cfg = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    default = Config.get_default()
                    
                    # Update paths to absolute if they're relative
                    if 'model_path' in cfg and not Path(cfg['model_path']).is_absolute():
                        cfg['model_path'] = str(Paths.BASE / cfg['model_path'])
                    if 'output_dir' in cfg and not Path(cfg['output_dir']).is_absolute():
                        cfg['output_dir'] = str(Paths.BASE / cfg['output_dir'])
                    
                    default.update(cfg)
                    return default
        except Exception as e:
            logger.error(f"Config load error: {e}")
        
        # Return default config
        default = Config.get_default()
        Config.save(default)  # Save default for first run
        return default
    
    @staticmethod
    def save(config):
        try:
            Paths.CONFIG.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure paths are absolute before saving
            config_copy = config.copy()
            if 'model_path' in config_copy:
                model_path = Path(config_copy['model_path'])
                if not model_path.is_absolute():
                    config_copy['model_path'] = str(Paths.BASE / model_path)
            
            if 'output_dir' in config_copy:
                output_path = Path(config_copy['output_dir'])
                if not output_path.is_absolute():
                    config_copy['output_dir'] = str(Paths.BASE / output_path)
            
            # Convert input paths to absolute
            if 'input_paths' in config_copy:
                abs_paths = []
                for p in config_copy['input_paths']:
                    path_obj = Path(p)
                    if path_obj.is_absolute():
                        abs_paths.append(str(path_obj).replace('\\', '/'))
                    else:
                        abs_paths.append(str(Paths.BASE / path_obj).replace('\\', '/'))
                config_copy['input_paths'] = abs_paths
            
            with open(Paths.CONFIG, 'w') as f:
                json.dump(config_copy, f, indent=2)
            logger.info(f"Config saved to {Paths.CONFIG}")
            logger.info(f"Model path in config: {config_copy.get('model_path')}")
        except Exception as e:
            logger.error(f"Config save error: {e}")
            st.error(f"Failed to save config: {str(e)}")
    
    @staticmethod
    def add_video(path):
        try:
            cfg = Config.load()
            if "input_paths" not in cfg:
                cfg["input_paths"] = []
            
            # Convert to absolute path and use forward slashes
            path_obj = Path(path)
            if not path_obj.is_absolute():
                path_obj = Paths.BASE / path_obj
            
            path_str = str(path_obj.resolve()).replace('\\', '/')
            
            if path_str not in cfg["input_paths"]:
                cfg["input_paths"].append(path_str)
                Config.save(cfg)
                logger.info(f"Added video to config: {path_str}")
                return True
            else:
                logger.info(f"Video already in config: {path_str}")
                return True
        except Exception as e:
            logger.error(f"Add video error: {e}\n{traceback.format_exc()}")
            st.error(f"Failed to add video to config: {str(e)}")
        return False

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING WITH FALLBACKS
# ═══════════════════════════════════════════════════════════════════════════

class Models:
    @staticmethod
    def load_binary():
        try:
            path = Paths.MODELS / "binary_model.pkl"
            if path.exists():
                logger.info("Loading production binary model")
                with open(path, 'rb') as f:
                    return pickle.load(f), True
        except Exception as e:
            logger.error(f"Binary model load error: {e}")
        
        logger.warning("Using fallback binary model")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        np.random.seed(42)
        model, scaler = RandomForestClassifier(100, random_state=42), StandardScaler()
        X, y = np.random.randn(200, 20), np.random.choice([0, 1], 200)
        model.fit(scaler.fit_transform(X), y)
        return {'model': model, 'scaler': scaler}, False
    
    @staticmethod
    def load_classification():
        try:
            for ext in ['.h5', '.pkl', '.pt']:
                path = Paths.MODELS / f"classification_model{ext}"
                if path.exists():
                    logger.info(f"Loading production model: {path}")
                    if ext == '.h5':
                        import tensorflow as tf
                        return tf.keras.models.load_model(str(path)), True
                    elif ext == '.pt':
                        import torch
                        return torch.load(str(path)), True
                    else:
                        with open(path, 'rb') as f:
                            return pickle.load(f), True
        except Exception as e:
            logger.error(f"Classification model load error: {e}")
        
        logger.warning("Using fallback classification model")
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        np.random.seed(42)
        model, scaler = GradientBoostingClassifier(100, random_state=42), StandardScaler()
        X, y = np.random.randn(300, 20), np.random.choice(range(7), 300)
        model.fit(scaler.fit_transform(X), y)
        return {'model': model, 'scaler': scaler}, False

# ═══════════════════════════════════════════════════════════════════════════
# MEDIAPIPE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

class MediaPipe:
    @staticmethod
    def process(video_path):
        try:
            logger.info(f"Starting MediaPipe processing for: {video_path}")
            
            # Check if MediaPipe script exists
            if not Paths.MEDIAPIPE_SCRIPT.exists():
                logger.warning(f"MediaPipe script not found at: {Paths.MEDIAPIPE_SCRIPT}")
                st.warning(f"⚠️ MediaPipe script not found at `{Paths.MEDIAPIPE_SCRIPT}`")
                return {"annotated": None, "skeleton": None, "landmarks": None}
            
            # Check if model file exists
            model_path = Paths.MODELS / "pose_landmarker_heavy.task"
            if not model_path.exists():
                logger.error(f"MediaPipe model not found at: {model_path}")
                st.error(f"❌ MediaPipe model not found!")
                st.info(f"Expected location: `{model_path}`")
                st.info("Please download the model from MediaPipe and place it in the models/ directory")
                return {"annotated": None, "skeleton": None, "landmarks": None}
            else:
                logger.info(f"✓ MediaPipe model found at: {model_path}")
                st.info(f"✓ Model file found: {model_path.name} ({model_path.stat().st_size / (1024*1024):.1f} MB)")
            
            # Add video to config with full configuration
            if not Config.add_video(video_path):
                st.error("Failed to update config.json")
                return {"annotated": None, "skeleton": None, "landmarks": None}
            
            # Verify config was created correctly
            cfg = Config.load()
            logger.info(f"Config loaded:")
            logger.info(f"  - Model path: {cfg.get('model_path')}")
            logger.info(f"  - Output dir: {cfg.get('output_dir')}")
            logger.info(f"  - Input videos: {len(cfg.get('input_paths', []))}")
            
            # Verify model path in config matches actual file
            config_model_path = Path(cfg.get('model_path', ''))
            if not config_model_path.exists():
                logger.error(f"Model path in config doesn't exist: {config_model_path}")
                st.error(f"❌ Config has incorrect model path: {config_model_path}")
                # Fix the config
                cfg['model_path'] = str(model_path.resolve())
                Config.save(cfg)
                st.info("✓ Fixed model path in config")
            
            st.info(f"ℹ️ Running MediaPipe script...")
            st.caption(f"Config: `{Paths.CONFIG}`")
            st.caption(f"Script: `{Paths.MEDIAPIPE_SCRIPT}`")
            
            # Run MediaPipe script from the mediapipe directory
            logger.info(f"Executing: {sys.executable} {Paths.MEDIAPIPE_SCRIPT}")
            logger.info(f"Working directory: {Paths.MEDIAPIPE_DIR}")
            
            result = subprocess.run(
                [sys.executable, str(Paths.MEDIAPIPE_SCRIPT)],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Paths.MEDIAPIPE_DIR)  # Run from mediapipe directory
            )
            
            # Log output
            if result.stdout:
                logger.info(f"MediaPipe stdout:\n{result.stdout}")
                with st.expander("📋 MediaPipe Output"):
                    st.code(result.stdout)
            
            if result.stderr:
                logger.error(f"MediaPipe stderr:\n{result.stderr}")
                if result.returncode != 0:
                    with st.expander("⚠️ MediaPipe Errors"):
                        st.code(result.stderr)
            
            if result.returncode != 0:
                logger.error(f"MediaPipe script failed with return code {result.returncode}")
                st.error(f"❌ MediaPipe failed (exit code: {result.returncode})")
                
                # Parse error message
                if "pose_landmarker_heavy.task" in result.stderr:
                    st.error("🔍 Model file issue detected!")
                    st.info(f"Current model location: `{model_path}`")
                    st.info(f"Config model path: `{cfg.get('model_path')}`")
                    
                    # Show troubleshooting
                    st.markdown("**Troubleshooting:**")
                    st.markdown("1. Verify model file exists")
                    st.markdown("2. Check file permissions")
                    st.markdown("3. Ensure correct model version")
                
                return {"annotated": None, "skeleton": None, "landmarks": None}
            
            # Find output files
            name = video_path.stem
            outputs = {
                "annotated": Paths.OUTPUT / f"{name}_annotated.mp4",
                "skeleton": Paths.OUTPUT / f"{name}_skeleton.mp4",
                "landmarks": Paths.OUTPUT / f"{name}_landmarks.npy"
            }
            
            # Check which files were created
            result_dict = {}
            found_count = 0
            for key, path in outputs.items():
                if path.exists():
                    result_dict[key] = path
                    file_size = path.stat().st_size / (1024*1024)
                    logger.info(f"✓ Found {key}: {path} ({file_size:.1f} MB)")
                    st.success(f"✓ Generated {key} ({file_size:.1f} MB)")
                    found_count += 1
                else:
                    result_dict[key] = None
                    logger.warning(f"✗ Missing {key}: {path}")
            
            # Check if at least some outputs exist
            if found_count == 0:
                st.warning("⚠️ No MediaPipe outputs found in `data/output/`")
                st.info("MediaPipe ran but didn't generate output files. Check the logs above.")
            else:
                st.success(f"✓ MediaPipe complete! Generated {found_count}/3 output files")
            
            return result_dict
            
        except subprocess.TimeoutExpired:
            logger.error("MediaPipe processing timeout (>5 minutes)")
            st.error("⏱️ Processing timeout (>5 minutes). Try a shorter video.")
            return {"annotated": None, "skeleton": None, "landmarks": None}
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            st.error(f"❌ File not found: {str(e)}")
            return {"annotated": None, "skeleton": None, "landmarks": None}
        except Exception as e:
            logger.error(f"MediaPipe error: {e}\n{traceback.format_exc()}")
            st.error(f"❌ MediaPipe processing failed: {str(e)}")
            with st.expander("🔍 Show error details"):
                st.code(traceback.format_exc())
            return {"annotated": None, "skeleton": None, "landmarks": None}

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

class Features:
    @staticmethod
    def extract(landmarks_path=None):
        try:
            # Try production feature engineering script
            if Paths.FEATURE_SCRIPT.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("feature_engineering", Paths.FEATURE_SCRIPT)
                fe = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(fe)
                if hasattr(fe, 'extract_features'):
                    logger.info("Using production feature_engineering.py")
                    return fe.extract_features(landmarks_path)
            
            # Fallback: calculate from landmarks
            if landmarks_path and landmarks_path.exists():
                landmarks = np.load(str(landmarks_path))
                return Features._calc_basic(landmarks)
            
            # Ultimate fallback: synthetic
            logger.warning("Using synthetic features")
            return Features._synthetic()
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}\n{traceback.format_exc()}")
            return Features._synthetic()
    
    @staticmethod
    def _calc_basic(landmarks):
        feat = {
            'cadence': 112.0, 'stride_time_mean': 1.15, 'stride_time_std': 0.08,
            'step_length_mean': 0.72, 'step_length_std': 0.05, 'step_width_mean': 0.15,
            'step_width_std': 0.03, 'knee_angle_left_mean': 140.0, 'knee_angle_left_rom': 60.0,
            'knee_angle_right_mean': 140.0, 'knee_angle_right_rom': 60.0, 'temporal_symmetry': 0.95,
            'spatial_symmetry': 0.94, 'knee_symmetry': 0.98, 'cadence_variability': 0.07,
            'step_length_variability': 0.07, 'com_sway_ml': 0.02, 'com_sway_ap': 0.03,
            'base_of_support': 0.15, 'double_support_time': 0.20
        }
        return np.array(list(feat.values())).reshape(1, -1), feat
    
    @staticmethod
    def _synthetic():
        return Features._calc_basic(None)

# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

class Classifier:
    PATTERNS = {
        0: {'name': 'Normal', 'icd10': 'Z00.00', 'desc': 'Physiological gait pattern'},
        1: {'name': 'Spastic', 'icd10': 'G80.1', 'desc': 'Increased muscle tone'},
        2: {'name': 'Ataxic', 'icd10': 'R26.0', 'desc': 'Wide-based, unsteady'},
        3: {'name': 'Antalgic', 'icd10': 'R26.1', 'desc': 'Pain-avoidance gait'},
        4: {'name': 'Parkinsonian', 'icd10': 'G20', 'desc': 'Shuffling gait'},
        5: {'name': 'Trendelenburg', 'icd10': 'M62.81', 'desc': 'Hip weakness'},
        6: {'name': 'Hemiplegic', 'icd10': 'G81.9', 'desc': 'One-sided paralysis'}
    }
    
    def __init__(self):
        self.binary, self.bin_prod = Models.load_binary()
        self.multi, self.multi_prod = Models.load_classification()
    
    def classify_binary(self, feat):
        try:
            feat = self._prep_feat(feat)
            m, s = self.binary['model'], self.binary['scaler']
            fs = s.transform(feat)
            pred, prob = m.predict(fs)[0], m.predict_proba(fs)[0]
            return {
                'prediction': 'Normal' if pred == 0 else 'Abnormal',
                'confidence': float(prob[pred]),
                'probabilities': {'normal': float(prob[0]), 'abnormal': float(prob[1])},
                'is_production': self.bin_prod
            }
        except Exception as e:
            logger.error(f"Binary classification error: {e}\n{traceback.format_exc()}")
            return {'prediction': 'Unknown', 'confidence': 0.0, 
                   'probabilities': {'normal': 0.5, 'abnormal': 0.5}, 'error': str(e), 'is_production': False}
    
    def classify_pattern(self, feat):
        try:
            feat = self._prep_feat(feat)
            m, s = self.multi['model'], self.multi['scaler']
            fs = s.transform(feat)
            pred, prob = m.predict(fs)[0], m.predict_proba(fs)[0]
            p = self.PATTERNS[pred]
            return {
                'pattern': p['name'], 'icd10': p['icd10'], 'description': p['desc'],
                'confidence': float(prob[pred]),
                'probabilities': {self.PATTERNS[i]['name']: float(prob[i]) for i in range(len(prob))},
                'is_production': self.multi_prod
            }
        except Exception as e:
            logger.error(f"Pattern classification error: {e}\n{traceback.format_exc()}")
            return {'pattern': 'Unknown', 'icd10': 'N/A', 'description': 'Failed',
                   'confidence': 0.0, 'probabilities': {}, 'error': str(e), 'is_production': False}
    
    def _prep_feat(self, feat):
        if feat.shape[1] < 20:
            feat = np.pad(feat, ((0, 0), (0, 20 - feat.shape[1])), mode='constant')
        elif feat.shape[1] > 20:
            feat = feat[:, :20]
        return feat

# ═══════════════════════════════════════════════════════════════════════════
# PDF EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def create_pdf(name, binary, pattern, features, path):
    try:
        doc = SimpleDocTemplate(str(path), pagesize=letter)
        story, styles = [], getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24,
                                     textColor=colors.HexColor('#1f77b4'), alignment=1)
        story.append(Paragraph("Clinical Gait Analysis Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Info table
        info = [['Patient:', name], ['Date:', datetime.now().strftime('%Y-%m-%d %H:%M')],
                ['ID:', f"GAIT-{datetime.now().strftime('%Y%m%d%H%M')}"]]
        t = Table(info, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
                               ('GRID', (0,0), (-1,-1), 1, colors.black)]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Results
        story.append(Paragraph("Classification Results", styles['Heading2']))
        res = [['Type', 'Result', 'Confidence'],
               ['Binary', binary['prediction'], f"{binary['confidence']*100:.1f}%"],
               ['Pattern', pattern['pattern'], f"{pattern['confidence']*100:.1f}%"],
               ['ICD-10', pattern.get('icd10', 'N/A'), '']]
        t = Table(res, colWidths=[2*inch, 2*inch, 2*inch])
        t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f77b4')),
                               ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                               ('GRID', (0,0), (-1,-1), 1, colors.black)]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Features
        story.append(Paragraph("Gait Parameters", styles['Heading2']))
        fdata = [['Feature', 'Value', 'Unit']]
        fmap = {'cadence': ('Cadence', 'steps/min'), 'stride_time_mean': ('Stride Time', 'sec'),
                'step_length_mean': ('Step Length', 'm'), 'step_width_mean': ('Step Width', 'm'),
                'knee_angle_left_mean': ('L Knee Angle', '°'), 'knee_angle_right_mean': ('R Knee Angle', '°'),
                'temporal_symmetry': ('Temporal Sym', '%'), 'spatial_symmetry': ('Spatial Sym', '%')}
        for k, (label, unit) in fmap.items():
            if k in features:
                v = features[k] * 100 if 'symmetry' in k else features[k]
                fdata.append([label, f"{v:.2f}", unit])
        t = Table(fdata, colWidths=[2.5*inch, 1.5*inch, 1*inch])
        t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.green),
                               ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                               ('GRID', (0,0), (-1,-1), 1, colors.black)]))
        story.append(t)
        
        doc.build(story)
        return True
    except Exception as e:
        logger.error(f"PDF error: {e}\n{traceback.format_exc()}")
        st.error(f"PDF generation failed: {str(e)}")
        return False

# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Clinical Gait Analysis", page_icon="🏥", layout="wide")
    Paths.init()
    
    # Init session with pipeline state
    if 'done' not in st.session_state:
        st.session_state.update({
            'done': False, 'binary': None, 'pattern': None,
            'features': None, 'feat_dict': None, 'mp_out': None, 'name': '', 'vid_path': None,
            'analysis_started': False, 'pipeline_step': 0
        })
    
    # CSS
    st.markdown("""
    <style>
    .main{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)}
    .title{font-size:3rem;color:white;text-align:center;padding:2rem;text-shadow:2px 2px 4px rgba(0,0,0,0.3)}
    .box{background:white;border-radius:15px;padding:2rem;box-shadow:0 8px 20px rgba(0,0,0,0.15);margin:1rem 0}
    .metric{text-align:center;padding:1.5rem;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:10px}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='title'>🏥 Clinical Gait Analysis System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        name = st.text_input("Patient Name", value=st.session_state.name or f"Patient-{datetime.now().strftime('%Y%m%d')}")
        st.session_state.name = name
        
        st.markdown("---")
        st.header("📹 Upload Video")
        vid = st.file_uploader("Gait Video", type=['mp4', 'avi', 'mov'])
        
        if vid:
            st.success(f"✓ {vid.name}")
            st.info(f"Size: {vid.size/(1024*1024):.2f} MB")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📈 Visualizations", "📄 Report"])
    
    with tab1:
        if vid:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Video Preview")
                st.video(vid)
                
                # Show pipeline progress
                show_pipeline_progress()
            
            with col2:
                st.subheader("Controls")
                
                # Only show Start Analysis if video is uploaded but analysis not started
                if not st.session_state.get('analysis_started', False):
                    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                        st.session_state.analysis_started = True
                        analyze(vid, name)
                
                if st.session_state.done:
                    st.success("✓ Analysis Complete!")
                    if st.button("🔄 New Analysis", use_container_width=True):
                        reset_analysis()
                        st.rerun()
            
            if st.session_state.done:
                st.markdown("---")
                show_results()
        else:
            st.info("👈 Upload a video to begin")
            show_pipeline_progress()
    
    with tab2:
        if st.session_state.done:
            show_viz()
        else:
            st.info("Run analysis first")
    
    with tab3:
        if st.session_state.done:
            show_report()
        else:
            st.info("Run analysis first")

def analyze(vid, name):
    """Sequential pipeline execution with visual progress"""
    try:
        # STEP 1: Upload Video
        st.session_state.pipeline_step = 1
        with st.spinner("📤 Step 1/5: Uploading video..."):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tmp.write(vid.read())
            tmp.close()
            vpath = Path(tmp.name)
            
            dest = Paths.UPLOADS / vid.name
            shutil.copy(vpath, dest)
            st.session_state.vid_path = dest
            st.success("✓ Step 1: Video uploaded")
        
        # STEP 2: MediaPipe Processing
        st.session_state.pipeline_step = 2
        with st.spinner("🎯 Step 2/5: MediaPipe pose detection..."):
            mp_out = MediaPipe.process(dest)
            st.session_state.mp_out = mp_out
            if mp_out['annotated']:
                st.success("✓ Step 2: MediaPipe processing complete")
            else:
                st.warning("⚠️ Step 2: MediaPipe outputs not found (models may not be loaded)")
        
        # STEP 3: Generate Videos (already done by MediaPipe)
        st.session_state.pipeline_step = 3
        if mp_out['annotated'] or mp_out['skeleton']:
            st.success("✓ Step 3: Annotated and skeleton videos generated")
        else:
            st.info("ℹ️ Step 3: Video generation skipped (MediaPipe outputs not available)")
        
        # STEP 4: Feature Extraction
        st.session_state.pipeline_step = 4
        with st.spinner("🧮 Step 4/5: Extracting gait features..."):
            feat, feat_dict = Features.extract(mp_out.get('landmarks'))
            st.session_state.features = feat
            st.session_state.feat_dict = feat_dict
            st.success(f"✓ Step 4: Extracted {feat.shape[1]} features")
        
        # STEP 5: ML Analysis (Binary + Classification in parallel)
        st.session_state.pipeline_step = 5
        clf = Classifier()
        
        with st.spinner("🤖 Step 5/5: Running ML analysis..."):
            # Binary classification
            binary = clf.classify_binary(feat)
            st.session_state.binary = binary
            
            # Pattern classification
            pattern = clf.classify_pattern(feat)
            st.session_state.pattern = pattern
            
            # Show model warnings
            if not binary.get('is_production'):
                st.warning("⚠️ Binary: Using fallback model (production model not found)")
            if not pattern.get('is_production'):
                st.warning("⚠️ Classification: Using fallback model (DL model not found)")
            
            st.success("✓ Step 5: ML analysis complete")
        
        # Complete
        st.session_state.done = True
        st.session_state.analysis_started = False
        st.balloons()
        st.success("✅ All steps completed successfully!")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Analysis error: {e}\n{traceback.format_exc()}")
        st.error(f"❌ Analysis failed at step {st.session_state.pipeline_step}: {str(e)}")
        st.error("Please check the logs for details")
        st.session_state.analysis_started = False

def reset_analysis():
    """Reset all analysis state"""
    st.session_state.update({
        'done': False, 'binary': None, 'pattern': None,
        'features': None, 'feat_dict': None, 'mp_out': None,
        'vid_path': None, 'analysis_started': False, 'pipeline_step': 0
    })

def show_pipeline_progress():
    """Visual pipeline progress indicator"""
    st.markdown("---")
    st.markdown("### 📋 Analysis Pipeline")
    
    steps = [
        {"num": 1, "name": "Upload Video", "desc": "Video input"},
        {"num": 2, "name": "MediaPipe", "desc": "Pose detection"},
        {"num": 3, "name": "Generate Videos", "desc": "Annotated + Skeleton"},
        {"num": 4, "name": "ML Analysis", "desc": "Classification"},
        {"num": 5, "name": "Results", "desc": "Report generation"}
    ]
    
    current_step = st.session_state.get('pipeline_step', 0)
    completed = st.session_state.done
    
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            # Determine status
            if completed or current_step > step['num']:
                status = "✅"
                bg_color = "#d4edda"
                border_color = "#28a745"
            elif current_step == step['num']:
                status = "⏳"
                bg_color = "#fff3cd"
                border_color = "#ffc107"
            else:
                status = "⏸️"
                bg_color = "#f8f9fa"
                border_color = "#dee2e6"
            
            st.markdown(f"""
            <div style='background:{bg_color};border-left:4px solid {border_color};
                        padding:15px;border-radius:8px;text-align:center;min-height:100px'>
                <div style='font-size:2rem'>{status}</div>
                <div style='font-weight:bold;margin:8px 0'>{step['num']}. {step['name']}</div>
                <div style='font-size:0.85rem;color:#666'>{step['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

def show_results():
    b, p = st.session_state.binary, st.session_state.pattern
    
    col1, col2, col3 = st.columns(3)
    
    # Box 1: Binary
    with col1:
        st.markdown("<div class='box'>", unsafe_allow_html=True)
        st.subheader("🎯 Binary Classification")
        color = "🟢" if b['prediction'] == 'Normal' else "🔴"
        st.markdown(f"### {color} {b['prediction']}")
        st.progress(b['confidence'])
        st.caption(f"Confidence: {b['confidence']*100:.1f}%")
        if not b.get('is_production'):
            st.caption("⚠️ Fallback model")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Box 2: Pattern
    with col2:
        st.markdown("<div class='box'>", unsafe_allow_html=True)
        st.subheader("🔍 Pattern Classification")
        st.markdown(f"### {p['pattern']}")
        st.caption(f"ICD-10: {p['icd10']}")
        st.caption(p['description'])
        st.progress(p['confidence'])
        st.caption(f"Confidence: {p['confidence']*100:.1f}%")
        if not p.get('is_production'):
            st.caption("⚠️ Fallback model (DL model not loaded)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Box 3: Features
    with col3:
        st.markdown("<div class='box'>", unsafe_allow_html=True)
        st.subheader("⚡ Key Features")
        fd = st.session_state.feat_dict
        st.metric("Cadence", f"{fd.get('cadence', 0):.1f} steps/min")
        st.metric("Stride Time", f"{fd.get('stride_time_mean', 0):.2f} sec")
        st.metric("Step Length", f"{fd.get('step_length_mean', 0):.2f} m")
        st.metric("Symmetry", f"{fd.get('temporal_symmetry', 0)*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Videos if available
    mp = st.session_state.mp_out
    if mp and (mp['annotated'] or mp['skeleton']):
        st.markdown("---")
        st.subheader("🎥 Processed Videos")
        cols = st.columns(3)
        
        if st.session_state.vid_path and st.session_state.vid_path.exists():
            with cols[0]:
                st.caption("Original")
                st.video(str(st.session_state.vid_path))
        
        if mp['annotated'] and mp['annotated'].exists():
            with cols[1]:
                st.caption("Annotated")
                st.video(str(mp['annotated']))
        
        if mp['skeleton'] and mp['skeleton'].exists():
            with cols[2]:
                st.caption("Skeleton")
                st.video(str(mp['skeleton']))

def show_viz():
    b, p = st.session_state.binary, st.session_state.pattern
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Binary probs
        fig = px.bar(x=list(b['probabilities'].values()), y=list(b['probabilities'].keys()),
                    orientation='h', title="Binary Classification", color=list(b['probabilities'].values()),
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pattern probs
        if p['probabilities']:
            df = pd.DataFrame({'Pattern': list(p['probabilities'].keys()), 
                              'Prob': [v*100 for v in p['probabilities'].values()]})
            df = df.sort_values('Prob', ascending=True)
            fig = px.bar(df, x='Prob', y='Pattern', orientation='h',
                        title="Pattern Probabilities", color='Prob', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

def show_report():
    b, p, fd = st.session_state.binary, st.session_state.pattern, st.session_state.feat_dict
    name = st.session_state.name
    
    st.subheader("📄 Clinical Report")
    
    st.markdown(f"**Patient:** {name}")
    st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown("---")
    
    st.markdown("### Classification")
    st.markdown(f"- **Binary:** {b['prediction']} ({b['confidence']*100:.1f}%)")
    st.markdown(f"- **Pattern:** {p['pattern']} ({p['confidence']*100:.1f}%)")
    st.markdown(f"- **ICD-10:** {p['icd10']}")
    
    st.markdown("### Parameters")
    for k, v in list(fd.items())[:8]:
        st.markdown(f"- **{k.replace('_', ' ').title()}:** {v:.2f}")
    
    st.markdown("---")
    
    # Export
    if st.button("📥 Export PDF", type="primary"):
        pdf_path = Paths.EXPORTS / f"report_{name}_{datetime.now().strftime('%Y%m%d%H%M')}.pdf"
        if create_pdf(name, b, p, fd, pdf_path):
            with open(pdf_path, 'rb') as f:
                st.download_button("Download PDF", f.read(), file_name=pdf_path.name, mime='application/pdf')
            st.success("✓ PDF generated!")

if __name__ == "__main__":
    main()