#!/usr/bin/env python3
"""
MEDIAPIPE POSE DETECTION PIPELINE - PRODUCTION GRADE APPLICATION
Complete implementation with robust video rendering and export functionality
"""

import os
import sys
import warnings
import logging

# Environment configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import streamlit as st
import json
import importlib.util
from pathlib import Path
from datetime import datetime
import traceback
import hashlib
import subprocess
import time
import zipfile
from io import BytesIO
from typing import Optional, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.absolute()
CONFIG_PATH = PROJECT_ROOT / "config.json"
MEDIAPIPE_SCRIPT = PROJECT_ROOT / "pre-processing-models" / "mediapipe" / "pre_mediapipe.py"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# VIDEO CODEC DETECTION & CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

class VideoConverter:
    """Production-grade video converter with multiple fallback strategies"""
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5, text=True)
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def get_video_codec(video_path: Path) -> Optional[str]:
        """Get codec information from video"""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            cap.release()
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            return codec.strip()
        except Exception as e:
            logger.error(f"Failed to get codec: {e}")
            return None
    
    @staticmethod
    def is_web_compatible(codec: str) -> bool:
        """Check if codec is web browser compatible"""
        if not codec:
            return False
        codec_upper = codec.upper()
        compatible = ['AVC1', 'H264', 'X264']
        incompatible = ['MP4V', 'XVID', 'DIVX', 'FMP4']
        
        if any(c in codec_upper for c in compatible):
            return True
        if any(c in codec_upper for c in incompatible):
            return False
        return True
    
    @staticmethod
    def convert_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
        """Convert video using FFmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                '-c:a', 'aac', '-b:a', '128k', '-y', str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300, text=True)
            
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"FFmpeg conversion successful: {output_path.name}")
                return True
            else:
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"FFmpeg error: {e}")
            return False
    
    @staticmethod
    def convert_with_opencv(input_path: Path, output_path: Path) -> bool:
        """Fallback: Convert using OpenCV"""
        try:
            import cv2
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                return False
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            codec_options = [('avc1', 'H.264'), ('H264', 'H.264'), ('X264', 'X264'), ('mp4v', 'MPEG-4')]
            
            for codec_str, codec_name in codec_options:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_str)
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    
                    if not out.isOpened():
                        continue
                    
                    logger.info(f"Using {codec_name} for conversion")
                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                        frame_count += 1
                    
                    cap.release()
                    out.release()
                    
                    if output_path.exists() and output_path.stat().st_size > 0:
                        logger.info(f"OpenCV conversion successful with {codec_name}")
                        return True
                except:
                    continue
            
            cap.release()
            return False
        except Exception as e:
            logger.error(f"OpenCV error: {e}")
            return False
    
    @classmethod
    def ensure_web_compatible(cls, video_path: Path) -> Path:
        """Ensure video is web-compatible, convert if necessary"""
        if not video_path or not video_path.exists():
            return video_path
        
        web_path = video_path.parent / f"{video_path.stem}_h264.mp4"
        if web_path.exists() and web_path.stat().st_size > 0:
            logger.info(f"Using cached H.264: {web_path.name}")
            return web_path
        
        codec = cls.get_video_codec(video_path)
        logger.info(f"Video codec: {codec}")
        
        if cls.is_web_compatible(codec):
            logger.info(f"Already compatible: {video_path.name}")
            return video_path
        
        logger.warning(f"Converting from {codec} to H.264")
        
        if cls.check_ffmpeg():
            if cls.convert_with_ffmpeg(video_path, web_path):
                return web_path
        
        if cls.convert_with_opencv(video_path, web_path):
            return web_path
        
        logger.error(f"All conversions failed for {video_path.name}")
        return video_path


# ═══════════════════════════════════════════════════════════════════════════
# FILE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class FileManager:
    """Production-grade file management"""
    
    @staticmethod
    def get_file_hash(file_bytes: bytes) -> str:
        return hashlib.md5(file_bytes).hexdigest()
    
    @staticmethod
    def cleanup_old_uploads(keep_latest: int = 1) -> None:
        try:
            uploads = sorted(UPLOAD_DIR.glob("*.*"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_file in uploads[keep_latest:]:
                try:
                    old_file.unlink()
                    logger.info(f"Deleted: {old_file.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {old_file.name}: {e}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    @staticmethod
    def cleanup_old_outputs(video_stem: str) -> None:
        try:
            patterns = [
                f"*{video_stem}*_annotated*.mp4",
                f"*{video_stem}*_skeleton*.mp4",
                f"*{video_stem}*_landmarks*.csv",
                f"*{video_stem}*_h264*.mp4",
            ]
            for pattern in patterns:
                for old_file in OUTPUT_DIR.glob(pattern):
                    try:
                        old_file.unlink()
                        logger.info(f"Deleted: {old_file.name}")
                    except Exception as e:
                        logger.warning(f"Could not delete {old_file.name}: {e}")
        except Exception as e:
            logger.error(f"Output cleanup failed: {e}")
    
    @staticmethod
    def save_uploaded_video(uploaded_file) -> Tuple[Optional[Path], bool]:
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension not in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']:
            logger.error(f"Unsupported: {file_extension}")
            return None, False
        
        file_bytes = uploaded_file.getvalue()
        file_hash = FileManager.get_file_hash(file_bytes)
        
        for existing_file in UPLOAD_DIR.glob(f"*{file_extension}"):
            try:
                with open(existing_file, 'rb') as f:
                    if FileManager.get_file_hash(f.read()) == file_hash:
                        logger.info(f"Duplicate: {existing_file.name}")
                        return existing_file, True
            except:
                pass
        
        FileManager.cleanup_old_uploads(keep_latest=0)
        
        clean_name = Path(uploaded_file.name).stem
        video_path = UPLOAD_DIR / f"{clean_name}{file_extension}"
        
        if video_path.exists():
            video_path.unlink()
        
        FileManager.cleanup_old_outputs(clean_name)
        
        with open(video_path, 'wb') as f:
            f.write(file_bytes)
        
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error("Invalid video")
                video_path.unlink()
                return None, False
            cap.release()
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            if video_path.exists():
                video_path.unlink()
            return None, False
        
        logger.info(f"Saved: {video_path.name}")
        return video_path, False
    
    @staticmethod
    def find_output_videos(video_path: Path) -> Dict[str, Optional[Path]]:
        video_stem = video_path.stem
        
        if '_' in video_stem:
            parts = video_stem.split('_')
            if len(parts[0]) == 8 and parts[0].isdigit():
                if len(parts) > 2:
                    video_stem = '_'.join(parts[2:])
                elif len(parts) > 1:
                    video_stem = parts[1]
        
        logger.info(f"Searching for: {video_stem}")
        
        results = {'annotated': None, 'skeleton': None, 'csv': None}
        
        candidates = {
            'annotated': [
                OUTPUT_DIR / f"{video_stem}_annotated.mp4",
                OUTPUT_DIR / f"{video_path.stem}_annotated.mp4",
            ],
            'skeleton': [
                OUTPUT_DIR / f"{video_stem}_skeleton.mp4",
                OUTPUT_DIR / f"{video_path.stem}_skeleton.mp4",
            ],
            'csv': [
                OUTPUT_DIR / f"{video_stem}_landmarks.csv",
                OUTPUT_DIR / f"{video_path.stem}_landmarks.csv",
            ]
        }
        
        for key, paths in candidates.items():
            for candidate in paths:
                if candidate.exists() and candidate.stat().st_size > 0:
                    results[key] = candidate
                    logger.info(f"Found {key}: {candidate.name}")
                    break
        
        for file in OUTPUT_DIR.iterdir():
            name_lower = file.name.lower()
            stem_lower = video_stem.lower()
            
            if stem_lower in name_lower or video_path.stem.lower() in name_lower:
                if 'annotated' in name_lower and file.suffix == '.mp4' and not results['annotated']:
                    results['annotated'] = file
                    logger.info(f"Fuzzy found annotated: {file.name}")
                elif 'skeleton' in name_lower and file.suffix == '.mp4' and not results['skeleton']:
                    results['skeleton'] = file
                    logger.info(f"Fuzzy found skeleton: {file.name}")
                elif 'landmark' in name_lower and file.suffix == '.csv' and not results['csv']:
                    results['csv'] = file
                    logger.info(f"Fuzzy found CSV: {file.name}")
        
        if not results['annotated']:
            annotated = sorted(OUTPUT_DIR.glob("*annotated*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
            if annotated:
                results['annotated'] = annotated[0]
                logger.info(f"Most recent annotated: {annotated[0].name}")
        
        if not results['skeleton']:
            skeleton = sorted(OUTPUT_DIR.glob("*skeleton*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
            if skeleton:
                results['skeleton'] = skeleton[0]
                logger.info(f"Most recent skeleton: {skeleton[0].name}")
        
        if not results['csv']:
            csv_files = sorted(OUTPUT_DIR.glob("*landmark*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
            if csv_files:
                results['csv'] = csv_files[0]
                logger.info(f"Most recent CSV: {csv_files[0].name}")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class PipelineManager:
    
    @staticmethod
    def load_config() -> Optional[dict]:
        if not CONFIG_PATH.exists():
            logger.error(f"Config not found: {CONFIG_PATH}")
            return None
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("Config loaded")
            return config
        except Exception as e:
            logger.error(f"Load config failed: {e}")
            return None
    
    @staticmethod
    def save_config(config: dict) -> bool:
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            logger.info("Config saved")
            return True
        except Exception as e:
            logger.error(f"Save config failed: {e}")
            return False
    
    @staticmethod
    def update_config_with_video(video_path: Path) -> bool:
        config = PipelineManager.load_config()
        if not config:
            return False
        
        try:
            rel_path = str(video_path.relative_to(PROJECT_ROOT))
        except:
            rel_path = str(video_path)
        
        config["input_paths"] = [rel_path]
        config.setdefault("output_dir", "data/output")
        
        return PipelineManager.save_config(config)
    
    @staticmethod
    def load_mediapipe_module():
        if not MEDIAPIPE_SCRIPT.exists():
            logger.error(f"Script not found: {MEDIAPIPE_SCRIPT}")
            return None
        
        try:
            spec = importlib.util.spec_from_file_location("mediapipe_module", MEDIAPIPE_SCRIPT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info("MediaPipe loaded")
            return module
        except Exception as e:
            logger.error(f"Load module failed: {e}")
            return None
    
    @staticmethod
    def run_pipeline() -> Optional[list]:
        mp_module = PipelineManager.load_mediapipe_module()
        if not mp_module:
            return None
        
        try:
            config = mp_module.PipelineConfig.from_json(CONFIG_PATH)
            pipeline = mp_module.PoseDetectionPipeline(config)
            results = pipeline.run()
            logger.info(f"Pipeline complete: {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            return None


# ═══════════════════════════════════════════════════════════════════════════
# VIDEO DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

class VideoDisplay:
    
    @staticmethod
    def get_video_info(video_path: Path) -> Optional[dict]:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)]).strip()
            
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
                'size_mb': video_path.stat().st_size / (1024 * 1024),
                'codec': codec
            }
            cap.release()
            return info
        except Exception as e:
            logger.error(f"Get info failed: {e}")
            return None
    
    @staticmethod
    def display_video_with_download(video_path: Optional[Path], label: str, key_suffix: str) -> None:
        st.markdown(f"**{label}**")
        
        if not video_path or not video_path.exists():
            st.warning("⚠️ Video not found")
            return
        
        try:
            web_video = VideoConverter.ensure_web_compatible(video_path)
            
            with open(web_video, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                info = VideoDisplay.get_video_info(web_video)
                if info:
                    codec_status = "✅" if VideoConverter.is_web_compatible(info['codec']) else "⚠️"
                    st.caption(f"📊 {info['width']}×{info['height']} | {info['fps']:.0f} FPS | {info['duration']:.1f}s | {info['size_mb']:.1f} MB | {codec_status} {info['codec']}")
            
            with col2:
                st.download_button(
                    "📥 Download",
                    video_bytes,
                    file_name=web_video.name,
                    mime="video/mp4",
                    key=f"download_{key_suffix}",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Display error: {str(e)}")
            logger.error(f"Display error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class ExportManager:
    
    @staticmethod
    def create_results_zip(video_path: Path, output_files: dict) -> Optional[BytesIO]:
        try:
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                if video_path and video_path.exists():
                    zf.write(video_path, f"original/{video_path.name}")
                    logger.info(f"Added to ZIP: {video_path.name}")
                
                for key, path in output_files.items():
                    if path and path.exists():
                        if key == 'csv':
                            zf.write(path, f"data/{path.name}")
                        else:
                            zf.write(path, f"videos/{path.name}")
                        logger.info(f"Added to ZIP: {path.name}")
                
                metadata = {
                    'generated_at': datetime.now().isoformat(),
                    'original_video': video_path.name if video_path else None,
                    'files': {key: path.name if path else None for key, path in output_files.items()}
                }
                zf.writestr('metadata.json', json.dumps(metadata, indent=2))
            
            zip_buffer.seek(0)
            logger.info("ZIP created")
            return zip_buffer
        except Exception as e:
            logger.error(f"ZIP failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="MediaPipe Pose Detection",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .stAlert > div { padding: 1rem; }
    .stVideo { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎥 MediaPipe Pose Detection Pipeline")
    st.markdown("**Production-Grade Gait Analysis Application**")
    st.markdown("---")
    
    # Initialize session state
    if 'uploaded_video_path' not in st.session_state:
        st.session_state.uploaded_video_path = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'output_videos' not in st.session_state:
        st.session_state.output_videos = {}
    
    # Sidebar
    with st.sidebar:
        st.header("📋 System Status")
        
        if CONFIG_PATH.exists():
            st.success("✅ Config file")
        else:
            st.error("❌ Config missing")
        
        if MEDIAPIPE_SCRIPT.exists():
            st.success("✅ MediaPipe script")
        else:
            st.error("❌ Script missing")
        
        if VideoConverter.check_ffmpeg():
            st.success("✅ FFmpeg available")
        else:
            st.warning("⚠️ FFmpeg unavailable")
        
        st.markdown("---")
        st.header("📊 Current Session")
        
        if st.session_state.uploaded_video_path:
            st.info(f"📹 {st.session_state.uploaded_video_path.name}")
            if st.session_state.processing_complete:
                st.success("✅ Processed")
        else:
            st.info("No video uploaded")
        
        st.markdown("---")
        
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.uploaded_video_path = None
            st.session_state.processing_complete = False
            st.session_state.output_videos = {}
            st.rerun()
        
        with st.expander("🐛 Debug"):
            st.write(f"Uploads: {len(list(UPLOAD_DIR.glob('*')))}")
            st.write(f"Outputs: {len(list(OUTPUT_DIR.glob('*')))}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "⚙️ Process", "📊 Results"])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: UPLOAD
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.subheader("Upload Video for Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'],
                help="Upload walking/gait video"
            )
            
            if uploaded_file:
                st.info(f"📄 {uploaded_file.name}")
                st.info(f"📏 {uploaded_file.size / (1024*1024):.2f} MB")
                
                with st.spinner("Saving video..."):
                    video_path, is_duplicate = FileManager.save_uploaded_video(uploaded_file)
                
                if video_path:
                    st.session_state.uploaded_video_path = video_path
                    st.session_state.processing_complete = False
                    st.session_state.output_videos = {}
                    
                    if is_duplicate:
                        st.warning("⚠️ This video was previously uploaded")
                    else:
                        st.success("✅ Video saved successfully!")
                else:
                    st.error("❌ Failed to save video")
        
        with col2:
            if st.session_state.uploaded_video_path:
                st.metric("Status", "Ready for processing")
                st.metric("Video", st.session_state.uploaded_video_path.name)
        
        # Display uploaded video
        if st.session_state.uploaded_video_path:
            st.markdown("---")
            st.subheader("📹 Video Preview")
            VideoDisplay.display_video_with_download(
                st.session_state.uploaded_video_path,
                "Original Video",
                "tab1_original"
            )
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: PROCESS
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.subheader("Process Video with MediaPipe")
        
        if not st.session_state.uploaded_video_path:
            st.info("👈 Please upload a video in the Upload tab first")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"📹 **Ready to process:** {st.session_state.uploaded_video_path.name}")
                
                # Show config preview
                with st.expander("⚙️ View Current Configuration"):
                    config = PipelineManager.load_config()
                    if config:
                        st.json(config)
            
            with col2:
                if st.button("📝 Update Config", use_container_width=True, type="secondary"):
                    with st.spinner("Updating configuration..."):
                        if PipelineManager.update_config_with_video(st.session_state.uploaded_video_path):
                            st.success("✅ Config updated!")
                        else:
                            st.error("❌ Failed to update config")
                
                st.markdown("")
                
                if st.button("🚀 Run Pipeline", use_container_width=True, type="primary"):
                    # Update config first
                    with st.spinner("Preparing pipeline..."):
                        if not PipelineManager.update_config_with_video(st.session_state.uploaded_video_path):
                            st.error("❌ Failed to update configuration")
                            st.stop()
                    
                    # Run pipeline
                    st.markdown("---")
                    st.subheader("⏳ Processing Video...")
                    
                    progress_bar = st.progress(0, text="Initializing MediaPipe...")
                    status_placeholder = st.empty()
                    
                    try:
                        progress_bar.progress(10, text="Loading MediaPipe module...")
                        time.sleep(0.5)
                        
                        progress_bar.progress(20, text="Starting pose detection...")
                        start_time = time.time()
                        
                        results = PipelineManager.run_pipeline()
                        
                        progress_bar.progress(80, text="Processing complete, locating outputs...")
                        
                        if results and len(results) > 0:
                            # Find output files
                            time.sleep(2)  # Give filesystem time to sync
                            output_files = FileManager.find_output_videos(st.session_state.uploaded_video_path)
                            
                            progress_bar.progress(90, text="Converting videos for web compatibility...")
                            
                            # Convert videos to web-compatible format
                            if output_files.get('annotated'):
                                status_placeholder.info("🔄 Converting annotated video...")
                                VideoConverter.ensure_web_compatible(output_files['annotated'])
                            
                            if output_files.get('skeleton'):
                                status_placeholder.info("🔄 Converting skeleton video...")
                                VideoConverter.ensure_web_compatible(output_files['skeleton'])
                            
                            progress_bar.progress(100, text="Complete!")
                            
                            # Update session state
                            st.session_state.output_videos = output_files
                            st.session_state.processing_complete = True
                            
                            end_time = time.time()
                            
                            status_placeholder.empty()
                            st.success(f"✅ Pipeline completed in {end_time - start_time:.1f} seconds!")
                            st.balloons()
                            
                            # Show summary
                            st.markdown("---")
                            st.subheader("📋 Processing Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Frames Processed", results[0].frames_processed if results else "N/A")
                            with col2:
                                st.metric("Landmarks Detected", results[0].landmarks_detected if results else "N/A")
                            with col3:
                                st.metric("Processing Time", f"{end_time - start_time:.1f}s")
                            
                            st.info("✨ Check the Results tab to view and download outputs")
                            
                        else:
                            progress_bar.empty()
                            status_placeholder.empty()
                            st.error("❌ Pipeline failed to produce results. Check logs for details.")
                    
                    except Exception as e:
                        progress_bar.empty()
                        status_placeholder.empty()
                        st.error(f"❌ Pipeline error: {str(e)}")
                        logger.error(f"Pipeline execution error: {e}")
                        logger.error(traceback.format_exc())
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab3:
        st.subheader("Analysis Results")
        
        if not st.session_state.processing_complete:
            st.info("⏳ No results available yet. Process a video in the Process tab.")
        else:
            output_files = st.session_state.output_videos
            
            # Refresh button
            if st.button("🔄 Refresh Output Files"):
                with st.spinner("Refreshing..."):
                    output_files = FileManager.find_output_videos(st.session_state.uploaded_video_path)
                    st.session_state.output_videos = output_files
                    st.rerun()
            
            st.markdown("---")
            
            # Display videos in 3 columns
            st.subheader("📹 Video Outputs")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                VideoDisplay.display_video_with_download(
                    st.session_state.uploaded_video_path,
                    "📹 Original Video",
                    "result_original"
                )
            
            with col2:
                VideoDisplay.display_video_with_download(
                    output_files.get('annotated'),
                    "🎨 Annotated Video",
                    "result_annotated"
                )
            
            with col3:
                VideoDisplay.display_video_with_download(
                    output_files.get('skeleton'),
                    "🦴 Skeleton Video",
                    "result_skeleton"
                )
            
            # CSV Data section
            st.markdown("---")
            st.subheader("📄 Landmarks Data")
            
            if output_files.get('csv'):
                csv_path = output_files['csv']
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.info(f"**CSV File:** {csv_path.name}")
                    st.caption(f"📏 Size: {csv_path.stat().st_size / 1024:.1f} KB")
                
                with col2:
                    with open(csv_path, 'rb') as f:
                        st.download_button(
                            "📥 Download CSV",
                            f.read(),
                            file_name=csv_path.name,
                            mime="text/csv",
                            key="download_csv",
                            use_container_width=True
                        )
                
                # Preview CSV
                with st.expander("👁️ Preview CSV Data"):
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_path)
                        st.dataframe(df.head(10), use_container_width=True)
                        st.caption(f"Showing first 10 rows of {len(df)} total rows")
                    except Exception as e:
                        st.error(f"Could not preview CSV: {e}")
            else:
                st.warning("⚠️ CSV file not found")
            
            # Export all results
            st.markdown("---")
            st.subheader("📦 Export All Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("Download all results (videos + CSV) as a ZIP archive")
            
            with col2:
                if st.button("📦 Create ZIP", use_container_width=True, type="primary"):
                    with st.spinner("Creating ZIP archive..."):
                        zip_buffer = ExportManager.create_results_zip(
                            st.session_state.uploaded_video_path,
                            output_files
                        )
                        
                        if zip_buffer:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                "📥 Download ZIP Archive",
                                zip_buffer.getvalue(),
                                file_name=f"mediapipe_results_{timestamp}.zip",
                                mime="application/zip",
                                key="download_zip",
                                use_container_width=True
                            )
                            st.success("✅ ZIP archive ready!")
                        else:
                            st.error("❌ Failed to create ZIP archive")
            
            # Debug section
            with st.expander("🐛 Debug: File Locations"):
                st.write("**Output files found:**")
                for key, path in output_files.items():
                    if path:
                        st.write(f"- {key}: `{path}` (exists: {path.exists()})")
                    else:
                        st.write(f"- {key}: Not found")


if __name__ == "__main__":
    main()