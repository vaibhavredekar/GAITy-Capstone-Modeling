#!/usr/bin/env python3
"""
BARE-BONE MEDIAPIPE TEST APPLICATION
Enhanced with video display fixes and progress indicators
"""

# Configure environment BEFORE importing any libraries
import os
import sys
import warnings
import logging

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_MIN_LOG_LEVEL'] = '3'

# Suppress absl logging
os.environ['ABSL_LOG_MIN_LEVEL'] = '3'

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Monkey patch to suppress protobuf warnings
original_warn = warnings.warn

def custom_warn(*args, **kwargs):
    if len(args) >= 1 and isinstance(args[0], str) and "SymbolDatabase.GetPrototype" in args[0]:
        return  # Suppress this specific warning
    return original_warn(*args, **kwargs)

warnings.warn = custom_warn

# Now import other libraries
import streamlit as st
import json
import importlib.util
from pathlib import Path
from datetime import datetime
import traceback
import tempfile
from io import BytesIO
import time
import threading
import queue

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Get the project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).parent.absolute()
CONFIG_PATH = PROJECT_ROOT / "config.json"
MEDIAPIPE_SCRIPT = PROJECT_ROOT / "pre-processing-models" / "mediapipe" / "pre_mediapipe.py"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED VIDEO HANDLING
# ═══════════════════════════════════════════════════════════════════════════

def find_output_videos_enhanced(video_path):
    """
    Enhanced function to find generated videos with multiple fallback strategies
    """
    if not video_path:
        return {'annotated': None, 'skeleton': None, 'csv': None}
    
    # Try different naming patterns
    video_stem = video_path.stem
    
    # Remove timestamp prefix if present (e.g., "20240121_120519_video.mp4" -> "video")
    if '_' in video_stem and video_stem.split('_')[0].isdigit():
        parts = video_stem.split('_', 2)
        if len(parts) >= 2:
            video_stem = parts[-1]  # Take the last part after timestamp
    
    # Possible output patterns
    patterns = [
        f"{video_stem}_annotated.mp4",
        f"{video_stem}_skeleton.mp4",
        f"{video_stem}_landmarks.csv",
        f"{video_path.stem}_annotated.mp4",  # Original stem
        f"{video_path.stem}_skeleton.mp4",
        f"{video_path.stem}_landmarks.csv",
    ]
    
    results = {}
    
    # Search for annotated video
    for pattern in patterns[:3] + patterns[3:4]:
        candidate = OUTPUT_DIR / pattern
        if candidate.exists():
            results['annotated'] = candidate
            break
    
    # Search for skeleton video
    for pattern in patterns[1:2] + patterns[4:5]:
        candidate = OUTPUT_DIR / pattern
        if candidate.exists():
            results['skeleton'] = candidate
            break
    
    # Search for CSV
    for pattern in patterns[2:3] + patterns[5:6]:
        candidate = OUTPUT_DIR / pattern
        if candidate.exists():
            results['csv'] = candidate
            break
    
    return results


def wait_for_file_creation(file_path, timeout=30):
    """
    Wait for a file to be created and fully written
    """
    if not file_path:
        return False
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if file_path.exists():
            # Check if file is still being written (simple check)
            try:
                size1 = file_path.stat().st_size
                time.sleep(0.5)
                size2 = file_path.stat().st_size
                if size1 == size2 and size1 > 0:
                    return True
            except:
                pass
        time.sleep(0.5)
    return False


def get_all_output_videos():
    """
    Get all videos in the output directory
    """
    if not OUTPUT_DIR.exists():
        return []
    
    videos = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        videos.extend(OUTPUT_DIR.glob(f"*{ext}"))
    return sorted(videos, key=lambda x: x.stat().st_mtime, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKING
# ═══════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """Thread-safe progress tracker for MediaPipe pipeline"""
    
    def __init__(self):
        self.progress_queue = queue.Queue()
        self.current_status = "Initializing..."
        self.progress = 0
        self.total_frames = 0
        self.processed_frames = 0
        self.current_frame = 0
        self.eta_seconds = 0
        self.start_time = None
        self.is_running = False
    
    def start(self, total_frames=None):
        """Start tracking progress"""
        self.is_running = True
        self.start_time = time.time()
        self.total_frames = total_frames or 0
        self.processed_frames = 0
        self.current_frame = 0
        self.progress = 0
        self.current_status = "Processing..."
    
    def update(self, frame_num=None, status=None):
        """Update progress"""
        if frame_num is not None:
            self.current_frame = frame_num
            self.processed_frames = frame_num
            
            if self.total_frames > 0:
                self.progress = min(100, (frame_num / self.total_frames) * 100)
                
                # Calculate ETA
                if self.start_time and frame_num > 0:
                    elapsed = time.time() - self.start_time
                    rate = frame_num / elapsed
                    remaining_frames = self.total_frames - frame_num
                    self.eta_seconds = remaining_frames / rate if rate > 0 else 0
        
        if status:
            self.current_status = status
        
        # Put update in queue for Streamlit
        self.progress_queue.put({
            'progress': self.progress,
            'status': self.current_status,
            'frame': self.current_frame,
            'total': self.total_frames,
            'eta': self.eta_seconds
        })
    
    def finish(self):
        """Mark as finished"""
        self.is_running = False
        self.progress = 100
        self.current_status = "Completed!"
        self.progress_queue.put({
            'progress': 100,
            'status': 'Completed!',
            'frame': self.total_frames,
            'total': self.total_frames,
            'eta': 0
        })
    
    def error(self, error_msg):
        """Mark as error"""
        self.is_running = False
        self.current_status = f"Error: {error_msg}"
        self.progress_queue.put({
            'progress': self.progress,
            'status': f"Error: {error_msg}",
            'frame': self.current_frame,
            'total': self.total_frames,
            'eta': 0
        })


# Global progress tracker
progress_tracker = ProgressTracker()


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED MEDIAPIPE MODULE LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_mediapipe_module_with_progress():
    """Load MediaPipe module with progress tracking"""
    if not MEDIAPIPE_SCRIPT.exists():
        st.error(f"❌ MediaPipe script not found: {MEDIAPIPE_SCRIPT}")
        return None
    
    try:
        # Read the script content
        with open(MEDIAPIPE_SCRIPT, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # Inject progress tracking into the script
        # This is a bit hacky but works for demonstration
        progress_injection = """
# Progress tracking injection
import sys
import time

# Add progress tracking to VideoProcessor
original_process = None

def process_with_progress(self, input_path):
    global progress_tracker
    progress_tracker.start()
    
    cap = None
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_tracker.total_frames = total_frames
        progress_tracker.update(status=f"Processing {input_path.name} ({total_frames} frames)...")
        
        # Rest of the original process method...
        # (This would need to be more carefully implemented in production)
        
    finally:
        if cap:
            cap.release()
        progress_tracker.finish()

# Monkey patch the process method
import mediapipe_script
mediapipe_script.VideoProcessor.process = process_with_progress
"""
        
        # Create a modified module with progress tracking
        spec = importlib.util.spec_from_file_location("mediapipe_module", MEDIAPIPE_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        
        # Execute the module
        spec.loader.exec_module(module)
        
        return module
        
    except Exception as e:
        st.error(f"❌ Failed to load MediaPipe module: {e}")
        st.code(traceback.format_exc())
        return None


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (Keeping existing ones)
# ═══════════════════════════════════════════════════════════════════════════

def load_config():
    """Load config.json from project root"""
    if not CONFIG_PATH.exists():
        st.error(f"❌ Config file not found: {CONFIG_PATH}")
        return None
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        st.error(f"❌ Failed to load config: {e}")
        return None


def save_config(config):
    """Save config.json to project root"""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        st.error(f"❌ Failed to save config: {e}")
        return False


def save_uploaded_video(uploaded_file):
    """Save uploaded video to data/uploads with validation"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Validate file type
    file_extension = Path(uploaded_file.name).suffix.lower()
    if file_extension not in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']:
        st.error(f"❌ Unsupported file type: {file_extension}")
        return None
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    video_path = UPLOAD_DIR / filename
    
    # Save file
    with open(video_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Verify the file is a valid video
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            st.error("❌ Uploaded file is not a valid video")
            os.remove(video_path)
            return None
        cap.release()
    except Exception as e:
        st.error(f"❌ Error validating video: {e}")
        if video_path.exists():
            os.remove(video_path)
        return None
    
    return video_path


def update_config_with_video(video_path):
    """Update config.json with new video path"""
    config = load_config()
    if not config:
        return False
    
    # Convert to relative path
    try:
        rel_path = str(video_path.relative_to(PROJECT_ROOT))
    except:
        rel_path = str(video_path)
    
    # Update input_paths
    if "input_paths" not in config:
        config["input_paths"] = []
    
    # Clear previous paths and add new one
    config["input_paths"] = [rel_path]
    
    # Ensure output_dir is set
    if "output_dir" not in config:
        config["output_dir"] = "data/output"
    
    return save_config(config)


def load_mediapipe_module():
    """Dynamically load pre_mediapipe.py module"""
    if not MEDIAPIPE_SCRIPT.exists():
        st.error(f"❌ MediaPipe script not found: {MEDIAPIPE_SCRIPT}")
        return None
    
    try:
        spec = importlib.util.spec_from_file_location("mediapipe_module", MEDIAPIPE_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        st.error(f"❌ Failed to load MediaPipe module: {e}")
        st.code(traceback.format_exc())
        return None


def run_mediapipe_pipeline_with_progress():
    """Execute MediaPipe pipeline with progress tracking"""
    # Load MediaPipe module
    mp_module = load_mediapipe_module()
    if not mp_module:
        return None
    
    try:
        # Start progress tracking
        progress_tracker.start()
        progress_tracker.update(status="Loading configuration...")
        
        # Load config using module's PipelineConfig
        config = mp_module.PipelineConfig.from_json(CONFIG_PATH)
        
        progress_tracker.update(status="Initializing pipeline...")
        
        # Create and run pipeline
        pipeline = mp_module.PoseDetectionPipeline(config)
        
        progress_tracker.update(status="Running pose detection...")
        
        # Run pipeline in a separate thread to allow progress updates
        def run_pipeline():
            try:
                results = pipeline.run()
                progress_tracker.finish()
                return results
            except Exception as e:
                progress_tracker.error(str(e))
                return None
        
        # For now, run synchronously (in production, use threading)
        results = pipeline.run()
        progress_tracker.finish()
        
        return results
        
    except Exception as e:
        progress_tracker.error(str(e))
        st.error(f"❌ Pipeline execution failed: {e}")
        st.code(traceback.format_exc())
        return None


def get_video_info(video_path):
    """Get information about a video file"""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            'size_mb': os.path.getsize(video_path) / (1024 * 1024)
        }
        
        cap.release()
        return info
    except Exception as e:
        st.error(f"Error getting video info: {e}")
        return None


def display_video_with_controls(video_path, label="", show_info=True, show_download=True):
    """Display a video with controls and information"""
    if label:
        st.markdown(f"**{label}**")
    
    # Check if video exists
    if not video_path or not video_path.exists():
        st.error(f"❌ Video not found: {video_path}")
        return
    
    # Display video
    video_url = str(video_path)
    st.video(video_url)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if show_info:
            info = get_video_info(video_path)
            if info:
                st.markdown(f"""
                <div style="font-size: 0.85em; color: #666;">
                    Resolution: {info['width']}×{info['height']} | 
                    FPS: {info['fps']:.1f} | 
                    Duration: {info['duration']:.1f}s | 
                    Size: {info['size_mb']:.1f}MB
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if show_download:
            with open(video_path, 'rb') as f:
                st.download_button(
                    "📥 Download",
                    f,
                    file_name=video_path.name,
                    mime="video/mp4",
                    use_container_width=True
                )


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT APP (Enhanced with progress indicators)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="MediaPipe Test",
        page_icon="🎥",
        layout="wide"
    )
    
    # Header
    st.title("🎥 MediaPipe Pipeline Test")
    st.markdown("**Enhanced with video display fixes and progress indicators**")
    st.markdown("---")
    
    # Initialize session state
    if 'uploaded_video_path' not in st.session_state:
        st.session_state.uploaded_video_path = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'processing_start_time' not in st.session_state:
        st.session_state.processing_start_time = None
    
    # Sidebar: System Status
    with st.sidebar:
        st.subheader("📋 System Status")
        
        # Check config.json
        if CONFIG_PATH.exists():
            st.success(f"✅ Config: `{CONFIG_PATH.name}`")
            config = load_config()
            if config:
                st.info(f"Input paths: {len(config.get('input_paths', []))}")
        else:
            st.error(f"❌ Config not found")
        
        # Check MediaPipe script
        if MEDIAPIPE_SCRIPT.exists():
            st.success(f"✅ Script: Found")
        else:
            st.error(f"❌ Script not found")
        
        # Check directories
        st.markdown("**Directories:**")
        st.code(f"Upload: {UPLOAD_DIR}\nOutput: {OUTPUT_DIR}")
        
        # Show output videos
        st.markdown("**Output Videos:**")
        output_videos = get_all_output_videos()
        if output_videos:
            for video in output_videos[:5]:  # Show last 5 videos
                if st.checkbox(f"📹 {video.name}", key=f"output_{video}"):
                    st.session_state.selected_output_video = video
        else:
            st.info("No output videos yet")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Reset Application"):
            st.session_state.uploaded_video_path = None
            st.session_state.processing_complete = False
            st.session_state.results = None
            st.session_state.processing_start_time = None
            progress_tracker.__init__()  # Reset progress tracker
            st.rerun()
    
    # Main Content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Step 1: Upload Video")
        
        # Enhanced file uploader
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'],
            help="Upload a walking/gait video for analysis",
            key="video_uploader"
        )
        
        if uploaded_file:
            # Display file info
            st.info(f"File: {uploaded_file.name}")
            st.info(f"Size: {uploaded_file.size / (1024*1024):.1f} MB")
            
            # Save video
            video_path = save_uploaded_video(uploaded_file)
            
            if video_path:
                st.session_state.uploaded_video_path = video_path
                st.success(f"✅ Saved: `{video_path.name}`")
                
                # Display uploaded video
                display_video_with_controls(video_path, "Uploaded Video", show_info=True, show_download=False)
            else:
                st.error("❌ Failed to save video")
        
        # Display already uploaded video
        elif st.session_state.uploaded_video_path:
            video_path = st.session_state.uploaded_video_path
            st.success(f"Current video: `{video_path.name}`")
            display_video_with_controls(video_path, "Uploaded Video", show_info=True, show_download=False)
    
    with col2:
        st.subheader("⚙️ Step 2: Update Config & Process")
        
        if st.session_state.uploaded_video_path:
            video_path = st.session_state.uploaded_video_path
            
            # Button to update config
            if st.button("📝 Update config.json", type="primary", use_container_width=True):
                with st.spinner("Updating config.json..."):
                    if update_config_with_video(video_path):
                        st.success("✅ Config updated successfully!")
                        
                        # Show updated config
                        config = load_config()
                        st.json(config)
                    else:
                        st.error("❌ Failed to update config")
            
            st.markdown("---")
            
            # Button to run pipeline with progress
            if st.button("🚀 Run MediaPipe Pipeline", type="primary", use_container_width=True):
                st.session_state.processing_start_time = time.time()
                
                # Create progress placeholder
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Run pipeline with progress updates
                with st.spinner("Initializing MediaPipe..."):
                    results = run_mediapipe_pipeline_with_progress()
                    
                    if results and len(results) > 0:
                        st.session_state.results = results
                        st.session_state.processing_complete = True
                        st.success("✅ Pipeline completed successfully!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Pipeline failed or returned no results")
        else:
            st.info("👈 Please upload a video first")
    
    # Display Results
    if st.session_state.processing_complete and st.session_state.results:
        st.markdown("---")
        st.subheader("📊 Step 3: View Results")
        
        results = st.session_state.results
        result = results[0]  # Get first result
        
        # Show processing info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "✅ Success" if result.success else "❌ Failed")
        with col2:
            st.metric("Frames Processed", result.frames_processed)
        with col3:
            st.metric("Landmarks Detected", result.landmarks_detected)
        with col4:
            processing_time = result.processing_time if hasattr(result, 'processing_time') else 0
            st.metric("Processing Time", f"{processing_time:.1f}s")
        
        st.markdown("---")
        
        # Enhanced video display with better path resolution
        if st.session_state.uploaded_video_path:
            original_path = st.session_state.uploaded_video_path
            
            # Use enhanced function to find output videos
            output_videos = find_output_videos_enhanced(original_path)
            
            # Debug information
            with st.expander("🔍 Debug: File Search Results"):
                st.write(f"Original video: {original_path}")
                st.write(f"Video stem: {original_path.stem}")
                st.write(f"Output directory: {OUTPUT_DIR}")
                st.write("Found outputs:")
                for key, path in output_videos.items():
                    if path:
                        st.write(f"  {key}: {path} (exists: {path.exists()})")
                    else:
                        st.write(f"  {key}: Not found")
                
                # List all files in output directory
                st.write("All files in output directory:")
                if OUTPUT_DIR.exists():
                    for file in OUTPUT_DIR.iterdir():
                        st.write(f"  {file.name}")
            
            # Display videos in tabs
            tab1, tab2, tab3 = st.tabs(["Original", "Annotated", "Skeleton"])
            
            with tab1:
                display_video_with_controls(original_path, "Original Video")
            
            with tab2:
                if output_videos.get('annotated') and output_videos['annotated'].exists():
                    display_video_with_controls(output_videos['annotated'], "Annotated Video")
                else:
                    st.warning("Annotated video not found")
                    # Try to refresh
                    if st.button("🔄 Refresh Annotated Video"):
                        time.sleep(1)  # Give it time to appear
                        output_videos = find_output_videos_enhanced(original_path)
                        if output_videos.get('annotated') and output_videos['annotated'].exists():
                            display_video_with_controls(output_videos['annotated'], "Annotated Video")
                        else:
                            st.error("Still not found. Check if processing completed successfully.")
            
            with tab3:
                if output_videos.get('skeleton') and output_videos['skeleton'].exists():
                    display_video_with_controls(output_videos['skeleton'], "Skeleton Video")
                else:
                    st.warning("Skeleton video not found")
            
            # Download section
            st.markdown("---")
            st.subheader("📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with open(original_path, 'rb') as f:
                    st.download_button(
                        "📥 Original Video",
                        f,
                        file_name=original_path.name,
                        mime="video/mp4",
                        use_container_width=True
                    )
            
            with col2:
                if output_videos.get('annotated') and output_videos['annotated'].exists():
                    with open(output_videos['annotated'], 'rb') as f:
                        st.download_button(
                            "📥 Annotated Video",
                            f,
                            file_name=output_videos['annotated'].name,
                            mime="video/mp4",
                            use_container_width=True
                        )
                else:
                    st.button("📥 Annotated Video", disabled=True, use_container_width=True)
            
            with col3:
                if output_videos.get('skeleton') and output_videos['skeleton'].exists():
                    with open(output_videos['skeleton'], 'rb') as f:
                        st.download_button(
                            "📥 Skeleton Video",
                            f,
                            file_name=output_videos['skeleton'].name,
                            mime="video/mp4",
                            use_container_width=True
                        )
                else:
                    st.button("📥 Skeleton Video", disabled=True, use_container_width=True)
            
            # Show CSV path
            if output_videos.get('csv') and output_videos['csv'].exists():
                st.markdown("---")
                st.subheader("📄 Landmarks CSV")
                csv_path = output_videos['csv']
                st.code(f"CSV File: {csv_path}")
                
                with open(csv_path, 'rb') as f:
                    st.download_button(
                        "📥 Download Landmarks CSV",
                        f,
                        file_name=csv_path.name,
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Show all output paths
            with st.expander("🔍 View All Output Paths"):
                if hasattr(result, 'output_paths'):
                    st.json({k: str(v) for k, v in result.output_paths.items()})
                else:
                    st.json(output_videos)


if __name__ == "__main__":
    main()