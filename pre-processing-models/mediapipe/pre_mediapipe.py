"""
MediaPipe Pose Detection Pipeline
Production-grade, OOP implementation with batch processing support
FIXED: Unicode encoding errors and path validation
"""

import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import traceback


# ============================================================
# PATH RESOLUTION
# ============================================================

def get_project_root() -> Path:
    """Get the project root directory (where the script is located)"""
    if '__file__' in globals():
        script_dir = Path(__file__).resolve().parent
    else:
        script_dir = Path.cwd()
    
    current = script_dir
    
    while current != current.parent:
        if (current / 'models').exists() or (current / 'config.json').exists():
            return current
        current = current.parent
    
    return script_dir


def resolve_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to base_dir or project root
    Handles both absolute and relative paths
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    if base_dir is None:
        base_dir = get_project_root()
    
    return (base_dir / path).resolve()


# ============================================================
# CONFIGURATION
# ============================================================

class ProcessMode(Enum):
    """Processing mode enumeration"""
    IMAGE = "image"
    VIDEO = "video"


@dataclass
class PipelineConfig:
    """Configuration for pose detection pipeline"""
    
    # Model settings
    model_path: Path = Path("models/pose_landmarker_heavy.task")
    
    # Detection parameters for accuracy
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_poses: int = 1
    
    # Input/Output
    input_paths: List[Path] = field(default_factory=list)
    output_dir: Path = Path("data/output")
    
    # Processing options
    batch_mode: bool = False
    save_annotated: bool = True
    save_csv: bool = True
    auto_open: bool = False
    
    # Visualization
    landmark_color: Tuple[int, int, int] = (0, 255, 0)
    connection_color: Tuple[int, int, int] = (255, 0, 0)
    landmark_thickness: int = 2
    connection_thickness: int = 2
    landmark_radius: int = 2
    
    # Video codec
    video_codec: str = "mp4v"
    
    # Supported extensions
    image_extensions: set = field(default_factory=lambda: {".jpg", ".jpeg", ".png", ".bmp"})
    video_extensions: set = field(default_factory=lambda: {".mp4", ".mov", ".avi", ".mkv"})
    
    def __post_init__(self):
        """Resolve all paths relative to project root"""
        project_root = get_project_root()
        
        if not self.model_path.is_absolute():
            candidate = (project_root / self.model_path).resolve()
            if not candidate.exists():
                parts = self.model_path.parts
                if parts and parts[0] in ('..', '.'):
                    clean_path = Path(*[p for p in parts if p not in ('..', '.')])
                    candidate = (project_root / clean_path).resolve()
            self.model_path = candidate
        
        self.output_dir = resolve_path(self.output_dir, project_root)
        
        resolved_inputs = []
        for path in self.input_paths:
            resolved_inputs.append(resolve_path(path, project_root))
        self.input_paths = resolved_inputs
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'PipelineConfig':
        """Load configuration from JSON file"""
        json_path = resolve_path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'model_path' in data:
            data['model_path'] = Path(data['model_path'])
        if 'output_dir' in data:
            data['output_dir'] = Path(data['output_dir'])
        if 'input_paths' in data:
            data['input_paths'] = [Path(p) for p in data['input_paths']]
        
        if 'landmark_color' in data:
            data['landmark_color'] = tuple(data['landmark_color'])
        if 'connection_color' in data:
            data['connection_color'] = tuple(data['connection_color'])
        
        if 'image_extensions' in data:
            data['image_extensions'] = set(data['image_extensions'])
        if 'video_extensions' in data:
            data['video_extensions'] = set(data['video_extensions'])
        
        # Set defaults for detection parameters if not present
        defaults = {
            'min_pose_detection_confidence': 0.5,
            'min_pose_presence_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'num_poses': 1
        }
        for key, value in defaults.items():
            if key not in data:
                data[key] = value
        
        return cls(**data)
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        json_path = resolve_path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        project_root = get_project_root()
        
        try:
            model_rel = self.model_path.relative_to(project_root)
        except ValueError:
            model_rel = self.model_path
        
        try:
            output_rel = self.output_dir.relative_to(project_root)
        except ValueError:
            output_rel = self.output_dir
        
        input_rels = []
        for p in self.input_paths:
            try:
                input_rels.append(str(p.relative_to(project_root)))
            except ValueError:
                input_rels.append(str(p))
        
        data = {
            'model_path': str(model_rel),
            'output_dir': str(output_rel),
            'input_paths': input_rels,
            'batch_mode': self.batch_mode,
            'save_annotated': self.save_annotated,
            'save_csv': self.save_csv,
            'auto_open': self.auto_open,
            'min_pose_detection_confidence': self.min_pose_detection_confidence,
            'min_pose_presence_confidence': self.min_pose_presence_confidence,
            'min_tracking_confidence': self.min_tracking_confidence,
            'num_poses': self.num_poses,
            'landmark_color': list(self.landmark_color),
            'connection_color': list(self.connection_color),
            'landmark_thickness': self.landmark_thickness,
            'connection_thickness': self.connection_thickness,
            'landmark_radius': self.landmark_radius,
            'video_codec': self.video_codec,
            'image_extensions': list(self.image_extensions),
            'video_extensions': list(self.video_extensions)
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


# ============================================================
# DEPENDENCY CHECKER
# ============================================================

def check_dependencies() -> List[str]:
    """Check if all required dependencies are available"""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import mediapipe
    except ImportError:
        missing.append("mediapipe")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    return missing


# ============================================================
# LOGGING SETUP (FIXED FOR UNICODE)
# ============================================================

def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration with UTF-8 encoding"""
    if log_dir is None:
        log_dir = get_project_root() / "logs"
    else:
        log_dir = resolve_path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pose_pipeline_{timestamp}.log"
    
    # Create handlers with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Try to set UTF-8 encoding for console on Windows
    try:
        if sys.platform == 'win32':
            import io
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True
            )
    except Exception:
        pass  # Fallback to ASCII-safe logging
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Project root: {get_project_root()}")
    logger.info(f"Log file: {log_file}")
    
    return logger


# ============================================================
# RESULT CLASSES
# ============================================================

@dataclass
class ProcessingResult:
    """Result from processing a single file"""
    input_path: Path
    success: bool
    mode: ProcessMode
    output_paths: Dict[str, Path] = field(default_factory=dict)
    error: Optional[str] = None
    frames_processed: int = 0
    landmarks_detected: int = 0
    processing_time: float = 0.0


# ============================================================
# CSV WRITER
# ============================================================

class LandmarkCSVWriter:
    """Handles CSV writing for pose landmarks"""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.file = None
        self.writer = None
        
    def __enter__(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "frame",
            "timestamp_ms",
            "landmark_id",
            "x_norm",
            "y_norm",
            "z_norm",
            "visibility",
            "x_px",
            "y_px",
        ])
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
    
    def write_landmarks(
        self,
        landmarks: List,
        frame_idx: int,
        timestamp_ms: int,
        width: int,
        height: int
    ) -> int:
        """Write landmarks for a frame, returns count"""
        count = 0
        for lm_id, lm in enumerate(landmarks):
            self.writer.writerow([
                frame_idx,
                timestamp_ms,
                lm_id,
                lm.x,
                lm.y,
                lm.z,
                lm.visibility,
                int(lm.x * width),
                int(lm.y * height)
            ])
            count += 1
        return count


# ============================================================
# POSE DETECTOR
# ============================================================

class PoseDetector:
    """Handles MediaPipe pose detection"""
    
    def __init__(self, model_path: Path, mode: ProcessMode, config: PipelineConfig):
        self.model_path = model_path
        self.mode = mode
        self.config = config
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize MediaPipe detector with accuracy settings"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Please download the model and place it at this location.\n"
                f"Expected location: {self.model_path}"
            )
        
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        
        running_mode = (
            vision.RunningMode.IMAGE if self.mode == ProcessMode.IMAGE
            else vision.RunningMode.VIDEO
        )
        
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=self.config.num_poses,
            min_pose_detection_confidence=self.config.min_pose_detection_confidence,
            min_pose_presence_confidence=self.config.min_pose_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        
        self.detector = vision.PoseLandmarker.create_from_options(options)
    
    def detect_image(self, image_rgb: np.ndarray):
        """Detect pose in image"""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )
        return self.detector.detect(mp_image)
    
    def detect_video_frame(self, image_rgb: np.ndarray, timestamp_ms: int):
        """Detect pose in video frame"""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )
        return self.detector.detect_for_video(mp_image, timestamp_ms)
    
    def close(self):
        """Cleanup detector"""
        if self.detector:
            self.detector.close()


# ============================================================
# VISUALIZER
# ============================================================

class PoseVisualizer:
    """Handles pose visualization"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def draw_landmarks(
        self,
        image_rgb: np.ndarray,
        landmarks: List
    ) -> np.ndarray:
        """Draw landmarks on image"""
        annotated = image_rgb.copy()
        
        pose_proto = landmark_pb2.NormalizedLandmarkList()
        pose_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in landmarks
        ])
        
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            pose_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=self.config.landmark_color,
                thickness=self.config.landmark_thickness,
                circle_radius=self.config.landmark_radius
            ),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=self.config.connection_color,
                thickness=self.config.connection_thickness
            ),
        )
        
        return annotated


# ============================================================
# PROCESSORS
# ============================================================

class ImageProcessor:
    """Processes single images"""
    
    def __init__(
        self,
        config: PipelineConfig,
        detector: PoseDetector,
        visualizer: PoseVisualizer,
        logger: logging.Logger
    ):
        self.config = config
        self.detector = detector
        self.visualizer = visualizer
        self.logger = logger
    
    def process(self, input_path: Path) -> ProcessingResult:
        """Process single image"""
        start_time = datetime.now()
        result = ProcessingResult(
            input_path=input_path,
            success=False,
            mode=ProcessMode.IMAGE
        )
        
        try:
            frame = cv2.imread(str(input_path))
            if frame is None:
                raise ValueError(f"Could not read image: {input_path}")
            
            height, width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            detection_result = self.detector.detect_image(frame_rgb)
            
            stem = input_path.stem
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.config.output_dir / f"{stem}_landmarks.csv"
            img_path = self.config.output_dir / f"{stem}_annotated.png"
            
            landmarks_count = 0
            
            if self.config.save_csv:
                with LandmarkCSVWriter(csv_path) as csv_writer:
                    if detection_result.pose_landmarks:
                        landmarks_count = csv_writer.write_landmarks(
                            detection_result.pose_landmarks[0],
                            frame_idx=0,
                            timestamp_ms=0,
                            width=width,
                            height=height
                        )
                result.output_paths['csv'] = csv_path
            
            if self.config.save_annotated:
                annotated_rgb = frame_rgb.copy()
                
                if detection_result.pose_landmarks:
                    annotated_rgb = self.visualizer.draw_landmarks(
                        annotated_rgb,
                        detection_result.pose_landmarks[0]
                    )
                
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_path), annotated_bgr)
                result.output_paths['annotated'] = img_path
            
            result.success = True
            result.frames_processed = 1
            result.landmarks_detected = landmarks_count
            
        except Exception as e:
            result.error = str(e)
            self.logger.error(f"Error processing image {input_path}: {e}")
            self.logger.debug(traceback.format_exc())
        
        finally:
            result.processing_time = (datetime.now() - start_time).total_seconds()
        
        return result


class VideoProcessor:
    """Processes videos"""
    
    def __init__(
        self,
        config: PipelineConfig,
        detector: PoseDetector,
        visualizer: PoseVisualizer,
        logger: logging.Logger
    ):
        self.config = config
        self.detector = detector
        self.visualizer = visualizer
        self.logger = logger
    
    def process(self, input_path: Path) -> ProcessingResult:
        """Process single video"""
        start_time = datetime.now()
        result = ProcessingResult(
            input_path=input_path,
            success=False,
            mode=ProcessMode.VIDEO
        )
        
        cap = None
        writer = None
        csv_writer_ctx = None
        
        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            stem = input_path.stem
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.config.output_dir / f"{stem}_landmarks.csv"
            video_path = self.config.output_dir / f"{stem}_annotated.mp4"
            
            if self.config.save_csv:
                csv_writer_ctx = LandmarkCSVWriter(csv_path)
                csv_writer = csv_writer_ctx.__enter__()
                result.output_paths['csv'] = csv_path
            
            if self.config.save_annotated:
                fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
                writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    fps,
                    (width, height)
                )
                result.output_paths['annotated'] = video_path
            
            frame_idx = 0
            landmarks_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((frame_idx / fps) * 1000)
                
                detection_result = self.detector.detect_video_frame(
                    frame_rgb,
                    timestamp_ms
                )
                
                if self.config.save_csv and detection_result.pose_landmarks:
                    landmarks_count += csv_writer.write_landmarks(
                        detection_result.pose_landmarks[0],
                        frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms,
                        width=width,
                        height=height
                    )
                
                if self.config.save_annotated:
                    annotated_rgb = frame_rgb.copy()
                    
                    if detection_result.pose_landmarks:
                        annotated_rgb = self.visualizer.draw_landmarks(
                            annotated_rgb,
                            detection_result.pose_landmarks[0]
                        )
                    
                    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                    writer.write(annotated_bgr)
                
                frame_idx += 1
                
                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    self.logger.info(
                        f"Processing {input_path.name}: "
                        f"Frame {frame_idx}/{total_frames} ({progress:.1f}%)"
                    )
            
            result.success = True
            result.frames_processed = frame_idx
            result.landmarks_detected = landmarks_count
            
        except Exception as e:
            result.error = str(e)
            self.logger.error(f"Error processing video {input_path}: {e}")
            self.logger.debug(traceback.format_exc())
        
        finally:
            if cap:
                cap.release()
            if writer:
                writer.release()
            if csv_writer_ctx:
                csv_writer_ctx.__exit__(None, None, None)
            
            result.processing_time = (datetime.now() - start_time).total_seconds()
        
        return result


# ============================================================
# MAIN PIPELINE
# ============================================================

class PoseDetectionPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizer = PoseVisualizer(config)
        self.results: List[ProcessingResult] = []
    
    def _determine_mode(self, path: Path) -> ProcessMode:
        """Determine processing mode from file extension"""
        ext = path.suffix.lower()
        
        if ext in self.config.image_extensions:
            return ProcessMode.IMAGE
        elif ext in self.config.video_extensions:
            return ProcessMode.VIDEO
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def _validate_inputs(self, paths: List[Path]) -> List[Path]:
        """Validate and filter input paths"""
        valid_paths = []
        
        for path in paths:
            if not path.exists():
                self.logger.warning(f"File not found: {path}")
                continue
            
            if not path.is_file():
                self.logger.warning(f"Not a file: {path}")
                continue
            
            try:
                self._determine_mode(path)
                valid_paths.append(path)
            except ValueError as e:
                self.logger.warning(f"Skipping {path}: {e}")
        
        return valid_paths
    
    def process_single(self, input_path: Path) -> ProcessingResult:
        """Process a single file"""
        self.logger.info(f"Processing: {input_path}")
        
        mode = self._determine_mode(input_path)
        detector = PoseDetector(self.config.model_path, mode, self.config)
        
        try:
            if mode == ProcessMode.IMAGE:
                processor = ImageProcessor(
                    self.config,
                    detector,
                    self.visualizer,
                    self.logger
                )
            else:
                processor = VideoProcessor(
                    self.config,
                    detector,
                    self.visualizer,
                    self.logger
                )
            
            result = processor.process(input_path)
            self.results.append(result)
            
            if result.success:
                self.logger.info(
                    f"[SUCCESS] {input_path.name} - "
                    f"{result.frames_processed} frames, "
                    f"{result.landmarks_detected} landmarks, "
                    f"{result.processing_time:.2f}s"
                )
            else:
                self.logger.error(f"[FAILED] {input_path.name} - {result.error}")
            
            return result
            
        finally:
            detector.close()
    
    def process_batch(self, input_paths: List[Path]) -> List[ProcessingResult]:
        """Process multiple files"""
        valid_paths = self._validate_inputs(input_paths)
        
        if not valid_paths:
            self.logger.error("No valid input files found")
            return []
        
        self.logger.info(f"Processing {len(valid_paths)} files...")
        
        results = []
        for i, path in enumerate(valid_paths, 1):
            self.logger.info(f"\n[{i}/{len(valid_paths)}] Processing: {path.name}")
            result = self.process_single(path)
            results.append(result)
        
        return results
    
    def run(self) -> List[ProcessingResult]:
        """Run the pipeline"""
        if not self.config.input_paths:
            raise ValueError("No input paths specified")
        
        if self.config.batch_mode or len(self.config.input_paths) > 1:
            results = self.process_batch(self.config.input_paths)
        else:
            results = [self.process_single(self.config.input_paths[0])]
        
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: List[ProcessingResult]):
        """Print processing summary"""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        total_frames = sum(r.frames_processed for r in results)
        total_landmarks = sum(r.landmarks_detected for r in results)
        total_time = sum(r.processing_time for r in results)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total files: {len(results)}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Total frames: {total_frames}")
        self.logger.info(f"Total landmarks: {total_landmarks}")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info("=" * 60)


# ============================================================
# EXAMPLE USAGE
# ============================================================

def main():
    """Example usage of the pipeline"""
    
    missing_deps = check_dependencies()
    if missing_deps:
        print("ERROR: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing_deps))
        sys.exit(1)
    
    logger = setup_logging()
    
    try:
        config_path = resolve_path("config.json")
        
        if config_path.exists():
            logger.info(f"Loading config from: {config_path}")
            config = PipelineConfig.from_json(config_path)
        else:
            logger.info("Creating default configuration")
            config = PipelineConfig(
                model_path=Path("models/pose_landmarker_heavy.task"),
                input_paths=[
                    Path("data/videos/video.mp4"),
                    Path("data/videos/3044091-sd_640_360_24fps.mp4"),
                ],
                output_dir=Path("data/output"),
                batch_mode=True,
                save_annotated=True,
                save_csv=True,
                auto_open=False,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                num_poses=1
            )
            
            logger.info(f"Saving config to: {config_path}")
            config.to_json(config_path)
        
        if not config.model_path.exists():
            logger.error(f"Model file not found: {config.model_path}")
            logger.error("Please download the pose_landmarker_heavy.task model")
            logger.error("and place it in the models/ directory")
            sys.exit(1)
        
        missing_inputs = [p for p in config.input_paths if not p.exists()]
        if missing_inputs:
            logger.warning("Some input files not found:")
            for p in missing_inputs:
                logger.warning(f"  - {p}")
        
        valid_inputs = [p for p in config.input_paths if p.exists()]
        if not valid_inputs:
            logger.error("No valid input files found")
            logger.error("Please update config.json with valid input paths")
            sys.exit(1)
        
        config.input_paths = valid_inputs
        
        pipeline = PoseDetectionPipeline(config, logger)
        results = pipeline.run()
        
        logger.info("\nDETAILED RESULTS:")
        for result in results:
            if result.success:
                logger.info(f"\n[SUCCESS] {result.input_path.name}")
                for output_type, output_path in result.output_paths.items():
                    logger.info(f"   {output_type}: {output_path}")
            else:
                logger.error(f"\n[FAILED] {result.input_path.name}: {result.error}")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()