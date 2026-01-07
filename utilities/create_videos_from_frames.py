"""
Multi-Modal Dataset Video Creator with Automatic Directory Discovery
Recursively scans dataset and creates videos with optional overlays
Production-grade implementation with comprehensive error handling
"""

import cv2
import numpy as np
import json
import csv
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import traceback
import sys


# ============================================================
# CONFIGURATION
# ============================================================

class VideoType(Enum):
    """Base video type"""
    FROM_IMAGES = "from_images"
    FROM_SKELETON = "from_skeleton"


@dataclass
class DatasetConfig:
    """Configuration for dataset video creation"""
    
    # Dataset root path - will be auto-detected if None
    dataset_root: Optional[Path] = None
    
    # Output root (mirrors input structure)
    output_root: Optional[Path] = None
    
    # Video creation options
    video_type: VideoType = VideoType.FROM_IMAGES
    overlay_skeleton: bool = True  # Add skeleton overlay to images
    
    # Export options
    export_csv: bool = True
    
    # Video properties
    fps: float = 30.0
    video_codec: str = "mp4v"
    
    # Visualization settings
    skeleton_color: Tuple[int, int, int] = (0, 255, 0)  # BGR
    skeleton_thickness: int = 3
    joint_radius: int = 5
    joint_color: Tuple[int, int, int] = (0, 0, 255)  # BGR
    
    # File extensions to look for
    image_extensions: Set[str] = field(default_factory=lambda: {".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG"})
    
    def __post_init__(self):
        """Auto-detect paths if not provided"""
        if self.dataset_root is None:
            self.dataset_root = self._find_dataset_root()
        else:
            self.dataset_root = Path(self.dataset_root).resolve()
        
        if self.output_root is None:
            self.output_root = Path.cwd() / "data" / "output" / "dataset_samples"
        else:
            self.output_root = Path(self.output_root).resolve()
    
    def _find_dataset_root(self) -> Path:
        """Auto-detect dataset_samples location"""
        cwd = Path.cwd()
        
        possible_paths = [
            cwd / "dataset_samples",
            cwd.parent / "dataset_samples",
            cwd / "data" / "dataset_samples",
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                return path.resolve()
        
        # Default to current directory / dataset_samples
        return cwd / "dataset_samples"
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'DatasetConfig':
        """Load configuration from JSON file"""
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert paths
        if 'dataset_root' in data:
            data['dataset_root'] = Path(data['dataset_root'])
        if 'output_root' in data:
            data['output_root'] = Path(data['output_root'])
        
        # Convert enums
        if 'video_type' in data:
            data['video_type'] = VideoType(data['video_type'])
        
        # Convert tuples
        if 'skeleton_color' in data:
            data['skeleton_color'] = tuple(data['skeleton_color'])
        if 'joint_color' in data:
            data['joint_color'] = tuple(data['joint_color'])
        
        # Convert sets
        if 'image_extensions' in data:
            data['image_extensions'] = set(data['image_extensions'])
        
        return cls(**data)


# ============================================================
# POSE SKELETON CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [
    # Face
    ("nose", "l_eye"),
    ("nose", "r_eye"),
    ("l_eye", "l_ear"),
    ("r_eye", "r_ear"),
    
    # Torso
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"),
    ("l_hip", "r_hip"),
    
    # Left arm
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    
    # Right arm
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    
    # Left leg
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    
    # Right leg
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
]


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration"""
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dataset_video_creator_{timestamp}.log"
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    
    return logger


# ============================================================
# DIRECTORY SCANNER
# ============================================================

@dataclass
class DataFolder:
    """Represents a discovered data folder"""
    path: Path
    relative_path: Path  # Relative to dataset_root
    folder_name: str  # Combined name from path hierarchy
    has_images: bool = False
    has_json: bool = False
    image_files: List[Path] = field(default_factory=list)
    json_file: Optional[Path] = None


class DirectoryScanner:
    """Recursively scans dataset directory structure"""
    
    def __init__(self, dataset_root: Path, image_extensions: Set[str], logger: logging.Logger):
        self.dataset_root = dataset_root.resolve()
        self.image_extensions = image_extensions
        self.logger = logger
    
    def scan(self) -> List[DataFolder]:
        """Scan dataset and find all processable folders"""
        self.logger.info(f"Scanning dataset: {self.dataset_root}")
        
        if not self.dataset_root.exists():
            self.logger.error(f"❌ Dataset root does not exist: {self.dataset_root}")
            self.logger.error(f"   Current working directory: {Path.cwd()}")
            return []
        
        if not self.dataset_root.is_dir():
            self.logger.error(f"❌ Dataset root is not a directory: {self.dataset_root}")
            return []
        
        discovered_folders = []
        scanned_dirs = 0
        
        try:
            # Walk through all directories
            for dirpath in self.dataset_root.rglob("*"):
                if not dirpath.is_dir():
                    continue
                
                scanned_dirs += 1
                
                # Check if this directory contains images or JSON
                image_files = self._find_images(dirpath)
                json_files = self._find_json_files(dirpath)
                
                if image_files or json_files:
                    # Calculate relative path and create folder name
                    relative_path = dirpath.relative_to(self.dataset_root)
                    folder_name = self._create_folder_name(relative_path)
                    
                    data_folder = DataFolder(
                        path=dirpath,
                        relative_path=relative_path,
                        folder_name=folder_name,
                        has_images=len(image_files) > 0,
                        has_json=len(json_files) > 0,
                        image_files=sorted(image_files),
                        json_file=json_files[0] if json_files else None
                    )
                    
                    discovered_folders.append(data_folder)
                    
                    self.logger.debug(
                        f"Found: {folder_name} "
                        f"(images: {len(image_files)}, json: {len(json_files)})"
                    )
        
        except Exception as e:
            self.logger.error(f"Error during scan: {e}")
            self.logger.debug(traceback.format_exc())
        
        self.logger.info(f"Scanned {scanned_dirs} directories")
        self.logger.info(f"Total folders discovered: {len(discovered_folders)}")
        
        if discovered_folders:
            self.logger.info(f"\n{'='*70}")
            self.logger.info("DISCOVERED FOLDERS:")
            self.logger.info(f"{'='*70}")
            for i, folder in enumerate(discovered_folders, 1):
                img_icon = "🖼️ " if folder.has_images else "   "
                json_icon = "📄" if folder.has_json else "  "
                self.logger.info(f"{i:3}. {img_icon}{json_icon} {folder.folder_name}")
                if folder.has_images:
                    self.logger.info(f"     └─ {len(folder.image_files)} images")
                if folder.has_json:
                    self.logger.info(f"     └─ JSON: {folder.json_file.name}")
            self.logger.info(f"{'='*70}\n")
        else:
            self.logger.warning(
                f"\n{'='*70}\n"
                "⚠️  NO PROCESSABLE FOLDERS FOUND!\n"
                f"{'='*70}\n"
                "Possible reasons:\n"
                f"  1. No image files in subdirectories (looking for: {', '.join(self.image_extensions)})\n"
                "  2. No JSON files in subdirectories\n"
                f"  3. Dataset path is incorrect: {self.dataset_root}\n"
                f"  4. Current working directory: {Path.cwd()}\n"
                "\n"
                "💡 Quick fixes:\n"
                "  - Run: python find_dataset.py\n"
                "  - Check if dataset_samples folder exists\n"
                "  - Verify files are in subdirectories (not root)\n"
                f"{'='*70}\n"
            )
        
        return discovered_folders
    
    def _find_images(self, directory: Path) -> List[Path]:
        """Find all image files in directory"""
        images = []
        for ext in self.image_extensions:
            images.extend(directory.glob(f"*{ext}"))
        
        # Sort by name (try to extract frame number)
        def sort_key(path):
            try:
                # Try to extract number from filename
                return int(''.join(filter(str.isdigit, path.stem)))
            except ValueError:
                return path.stem
        
        return sorted(images, key=sort_key)
    
    def _find_json_files(self, directory: Path) -> List[Path]:
        """Find JSON files in directory"""
        return list(directory.glob("*.json"))
    
    def _create_folder_name(self, relative_path: Path) -> str:
        """Create folder name from path hierarchy"""
        # Join all path parts with underscores
        parts = list(relative_path.parts)
        
        # Clean up parts (remove special characters, make filesystem-safe)
        cleaned_parts = []
        for part in parts:
            # Remove or replace problematic characters
            cleaned = part.replace(" ", "_").replace("/", "_").replace("\\", "_")
            cleaned_parts.append(cleaned)
        
        return "_".join(cleaned_parts)


# ============================================================
# DATA LOADERS
# ============================================================

class PoseDataLoader:
    """Loads pose data from JSON files"""
    
    @staticmethod
    def load_pose_json(json_path: Path) -> List[Dict]:
        """Load pose JSON file"""
        if not json_path.exists():
            raise FileNotFoundError(f"Pose JSON not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    @staticmethod
    def get_frame_pose(pose_data: List[Dict], frame_idx: int) -> Optional[Dict]:
        """Get pose for specific frame"""
        for frame_data in pose_data:
            if frame_data.get('frame') == frame_idx:
                return frame_data.get('joints')
        return None


# ============================================================
# CSV EXPORTER
# ============================================================

class PoseLandmarkExporter:
    """Exports pose landmarks to CSV"""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.writer = None
    
    def __enter__(self):
        self.file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "frame",
            "timestamp_ms",
            "joint_name",
            "x",
            "y",
            "has_detection"
        ])
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
    
    def write_frame_joints(
        self,
        frame_idx: int,
        timestamp_ms: int,
        joints: Optional[Dict]
    ) -> int:
        """Write joints for a frame, returns count"""
        if not joints:
            self.writer.writerow([
                frame_idx,
                timestamp_ms,
                "none",
                0,
                0,
                False
            ])
            return 0
        
        count = 0
        for joint_name, coords in joints.items():
            if coords:
                self.writer.writerow([
                    frame_idx,
                    timestamp_ms,
                    joint_name,
                    coords.get('x', 0),
                    coords.get('y', 0),
                    True
                ])
                count += 1
        
        return count


# ============================================================
# VISUALIZER
# ============================================================

class PoseVisualizer:
    """Handles pose visualization"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def draw_skeleton(
        self,
        image: np.ndarray,
        joints: Optional[Dict]
    ) -> np.ndarray:
        """Draw pose skeleton on image"""
        if not joints:
            return image
        
        result = image.copy()
        
        # Draw connections
        for joint1, joint2 in POSE_CONNECTIONS:
            if joint1 in joints and joint2 in joints:
                coord1 = joints[joint1]
                coord2 = joints[joint2]
                
                if coord1 and coord2:
                    try:
                        pt1 = (int(coord1['x']), int(coord1['y']))
                        pt2 = (int(coord2['x']), int(coord2['y']))
                        
                        cv2.line(
                            result,
                            pt1,
                            pt2,
                            self.config.skeleton_color,
                            self.config.skeleton_thickness
                        )
                    except (KeyError, ValueError, TypeError):
                        continue
        
        # Draw joints
        for joint_name, coords in joints.items():
            if coords:
                try:
                    pt = (int(coords['x']), int(coords['y']))
                    cv2.circle(
                        result,
                        pt,
                        self.config.joint_radius,
                        self.config.joint_color,
                        -1
                    )
                except (KeyError, ValueError, TypeError):
                    continue
        
        return result
    
    def create_skeleton_only_frame(
        self,
        joints: Optional[Dict],
        width: int,
        height: int
    ) -> np.ndarray:
        """Create frame with skeleton only (black background)"""
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if joints:
            frame = self.draw_skeleton(frame, joints)
        
        return frame


# ============================================================
# VIDEO CREATOR
# ============================================================

@dataclass
class VideoCreationResult:
    """Result from video creation"""
    folder_name: str
    success: bool
    output_video: Optional[Path] = None
    output_csv: Optional[Path] = None
    frames_processed: int = 0
    error: Optional[str] = None
    processing_time: float = 0.0


# class VideoCreator:
#     """Creates videos from data folders"""
    
#     def __init__(
#         self,
#         config: DatasetConfig,
#         logger: logging.Logger
#     ):
#         self.config = config
#         self.logger = logger
#         self.visualizer = PoseVisualizer(config)
    
#     def create_video(self, data_folder: DataFolder) -> VideoCreationResult:
#         """Create video from a data folder"""
#         start_time = datetime.now()
#         result = VideoCreationResult(
#             folder_name=data_folder.folder_name,
#             success=False
#         )
        
#         writer = None
#         csv_exporter = None
        
#         try:
#             self.logger.info(f"\n{'='*70}")
#             self.logger.info(f"Processing: {data_folder.folder_name}")
#             self.logger.info(f"{'='*70}")
            
#             # Load pose data if available
#             pose_data = None
#             if data_folder.has_json and data_folder.json_file:
#                 try:
#                     pose_data = PoseDataLoader.load_pose_json(data_folder.json_file)
#                     self.logger.info(f"✅ Loaded pose data: {len(pose_data)} frames")
#                 except Exception as e:
#                     self.logger.warning(f"⚠️  Could not load pose JSON: {e}")
            
#             # Determine video dimensions and frame count
#             if self.config.video_type == VideoType.FROM_SKELETON:
#                 # Skeleton-only video
#                 if not pose_data:
#                     raise ValueError("No pose data available for skeleton-only video")
                
#                 # Default dimensions or get from first valid pose
#                 width, height = 1920, 1080
#                 for frame_data in pose_data:
#                     joints = frame_data.get('joints')
#                     if joints:
#                         # Try to estimate dimensions from joint coordinates
#                         max_x = max(j.get('x', 0) for j in joints.values() if j)
#                         max_y = max(j.get('y', 0) for j in joints.values() if j)
#                         if max_x > 0 and max_y > 0:
#                             width = int(max_x * 1.2)
#                             height = int(max_y * 1.2)
#                             break
                
#                 total_frames = len(pose_data)
#                 self.logger.info(f"📐 Video dimensions: {width}x{height} (skeleton-only)")
                
#             else:
#                 # From images
#                 if not data_folder.has_images or not data_folder.image_files:
#                     raise ValueError("No images found in folder")
                
#                 # Get dimensions from first image
#                 first_image = cv2.imread(str(data_folder.image_files[0]))
#                 if first_image is None:
#                     raise ValueError(f"Could not read first image: {data_folder.image_files[0]}")
                
#                 height, width = first_image.shape[:2]
#                 total_frames = max(
#                     len(data_folder.image_files),
#                     len(pose_data) if pose_data else 0
#                 )
#                 self.logger.info(f"📐 Video dimensions: {width}x{height}")
#                 self.logger.info(f"🎞️  Total frames: {total_frames}")
            
#             # Create output directory (mirrors input structure)
#             output_dir = self.config.output_root / data_folder.relative_path.parent / data_folder.folder_name
#             output_dir.mkdir(parents=True, exist_ok=True)
            
#             # Setup output paths
#             video_path = output_dir / f"{data_folder.folder_name}.mp4"
#             csv_path = output_dir / f"{data_folder.folder_name}_landmarks.csv"
            
#             self.logger.info(f"📹 Output video: {video_path.relative_to(Path.cwd())}")
#             if self.config.export_csv:
#                 self.logger.info(f"📊 Output CSV: {csv_path.relative_to(Path.cwd())}")
            
#             # Setup video writer
#             fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
#             writer = cv2.VideoWriter(
#                 str(video_path),
#                 fourcc,
#                 self.config.fps,
#                 (width, height)
#             )
            
#             if not writer.isOpened():
#                 raise ValueError(f"Could not open video writer for {video_path}")
            
#             # Setup CSV exporter
#             if self.config.export_csv:
#                 csv_exporter = PoseLandmarkExporter(csv_path)
#                 csv_exporter.__enter__()
            
#             # Process frames
#             frames_written = 0
#             frames_with_pose = 0
            
#             self.logger.info(f"\n⏳ Processing frames...")
            
#             for frame_idx in range(total_frames):
#                 timestamp_ms = int((frame_idx / self.config.fps) * 1000)
                
#                 # Get pose for this frame
#                 joints = None
#                 if pose_data:
#                     joints = PoseDataLoader.get_frame_pose(pose_data, frame_idx)
#                     if joints:
#                         frames_with_pose += 1
                
#                 # Create frame
#                 if self.config.video_type == VideoType.FROM_SKELETON:
#                     frame = self.visualizer.create_skeleton_only_frame(joints, width, height)
#                 else:
#                     # Load image frame
#                     if frame_idx < len(data_folder.image_files):
#                         frame = cv2.imread(str(data_folder.image_files[frame_idx]))
#                         if frame is None:
#                             self.logger.warning(f"⚠️  Could not read frame {frame_idx}, using black frame")
#                             frame = np.zeros((height, width, 3), dtype=np.uint8)
#                     else:
#                         # Use last frame or black frame
#                         if data_folder.image_files:
#                             frame = cv2.imread(str(data_folder.image_files[-1]))
#                             if frame is None:
#                                 frame = np.zeros((height, width, 3), dtype=np.uint8)
#                         else:
#                             frame = np.zeros((height, width, 3), dtype=np.uint8)
                    
#                     # Ensure correct dimensions
#                     if frame.shape[:2] != (height, width):
#                         frame = cv2.resize(frame, (width, height))
                    
#                     # Add skeleton overlay if requested
#                     if self.config.overlay_skeleton and joints:
#                         frame = self.visualizer.draw_skeleton(frame, joints)
                
#                 # Write frame
#                 writer.write(frame)
#                 frames_written += 1
                
#                 # Export to CSV
#                 if csv_exporter:
#                     csv_exporter.write_frame_joints(frame_idx, timestamp_ms, joints)
                
#                 # Progress logging
#                 if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
#                     progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
#                     self.logger.info(f"   Frame {frame_idx+1}/{total_frames} ({progress:.1f}%)")
            
#             result.success = True
#             result.output_video = video_path
#             result.output_csv = csv_path if self.config.export_csv else None
#             result.frames_processed = frames_written
            
#             self.logger.info(f"\n{'='*70}")
#             self.logger.info(f"✅ SUCCESS: {data_folder.folder_name}")
#             self.logger.info(f"{'='*70}")
#             self.logger.info(f"📹 Video: {frames_written} frames written")
#             if pose_data:
#                 self.logger.info(f"🦴 Pose: {frames_with_pose} frames with skeleton")
#             self.logger.info(f"⏱️  Time: {(datetime.now() - start_time).total_seconds():.2f}s")
#             self.logger.info(f"{'='*70}\n")
            
#         except Exception as e:
#             result.error = str(e)
#             self.logger.error(f"\n{'='*70}")
#             self.logger.error(f"❌ ERROR: {data_folder.folder_name}")
#             self.logger.error(f"{'='*70}")
#             self.logger.error(f"Error: {e}")
#             self.logger.debug(traceback.format_exc())
#             self.logger.error(f"{'='*70}\n")
        
#         finally:
#             # Cleanup
#             if writer:
#                 writer.release()
#             if csv_exporter:
#                 csv_exporter.__exit__(None, None, None)
            
#             result.processing_time = (datetime.now() - start_time).total_seconds()
        
#         return result

class VideoCreator:
    """Creates videos from data folders"""
    
    def __init__(
        self,
        config: DatasetConfig,
        logger: logging.Logger
    ):
        self.config = config
        self.logger = logger
        self.visualizer = PoseVisualizer(config)
    
    def create_video(self, data_folder: DataFolder) -> VideoCreationResult:
        """Create video from a data folder"""
        start_time = datetime.now()
        result = VideoCreationResult(
            folder_name=data_folder.folder_name,
            success=False
        )
        
        writer = None
        csv_exporter = None
        
        try:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Processing: {data_folder.folder_name}")
            self.logger.info(f"{'='*70}")
            
            # Load pose data if available
            pose_data = None
            if data_folder.has_json and data_folder.json_file:
                try:
                    pose_data = PoseDataLoader.load_pose_json(data_folder.json_file)
                    self.logger.info(f"✅ Loaded pose data: {len(pose_data)} frames")
                except Exception as e:
                    self.logger.warning(f"⚠️  Could not load pose JSON: {e}")
            
            # Determine video dimensions and frame count
            if self.config.video_type == VideoType.FROM_SKELETON:
                # Skeleton-only video
                if not pose_data:
                    raise ValueError("No pose data available for skeleton-only video")
                
                # --- FIX IS HERE ---
                # Default dimensions
                width, height = 1920, 1080
                valid_joints_found = False
                
                # Iterate through pose_data to find the first frame with valid joints
                for frame_data in pose_data:
                    joints = frame_data.get('joints')
                    if joints and isinstance(joints, dict): # Check if joints is a valid dictionary
                        # Collect all valid (non-None) coordinates
                        valid_coords = [
                            (j.get('x', 0), j.get('y', 0)) 
                            for j in joints.values() 
                            if j and 'x' in j and 'y' in j
                        ]
                        
                        if valid_coords:
                            valid_joints_found = True
                            # Unzip the list of tuples into two separate lists
                            x_coords, y_coords = zip(*valid_coords)
                            
                            # Find the maximum x and y to determine the canvas size
                            max_x = max(x_coords)
                            max_y = max(y_coords)
                            
                            # Add a padding (e.g., 20%) to ensure joints aren't on the edge
                            padding = 1.2
                            width = int(max_x * padding)
                            height = int(max_y * padding)
                            break # Found dimensions, no need to check further
                
                if not valid_joints_found:
                    self.logger.warning("⚠️  Could not find any frames with valid joint coordinates. Using default dimensions.")
                
                total_frames = len(pose_data)
                self.logger.info(f"📐 Video dimensions: {width}x{height} (skeleton-only)")
                
            else:
                # From images (this part was already working)
                if not data_folder.has_images or not data_folder.image_files:
                    raise ValueError("No images found in folder")
                
                # Get dimensions from first image
                first_image = cv2.imread(str(data_folder.image_files[0]))
                if first_image is None:
                    raise ValueError(f"Could not read first image: {data_folder.image_files[0]}")
                
                height, width = first_image.shape[:2]
                total_frames = max(
                    len(data_folder.image_files),
                    len(pose_data) if pose_data else 0
                )
                self.logger.info(f"📐 Video dimensions: {width}x{height}")
                self.logger.info(f"🎞️  Total frames: {total_frames}")
            
            # The rest of the method remains the same...
            # Create output directory (mirrors input structure)
            output_dir = self.config.output_root / data_folder.relative_path.parent / data_folder.folder_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup output paths
            video_path = output_dir / f"{data_folder.folder_name}.mp4"
            csv_path = output_dir / f"{data_folder.folder_name}_landmarks.csv"
            
            self.logger.info(f"📹 Output video: {video_path.relative_to(Path.cwd())}")
            if self.config.export_csv:
                self.logger.info(f"📊 Output CSV: {csv_path.relative_to(Path.cwd())}")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.config.fps,
                (width, height)
            )
            
            if not writer.isOpened():
                raise ValueError(f"Could not open video writer for {video_path}")
            
            # Setup CSV exporter
            if self.config.export_csv:
                csv_exporter = PoseLandmarkExporter(csv_path)
                csv_exporter.__enter__()
            
            # Process frames
            frames_written = 0
            frames_with_pose = 0
            
            self.logger.info(f"\n⏳ Processing frames...")
            
            for frame_idx in range(total_frames):
                timestamp_ms = int((frame_idx / self.config.fps) * 1000)
                
                # Get pose for this frame
                joints = None
                if pose_data:
                    joints = PoseDataLoader.get_frame_pose(pose_data, frame_idx)
                    if joints:
                        frames_with_pose += 1
                
                # Create frame
                if self.config.video_type == VideoType.FROM_SKELETON:
                    frame = self.visualizer.create_skeleton_only_frame(joints, width, height)
                else:
                    # Load image frame
                    if frame_idx < len(data_folder.image_files):
                        frame = cv2.imread(str(data_folder.image_files[frame_idx]))
                        if frame is None:
                            self.logger.warning(f"⚠️  Could not read frame {frame_idx}, using black frame")
                            frame = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        # Use last frame or black frame
                        if data_folder.image_files:
                            frame = cv2.imread(str(data_folder.image_files[-1]))
                            if frame is None:
                                frame = np.zeros((height, width, 3), dtype=np.uint8)
                        else:
                            frame = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Ensure correct dimensions
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    
                    # Add skeleton overlay if requested
                    if self.config.overlay_skeleton and joints:
                        frame = self.visualizer.draw_skeleton(frame, joints)
                
                # Write frame
                writer.write(frame)
                frames_written += 1
                
                # Export to CSV
                if csv_exporter:
                    csv_exporter.write_frame_joints(frame_idx, timestamp_ms, joints)
                
                # Progress logging
                if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
                    progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    self.logger.info(f"   Frame {frame_idx+1}/{total_frames} ({progress:.1f}%)")
            
            result.success = True
            result.output_video = video_path
            result.output_csv = csv_path if self.config.export_csv else None
            result.frames_processed = frames_written
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✅ SUCCESS: {data_folder.folder_name}")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"📹 Video: {frames_written} frames written")
            if pose_data:
                self.logger.info(f"🦴 Pose: {frames_with_pose} frames with skeleton")
            self.logger.info(f"⏱️  Time: {(datetime.now() - start_time).total_seconds():.2f}s")
            self.logger.info(f"{'='*70}\n")
            
        except Exception as e:
            result.error = str(e)
            self.logger.error(f"\n{'='*70}")
            self.logger.error(f"❌ ERROR: {data_folder.folder_name}")
            self.logger.error(f"{'='*70}")
            self.logger.error(f"Error: {e}")
            self.logger.debug(traceback.format_exc())
            self.logger.error(f"{'='*70}\n")
        
        finally:
            # Cleanup
            if writer:
                writer.release()
            if csv_exporter:
                csv_exporter.__exit__(None, None, None)
            
            result.processing_time = (datetime.now() - start_time).total_seconds()
        
        return result
    

# ============================================================
# MAIN PIPELINE
# ============================================================

class DatasetVideoPipeline:
    """Main pipeline for dataset video creation"""
    
    def __init__(
        self,
        config: DatasetConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.scanner = DirectoryScanner(
            config.dataset_root,
            config.image_extensions,
            self.logger
        )
        self.creator = VideoCreator(config, self.logger)
        self.results: List[VideoCreationResult] = []
    
    def run(self) -> List[VideoCreationResult]:
        """Run the pipeline"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("DATASET VIDEO CREATION PIPELINE - RECURSIVE MODE")
        self.logger.info("=" * 70)
        self.logger.info(f"📂 Dataset: {self.config.dataset_root}")
        self.logger.info(f"📁 Output: {self.config.output_root}")
        self.logger.info(f"🎬 Mode: {self.config.video_type.value}")
        self.logger.info(f"🦴 Skeleton overlay: {self.config.overlay_skeleton}")
        self.logger.info(f"📊 Export CSV: {self.config.export_csv}")
        self.logger.info("=" * 70 + "\n")
        
        # Scan dataset
        data_folders = self.scanner.scan()
        
        if not data_folders:
            self.logger.warning("No processable folders found! Exiting.")
            return []
        
        self.logger.info(f"\n🚀 Starting processing of {len(data_folders)} folders...\n")
        
        # Process each folder
        results = []
        for i, data_folder in enumerate(data_folders, 1):
            self.logger.info(f"\n{'#'*70}")
            self.logger.info(f"# FOLDER {i}/{len(data_folders)}")
            self.logger.info(f"{'#'*70}")
            
            result = self.creator.create_video(data_folder)
            results.append(result)
        
        self._print_summary(results)
        return results
    
    def _print_summary(self, results: List[VideoCreationResult]):
        """Print processing summary"""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        total_frames = sum(r.frames_processed for r in results)
        total_time = sum(r.processing_time for r in results)
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FINAL SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Total folders processed: {len(results)}")
        self.logger.info(f"✅ Successful: {successful}")
        self.logger.info(f"❌ Failed: {failed}")
        self.logger.info(f"🎞️  Total frames: {total_frames:,}")
        self.logger.info(f"⏱️  Total time: {total_time:.2f}s")
        
        if successful > 0:
            avg_time = total_time / successful
            self.logger.info(f"📊 Average time per video: {avg_time:.2f}s")
            
            self.logger.info(f"\n✅ Successfully created {successful} videos:")
            self.logger.info(f"   Output directory: {self.config.output_root}")
            
            for r in results:
                if r.success and r.output_video:
                    rel_path = r.output_video.relative_to(self.config.output_root)
                    self.logger.info(f"   ✓ {rel_path}")
        
        if failed > 0:
            self.logger.info(f"\n❌ Failed folders ({failed}):")
            for r in results:
                if not r.success:
                    self.logger.error(f"   ✗ {r.folder_name}")
                    self.logger.error(f"     Error: {r.error}")
        
        self.logger.info("=" * 70)


# ============================================================
# EXAMPLE USAGE
# ============================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create videos from multi-modal dataset with automatic directory discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect dataset and create videos with skeleton overlay
  python dataset_video_creator.py
  
  # Specify custom dataset path
  python dataset_video_creator.py --dataset /path/to/dataset
  
  # Create skeleton-only videos
  python dataset_video_creator.py --video-type from_skeleton
  
  # Load configuration from JSON
  python dataset_video_creator.py --config config.json
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        help="Path to dataset root directory (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for videos (mirrors input structure if not specified)"
    )
    
    parser.add_argument(
        "--video-type", "-t",
        choices=["from_images", "from_skeleton"],
        default="from_images",
        help="Type of video to create (default: from_images)"
    )
    
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Disable skeleton overlay on image videos"
    )
    
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV export"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS (default: 30.0)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()


def main():
    """Main function with command-line support"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(level=log_level)
    
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-MODAL DATASET VIDEO CREATOR")
    logger.info("=" * 70)
    logger.info(f"Current directory: {Path.cwd()}")
    logger.info("=" * 70 + "\n")
    
    try:
        # Create configuration
        if args.config:
            # Load from JSON
            logger.info(f"Loading configuration from: {args.config}")
            config = DatasetConfig.from_json(args.config)
            
            # Override with command line arguments
            if args.dataset:
                config.dataset_root = args.dataset
            if args.output:
                config.output_root = args.output
            if args.video_type:
                config.video_type = VideoType(args.video_type)
            if args.no_skeleton:
                config.overlay_skeleton = False
            if args.no_csv:
                config.export_csv = False
            if args.fps:
                config.fps = args.fps
        else:
            # Create from command line arguments or defaults
            config = DatasetConfig(
                dataset_root=args.dataset,
                output_root=args.output,
                video_type=VideoType(args.video_type),
                overlay_skeleton=not args.no_skeleton,
                export_csv=not args.no_csv,
                fps=args.fps
            )
        
        # Save configuration for reference
        config_path = Path.cwd() / "last_dataset_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'dataset_root': str(config.dataset_root),
                'output_root': str(config.output_root),
                'video_type': config.video_type.value,
                'overlay_skeleton': config.overlay_skeleton,
                'export_csv': config.export_csv,
                'fps': config.fps
            }, f, indent=2)
        logger.info(f"Saved configuration to: {config_path}")
        
        # Create and run pipeline
        pipeline = DatasetVideoPipeline(config, logger)
        results = pipeline.run()
        
        # Return appropriate exit code
        failed_count = sum(1 for r in results if not r.success)
        if failed_count > 0:
            logger.warning(f"{failed_count} folders failed to process")
            return 1
        
        logger.info("All folders processed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())


# Currently works with all the images file folder 
# Does not work for json yolo pose files need to check the work around for them.
