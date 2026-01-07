#!/usr/bin/env python3
"""
Visualize pose skeleton from a landmark CSV file.

This script is self-contained. Edit the settings in the main() function
to specify the input CSV, output file, and dimensions.
"""

import csv
import sys
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class VisualizationConfig:
    """Configuration for skeleton visualization from CSV"""
    
    # Skeleton appearance
    background_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    landmark_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    connection_color: Tuple[int, int, int] = (255, 0, 0)  # Red
    landmark_thickness: int = 3
    connection_thickness: int = 2
    landmark_radius: int = 4

# ============================================================
# CSV VISUALIZER
# ============================================================

class CSVVisualizer:
    """Visualizes pose landmarks from a CSV file."""

    def __init__(self, config: VisualizationConfig, width: int, height: int):
        self.config = config
        self.width = width
        self.height = height
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def _load_landmarks(self, csv_path: Path) -> Dict[int, List[landmark_pb2.NormalizedLandmark]]:
        """
        Load landmarks from a CSV file and group them by frame.
        
        Returns:
            A dictionary where keys are frame indices and values are lists of
            NormalizedLandmark objects.
        """
        landmarks_by_frame: Dict[int, List[landmark_pb2.NormalizedLandmark]] = {}
        
        print(f"Loading landmarks from {csv_path}...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row['frame'])
                landmark = landmark_pb2.NormalizedLandmark(
                    x=float(row['x_norm']),
                    y=float(row['y_norm']),
                    z=float(row['z_norm']),
                    visibility=float(row['visibility'])
                )
                
                if frame_idx not in landmarks_by_frame:
                    landmarks_by_frame[frame_idx] = []
                landmarks_by_frame[frame_idx].append(landmark)
        
        if not landmarks_by_frame:
            raise ValueError("No landmark data found in the CSV file.")
            
        print(f"Found data for {len(landmarks_by_frame)} frames.")
        return landmarks_by_frame

    def _create_skeleton_frame(self, landmarks: List[landmark_pb2.NormalizedLandmark]) -> np.ndarray:
        """Draws a single skeleton frame on a blank canvas."""
        # Create a blank canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.config.background_color
        
        # Create the landmark list protobuf
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.extend(landmarks)
        
        # Draw the skeleton
        self.mp_drawing.draw_landmarks(
            canvas,
            landmark_list,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=self.config.landmark_color,
                thickness=self.config.landmark_thickness,
                circle_radius=self.config.landmark_radius
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=self.config.connection_color,
                thickness=self.config.connection_thickness
            )
        )
        return canvas

    def create_image(self, csv_path: Path, output_path: Path, landmarks_by_frame: Dict):
        """Generates a single skeleton image from the first frame of the CSV."""
        
        if len(landmarks_by_frame) > 1:
            print(f"Warning: CSV contains {len(landmarks_by_frame)} frames. "
                  f"Generating image from the first frame (frame {min(landmarks_by_frame)}).")
        
        first_frame_idx = min(landmarks_by_frame)
        landmarks = landmarks_by_frame[first_frame_idx]
        
        print(f"Drawing skeleton for frame {first_frame_idx}...")
        skeleton_image = self._create_skeleton_frame(landmarks)
        
        print(f"Saving image to {output_path}...")
        cv2.imwrite(str(output_path), skeleton_image)
        print("Image saved successfully.")

    def create_video(self, csv_path: Path, output_path: Path, fps: float, landmarks_by_frame: Dict):
        """Generates a skeleton video from all frames in the CSV."""
        
        total_frames = max(landmarks_by_frame) + 1 # Assuming frames start at 0
        print(f"Creating video with {len(landmarks_by_frame)} frames...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (self.width, self.height))
        
        if not writer.isOpened():
            raise IOError(f"Could not open video writer for {output_path}")

        try:
            for frame_idx in sorted(landmarks_by_frame.keys()):
                landmarks = landmarks_by_frame[frame_idx]
                skeleton_frame = self._create_skeleton_frame(landmarks)
                writer.write(skeleton_frame)
                
                # Progress update
                if (frame_idx + 1) % 30 == 0 or (frame_idx + 1) == total_frames:
                    progress = ((frame_idx + 1) / total_frames) * 100
                    print(f"Processing frame {frame_idx + 1}/{total_frames} ({progress:.1f}%)")
        finally:
            writer.release()
        
        print(f"Video saved successfully to {output_path}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    --- EDIT THESE SETTINGS ---
    This is the only section you need to change.
    """
    
    # 1. INPUT: Path to your landmark CSV file.
    #    Use a relative path (like below) if the CSV is in the same folder.
    #    Use an absolute path (like 'C:/Users/YourName/Desktop/video_landmarks.csv') if it's elsewhere.
    INPUT_CSV_PATH = "data/output/video_landmarks.csv"

    # 2. OUTPUT: Path for your new image or video file.
    #    The extension (.png or .mp4) determines the output type.
    OUTPUT_PATH = "data/output/skeleton_reconstruction.mp4"

    # 3. DIMENSIONS: Width and height of the ORIGINAL video/image in pixels.
    #    THIS IS REQUIRED. If the dimensions are wrong, the skeleton will be distorted.
    WIDTH = 1280
    HEIGHT = 720

    # 4. VIDEO SETTINGS (only used if OUTPUT_PATH is a video)
    FPS = 1.0 #30.0  # Frames per second for the output video

    # 5. VISUALIZATION SETTINGS (optional)
    #    You can change colors and thickness here.
    #    Colors are in BGR format (Blue, Green, Red).
    config = VisualizationConfig(
        background_color=(0, 0, 0),      # Black
        landmark_color=(0, 255, 0),      # Green
        connection_color=(0, 0, 255),    # Red
        landmark_thickness=3,
        connection_thickness=2,
        landmark_radius=4
    )

    # --- DO NOT EDIT BELOW THIS LINE ---
    
    # --- Validation ---
    input_path = Path(INPUT_CSV_PATH)
    output_path = Path(OUTPUT_PATH)

    if not input_path.exists():
        print(f"Error: Input CSV file not found at '{input_path.resolve()}'")
        print("Please check the INPUT_CSV_PATH setting in the script.")
        sys.exit(1)
        
    if WIDTH <= 0 or HEIGHT <= 0:
        print("Error: Width and height must be positive integers.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Determine Output Type ---
    output_ext = output_path.suffix.lower()
    is_image = output_ext in {'.png', '.jpg', '.jpeg', '.bmp'}
    is_video = output_ext in {'.mp4', '.avi', '.mov', '.mkv'}

    if not is_image and not is_video:
        print(f"Error: Unsupported output file extension '{output_ext}'. "
              "Please use .png, .jpg for images or .mp4, .avi for videos.")
        sys.exit(1)

    # --- Run Visualization ---
    try:
        visualizer = CSVVisualizer(config, WIDTH, HEIGHT)
        landmarks_by_frame = visualizer._load_landmarks(input_path)
        
        if is_image:
            visualizer.create_image(input_path, output_path, landmarks_by_frame)
        else: # is_video
            visualizer.create_video(input_path, output_path, FPS, landmarks_by_frame)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()