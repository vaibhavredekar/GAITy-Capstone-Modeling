#!/usr/bin/env python3
"""
Video File Finder - Recursively finds all video files in a directory and exports their paths to a config file.

This script searches through a specified parent directory and all its subdirectories to find
video files, then exports their complete paths to a configuration file that can be used by
other scripts for further processing.
"""

import os
import argparse
import json
import yaml
import logging
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional, Union
from datetime import datetime
import mimetypes

class VideoFileFinder:
    """
    A production-grade tool to find video files recursively and export their paths to a config file.
    """
    
    # Default video file extensions
    DEFAULT_VIDEO_EXTENSIONS = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', 
        '.m4v', '.3gp', '.ogv', '.ts', '.mts', '.m2ts', '.vob'
    }
    
    def __init__(self, root_dir: str, output_file: str, output_format: str = 'json',
                 video_extensions: Optional[Set[str]] = None, 
                 use_mimetypes: bool = False, log_file: str = None, 
                 verbose: bool = False):
        """
        Initialize the VideoFileFinder.
        
        Args:
            root_dir: Root directory to search for video files
            output_file: Path to the output config file
            output_format: Format of the output file ('json', 'yaml', or 'text')
            video_extensions: Set of video file extensions to search for
            use_mimetypes: If True, also use MIME types to identify video files
            log_file: Path to log file (optional)
            verbose: Enable verbose logging
        """
        self.root_dir = Path(root_dir).resolve()
        self.output_file = Path(output_file)
        self.output_format = output_format.lower()
        self.use_mimetypes = use_mimetypes
        self.verbose = verbose
        
        # Set video extensions
        self.video_extensions = video_extensions or self.DEFAULT_VIDEO_EXTENSIONS
        
        # Initialize mimetypes
        if use_mimetypes:
            mimetypes.init()
        
        # Set up logging
        self.setup_logging(log_file)
        
        # Statistics
        self.stats = {
            'directories_scanned': 0,
            'files_scanned': 0,
            'video_files_found': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Results storage
        self.video_files = []
        
    def setup_logging(self, log_file: str = None):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up root logger
        self.logger = logging.getLogger('VideoFileFinder')
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def is_video_file(self, file_path: Path) -> bool:
        """
        Check if a file is a video file based on its extension or MIME type.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file is a video file, False otherwise
        """
        # Check by extension
        if file_path.suffix.lower() in self.video_extensions:
            return True
            
        # Check by MIME type if enabled
        if self.use_mimetypes:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type and mime_type.startswith('video/'):
                return True
                
        return False
    
    def find_video_files(self) -> None:
        """Recursively find all video files in the directory tree."""
        self.logger.info(f"Starting video file search in: {self.root_dir}")
        self.stats['start_time'] = datetime.now()
        
        if not self.root_dir.exists():
            self.logger.error(f"Root directory does not exist: {self.root_dir}")
            sys.exit(1)
            
        if not self.root_dir.is_dir():
            self.logger.error(f"Root path is not a directory: {self.root_dir}")
            sys.exit(1)
        
        # Start the recursive search
        self._search_directory(self.root_dir)
        
        self.stats['end_time'] = datetime.now()
        self._report_results()
    
    def _search_directory(self, directory: Path) -> None:
        """
        Recursively search a directory for video files.
        
        Args:
            directory: Directory to search
        """
        try:
            self.stats['directories_scanned'] += 1
            
            for item in directory.iterdir():
                if item.is_dir():
                    # Recursively search subdirectories
                    self._search_directory(item)
                elif item.is_file():
                    self.stats['files_scanned'] += 1
                    
                    if self.is_video_file(item):
                        self.video_files.append(str(item))
                        self.stats['video_files_found'] += 1
                        self.logger.debug(f"Found video file: {item}")
                        
                        # Report progress every 100 files
                        if self.stats['video_files_found'] % 100 == 0:
                            self.logger.info(f"Found {self.stats['video_files_found']} video files so far...")
                            
        except PermissionError:
            self.logger.warning(f"Permission denied accessing: {directory}")
            self.stats['errors'] += 1
        except Exception as e:
            self.logger.error(f"Error scanning {directory}: {e}")
            self.stats['errors'] += 1
    
    def _report_results(self) -> None:
        """Report the search results."""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        self.logger.info("Search completed. Summary:")
        self.logger.info(f"  Directories scanned: {self.stats['directories_scanned']}")
        self.logger.info(f"  Files scanned: {self.stats['files_scanned']}")
        self.logger.info(f"  Video files found: {self.stats['video_files_found']}")
        self.logger.info(f"  Errors encountered: {self.stats['errors']}")
        self.logger.info(f"  Search duration: {duration:.2f} seconds")
        
        if self.verbose:
            self.logger.debug("Video files found:")
            for video_file in sorted(self.video_files):
                self.logger.debug(f"  {video_file}")
    
    def export_to_config(self) -> None:
        """Export the video file paths to a configuration file."""
        self.logger.info(f"Exporting {len(self.video_files)} video file paths to: {self.output_file}")
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.output_format == 'json':
                self._export_to_json()
            elif self.output_format == 'yaml':
                self._export_to_yaml()
            elif self.output_format == 'text':
                self._export_to_text()
            else:
                self.logger.error(f"Unsupported output format: {self.output_format}")
                sys.exit(1)
                
            self.logger.info(f"Successfully exported video file paths to {self.output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to config file: {e}")
            sys.exit(1)
    
    def _export_to_json(self) -> None:
        """Export video file paths to a JSON config file."""
        config_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_videos': len(self.video_files),
                'source_directory': str(self.root_dir)
            },
            'video_files': self.video_files
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def _export_to_yaml(self) -> None:
        """Export video file paths to a YAML config file."""
        config_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_videos': len(self.video_files),
                'source_directory': str(self.root_dir)
            },
            'video_files': self.video_files
        }
        
        with open(self.output_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def _export_to_text(self) -> None:
        """Export video file paths to a text config file."""
        with open(self.output_file, 'w') as f:
            f.write(f"# Video File List\n")
            f.write(f"# Generated at: {datetime.now().isoformat()}\n")
            f.write(f"# Source directory: {self.root_dir}\n")
            f.write(f"# Total videos: {len(self.video_files)}\n\n")
            
            for video_file in self.video_files:
                f.write(f"{video_file}\n")
    
    def run(self) -> None:
        """Run the complete video file finding and exporting process."""
        self.find_video_files()
        self.export_to_config()


def main():
    """Main function to parse arguments and run the VideoFileFinder."""
    parser = argparse.ArgumentParser(
        description="Find all video files in a directory and export their paths to a config file."
    )
    parser.add_argument(
        "directory",
        help="Root directory to search for video files"
    )
    parser.add_argument(
        "output",
        help="Path to the output config file"
    )
    parser.add_argument(
        "--format",
        choices=['json', 'yaml', 'text'],
        default='json',
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--extensions",
        nargs='+',
        help=f"Video file extensions to search for (default: {', '.join(sorted(VideoFileFinder.DEFAULT_VIDEO_EXTENSIONS))})"
    )
    parser.add_argument(
        "--use-mimetypes",
        action="store_true",
        help="Also use MIME types to identify video files"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Process extensions
    video_extensions = None
    if args.extensions:
        video_extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                           for ext in args.extensions}
    
    finder = VideoFileFinder(
        root_dir=args.directory,
        output_file=args.output,
        output_format=args.format,
        video_extensions=video_extensions,
        use_mimetypes=args.use_mimetypes,
        log_file=args.log_file,
        verbose=args.verbose
    )
    
    try:
        finder.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()