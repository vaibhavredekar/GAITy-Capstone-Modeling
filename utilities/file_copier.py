#!/usr/bin/env python3
"""
File Copier - Recursively copies files with specific extensions from source to destination.

This script searches through a specified source directory and all its subdirectories to find
files with specific extensions, then copies them to a destination directory.

Usage: python file_copier.py --extension .csv path/to/pickup_folder path/desired_folder
"""

import os
import argparse
import logging
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime

class FileCopier:
    """
    A production-grade tool to copy files with specific extensions from source to destination.
    """
    
    def __init__(self, source_dir: str, dest_dir: str, extensions: List[str], 
                 dry_run: bool = False, log_file: str = None, verbose: bool = False,
                 preserve_structure: bool = False, overwrite: bool = False):
        """
        Initialize the FileCopier.
        
        Args:
            source_dir: Source directory to search for files
            dest_dir: Destination directory to copy files to
            extensions: List of file extensions to copy
            dry_run: If True, only show what would be copied without actually copying
            log_file: Path to log file (optional)
            verbose: Enable verbose logging
            preserve_structure: If True, preserve directory structure in destination
            overwrite: If True, overwrite existing files in destination
        """
        self.source_dir = Path(source_dir).resolve()
        self.dest_dir = Path(dest_dir).resolve()
        self.extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                          for ext in extensions]
        self.dry_run = dry_run
        self.verbose = verbose
        self.preserve_structure = preserve_structure
        self.overwrite = overwrite
        
        # Set up logging
        self.setup_logging(log_file)
        
        # Statistics
        self.stats = {
            'directories_scanned': 0,
            'files_scanned': 0,
            'files_found': 0,
            'files_to_copy': 0,
            'files_copied': 0,
            'files_skipped': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Results storage
        self.copy_operations = []
        self.skipped_files = []
        
    def setup_logging(self, log_file: str = None):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up root logger
        self.logger = logging.getLogger('FileCopier')
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
    
    def should_copy_file(self, file_path: Path) -> bool:
        """
        Check if a file should be copied based on its extension.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file should be copied, False otherwise
        """
        return file_path.suffix.lower() in self.extensions
    
    def get_destination_path(self, source_path: Path) -> Path:
        """
        Get the destination path for a source file.
        
        Args:
            source_path: Path to the source file
            
        Returns:
            Path to the destination file
        """
        if self.preserve_structure:
            # Preserve the directory structure relative to source
            relative_path = source_path.relative_to(self.source_dir)
            return self.dest_dir / relative_path
        else:
            # Flatten structure - all files go directly to destination
            return self.dest_dir / source_path.name
    
    def find_files_to_copy(self) -> None:
        """Find all files that need to be copied."""
        self.logger.info(f"Starting file search in: {self.source_dir}")
        self.logger.info(f"Looking for files with extensions: {', '.join(self.extensions)}")
        self.logger.info(f"Destination: {self.dest_dir}")
        self.stats['start_time'] = datetime.now()
        
        if not self.source_dir.exists():
            self.logger.error(f"Source directory does not exist: {self.source_dir}")
            sys.exit(1)
            
        if not self.source_dir.is_dir():
            self.logger.error(f"Source path is not a directory: {self.source_dir}")
            sys.exit(1)
        
        # Create destination directory if it doesn't exist
        if not self.dry_run:
            self.dest_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created destination directory: {self.dest_dir}")
        
        # Start the recursive search
        self._search_directory(self.source_dir)
        
        self.stats['end_time'] = datetime.now()
        self._report_findings()
    
    def _search_directory(self, directory: Path) -> None:
        """
        Recursively search a directory for files to copy.
        
        Args:
            directory: Directory to search
        """
        try:
            self.stats['directories_scanned'] += 1
            
            for item in directory.iterdir():
                if item.is_dir():
                    # Continue recursion for subdirectories
                    self._search_directory(item)
                elif item.is_file():
                    self.stats['files_scanned'] += 1
                    
                    if self.should_copy_file(item):
                        self.stats['files_found'] += 1
                        dest_path = self.get_destination_path(item)
                        
                        # Check if destination file already exists
                        if dest_path.exists():
                            if not self.overwrite:
                                self.skipped_files.append((str(item), str(dest_path), "File already exists"))
                                self.stats['files_skipped'] += 1
                                self.logger.debug(f"Skipping (exists): {item.name} -> {dest_path}")
                                continue
                            else:
                                self.logger.debug(f"Will overwrite: {item.name} -> {dest_path}")
                        
                        self.copy_operations.append((item, dest_path))
                        self.stats['files_to_copy'] += 1
                        self.logger.debug(f"Will copy: {item.name} -> {dest_path}")
                        
                        # Report progress every 100 files
                        if self.stats['files_to_copy'] % 100 == 0:
                            self.logger.info(f"Found {self.stats['files_to_copy']} files to copy so far...")
                            
        except PermissionError:
            self.logger.warning(f"Permission denied accessing: {directory}")
            self.stats['errors'] += 1
        except Exception as e:
            self.logger.error(f"Error scanning {directory}: {e}")
            self.stats['errors'] += 1
    
    def _report_findings(self) -> None:
        """Report the findings."""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        self.logger.info("Search completed. Summary:")
        self.logger.info(f"  Directories scanned: {self.stats['directories_scanned']}")
        self.logger.info(f"  Files scanned: {self.stats['files_scanned']}")
        self.logger.info(f"  Files found with matching extensions: {self.stats['files_found']}")
        self.logger.info(f"  Files to copy: {self.stats['files_to_copy']}")
        self.logger.info(f"  Files skipped (already exist): {self.stats['files_skipped']}")
        self.logger.info(f"  Errors encountered: {self.stats['errors']}")
        self.logger.info(f"  Search duration: {duration:.2f} seconds")
        
        if self.skipped_files and self.verbose:
            self.logger.debug("Files skipped (already exist):")
            for source, dest, reason in self.skipped_files:
                self.logger.debug(f"  {source} -> {dest} ({reason})")
        
        if self.verbose:
            self.logger.debug("Files to copy:")
            for source_path, dest_path in self.copy_operations:
                self.logger.debug(f"  {source_path} -> {dest_path}")
    
    def execute_copies(self) -> None:
        """Execute the copy operations."""
        if not self.copy_operations:
            self.logger.info("No files to copy.")
            return
            
        self.logger.info(f"Starting to copy {len(self.copy_operations)} files...")
        
        for source_path, dest_path in self.copy_operations:
            try:
                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Would copy: {source_path.name} -> {dest_path}")
                else:
                    # Create destination directory if it doesn't exist (for preserve_structure mode)
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy the file
                    shutil.copy2(source_path, dest_path)
                    self.stats['files_copied'] += 1
                    self.logger.debug(f"Copied: {source_path.name} -> {dest_path}")
                    
                    # Report progress every 50 files
                    if self.stats['files_copied'] % 50 == 0:
                        self.logger.info(f"Copied {self.stats['files_copied']} files so far...")
                    
            except Exception as e:
                self.logger.error(f"Error copying {source_path}: {e}")
                self.stats['errors'] += 1
        
        self._report_results()
    
    def _report_results(self) -> None:
        """Report the results of the copy operation."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would have copied {len(self.copy_operations)} files")
        else:
            self.logger.info(f"Successfully copied {self.stats['files_copied']} files")
            
        if self.stats['errors'] > 0:
            self.logger.warning(f"Encountered {self.stats['errors']} errors during the process")
        
        if self.stats['files_skipped'] > 0:
            self.logger.info(f"Skipped {self.stats['files_skipped']} files (already exist)")
    
    def run(self) -> None:
        """Run the complete file copying process."""
        self.find_files_to_copy()
        self.execute_copies()


def main():
    """Main function to parse arguments and run the FileCopier."""
    parser = argparse.ArgumentParser(
        description="Copy files with specific extensions from source to destination."
    )
    parser.add_argument(
        "--extension",
        action="append",
        required=True,
        help="File extension to copy (can be used multiple times for multiple extensions)"
    )
    parser.add_argument(
        "source_dir",
        help="Source directory to search for files"
    )
    parser.add_argument(
        "dest_dir",
        help="Destination directory to copy files to"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying"
    )
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve directory structure in destination"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in destination"
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
    
    copier = FileCopier(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        extensions=args.extension,
        dry_run=args.dry_run,
        log_file=args.log_file,
        verbose=args.verbose,
        preserve_structure=args.preserve_structure,
        overwrite=args.overwrite
    )
    
    try:
        copier.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()