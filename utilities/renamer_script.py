#!/usr/bin/env python3
"""
File Renamer - Robust script to rename files based on folder name with duplicate handling.

This script searches through a specified directory and all its subdirectories to find
folders with numeric names, then renames files within those folders by adding a "PA"
prefix to the folder number while preserving the rest of the filename structure.

Features:
- Skips files that already have the correct PA prefix
- Fixes duplicates like PA114_PA114 by removing the duplicate
- Optional semantic_segmentation prefix
- Simple pattern: semantic_segmentation_PA{folder_name}_{old_filename} (when enabled)
- Robust error handling and logging
"""

import os
import argparse
import logging
import sys
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime

class FileRenamer:
    """
    A production-grade tool to rename files based on folder name with duplicate handling.
    """
    
    def __init__(self, root_dir: str, dry_run: bool = False, log_file: str = None, 
                 verbose: bool = False, file_extensions: Optional[List[str]] = None,
                 add_semantic_prefix: bool = False):
        """
        Initialize the FileRenamer.
        
        Args:
            root_dir: Root directory to process
            dry_run: If True, only show what would be renamed without actually renaming
            log_file: Path to log file (optional)
            verbose: Enable verbose logging
            file_extensions: List of file extensions to process (if None, process all)
            add_semantic_prefix: If True, add "semantic_segmentation_" prefix to filenames
        """
        self.root_dir = Path(root_dir).resolve()
        self.dry_run = dry_run
        self.verbose = verbose
        self.file_extensions = file_extensions
        self.add_semantic_prefix = add_semantic_prefix
        
        # Set up logging
        self.setup_logging(log_file)
        
        # Statistics
        self.stats = {
            'directories_scanned': 0,
            'numeric_folders_found': 0,
            'files_scanned': 0,
            'files_to_rename': 0,
            'files_renamed': 0,
            'files_skipped': 0,
            'duplicates_fixed': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Results storage
        self.rename_operations = []
        self.skipped_files = []
        
    def setup_logging(self, log_file: str = None):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up root logger
        self.logger = logging.getLogger('FileRenamer')
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
    
    def is_numeric_folder(self, folder_path: Path) -> bool:
        """
        Check if a folder name is numeric.
        
        Args:
            folder_path: Path to the folder to check
            
        Returns:
            True if the folder name is numeric, False otherwise
        """
        return folder_path.name.isdigit()
    
    def should_process_file(self, file_path: Path) -> bool:
        """
        Check if a file should be processed based on its extension.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file should be processed, False otherwise
        """
        if self.file_extensions is None:
            return True
            
        return file_path.suffix.lower() in [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                                           for ext in self.file_extensions]
    
    def has_correct_prefix(self, folder_name: str, filename: str) -> bool:
        """
        Check if the filename already has the correct PA prefix.
        
        Args:
            folder_name: Name of the folder (numeric)
            filename: Name of the file to check
            
        Returns:
            True if the file already has the correct PA prefix, False otherwise
        """
        if self.add_semantic_prefix:
            expected_prefix = f"semantic_segmentation_PA{folder_name}_"
        else:
            expected_prefix = f"PA{folder_name}_"
        return filename.startswith(expected_prefix)
    
    def has_duplicate_prefix(self, folder_name: str, filename: str) -> bool:
        """
        Check if the filename has a duplicate PA prefix like PA114_PA114.
        
        Args:
            folder_name: Name of the folder (numeric)
            filename: Name of the file to check
            
        Returns:
            True if the file has a duplicate PA prefix, False otherwise
        """
        duplicate_pattern = f"PA{folder_name}_PA{folder_name}_"
        return filename.startswith(duplicate_pattern)
    
    def fix_duplicate_prefix(self, folder_name: str, filename: str) -> str:
        """
        Fix a filename with duplicate PA prefix.
        
        Args:
            folder_name: Name of the folder (numeric)
            filename: Name of the file with duplicate prefix
            
        Returns:
            Fixed filename with only one PA prefix
        """
        duplicate_pattern = f"PA{folder_name}_PA{folder_name}_"
        if filename.startswith(duplicate_pattern):
            # Remove the duplicate prefix
            return filename[len(duplicate_pattern)-len(f"PA{folder_name}_"):]
        return filename
    
    def generate_new_filename(self, folder_name: str, original_filename: str) -> Tuple[str, str]:
        """
        Generate a new filename based on the folder name and original filename.
        
        Args:
            folder_name: Name of the folder (numeric)
            original_filename: Original filename
            
        Returns:
            Tuple of (new_filename, status)
            where status is one of: 'skip', 'fix_duplicate', 'rename'
        """
        # Check if the file already has the correct prefix
        if self.has_correct_prefix(folder_name, original_filename):
            return original_filename, 'skip'
        
        # Check if the file has a duplicate prefix
        if self.has_duplicate_prefix(folder_name, original_filename):
            fixed_filename = self.fix_duplicate_prefix(folder_name, original_filename)
            if self.has_correct_prefix(folder_name, fixed_filename):
                return fixed_filename, 'fix_duplicate'
        
        # Generate the new filename with the appropriate pattern
        prefixed_folder = f"PA{folder_name}"
        
        if self.add_semantic_prefix:
            new_filename = f"semantic_segmentation_{prefixed_folder}_{original_filename}"
        else:
            new_filename = f"{prefixed_folder}_{original_filename}"
        
        return new_filename, 'rename'
    
    def find_and_prepare_renames(self) -> None:
        """Find all files that need to be renamed and prepare the rename operations."""
        self.logger.info(f"Starting file rename preparation in: {self.root_dir}")
        if self.add_semantic_prefix:
            self.logger.info("Using pattern: semantic_segmentation_PA{folder_name}_{original_filename}")
        else:
            self.logger.info("Using pattern: PA{folder_name}_{original_filename}")
        
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
        self._report_findings()
    
    def _search_directory(self, directory: Path) -> None:
        """
        Recursively search a directory for files to rename.
        
        Args:
            directory: Directory to search
        """
        try:
            self.stats['directories_scanned'] += 1
            
            for item in directory.iterdir():
                if item.is_dir():
                    # Check if this is a numeric folder
                    if self.is_numeric_folder(item):
                        self.stats['numeric_folders_found'] += 1
                        self.logger.debug(f"Found numeric folder: {item.name}")
                        
                        # Process files in this folder
                        self._process_numeric_folder(item)
                    else:
                        # Continue recursion for non-numeric folders
                        self._search_directory(item)
                        
        except PermissionError:
            self.logger.warning(f"Permission denied accessing: {directory}")
            self.stats['errors'] += 1
        except Exception as e:
            self.logger.error(f"Error scanning {directory}: {e}")
            self.stats['errors'] += 1
    
    def _process_numeric_folder(self, folder_path: Path) -> None:
        """
        Process files in a numeric folder.
        
        Args:
            folder_path: Path to the numeric folder
        """
        try:
            for item in folder_path.iterdir():
                if item.is_file() and self.should_process_file(item):
                    self.stats['files_scanned'] += 1
                    
                    # Generate new filename and get status
                    new_filename, status = self.generate_new_filename(folder_path.name, item.name)
                    new_path = folder_path / new_filename
                    
                    if status == 'skip':
                        self.stats['files_skipped'] += 1
                        self.logger.debug(f"Skipping (already has correct prefix): {item.name}")
                        continue
                    
                    # Only add to rename operations if the filename would actually change
                    if new_path != item:
                        self.rename_operations.append((item, new_path, status))
                        self.stats['files_to_rename'] += 1
                        
                        if status == 'fix_duplicate':
                            self.stats['duplicates_fixed'] += 1
                            self.logger.debug(f"Will fix duplicate: {item.name} -> {new_filename}")
                        else:
                            self.logger.debug(f"Will rename: {item.name} -> {new_filename}")
                        
                        # Report progress every 100 files
                        if self.stats['files_to_rename'] % 100 == 0:
                            self.logger.info(f"Found {self.stats['files_to_rename']} files to rename so far...")
                            
        except PermissionError:
            self.logger.warning(f"Permission denied accessing: {folder_path}")
            self.stats['errors'] += 1
        except Exception as e:
            self.logger.error(f"Error processing folder {folder_path}: {e}")
            self.stats['errors'] += 1
    
    def _report_findings(self) -> None:
        """Report the findings."""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        self.logger.info("Search completed. Summary:")
        self.logger.info(f"  Directories scanned: {self.stats['directories_scanned']}")
        self.logger.info(f"  Numeric folders found: {self.stats['numeric_folders_found']}")
        self.logger.info(f"  Files scanned: {self.stats['files_scanned']}")
        self.logger.info(f"  Files to rename: {self.stats['files_to_rename']}")
        self.logger.info(f"  Files skipped (already correct): {self.stats['files_skipped']}")
        self.logger.info(f"  Duplicates to fix: {self.stats['duplicates_fixed']}")
        self.logger.info(f"  Errors encountered: {self.stats['errors']}")
        self.logger.info(f"  Search duration: {duration:.2f} seconds")
        
        if self.verbose:
            self.logger.debug("Files to rename:")
            for old_path, new_path, status in self.rename_operations:
                self.logger.debug(f"  {old_path} -> {new_path} ({status})")
    
    def execute_renames(self) -> None:
        """Execute the rename operations."""
        if not self.rename_operations:
            self.logger.info("No files to rename.")
            return
            
        self.logger.info(f"Starting to rename {len(self.rename_operations)} files...")
        
        for old_path, new_path, status in self.rename_operations:
            try:
                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Would rename ({status}): {old_path.name} -> {new_path.name}")
                else:
                    # Check if the new filename already exists
                    if new_path.exists():
                        self.logger.warning(f"Target file already exists, skipping: {new_path}")
                        continue
                        
                    # Rename the file
                    old_path.rename(new_path)
                    self.stats['files_renamed'] += 1
                    self.logger.debug(f"Renamed ({status}): {old_path.name} -> {new_path.name}")
                    
            except Exception as e:
                self.logger.error(f"Error renaming {old_path}: {e}")
                self.stats['errors'] += 1
        
        self._report_results()
    
    def _report_results(self) -> None:
        """Report the results of the rename operation."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would have renamed {len(self.rename_operations)} files")
            if self.stats['duplicates_fixed'] > 0:
                self.logger.info(f"[DRY RUN] Would have fixed {self.stats['duplicates_fixed']} duplicate prefixes")
        else:
            self.logger.info(f"Renamed {self.stats['files_renamed']} files successfully")
            if self.stats['duplicates_fixed'] > 0:
                self.logger.info(f"Fixed {self.stats['duplicates_fixed']} duplicate prefixes")
            
        if self.stats['errors'] > 0:
            self.logger.warning(f"Encountered {self.stats['errors']} errors during the process")
    
    def run(self) -> None:
        """Run the complete file renaming process."""
        self.find_and_prepare_renames()
        self.execute_renames()


def main():
    """Main function to parse arguments and run the FileRenamer."""
    parser = argparse.ArgumentParser(
        description="Rename files based on folder name with duplicate handling."
    )
    parser.add_argument(
        "directory",
        help="Root directory to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually renaming"
    )
    parser.add_argument(
        "--extensions",
        nargs='+',
        help="File extensions to process (e.g., .csv .mp4). If not specified, all files will be processed."
    )
    parser.add_argument(
        "--add-semantic-prefix",
        action="store_true",
        help="Add 'semantic_segmentation_' prefix to filenames"
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
    
    renamer = FileRenamer(
        root_dir=args.directory,
        dry_run=args.dry_run,
        log_file=args.log_file,
        verbose=args.verbose,
        file_extensions=args.extensions,
        add_semantic_prefix=args.add_semantic_prefix
    )
    
    try:
        renamer.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()