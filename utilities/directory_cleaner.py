#!/usr/bin/env python3
"""
Production-grade script to clean directory structure while preserving:
1. All folders matching a specified pattern (default: "semantic_segmentation")
2. All subfolders and files within those folders
3. All CSV files throughout the directory structure

All other folders and files will be deleted.
"""

import os
import shutil
import argparse
import logging
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple
from datetime import datetime
import json

class DirectoryCleaner:
    """
    A production-grade tool to clean directory structures while preserving specific folders and files.
    """
    
    def __init__(self, root_dir: str, pattern: str = "semantic_segmentation", 
                 dry_run: bool = False, no_confirm: bool = False, 
                 log_file: str = None, verbose: bool = False):
        """
        Initialize the DirectoryCleaner.
        
        Args:
            root_dir: Root directory to process
            pattern: Folder name pattern to preserve (default: "semantic_segmentation")
            dry_run: If True, only show what would be deleted without actually deleting
            no_confirm: If True, skip confirmation prompts
            log_file: Path to log file (optional)
            verbose: Enable verbose logging
        """
        self.root_dir = Path(root_dir).resolve()
        self.pattern = pattern
        self.dry_run = dry_run
        self.no_confirm = no_confirm
        self.verbose = verbose
        
        # Set up logging
        self.setup_logging(log_file)
        
        # Track what to keep and what to delete
        self.semantic_folders = set()  # All folders matching the pattern
        self.protected_folders = set()  # All folders under semantic folders
        self.protected_files = set()  # All files within semantic folders
        self.csv_files = set()  # All CSV files
        self.folders_to_delete = set()  # Folders to delete
        self.files_to_delete = set()  # Files to delete
        
        # Statistics
        self.stats = {
            'folders_scanned': 0,
            'files_scanned': 0,
            'semantic_folders_found': 0,
            'protected_subfolders': 0,
            'protected_files': 0,
            'csv_files_found': 0,
            'folders_to_delete': 0,
            'files_to_delete': 0,
            'folders_deleted': 0,
            'files_deleted': 0,
            'errors': 0
        }
        
    def setup_logging(self, log_file: str = None):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up root logger
        self.logger = logging.getLogger('DirectoryCleaner')
        self.logger.setLevel(log_level)
        
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
    
    def scan_directory(self) -> None:
        """Scan the directory structure to identify what to keep and what to delete."""
        self.logger.info(f"Scanning directory tree: {self.root_dir}")
        self.logger.info(f"Pattern to preserve: '{self.pattern}'")
        
        if not self.root_dir.exists():
            self.logger.error(f"Root directory does not exist: {self.root_dir}")
            sys.exit(1)
            
        if not self.root_dir.is_dir():
            self.logger.error(f"Root path is not a directory: {self.root_dir}")
            sys.exit(1)
        
        # First pass: identify all semantic folders and their contents
        self._find_and_protect_semantic_folders(self.root_dir)
        
        # Second pass: identify all CSV files outside protected folders
        self._find_csv_files(self.root_dir)
        
        # Third pass: identify what to delete
        self._identify_items_to_delete(self.root_dir)
        
        # Report findings
        self._report_findings()
    
    def _find_and_protect_semantic_folders(self, directory: Path) -> None:
        """Recursively find all semantic folders and protect their contents."""
        try:
            for item in directory.iterdir():
                self.stats['folders_scanned'] += 1
                
                if item.is_dir():
                    if item.name == self.pattern:
                        self.semantic_folders.add(item)
                        self.stats['semantic_folders_found'] += 1
                        # Add all subfolders and files to protected sets
                        self._protect_folder_contents(item)
                    else:
                        # Continue recursion
                        self._find_and_protect_semantic_folders(item)
        except PermissionError:
            self.logger.warning(f"Permission denied accessing: {directory}")
            self.stats['errors'] += 1
        except Exception as e:
            self.logger.error(f"Error scanning {directory}: {e}")
            self.stats['errors'] += 1
    
    def _protect_folder_contents(self, directory: Path) -> None:
        """Add a directory and all its subfolders and files to protected sets."""
        try:
            self.protected_folders.add(directory)
            for item in directory.iterdir():
                if item.is_dir():
                    self._protect_folder_contents(item)
                elif item.is_file():
                    self.protected_files.add(item)
                    self.stats['protected_files'] += 1
            self.stats['protected_subfolders'] += 1
        except PermissionError:
            self.logger.warning(f"Permission denied accessing: {directory}")
            self.stats['errors'] += 1
        except Exception as e:
            self.logger.error(f"Error scanning {directory}: {e}")
            self.stats['errors'] += 1
    
    def _find_csv_files(self, directory: Path) -> None:
        """Recursively find all CSV files outside protected folders."""
        try:
            for item in directory.iterdir():
                if item.is_file():
                    self.stats['files_scanned'] += 1
                    if item.suffix.lower() == '.csv' and item not in self.protected_files:
                        self.csv_files.add(item)
                        self.stats['csv_files_found'] += 1
                elif item.is_dir() and item not in self.protected_folders:
                    # Only recurse into non-protected folders
                    self._find_csv_files(item)
        except PermissionError:
            self.logger.warning(f"Permission denied accessing: {directory}")
            self.stats['errors'] += 1
        except Exception as e:
            self.logger.error(f"Error scanning {directory}: {e}")
            self.stats['errors'] += 1
    
    def _identify_items_to_delete(self, directory: Path) -> None:
        """Identify folders and files to delete."""
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    # Don't delete if it's a protected folder
                    if item not in self.protected_folders:
                        self.folders_to_delete.add(item)
                        self.stats['folders_to_delete'] += 1
                        # No need to check subfolders, they'll be deleted with the parent
                elif item.is_file():
                    # Don't delete if it's a protected file or a CSV file
                    if item not in self.protected_files and item not in self.csv_files:
                        self.files_to_delete.add(item)
                        self.stats['files_to_delete'] += 1
        except PermissionError:
            self.logger.warning(f"Permission denied accessing: {directory}")
            self.stats['errors'] += 1
        except Exception as e:
            self.logger.error(f"Error scanning {directory}: {e}")
            self.stats['errors'] += 1
    
    def _report_findings(self) -> None:
        """Report what was found during scanning."""
        self.logger.info("Scan completed. Summary:")
        self.logger.info(f"  Folders scanned: {self.stats['folders_scanned']}")
        self.logger.info(f"  Files scanned: {self.stats['files_scanned']}")
        self.logger.info(f"  '{self.pattern}' folders found: {self.stats['semantic_folders_found']}")
        self.logger.info(f"  Protected subfolders: {self.stats['protected_subfolders']}")
        self.logger.info(f"  Protected files: {self.stats['protected_files']}")
        self.logger.info(f"  CSV files found: {self.stats['csv_files_found']}")
        self.logger.info(f"  Folders to delete: {self.stats['folders_to_delete']}")
        self.logger.info(f"  Files to delete: {self.stats['files_to_delete']}")
        
        if self.verbose:
            self.logger.debug(f"'{self.pattern}' folders:")
            for folder in sorted(self.semantic_folders):
                self.logger.debug(f"  {folder}")
                
            self.logger.debug("CSV files:")
            for file in sorted(self.csv_files):
                self.logger.debug(f"  {file}")
                
            self.logger.debug("Folders to delete:")
            for folder in sorted(self.folders_to_delete):
                self.logger.debug(f"  {folder}")
                
            self.logger.debug("Files to delete:")
            for file in sorted(self.files_to_delete):
                self.logger.debug(f"  {file}")
    
    def confirm_deletion(self) -> bool:
        """Ask for confirmation before proceeding with deletion."""
        if self.no_confirm:
            return True
            
        if self.dry_run:
            self.logger.info("[DRY RUN] No actual deletion will be performed.")
            return True
            
        self.logger.warning(f"About to delete {self.stats['folders_to_delete']} folders and {self.stats['files_to_delete']} files.")
        response = input("Do you want to proceed? (y/n): ").strip().lower()
        return response in ['y', 'yes']
    
    def execute_deletion(self) -> None:
        """Execute the deletion process."""
        if not self.confirm_deletion():
            self.logger.info("Operation cancelled by user.")
            return
            
        self.logger.info("Starting deletion process...")
        
        # Delete files first (to avoid issues with deleting folders that contain files)
        for file_path in sorted(self.files_to_delete):
            try:
                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Would delete file: {file_path}")
                else:
                    file_path.unlink()
                    self.logger.debug(f"Deleted file: {file_path}")
                    self.stats['files_deleted'] += 1
            except Exception as e:
                self.logger.error(f"Error deleting file {file_path}: {e}")
                self.stats['errors'] += 1
        
        # Delete folders (sorted by length to delete subfolders first)
        for folder_path in sorted(self.folders_to_delete, key=lambda p: len(str(p)), reverse=True):
            try:
                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Would delete folder: {folder_path}")
                else:
                    shutil.rmtree(folder_path)
                    self.logger.debug(f"Deleted folder: {folder_path}")
                    self.stats['folders_deleted'] += 1
            except Exception as e:
                self.logger.error(f"Error deleting folder {folder_path}: {e}")
                self.stats['errors'] += 1
        
        # Report final statistics
        self._report_final_stats()
    
    def _report_final_stats(self) -> None:
        """Report final statistics."""
        self.logger.info("Operation completed. Final statistics:")
        self.logger.info(f"  Folders scanned: {self.stats['folders_scanned']}")
        self.logger.info(f"  Files scanned: {self.stats['files_scanned']}")
        self.logger.info(f"  '{self.pattern}' folders preserved: {self.stats['semantic_folders_found']}")
        self.logger.info(f"  Protected subfolders: {self.stats['protected_subfolders']}")
        self.logger.info(f"  Protected files: {self.stats['protected_files']}")
        self.logger.info(f"  CSV files preserved: {self.stats['csv_files_found']}")
        
        if self.dry_run:
            self.logger.info(f"  [DRY RUN] Folders that would be deleted: {self.stats['folders_to_delete']}")
            self.logger.info(f"  [DRY RUN] Files that would be deleted: {self.stats['files_to_delete']}")
        else:
            self.logger.info(f"  Folders deleted: {self.stats['folders_deleted']}")
            self.logger.info(f"  Files deleted: {self.stats['files_deleted']}")
            
        self.logger.info(f"  Errors encountered: {self.stats['errors']}")
        
        # Save statistics to file
        stats_file = self.root_dir / f"directory_cleaner_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            self.logger.info(f"Statistics saved to: {stats_file}")
        except Exception as e:
            self.logger.error(f"Error saving statistics: {e}")
    
    def run(self) -> None:
        """Run the complete directory cleaning process."""
        start_time = datetime.now()
        self.logger.info(f"Starting directory cleaning at {start_time}")
        
        self.scan_directory()
        self.execute_deletion()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.logger.info(f"Directory cleaning completed in {duration:.2f} seconds")


def main():
    """Main function to parse arguments and run the DirectoryCleaner."""
    parser = argparse.ArgumentParser(
        description="Clean directory structure while preserving specified folders and CSV files."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Root directory to process (default: current directory)"
    )
    parser.add_argument(
        "--pattern",
        default="semantic_segmentation",
        help="Folder name pattern to preserve (default: semantic_segmentation)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompts"
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
    
    cleaner = DirectoryCleaner(
        root_dir=args.directory,
        pattern=args.pattern,
        dry_run=args.dry_run,
        no_confirm=args.no_confirm,
        log_file=args.log_file,
        verbose=args.verbose
    )
    
    try:
        cleaner.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
