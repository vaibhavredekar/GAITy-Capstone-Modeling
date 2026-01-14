#!/usr/bin/env python3
"""
Pattern Adder - Adds a specific pattern to files if they don't already have it.

This script adds a prefix pattern to files in a directory if they don't already start with that pattern.

Usage: python pattern_adder.py semantic_segmentation /input_path
"""

import os
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

class PatternAdder:
    """
    A tool to add a specific pattern prefix to files if they don't already have it.
    """
    
    def __init__(self, input_dir: str, pattern: str, dry_run: bool = False, 
                 log_file: str = None, verbose: bool = False, 
                 file_extensions: Optional[List[str]] = None, recursive: bool = False):
        """
        Initialize the PatternAdder.
        
        Args:
            input_dir: Directory containing files to process
            pattern: Pattern to add as prefix
            dry_run: If True, only show what would be renamed without actually renaming
            log_file: Path to log file (optional)
            verbose: Enable verbose logging
            file_extensions: List of file extensions to process (if None, process all)
            recursive: If True, process files in subdirectories as well
        """
        self.input_dir = Path(input_dir).resolve()
        self.pattern = pattern
        self.dry_run = dry_run
        self.verbose = verbose
        self.file_extensions = file_extensions
        self.recursive = recursive
        
        # Ensure pattern ends with underscore if not already
        if not self.pattern.endswith('_'):
            self.pattern = f"{self.pattern}_"
        
        # Set up logging
        self.setup_logging(log_file)
        
        # Statistics
        self.stats = {
            'files_scanned': 0,
            'files_to_rename': 0,
            'files_renamed': 0,
            'files_skipped': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Results storage
        self.rename_operations = []
        
    def setup_logging(self, log_file: str = None):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up root logger
        self.logger = logging.getLogger('PatternAdder')
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
    
    def has_pattern(self, filename: str) -> bool:
        """
        Check if the filename already starts with the pattern.
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if the file already has the pattern, False otherwise
        """
        return filename.startswith(self.pattern)
    
    def generate_new_filename(self, filename: str) -> str:
        """
        Generate a new filename with the pattern added.
        
        Args:
            filename: Original filename
            
        Returns:
            New filename with pattern added
        """
        return f"{self.pattern}{filename}"
    
    def find_files_to_rename(self) -> None:
        """Find all files that need to be renamed."""
        self.logger.info(f"Starting file scan in: {self.input_dir}")
        self.logger.info(f"Pattern to add: '{self.pattern}'")
        self.logger.info(f"Recursive search: {self.recursive}")
        self.stats['start_time'] = datetime.now()
        
        if not self.input_dir.exists():
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            sys.exit(1)
            
        if not self.input_dir.is_dir():
            self.logger.error(f"Input path is not a directory: {self.input_dir}")
            sys.exit(1)
        
        # Start the search
        self._search_directory(self.input_dir)
        
        self.stats['end_time'] = datetime.now()
        self._report_findings()
    
    def _search_directory(self, directory: Path) -> None:
        """
        Search a directory for files to rename.
        
        Args:
            directory: Directory to search
        """
        try:
            for item in directory.iterdir():
                if item.is_file() and self.should_process_file(item):
                    self.stats['files_scanned'] += 1
                    
                    # Check if the file already has the pattern
                    if self.has_pattern(item.name):
                        self.stats['files_skipped'] += 1
                        self.logger.debug(f"Skipping (has pattern): {item.name}")
                        continue
                    
                    # Generate new filename
                    new_filename = self.generate_new_filename(item.name)
                    new_path = item.parent / new_filename
                    
                    # Only add to rename operations if the filename would actually change
                    if new_path != item:
                        self.rename_operations.append((item, new_path))
                        self.stats['files_to_rename'] += 1
                        self.logger.debug(f"Will rename: {item.name} -> {new_filename}")
                        
                        # Report progress every 100 files
                        if self.stats['files_to_rename'] % 100 == 0:
                            self.logger.info(f"Found {self.stats['files_to_rename']} files to rename so far...")
                
                # If recursive and item is a directory, search it as well
                elif item.is_dir() and self.recursive:
                    self._search_directory(item)
                            
        except PermissionError:
            self.logger.warning(f"Permission denied accessing: {directory}")
            self.stats['errors'] += 1
        except Exception as e:
            self.logger.error(f"Error scanning {directory}: {e}")
            self.stats['errors'] += 1
    
    def _report_findings(self) -> None:
        """Report the findings."""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        self.logger.info("Scan completed. Summary:")
        self.logger.info(f"  Files scanned: {self.stats['files_scanned']}")
        self.logger.info(f"  Files to rename: {self.stats['files_to_rename']}")
        self.logger.info(f"  Files skipped (already have pattern): {self.stats['files_skipped']}")
        self.logger.info(f"  Errors encountered: {self.stats['errors']}")
        self.logger.info(f"  Scan duration: {duration:.2f} seconds")
        
        if self.verbose:
            self.logger.debug("Files to rename:")
            for old_path, new_path in self.rename_operations:
                self.logger.debug(f"  {old_path} -> {new_path}")
    
    def execute_renames(self) -> None:
        """Execute the rename operations."""
        if not self.rename_operations:
            self.logger.info("No files to rename.")
            return
            
        self.logger.info(f"Starting to rename {len(self.rename_operations)} files...")
        
        for old_path, new_path in self.rename_operations:
            try:
                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Would rename: {old_path.name} -> {new_path.name}")
                else:
                    # Check if the new filename already exists
                    if new_path.exists():
                        self.logger.warning(f"Target file already exists, skipping: {new_path}")
                        continue
                        
                    # Rename the file
                    old_path.rename(new_path)
                    self.stats['files_renamed'] += 1
                    self.logger.debug(f"Renamed: {old_path.name} -> {new_path.name}")
                    
                    # Report progress every 50 files
                    if self.stats['files_renamed'] % 50 == 0:
                        self.logger.info(f"Renamed {self.stats['files_renamed']} files so far...")
                    
            except Exception as e:
                self.logger.error(f"Error renaming {old_path}: {e}")
                self.stats['errors'] += 1
        
        self._report_results()
    
    def _report_results(self) -> None:
        """Report the results of the rename operation."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would have renamed {len(self.rename_operations)} files")
        else:
            self.logger.info(f"Successfully renamed {self.stats['files_renamed']} files")
            
        if self.stats['errors'] > 0:
            self.logger.warning(f"Encountered {self.stats['errors']} errors during the process")
    
    def run(self) -> None:
        """Run the complete pattern adding process."""
        self.find_files_to_rename()
        self.execute_renames()


def main():
    """Main function to parse arguments and run the PatternAdder."""
    parser = argparse.ArgumentParser(
        description="Add a specific pattern to files if they don't already have it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pattern_adder.py semantic_segmentation /path/to/files
  python pattern_adder.py semantic_segmentation /path/to/files --dry-run
  python pattern_adder.py semantic_segmentation /path/to/files --extensions .csv .mp4
  python pattern_adder.py semantic_segmentation /path/to/files --recursive
        """
    )
    parser.add_argument(
        "pattern",
        help="Pattern to add as prefix (e.g., semantic_segmentation)"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing files to process"
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
        "--recursive",
        action="store_true",
        help="Process files in subdirectories as well"
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
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_path}")
        sys.exit(1)
    
    adder = PatternAdder(
        input_dir=args.input_dir,
        pattern=args.pattern,
        dry_run=args.dry_run,
        log_file=args.log_file,
        verbose=args.verbose,
        file_extensions=args.extensions,
        recursive=args.recursive
    )
    
    try:
        adder.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()