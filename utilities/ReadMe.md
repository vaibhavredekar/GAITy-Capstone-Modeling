# Directory Cleaner 
## Usage Examples:

Use the following utility script to clean the directory. 
Test on a dummy folder before usage on the complete folder or either do a 

- git init at the root folder<br> 
- commit all the files<br> 
- and then use the command so you can revert anytime to previous state<br>

``` bash
# Basic usage (preserves "semantic_segmentation" folders)
python script.py /path/to/directory

# Custom folder pattern
python script.py /path/to/directory --pattern "my_special_folder"

# Dry run to see what would be deleted
python script.py /path/to/directory --dry-run

# Skip confirmation (use with caution)
python script.py /path/to/directory --no-confirm

# Verbose output with log file
python script.py /path/to/directory --verbose --log-file cleanup.log
```

# Video finder
## Usage Examples:

The purpose is to find all the path of the video files to provide to a common configuration file to generate the pre-processed video from the mediapipe model

``` bash
# Basic usage (JSON output)
python video_finder.py /path/to/videos /path/to/output/config.json

# YAML output
python video_finder.py /path/to/videos /path/to/output/config.yaml --format yaml

# Text output
python video_finder.py /path/to/videos /path/to/output/videos.txt --format text

# Custom video extensions
python video_finder.py /path/to/videos /path/to/output/config.json --extensions .mp4 .avi .mov

# Use MIME type detection
python video_finder.py /path/to/videos /path/to/output/config.json --use-mimetypes

# Verbose output with log file
python video_finder.py /path/to/videos /path/to/output/config.json --verbose --log-file video_finder.log
```

# CSV to SQLite Database Processor

## Purpose

The CSV to SQLite Database Processor is a production-grade tool designed to efficiently import landmark data from CSV files into a SQLite database. It handles two types of data sources: semantic segmentation files and seq-based files, merging them into a unified database schema with proper metadata mapping.

## Features

- Sequential processing to maintain data order
- Automatic metadata extraction from filenames
- Merged data schema from multiple sources
- Comprehensive error handling and logging
- Database diagnostics and verification
- Configurable processing parameters

## Installation

```bash
# Clone or download the script to your project directory
# Ensure you have the required dependencies installed:
pip install pandas openpyxl pydantic psutil
```

## Configuration

Create a `config.json` file in the same directory as the script:

```json
{
  "_description": "Configuration for CSV to SQLite Database Processor",
  "db_path": "landmark_database.db",
  "log_path": "processing.log",
  "log_level": "INFO",
  "semantic_dir": "./semantic_segmentation",
  "seq_dir": "./seq_files",
  "summary_file_path": "./merged_summary_enriched_full.xlsx",
  "chunk_size": 1000,
  "retry_attempts": 3,
  "retry_delay": 1.0,
  "use_parallel": false
}
```

## Usage Examples

```bash
# Basic usage (uses config.json)
python csv_to_db_processor.py

# Custom configuration file
python csv_to_db_processor.py /path/to/custom_config.json

# Run with verbose logging
python csv_to_db_processor.py --verbose

# Run with debug output
python csv_to_db_processor.py --debug
```

## Directory Structure

Organize your files as follows or provide relevant paths to config.json:

```
project/
├── csv_to_db_processor.py          # The Python script
├── config.json                     # Configuration file
├── semantic_segmentation/          # Directory for semantic files
│   ├── semantic_segmentation_PA201_FGS_WJ_1_DensePose_landmarks.csv
│   └── ...
├── seq_files/                      # Directory for seq-based files
│   ├── cljar878f00c03n6ly2v2ay88_right_side_nan_Abnormal_Gait_abnormal_landmarks.csv
│   └── ...
└── merged_summary_enriched_full.xlsx  # Excel summary file
```

## Output

After processing, you'll get:

1. **SQLite Database** (`landmark_database.db`): Contains all landmark data with merged metadata
2. **Processing Log** (`processing.log`): Detailed processing information
3. **Console Output**: Real-time progress and final summary

## Database Schema

The created database contains:
- `landmarks` table: All landmark data with metadata
- `processing_log` table: Processing status for each file
- `processing_metrics` table: Performance metrics

## Troubleshooting

- If data order is incorrect, ensure `use_parallel` is set to `false` in config
- For large datasets, increase `chunk_size` to improve performance
- Check the log file for detailed error information

## Verification

After processing, run the script with the `--verify` flag to check data integrity:

```bash
python csv_to_db_processor.py --verify
```

This will run comprehensive diagnostics to verify:
- Data order integrity
- File processing status
- Database consistency

----


# Renamer Script

## Usage Example: 

# Basic usage (preview changes)
python file_renamer.py /path/to/directory --dry-run

# Execute the renaming
python file_renamer.py /path/to/directory

# Process only specific file types
python file_renamer.py /path/to/directory --extensions .csv .mp4

# Verbose output with log file
python file_renamer.py /path/to/directory --verbose --log-file rename.log


---

# File Copier

## Key Features of This Script:

- Flexible Extension Handling:
Support for multiple extensions using --extension multiple times
Automatic handling of extensions with or without the dot

- Two Copy Modes:
Flat Mode (default): All files go directly to the destination folder
Preserve Structure Mode: Maintains the directory hierarchy in the destination

- Production-Grade Features:
Dry-run mode to preview what would be copied
Detailed logging with different verbosity levels
Progress reporting during large operations
Error handling for permission issues and other exceptions
Statistics tracking and reporting

- Duplicate Handling:
Default behavior: Skip existing files
Option to overwrite existing files with --overwrite

```
Usage Examples:
1. Basic Usage (Copy CSV files):
python file_copier.py --extension .csv /path/to/source /path/to/destination

2. Copy Multiple File Types:
python file_copier.py --extension .csv --extension .mp4 --extension .txt /path/to/source /path/to/destination

3. Preserve Directory Structure:
python file_copier.py --extension .csv /path/to/source /path/to/destination --preserve-structure

4. Overwrite Existing Files:
python file_copier.py --extension .csv /path/to/source /path/to/destination --overwrite

5. Dry Run (Preview Changes):
python file_copier.py --extension .csv /path/to/source /path/to/destination --dry-run

6. Verbose Output with Log File:
python file_copier.py --extension .csv /path/to/source /path/to/destination --verbose --log-file copy.log
```
Example Directory Structures:

```
Before (Source):
/path/to/source/
├── folder1/
│   ├── file1.csv
│   ├── file2.txt
│   └── subfolder/
│       └── file3.csv
├── folder2/
│   ├── file4.mp4
│   └── file5.csv
└── file6.csv

After (Flat Mode - Default):
/path/to/destination/
├── file1.csv
├── file3.csv
├── file5.csv
└── file6.csv

After (Preserve Structure Mode):
/path/to/destination/
├── folder1/
│   ├── file1.csv
│   └── subfolder/
│       └── file3.csv
├── folder2/
│   └── file5.csv
└── file6.csv
```

- Safety Features:
- Dry-run Mode: Always check with --dry-run first to see what would be copied
- Duplicate Detection: Warns about existing files and can optionally overwrite
- Error Handling: Properly handles permission issues and other errors
- Detailed Logging: Provides clear information about what's being copied
