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
