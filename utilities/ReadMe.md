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