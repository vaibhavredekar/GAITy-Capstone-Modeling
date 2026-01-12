# Usage Examples:

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
