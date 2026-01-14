import sqlite3
import pandas as pd
import re
import logging
import json
import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import hashlib
import signal
import psutil
from pydantic import BaseModel, Field, field_validator

@dataclass
class FileMetadata:
    """Metadata extracted from filename"""
    patient_name: Optional[str]
    movement_type: Optional[str]
    jacket_status: Optional[str]
    side: Optional[str]
    model_name: Optional[str]
    original_filename: str

class ProcessingMetrics:
    """Track processing metrics for monitoring"""
    def __init__(self):
        self.start_time = time.time()
        self.files_processed = 0
        self.files_failed = 0
        self.rows_processed = 0
        self.processing_time = 0
        self.lock = threading.Lock()
    
    def increment_files_processed(self):
        with self.lock:
            self.files_processed += 1
    
    def increment_files_failed(self):
        with self.lock:
            self.files_failed += 1
    
    def add_rows_processed(self, count):
        with self.lock:
            self.rows_processed += count
    
    def get_metrics(self):
        with self.lock:
            return {
                "files_processed": self.files_processed,
                "files_failed": self.files_failed,
                "rows_processed": self.rows_processed,
                "processing_time": time.time() - self.start_time,
                "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024)
            }

class ConfigModel(BaseModel):
    """Configuration model with validation"""
    db_path: str = Field(..., description="Path to SQLite database")
    log_path: str = Field("processing.log", description="Path to log file")
    log_level: str = Field("INFO", description="Logging level")
    semantic_dir: Optional[str] = Field(None, description="Path to semantic segmentation CSV files")
    seq_dir: Optional[str] = Field(None, description="Path to seq-based CSV files")
    summary_file_path: Optional[str] = Field(None, description="Path to merged summary Excel file")
    chunk_size: int = Field(1000, description="Chunk size for database inserts")
    retry_attempts: int = Field(3, description="Number of retry attempts for failed operations")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    use_parallel: bool = Field(False, description="Whether to use parallel processing (may affect order)")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()

class EnhancedCSVDatabaseProcessor:
    """Production-grade CSV to SQLite database processor with column merging"""
    
    # Filename pattern regex for semantic segmentation files
    SEMANTIC_PATTERN = re.compile(
        r'semantic_segmentation_'
        r'(?P<patient>[A-Z]+\d+)_'
        r'(?P<movement>FGS|UGS)_'
        r'(?P<jacket>WJ|WoJ)_'
        r'(?P<side>[12])_'
        r'(?P<model>[^_]+)_landmarks'
    )
    
    # Simple pattern to extract seq hash from beginning of filename
    SEQ_PATTERN = re.compile(r'^(cl[a-z0-9]+)_')
    
    def __init__(self, config: Union[str, Dict, ConfigModel]):
        """Initialize processor with configuration"""
        # Load configuration
        if isinstance(config, str):
            self.config = self._load_config_from_file(config)
        elif isinstance(config, dict):
            self.config = ConfigModel(**config)
        else:
            self.config = config
        
        # Set up paths
        self.db_path = Path(self.config.db_path)
        self.log_path = Path(self.config.log_path)
        
        # Initialize metrics
        self.metrics = ProcessingMetrics()
        
        # Set up logging
        self._setup_logging()
        
        # Track processed files for diagnostics
        self.processed_files = []
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        # Flag for graceful shutdown
        self.shutdown_requested = False
        
        self.logger.info(f"Processor initialized with config: {self.config.model_dump()}")
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def _load_config_from_file(self, config_file: str) -> ConfigModel:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            return ConfigModel(**config_data)
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_file}: {e}")
            raise
    
    def _setup_logging(self):
        """Configure comprehensive logging"""
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with human-readable format
        file_handler = logging.FileHandler(self.log_path, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler with human-readable format
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info("=" * 80)
        self.logger.info("Processing session started")
        self.logger.info("=" * 80)
    
    @contextmanager
    def _db_connection(self, timeout=30.0):
        """Context manager for database connections with retry logic"""
        attempts = 0
        last_error = None
        
        while attempts < self.config.retry_attempts and not self.shutdown_requested:
            try:
                conn = sqlite3.connect(
                    self.db_path, 
                    timeout=timeout,
                    check_same_thread=False
                )
                conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
                conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and performance
                yield conn
                conn.close()
                return
            except sqlite3.Error as e:
                last_error = e
                attempts += 1
                self.logger.warning(f"Database connection attempt {attempts} failed: {e}")
                if attempts < self.config.retry_attempts:
                    time.sleep(self.config.retry_delay * attempts)  # Exponential backoff
        
        if last_error:
            raise last_error
        else:
            raise Exception("Shutdown requested during database connection")
    
    def _create_database_schema(self):
        """Create database schema with merged columns"""
        self.logger.info("Creating database schema...")
        
        with self._db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if table exists to preserve data
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='landmarks'
            """)
            
            if not cursor.fetchone():
                # Create new table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE landmarks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_name TEXT,           -- Merged: semantic patient_name OR seq
                        frame INTEGER,
                        movement_type TEXT,         -- Merged: semantic movement_type OR video_speed
                        jacket_status TEXT,
                        side TEXT,                  -- Merged: semantic side OR cam_view
                        model_name TEXT,
                        timestamp_ms REAL,
                        landmark_id INTEGER,
                        x_norm REAL,
                        y_norm REAL,
                        z_norm REAL,
                        visibility REAL,
                        x_px REAL,
                        y_px REAL,
                        source_file TEXT NOT NULL,
                        file_order INTEGER NOT NULL,
                        file_path TEXT,             -- Full path to source file
                        -- Additional columns from merged_summary
                        start_frame INTEGER,
                        end_frame INTEGER,
                        url TEXT,
                        gait_event TEXT,
                        dataset TEXT,
                        gait_pattern TEXT,
                        add_pattern_info TEXT,
                        title TEXT,
                        uploader TEXT,
                        fps REAL,
                        start_time TEXT,
                        end_time TEXT,
                        duration REAL,
                        checksum TEXT,
                        width INTEGER,
                        height INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create processing log table with extended info
                cursor.execute("""
                    CREATE TABLE processing_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT UNIQUE NOT NULL,
                        file_path TEXT NOT NULL,
                        file_order INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        rows_processed INTEGER,
                        error_message TEXT,
                        seq_matched TEXT,
                        data_source TEXT,           -- 'SEMANTIC' or 'SEQ'
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create metrics table
                cursor.execute("""
                    CREATE TABLE processing_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        files_processed INTEGER,
                        files_failed INTEGER,
                        rows_processed INTEGER,
                        processing_time REAL,
                        memory_usage_mb REAL
                    )
                """)
                
                conn.commit()
                self.logger.info("✓ Database schema created successfully")
            else:
                self.logger.info("✓ Database already exists, preserving data")
    
    def _parse_semantic_filename(self, filename: str) -> FileMetadata:
        """Extract metadata from semantic segmentation filename"""
        name_without_ext = filename.rsplit('.', 1)[0]
        match = self.SEMANTIC_PATTERN.search(name_without_ext)
        
        if match:
            groups = match.groupdict()
            
            movement_map = {'FGS': 'Fast Movement', 'UGS': 'Regular Movement'}
            jacket_map = {'WJ': 'With Jacket', 'WoJ': 'Without Jacket'}
            side_map = {'1': 'Right', '2': 'Left'}
            
            return FileMetadata(
                patient_name=groups.get('patient'),
                movement_type=movement_map.get(groups.get('movement')),
                jacket_status=jacket_map.get(groups.get('jacket')),
                side=side_map.get(groups.get('side')),
                model_name=groups.get('model'),
                original_filename=filename
            )
        else:
            self.logger.warning(f"Could not parse semantic filename: {filename}")
            return FileMetadata(
                patient_name=None,
                movement_type=None,
                jacket_status=None,
                side=None,
                model_name=None,
                original_filename=filename
            )
    
    def _extract_seq_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract seq hash identifier from filename
        
        CRITICAL: Extracts ONLY the seq hash (e.g., cljar9bqo00c43n6l2u5zmlru)
        from the beginning of the filename before the first underscore
        
        Example:
            Input: 'cljar9bqo00c43n6l2u5zmlru_left side_nan_Abnormal Gait_abnormal_landmarks.csv'
            Output: 'cljar9bqo00c43n6l2u5zmlru'
        """
        name_without_ext = filename.rsplit('.', 1)[0]
        match = self.SEQ_PATTERN.match(name_without_ext)
        
        if match:
            seq_id = match.group(1)
            self.logger.info(f"  ✓ Extracted seq hash: {seq_id}")
            return seq_id
        else:
            self.logger.error(f"  ✗ Could not extract seq hash from: {filename}")
            self.logger.error(f"     Filename must start with 'cljar' followed by alphanumeric characters")
            return None
    
    def _custom_sort_files(self, files: List[Path]) -> List[Path]:
        """Custom sort to ensure WJ comes before WoJ for semantic files"""
        def sort_key(file_path):
            filename = file_path.name
            
            # Parse the filename to extract components
            match = self.SEMANTIC_PATTERN.search(filename)
            if match:
                groups = match.groupdict()
                
                # WJ gets 0, WoJ gets 1 to ensure WJ comes first
                jacket_priority = 0 if groups.get('jacket') == 'WJ' else 1
                
                # Movement priority: FGS before UGS
                movement_priority = 0 if groups.get('movement') == 'FGS' else 1
                
                return (
                    groups.get('patient', ''),
                    movement_priority,
                    jacket_priority,
                    int(groups.get('side', '0')),
                    groups.get('model', '')
                )
            return (filename,)
        
        sorted_files = sorted(files, key=sort_key)
        
        self.logger.info("Semantic files sorting order:")
        for i, file in enumerate(sorted_files):
            self.logger.info(f"  [{i+1}] {file.name}")
        
        return sorted_files
    
    def _safe_format(self, value, format_str="", default="NULL"):
        """Safely format a value that might be None"""
        if value is None or pd.isna(value):
            return default
        try:
            if format_str:
                return format(value, format_str)
            return str(value)
        except:
            return str(value)
    
    def _load_merged_summary_data(self, summary_file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Load and process the merged summary data"""
        self.logger.info(f"Loading merged summary from: {summary_file_path}")
        
        try:
            # Read the Excel file
            summary_df = pd.read_excel(summary_file_path)
            self.logger.info(f"  ✓ Loaded {len(summary_df)} rows")
            
            # Clean column names - remove trailing spaces
            summary_df.columns = summary_df.columns.str.strip()
            
            # Log available columns
            self.logger.info(f"  Columns: {', '.join(summary_df.columns)}")
            
            # CRITICAL: Verify 'seq' column exists
            if 'seq' not in summary_df.columns:
                self.logger.error("  ✗ CRITICAL: 'seq' column not found!")
                self.logger.error(f"     Available columns: {list(summary_df.columns)}")
                raise ValueError("Missing 'seq' column in merged_summary file")
            
            # Clean seq values - remove whitespace and convert to string
            summary_df['seq'] = summary_df['seq'].astype(str).str.strip()
            
            # Create lookup dictionary
            summary_lookup = {}
            for _, row in summary_df.iterrows():
                seq_key = row['seq']
                summary_lookup[seq_key] = row.to_dict()
            
            self.logger.info(f"  ✓ Created lookup with {len(summary_lookup)} seq entries")
            self.logger.info(f"  Sample seq hashes: {list(summary_lookup.keys())[:3]}")
            
            return summary_df, summary_lookup
            
        except Exception as e:
            self.logger.error(f"  ✗ Error loading merged summary: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame(), {}
    
    def _validate_csv_structure(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Validate CSV has expected columns"""
        required_columns = {
            'frame', 'timestamp_ms', 'landmark_id', 
            'x_norm', 'y_norm', 'z_norm', 'visibility', 'x_px', 'y_px'
        }
        
        df_columns = set(df.columns)
        missing_columns = required_columns - df_columns
        
        if missing_columns:
            return False, f"Missing columns: {missing_columns}"
        
        if df.empty:
            return False, "CSV file is empty"
        
        return True, None
    
    def _process_semantic_file(self, file_path: Path, file_order: int) -> Dict[str, Any]:
        """Process a single semantic segmentation file"""
        filename = file_path.name
        full_path = str(file_path.absolute())
        result = {
            'filename': filename,
            'full_path': full_path,
            'file_order': file_order,
            'rows': 0,
            'status': 'SUCCESS',
            'source': 'SEMANTIC',
            'patient_name': None,
            'error': None
        }
        
        try:
            self.logger.info(f"Processing: {filename}")
            
            # Parse filename
            metadata = self._parse_semantic_filename(filename)
            result['patient_name'] = metadata.patient_name
            
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Validate CSV structure
            is_valid, error_msg = self._validate_csv_structure(df)
            if not is_valid:
                self.logger.warning(f"  ⚠ {error_msg} - adding missing columns")
                required_columns = {
                    'frame', 'timestamp_ms', 'landmark_id', 
                    'x_norm', 'y_norm', 'z_norm', 'visibility', 'x_px', 'y_px'
                }
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = None
            
            if df.empty:
                self.logger.warning(f"  ⚠ CSV file is empty - skipping")
                result['status'] = 'FAILED'
                result['error'] = 'CSV file is empty'
                return result
            
            # Add metadata from filename
            df['patient_name'] = metadata.patient_name
            df['movement_type'] = metadata.movement_type
            df['jacket_status'] = metadata.jacket_status
            df['side'] = metadata.side
            df['model_name'] = metadata.model_name
            df['source_file'] = filename
            df['file_path'] = full_path
            df['file_order'] = file_order
            
            # Initialize new columns with None
            new_columns = [
                'start_frame', 'end_frame', 'url', 'gait_event', 'dataset', 
                'gait_pattern', 'add_pattern_info', 'title', 'uploader',
                'fps', 'start_time', 'end_time', 'duration', 
                'checksum', 'width', 'height'
            ]
            
            for col in new_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Insert into database
            with self._db_connection() as conn:
                # Insert in chunks
                chunksize = self.config.chunk_size
                rows_inserted = 0
                
                for i in range(0, len(df), chunksize):
                    if self.shutdown_requested:
                        result['status'] = 'FAILED'
                        result['error'] = 'Processing interrupted by shutdown signal'
                        return result
                        
                    chunk = df.iloc[i:i+chunksize]
                    chunk.to_sql('landmarks', conn, if_exists='append', index=False)
                    rows_inserted += len(chunk)
                    conn.commit()
                
                result['rows'] = rows_inserted
                
                # Log to database
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO processing_log 
                    (filename, file_path, file_order, status, rows_processed, error_message, seq_matched, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (filename, full_path, file_order, result['status'], rows_inserted, result['error'], None, result['source']))
                conn.commit()
            
            self.logger.info(f"  ✓ SUCCESS: {rows_inserted:,} rows inserted")
            self.metrics.increment_files_processed()
            self.metrics.add_rows_processed(rows_inserted)
            
        except Exception as e:
            result['status'] = 'FAILED'
            result['error'] = str(e)
            self.logger.error(f"  ✗ FAILED: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.metrics.increment_files_failed()
            
            # Log failure to database
            try:
                with self._db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO processing_log 
                        (filename, file_path, file_order, status, rows_processed, error_message, seq_matched, data_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (filename, full_path, file_order, result['status'], 0, result['error'][:500], None, result['source']))
                    conn.commit()
            except:
                pass
        
        return result
    
    def _process_seq_file(self, file_path: Path, file_order: int, summary_lookup: Dict) -> Dict[str, Any]:
        """Process a single seq-based file"""
        filename = file_path.name
        full_path = str(file_path.absolute())
        result = {
            'filename': filename,
            'full_path': full_path,
            'file_order': file_order,
            'rows': 0,
            'status': 'SUCCESS',
            'source': 'SEQ',
            'seq': None,
            'gait_event': None,
            'dataset': None,
            'error': None
        }
        
        try:
            self.logger.info(f"Processing: {filename}")
            
            # Extract seq hash from filename
            seq_id = self._extract_seq_from_filename(filename)
            result['seq'] = seq_id
            
            if not seq_id:
                result['status'] = 'FAILED'
                result['error'] = "Could not extract seq hash from filename"
                self.logger.error(f"  ✗ {result['error']}")
                self.metrics.increment_files_failed()
                return result
            
            # Match with summary
            if seq_id not in summary_lookup:
                result['status'] = 'FAILED'
                result['error'] = f"seq '{seq_id}' not found in merged_summary"
                self.logger.error(f"  ✗ {result['error']}")
                self.logger.error(f"     Available seq samples: {list(summary_lookup.keys())[:5]}")
                self.metrics.increment_files_failed()
                return result
            
            matched_summary = summary_lookup[seq_id]
            result['gait_event'] = matched_summary.get('gait_event')
            result['dataset'] = matched_summary.get('dataset')
            
            self.logger.info(f"  ✓ Matched with seq: {seq_id}")
            
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Validate CSV structure
            is_valid, error_msg = self._validate_csv_structure(df)
            if not is_valid:
                self.logger.warning(f"  ⚠ {error_msg} - adding missing columns")
                required_columns = {
                    'frame', 'timestamp_ms', 'landmark_id', 
                    'x_norm', 'y_norm', 'z_norm', 'visibility', 'x_px', 'y_px'
                }
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = None
            
            if df.empty:
                self.logger.warning(f"  ⚠ CSV file is empty - skipping")
                result['status'] = 'FAILED'
                result['error'] = 'CSV file is empty'
                self.metrics.increment_files_failed()
                return result
            
            # COLUMN MERGING: Map summary to landmark columns
            df['patient_name'] = matched_summary.get('seq')
            df['movement_type'] = matched_summary.get('Video_Speed')  # Note: Changed from video_speed to Video_Speed
            df['side'] = matched_summary.get('cam_view')
            
            # Add file metadata
            df['source_file'] = filename
            df['file_path'] = full_path
            df['file_order'] = file_order
            
            # Add new columns from summary
            new_columns_mapping = {
                'start_frame': 'start_frame',
                'end_frame': 'end_frame',
                'url': 'url',
                'gait_event': 'gait_event',
                'dataset': 'dataset',
                'gait_pattern': 'gait_pattern',
                'add_pattern_info': 'add_pattern_info',
                'title': 'title',
                'uploader': 'uploader',
                'fps': 'fps',
                'start_time': 'start_time',
                'end_time': 'end_time',
                'duration': 'duration',
                'checksum': 'checksum',
                'width': 'width',
                'height': 'height'
            }
            
            for db_col, summary_col in new_columns_mapping.items():
                df[db_col] = matched_summary.get(summary_col)
            
            # Ensure required columns exist
            for col in ['jacket_status', 'model_name']:
                if col not in df.columns:
                    df[col] = None
            
            # Insert into database
            with self._db_connection() as conn:
                # Insert in chunks
                chunksize = self.config.chunk_size
                rows_inserted = 0
                
                for i in range(0, len(df), chunksize):
                    if self.shutdown_requested:
                        result['status'] = 'FAILED'
                        result['error'] = 'Processing interrupted by shutdown signal'
                        return result
                        
                    chunk = df.iloc[i:i+chunksize]
                    chunk.to_sql('landmarks', conn, if_exists='append', index=False)
                    rows_inserted += len(chunk)
                    conn.commit()
                
                result['rows'] = rows_inserted
                
                # Log to database
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO processing_log 
                    (filename, file_path, file_order, status, rows_processed, error_message, seq_matched, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (filename, full_path, file_order, result['status'], rows_inserted, result['error'], seq_id, result['source']))
                conn.commit()
            
            self.logger.info(f"  ✓ SUCCESS: {rows_inserted:,} rows inserted")
            self.metrics.increment_files_processed()
            self.metrics.add_rows_processed(rows_inserted)
            
        except Exception as e:
            result['status'] = 'FAILED'
            result['error'] = str(e)
            self.logger.error(f"  ✗ FAILED: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.metrics.increment_files_failed()
            
            # Log failure to database
            try:
                with self._db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO processing_log 
                        (filename, file_path, file_order, status, rows_processed, error_message, seq_matched, data_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (filename, full_path, file_order, result['status'], 0, result['error'][:500], result['seq'], result['source']))
                    conn.commit()
            except:
                pass
        
        return result
    
    def process_semantic_directory(self, directory: str, pattern: str = "*.csv", 
                                  start_file_order: int = 0):
        """
        Process all semantic segmentation CSV files with custom sorting
        
        Args:
            directory: Path to CSV files
            pattern: File pattern to match
            start_file_order: Starting file order number
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            self.logger.error(f"✗ Directory not found: {directory}")
            return 0, 0, 0
        
        # Find all CSV files
        csv_files = list(dir_path.glob(pattern))
        total_files = len(csv_files)
        
        if total_files == 0:
            self.logger.warning(f"⚠ No CSV files found in {directory}")
            return 0, 0, 0
        
        # Apply custom sorting
        csv_files = self._custom_sort_files(csv_files)
        
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"Found {total_files} semantic CSV files to process")
        self.logger.info(f"{'=' * 80}")
        
        # Process files sequentially to maintain order
        successful = 0
        failed = 0
        total_rows = 0
        
        for i, file_path in enumerate(csv_files):
            if self.shutdown_requested:
                self.logger.info("Shutdown requested, stopping processing...")
                break
                
            file_order = start_file_order + i
            result = self._process_semantic_file(file_path, file_order)
            self.processed_files.append(result)
            
            if result['status'] == 'SUCCESS':
                successful += 1
                total_rows += result['rows']
            else:
                failed += 1
        
        # Create indexes
        self.logger.info("\nCreating database indexes...")
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_order ON landmarks(file_order)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_frame ON landmarks(patient_name, frame)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_file ON landmarks(source_file)")
            conn.commit()
        
        # Summary
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"SEMANTIC PROCESSING COMPLETE")
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"Total files: {total_files}")
        self.logger.info(f"✓ Successful: {successful}")
        self.logger.info(f"✗ Failed: {failed}")
        self.logger.info(f"Total rows: {total_rows:,}")
        self.logger.info(f"{'=' * 80}")
        
        return successful, failed, total_rows
    
    def process_seq_directory(self, directory: str, pattern: str = "*.csv", 
                             summary_file_path: str = None,
                             start_file_order: int = 0):
        """
        Process seq-based CSV files in natural order (NO SORTING - F1→F2→F3→...→Fn)
        Matches files to merged_summary using seq hash extracted from filename
        
        Args:
            directory: Path to CSV files
            pattern: File pattern to match
            summary_file_path: Path to the merged summary Excel file
            start_file_order: Starting file order number
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            self.logger.error(f"✗ Directory not found: {directory}")
            return 0, 0, 0
        
        # Load merged summary data
        if not summary_file_path or not Path(summary_file_path).exists():
            self.logger.error(f"✗ Summary file not found: {summary_file_path}")
            return 0, 0, 0
        
        summary_data, summary_lookup = self._load_merged_summary_data(summary_file_path)
        if not summary_lookup:
            self.logger.error("✗ Failed to load summary data - cannot proceed")
            return 0, 0, 0
        
        # Find all CSV files
        csv_files = list(dir_path.glob(pattern))
        total_files = len(csv_files)
        
        if total_files == 0:
            self.logger.warning(f"⚠ No CSV files found in {directory}")
            return 0, 0, 0
        
        # Sort only alphabetically for consistent ordering (NO CUSTOM SORTING)
        csv_files = sorted(csv_files, key=lambda x: x.name)
        
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"Found {total_files} seq-based CSV files")
        self.logger.info(f"Processing in sequential order (F1→F2→F3→...)")
        self.logger.info(f"{'=' * 80}")
        
        # Show file order
        self.logger.info("\nSeq files processing order:")
        for i, file in enumerate(csv_files):
            self.logger.info(f"  [{i+1}] {file.name}")
        
        # Process files sequentially to maintain order
        successful = 0
        failed = 0
        total_rows = 0
        match_failures = []
        
        for i, file_path in enumerate(csv_files):
            if self.shutdown_requested:
                self.logger.info("Shutdown requested, stopping processing...")
                break
                
            file_order = start_file_order + i
            result = self._process_seq_file(file_path, file_order, summary_lookup)
            self.processed_files.append(result)
            
            if result['status'] == 'SUCCESS':
                successful += 1
                total_rows += result['rows']
            else:
                failed += 1
                if 'seq' in result and result['seq']:
                    match_failures.append({
                        'filename': result['filename'],
                        'issue': 'SEQ_NOT_IN_SUMMARY' if 'not found' in result.get('error', '') else 'OTHER_ERROR',
                        'seq': result['seq']
                    })
                else:
                    match_failures.append({
                        'filename': result['filename'],
                        'issue': 'NO_SEQ_EXTRACTED',
                        'seq': None
                    })
        
        # Create indexes
        self.logger.info("\nCreating database indexes...")
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gait_event ON landmarks(gait_event)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON landmarks(dataset)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON landmarks(file_path)")
            conn.commit()
        
        # Summary
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"SEQ PROCESSING COMPLETE")
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"Total files: {total_files}")
        self.logger.info(f"✓ Successful: {successful}")
        self.logger.info(f"✗ Failed: {failed}")
        self.logger.info(f"Total rows: {total_rows:,}")
        
        if match_failures:
            self.logger.warning(f"\n⚠️  Match Failures ({len(match_failures)}):")
            for failure in match_failures:
                self.logger.warning(f"  • {failure['filename']}")
                self.logger.warning(f"    Issue: {failure['issue']}")
                if failure['seq']:
                    self.logger.warning(f"    Seq: {failure['seq']}")
        
        self.logger.info(f"{'=' * 80}")
        
        return successful, failed, total_rows
    
    def _record_metrics(self):
        """Record current processing metrics to database"""
        metrics = self.metrics.get_metrics()
        
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO processing_metrics 
                    (files_processed, files_failed, rows_processed, processing_time, memory_usage_mb)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metrics['files_processed'],
                    metrics['files_failed'],
                    metrics['rows_processed'],
                    metrics['processing_time'],
                    metrics['memory_usage_mb']
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to record metrics: {e}")
    
    def process_all_sources(self, semantic_dir: str = None, seq_dir: str = None,
                           summary_file_path: str = None):
        """
        Process both semantic and seq-based CSV files
        
        Args:
            semantic_dir: Path to semantic segmentation CSV files
            seq_dir: Path to seq-based CSV files
            summary_file_path: Path to merged summary Excel file
        """
        # Create database schema
        if not self.db_path.exists():
            self._create_database_schema()
        
        total_successful = 0
        total_failed = 0
        total_rows = 0
        next_file_order = 0
        
        # Record initial metrics
        self._record_metrics()
        
        # Process semantic files
        if semantic_dir:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STAGE 1: PROCESSING SEMANTIC SEGMENTATION FILES")
            self.logger.info("=" * 80)
            
            successful, failed, rows = self.process_semantic_directory(
                directory=semantic_dir,
                start_file_order=next_file_order
            )
            
            total_successful += successful
            total_failed += failed
            total_rows += rows
            next_file_order += successful + failed
            
            # Record metrics after semantic processing
            self._record_metrics()
        
        # Process seq files
        if seq_dir and not self.shutdown_requested:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STAGE 2: PROCESSING SEQ-BASED FILES")
            self.logger.info("=" * 80)
            
            successful, failed, rows = self.process_seq_directory(
                directory=seq_dir,
                summary_file_path=summary_file_path,
                start_file_order=next_file_order
            )
            
            total_successful += successful
            total_failed += failed
            total_rows += rows
            
            # Record final metrics
            self._record_metrics()
        
        # Final summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total files processed: {total_successful + total_failed}")
        self.logger.info(f"  ✓ Successful: {total_successful}")
        self.logger.info(f"  ✗ Failed: {total_failed}")
        self.logger.info(f"Total rows inserted: {total_rows:,}")
        self.logger.info(f"Database: {self.db_path.absolute()}")
        self.logger.info(f"Log file: {self.log_path.absolute()}")
        self.logger.info("=" * 80)
        
        # Run diagnostics
        self.run_comprehensive_diagnostics()
        
        return total_successful, total_failed, total_rows
    
    def run_comprehensive_diagnostics(self):
        """Run comprehensive diagnostics with detailed file tracking"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE DATABASE DIAGNOSTICS")
        print("=" * 80)
        
        if not self.db_path.exists():
            print(f"❌ Database not found: {self.db_path}")
            return
        
        with self._db_connection() as conn:
            cursor = conn.cursor()
            
            # === SECTION 1: DATABASE INFO ===
            db_size_mb = round(self.db_path.stat().st_size / (1024 * 1024), 2)
            print(f"\n{'=' * 80}")
            print(f"1. DATABASE INFORMATION")
            print(f"{'=' * 80}")
            print(f"📁 File: {self.db_path}")
            print(f"📊 Size: {db_size_mb} MB")
            
            cursor.execute("SELECT COUNT(*) FROM landmarks")
            total_rows = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT source_file) FROM landmarks")
            unique_files = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT patient_name) FROM landmarks WHERE patient_name IS NOT NULL")
            unique_patients = cursor.fetchone()[0]
            
            print(f"📈 Total rows: {total_rows:,}")
            print(f"📂 Unique files: {unique_files}")
            print(f"👤 Unique patients/seqs: {unique_patients}")
            
            # === SECTION 2: DATA ORDER VERIFICATION ===
            print(f"\n{'=' * 80}")
            print(f"2. DATA ORDER VERIFICATION")
            print(f"{'=' * 80}")
            
            # Check if data is ordered by file_order
            cursor.execute("""
                SELECT file_order, MIN(id) as min_id, MAX(id) as max_id, COUNT(*) as rows
                FROM landmarks
                GROUP BY file_order
                ORDER BY file_order
            """)
            
            file_order_groups = cursor.fetchall()
            
            print(f"\nFile Order Groups:")
            print(f"{'Order':<8} {'Min ID':<8} {'Max ID':<8} {'Rows':<8} {'Sequential':<12}")
            print(f"{'-' * 50}")
            
            is_sequential = True
            previous_max_id = 0
            
            for file_order, min_id, max_id, rows in file_order_groups:
                sequential = "✓" if min_id > previous_max_id else "✗"
                if sequential == "✗":
                    is_sequential = False
                print(f"{file_order:<8} {min_id:<8} {max_id:<8} {rows:<8} {sequential:<12}")
                previous_max_id = max_id
            
            if is_sequential:
                print(f"\n✓ Data is properly ordered by file_order")
            else:
                print(f"\n✗ WARNING: Data is NOT properly ordered by file_order!")
                print(f"   This indicates files were processed out of sequence")
            
            # === SECTION 3: ALL PROCESSED FILES (IN ORDER) ===
            print(f"\n{'=' * 80}")
            print(f"3. ALL PROCESSED FILES (IN INSERTION ORDER)")
            print(f"{'=' * 80}")
            
            cursor.execute("""
                SELECT 
                    file_order,
                    filename,
                    file_path,
                    status,
                    rows_processed,
                    data_source,
                    seq_matched
                FROM processing_log
                ORDER BY file_order
            """)
            
            print(f"\n{'Order':<6} {'Status':<8} {'Source':<10} {'Rows':<10} {'Filename':<50} {'Seq/Patient':<30}")
            print(f"{'-' * 130}")
            
            for row in cursor.fetchall():
                file_order, filename, file_path, status, rows, source, seq = row
                status_icon = "✓" if status == "SUCCESS" else "✗"
                rows_str = f"{rows:,}" if rows else "0"
                seq_str = seq if seq else "N/A"
                
                # Truncate long filenames
                display_name = filename[:47] + "..." if len(filename) > 50 else filename
                
                print(f"{file_order:<6} {status_icon} {status:<6} {source:<10} {rows_str:<10} {display_name:<50} {seq_str:<30}")
            
            # === SECTION 4: FAILED FILES DETAIL ===
            cursor.execute("SELECT COUNT(*) FROM processing_log WHERE status='FAILED'")
            failed_count = cursor.fetchone()[0]
            
            if failed_count > 0:
                print(f"\n{'=' * 80}")
                print(f"4. FAILED FILES DETAIL")
                print(f"{'=' * 80}")
                
                cursor.execute("""
                    SELECT filename, file_path, error_message, seq_matched, data_source
                    FROM processing_log
                    WHERE status='FAILED'
                    ORDER BY file_order
                """)
                
                for i, row in enumerate(cursor.fetchall(), 1):
                    filename, file_path, error, seq, source = row
                    print(f"\n[{i}] {filename}")
                    print(f"    Path: {file_path}")
                    print(f"    Source: {source}")
                    if seq:
                        print(f"    Seq: {seq}")
                    print(f"    Error: {error}")
            
            # === SECTION 5: SAMPLE DATA VERIFICATION ===
            print(f"\n{'=' * 80}")
            print(f"5. SAMPLE DATA VERIFICATION")
            print(f"{'=' * 80}")
            
            # Show first 10 rows to verify order
            cursor.execute("""
                SELECT id, file_order, source_file, patient_name, frame
                FROM landmarks
                ORDER BY id
                LIMIT 10
            """)
            
            first_rows = cursor.fetchall()
            if first_rows:
                print(f"\nFirst 10 rows (by ID):")
                print(f"{'ID':<6} {'Order':<8} {'Source':<30} {'Patient':<15} {'Frame':<8}")
                print(f"{'-' * 75}")
                for row in first_rows:
                    id_val, file_order, source_file, patient_name, frame = row
                    source_display = source_file[:27] + "..." if len(source_file) > 30 else source_file
                    print(f"{id_val:<6} {file_order:<8} {source_display:<30} {patient_name:<15} {frame:<8}")
            
            # Show last 10 rows to verify order
            cursor.execute("""
                SELECT id, file_order, source_file, patient_name, frame
                FROM landmarks
                ORDER BY id DESC
                LIMIT 10
            """)
            
            last_rows = cursor.fetchall()
            if last_rows:
                print(f"\nLast 10 rows (by ID, reversed):")
                print(f"{'ID':<6} {'Order':<8} {'Source':<30} {'Patient':<15} {'Frame':<8}")
                print(f"{'-' * 75}")
                for row in reversed(last_rows):
                    id_val, file_order, source_file, patient_name, frame = row
                    source_display = source_file[:27] + "..." if len(source_file) > 30 else source_file
                    print(f"{id_val:<6} {file_order:<8} {source_display:<30} {patient_name:<15} {frame:<8}")
            
            print("\n" + "=" * 80)
            print("✅ DIAGNOSTICS COMPLETE")
            print("=" * 80)
            print(f"Detailed log saved to: {self.log_path}")
            print("=" * 80)


def main():
    """Main execution function"""
    # Default configuration file path
    config_file = "config.json"
    
    # If a config file is provided as argument, use it
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        print("Creating a default config.json file...")
        
        # Create default config
        default_config = {
            "db_path": "landmark_database.db",
            "log_path": "processing.log",
            "log_level": "INFO",
            "semantic_dir": "./data/csv/Health_Gait_0_397",
            "seq_dir": "./data/csv/MissionGait",
            "summary_file_path": "./data/csv/merged_summary_enriched_full.csv",
            "chunk_size": 1000,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "use_parallel": False
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"✓ Created {config_file} with default configuration")
        print("Please edit the configuration and run the script again.")
        return
    
    # Create processor
    try:
        processor = EnhancedCSVDatabaseProcessor(config_file)
        
        # Process data
        processor.process_all_sources(
            semantic_dir=processor.config.semantic_dir,
            seq_dir=processor.config.seq_dir,
            summary_file_path=processor.config.summary_file_path
        )
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()