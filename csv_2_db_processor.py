import sqlite3
import pandas as pd
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FileMetadata:
    """Metadata extracted from filename"""
    patient_name: Optional[str]
    movement_type: Optional[str]
    jacket_status: Optional[str]
    side: Optional[str]
    model_name: Optional[str]
    original_filename: str

class CSVDatabaseProcessor:
    """Single-threaded CSV to SQLite database processor with integrated diagnostics"""
    
    # Filename pattern regex
    FILENAME_PATTERN = re.compile(
        r'semantic_segmentation_'
        r'(?P<patient>[A-Z]+\d+)_'
        r'(?P<movement>FGS|UGS)_'
        r'(?P<jacket>WJ|WoJ)_'
        r'(?P<side>[12])_'
        r'(?P<model>[^_]+)_landmarks'
    )
    
    def __init__(self, db_path: str = "landmark_database.db", log_path: str = "processing.log"):
        """Initialize processor with database and log paths"""
        self.db_path = Path(db_path)
        self.log_path = Path(log_path)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"=" * 80)
        self.logger.info(f"Processing session started")
        self.logger.info(f"=" * 80)
        
    def _create_database_schema(self):
        """Create database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Drop tables if they exist to start fresh
            cursor.execute("DROP TABLE IF EXISTS landmarks")
            cursor.execute("DROP TABLE IF EXISTS processing_log")
            
            # Main landmarks table
            cursor.execute("""
                CREATE TABLE landmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT,
                    frame INTEGER,
                    movement_type TEXT,
                    jacket_status TEXT,
                    side TEXT,
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Processing log table
            cursor.execute("""
                CREATE TABLE processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    file_order INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    rows_processed INTEGER,
                    error_message TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def _parse_filename(self, filename: str) -> FileMetadata:
        """Extract metadata from filename"""
        name_without_ext = filename.rsplit('.', 1)[0]
        match = self.FILENAME_PATTERN.search(name_without_ext)
        
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
            self.logger.warning(f"Could not parse filename: {filename}")
            return FileMetadata(
                patient_name=None,
                movement_type=None,
                jacket_status=None,
                side=None,
                model_name=None,
                original_filename=filename
            )
    
    def _custom_sort_files(self, files: List[Path]) -> List[Path]:
        """Custom sort to ensure WJ comes before WoJ"""
        def sort_key(file_path):
            filename = file_path.name
            
            # Parse the filename to extract components
            match = self.FILENAME_PATTERN.search(filename)
            if match:
                groups = match.groupdict()
                
                # Create a tuple for sorting
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
        
        # Debug: print the sorted order
        self.logger.info("File sorting order:")
        for i, file in enumerate(sorted_files):
            self.logger.info(f"  {i}: {file.name}")
        
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
    
    def process_directory(self, directory: str, pattern: str = "*.csv"):
        """
        Process all CSV files sequentially in a single thread
        
        Args:
            directory: Path to CSV files
            pattern: File pattern to match
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            self.logger.error(f"Directory not found: {directory}")
            return
        
        # Find all CSV files
        csv_files = list(dir_path.glob(pattern))
        total_files = len(csv_files)
        
        if total_files == 0:
            self.logger.warning(f"No CSV files found in {directory}")
            return
        
        # Apply custom sorting
        csv_files = self._custom_sort_files(csv_files)
        
        self.logger.info(f"Found {total_files} CSV files to process")
        
        # Create database schema
        self._create_database_schema()
        
        # Process files sequentially
        successful = 0
        failed = 0
        total_rows = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for file_order, file_path in enumerate(csv_files):
                filename = file_path.name
                
                try:
                    self.logger.info(f"Processing file {file_order+1}/{total_files}: {filename}")
                    
                    # Parse filename
                    metadata = self._parse_filename(filename)
                    self.logger.info(f"  Parsed: Patient={metadata.patient_name}, "
                                   f"Movement={metadata.movement_type}, "
                                   f"Jacket={metadata.jacket_status}, "
                                   f"Side={metadata.side}")
                    
                    # Read CSV - don't skip any rows
                    df = pd.read_csv(file_path)
                    self.logger.info(f"  Read {len(df)} rows from CSV")
                    
                    # Check for required columns but don't fail if missing
                    required_columns = {
                        'frame', 'timestamp_ms', 'landmark_id', 
                        'x_norm', 'y_norm', 'z_norm', 'visibility', 'x_px', 'y_px'
                    }
                    
                    df_columns = set(df.columns)
                    missing_columns = required_columns - df_columns
                    
                    if missing_columns:
                        self.logger.warning(f"  Missing columns: {missing_columns}")
                        # Add missing columns with NaN values
                        for col in missing_columns:
                            df[col] = None
                    
                    if df.empty:
                        self.logger.warning(f"  CSV file is empty")
                        failed += 1
                        continue
                    
                    # Add metadata
                    df['patient_name'] = metadata.patient_name
                    df['movement_type'] = metadata.movement_type
                    df['jacket_status'] = metadata.jacket_status
                    df['side'] = metadata.side
                    df['model_name'] = metadata.model_name
                    df['source_file'] = filename
                    df['file_order'] = file_order
                    
                    # Reorder columns
                    column_order = [
                        'patient_name', 'frame', 'movement_type', 'jacket_status', 
                        'side', 'model_name', 'timestamp_ms', 'landmark_id',
                        'x_norm', 'y_norm', 'z_norm', 'visibility', 'x_px', 'y_px',
                        'source_file', 'file_order'
                    ]
                    
                    # Ensure all columns exist
                    for col in column_order:
                        if col not in df.columns:
                            df[col] = None
                    
                    df = df[column_order]
                    
                    # Insert into database - use smaller chunksize and no method='multi'
                    chunksize = 1000  # Reduced chunksize
                    rows_inserted = 0
                    
                    # Split dataframe into chunks
                    for i in range(0, len(df), chunksize):
                        chunk = df.iloc[i:i+chunksize]
                        # Remove method='multi' to avoid SQL variables error
                        chunk.to_sql('landmarks', conn, if_exists='append', index=False)
                        rows_inserted += len(chunk)
                        conn.commit()  # Commit after each chunk
                    
                    total_rows += rows_inserted
                    successful += 1
                    
                    # Log success
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR IGNORE INTO processing_log 
                        (filename, file_order, status, rows_processed, error_message)
                        VALUES (?, ?, ?, ?, ?)
                    """, (filename, file_order, 'SUCCESS', rows_inserted, None))
                    conn.commit()
                    
                    self.logger.info(f"  Successfully processed {filename}: {rows_inserted} rows inserted")
                    
                except Exception as e:
                    failed += 1
                    error_msg = str(e)
                    self.logger.error(f"  Failed to process {filename}: {error_msg}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
                    # Log failure
                    try:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT OR IGNORE INTO processing_log 
                            (filename, file_order, status, rows_processed, error_message)
                            VALUES (?, ?, ?, ?, ?)
                        """, (filename, file_order, 'FAILED', 0, error_msg))
                        conn.commit()
                    except:
                        pass
        
        # Create indexes on final database
        self.logger.info("Creating indexes on database...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_order 
                ON landmarks(file_order)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patient_frame 
                ON landmarks(patient_name, frame)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_file 
                ON landmarks(source_file)
            """)
            conn.commit()
        
        # Summary report
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"PROCESSING SUMMARY")
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"Total files: {total_files}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Total rows inserted: {total_rows:,}")
        self.logger.info(f"{'=' * 80}")
        
        # Run comprehensive diagnostics after processing
        self.run_comprehensive_diagnostics()
    
    def run_comprehensive_diagnostics(self):
        """Run comprehensive database diagnostics after processing"""
        print("\n" + "=" * 80)
        print("RUNNING COMPREHENSIVE DATABASE DIAGNOSTICS")
        print("=" * 80)
        
        if not self.db_path.exists():
            print(f"❌ Database file not found: {self.db_path}")
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Database file info
            db_size_mb = round(self.db_path.stat().st_size / (1024 * 1024), 2)
            print(f"\n📁 Database file: {self.db_path}")
            print(f"📊 Database size: {db_size_mb} MB")
            
            # 2. Table existence check
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('landmarks', 'processing_log')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            print(f"\n📋 Tables found: {tables}")
            
            if 'landmarks' not in tables:
                print("❌ landmarks table missing!")
                return
            
            # 3. Table schema
            cursor.execute("PRAGMA table_info(landmarks)")
            columns = cursor.fetchall()
            
            print(f"\n📝 landmarks table schema ({len(columns)} columns):")
            for col in columns:
                nullable = "NULL" if col[3] == 0 else "NOT NULL"
                default = f" DEFAULT {col[4]}" if col[4] is not None else ""
                print(f"  • {col[1]} ({col[2]}) {nullable}{default}")
            
            # 4. Check for critical columns
            critical_columns = ['id', 'patient_name', 'file_order', 'jacket_status', 'frame', 'landmark_id']
            existing_columns = [col[1] for col in columns]
            missing_critical = [col for col in critical_columns if col not in existing_columns]
            
            if missing_critical:
                print(f"\n❌ Missing critical columns: {missing_critical}")
            else:
                print(f"\n✅ All critical columns present")
            
            # 5. Row counts and statistics
            cursor.execute("SELECT COUNT(*) FROM landmarks")
            total_rows = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT source_file) FROM landmarks")
            unique_files = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT patient_name) FROM landmarks WHERE patient_name IS NOT NULL")
            unique_patients = cursor.fetchone()[0]
            
            print(f"\n📈 Database statistics:")
            print(f"  • Total rows: {total_rows:,}")
            print(f"  • Unique files: {unique_files}")
            print(f"  • Unique patients: {unique_patients}")
            
            # 6. File order verification
            if 'file_order' in existing_columns:
                print(f"\n🔢 File order verification:")
                order_df = pd.read_sql_query("""
                    SELECT 
                        file_order,
                        source_file,
                        jacket_status,
                        side,
                        COUNT(*) as rows,
                        MIN(id) as first_id,
                        MAX(id) as last_id
                    FROM landmarks
                    GROUP BY file_order, source_file, jacket_status, side
                    ORDER BY file_order
                """, conn)
                
                for _, row in order_df.iterrows():
                    print(f"  Order {row['file_order']}: {row['source_file']}")
                    print(f"    → {row['jacket_status']}, {row['side']}")
                    print(f"    → {row['rows']:,} rows (IDs: {row['first_id']}-{row['last_id']})")
                
                # 7. Order consistency check
                print(f"\n🔍 Order consistency check:")
                expected_order = ['With Jacket', 'With Jacket', 'Without Jacket', 'Without Jacket'] * 2
                actual_order = order_df['jacket_status'].tolist()
                
                if actual_order == expected_order[:len(actual_order)]:
                    print("✅ File order is correct (WJ before WoJ)")
                else:
                    print("❌ File order mismatch!")
                    print(f"  Expected: {expected_order[:len(actual_order)]}")
                    print(f"  Actual:   {actual_order}")
            
            # 8. Data integrity checks
            print(f"\n🔒 Data integrity checks:")
            
            # Check for null values in critical columns
            for col in ['patient_name', 'frame', 'landmark_id']:
                cursor.execute(f"SELECT COUNT(*) FROM landmarks WHERE {col} IS NULL")
                null_count = cursor.fetchone()[0]
                if null_count > 0:
                    print(f"  ⚠️  {null_count:,} rows with NULL {col}")
                else:
                    print(f"  ✅ No NULL values in {col}")
            
            # 9. Processing log analysis
            if 'processing_log' in tables:
                print(f"\n📋 Processing log analysis:")
                log_df = pd.read_sql_query("""
                    SELECT 
                        status,
                        COUNT(*) as count,
                        SUM(rows_processed) as total_rows
                    FROM processing_log
                    GROUP BY status
                """, conn)
                
                for _, row in log_df.iterrows():
                    print(f"  • {row['status']}: {row['count']} files, {row['total_rows'] or 0:,} rows")
                
                # Show failed files if any
                failed_df = pd.read_sql_query("""
                    SELECT filename, error_message
                    FROM processing_log
                    WHERE status = 'FAILED'
                """, conn)
                
                if not failed_df.empty:
                    print(f"\n❌ Failed files:")
                    for _, row in failed_df.iterrows():
                        print(f"  • {row['filename']}: {row['error_message']}")
            
            # 10. Sample data display
            print(f"\n👀 Sample data (first 3 rows from each file):")
            if 'file_order' in existing_columns:
                for file_order in range(min(3, order_df['file_order'].max() + 1)):
                    sample_df = pd.read_sql_query("""
                        SELECT id, frame, landmark_id, x_norm, y_norm, visibility
                        FROM landmarks
                        WHERE file_order = ?
                        ORDER BY id
                        LIMIT 3
                    """, conn, params=(file_order,))
                    
                    if not sample_df.empty:
                        file_info = order_df[order_df['file_order'] == file_order].iloc[0]
                        print(f"\n  File Order {file_order} ({file_info['jacket_status']}):")
                        for _, row in sample_df.iterrows():
                            # Use safe_format to handle None values
                            print(f"    ID:{self._safe_format(row['id'], '>5')} "
                                  f"Frame:{self._safe_format(row['frame'], '>4')} "
                                  f"Landmark:{self._safe_format(row['landmark_id'], '>3')} "
                                  f"X:{self._safe_format(row['x_norm'], '.3f', 'NaN')} "
                                  f"Y:{self._safe_format(row['y_norm'], '.3f', 'NaN')} "
                                  f"Vis:{self._safe_format(row['visibility'], '.3f', 'NaN')}")
            
            print("\n" + "=" * 80)
            print("✅ DIAGNOSTICS COMPLETE")
            print("=" * 80)
    
    def test_filename_parsing(self, directory: str, pattern: str = "*.csv", sample_size: int = 10):
        """Test filename parsing on sample files"""
        dir_path = Path(directory)
        csv_files = list(dir_path.glob(pattern))[:sample_size]
        
        print(f"\n{'=' * 80}")
        print(f"TESTING FILENAME PARSING ON {len(csv_files)} FILES")
        print(f"{'=' * 80}\n")
        
        # Test the custom sort
        csv_files = self._custom_sort_files(csv_files)
        
        for file in csv_files:
            metadata = self._parse_filename(file.name)
            print(f"File: {file.name}")
            print(f"  Patient: {metadata.patient_name}")
            print(f"  Movement: {metadata.movement_type}")
            print(f"  Jacket: {metadata.jacket_status}")
            print(f"  Side: {metadata.side}")
            print(f"  Model: {metadata.model_name}")
            print()
    
    def get_processing_report(self) -> pd.DataFrame:
        """Get detailed processing report"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT 
                    file_order,
                    filename,
                    status,
                    rows_processed,
                    error_message,
                    processed_at
                FROM processing_log
                ORDER BY file_order
            """, conn)
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute("SELECT COUNT(*) FROM landmarks")
            stats['total_rows'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT patient_name) FROM landmarks WHERE patient_name IS NOT NULL")
            stats['unique_patients'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT source_file) FROM landmarks")
            stats['unique_files'] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT movement_type, COUNT(*) as count
                FROM landmarks
                GROUP BY movement_type
            """)
            stats['by_movement'] = dict(cursor.fetchall())
            
            cursor.execute("""
                SELECT jacket_status, COUNT(*) as count
                FROM landmarks
                GROUP BY jacket_status
            """)
            stats['by_jacket'] = dict(cursor.fetchall())
            
            stats['database_size_mb'] = round(self.db_path.stat().st_size / (1024 * 1024), 2)
            
        return stats
    
    def verify_insertion_order(self) -> pd.DataFrame:
        """Verify data insertion order"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT 
                    file_order,
                    source_file,
                    MIN(id) as first_row_id,
                    MAX(id) as last_row_id,
                    COUNT(*) as row_count
                FROM landmarks
                GROUP BY file_order, source_file
                ORDER BY file_order
            """, conn)
    
    def verify_data_integrity(self):
        """Detailed verification of data insertion order and content"""
        print("\n" + "=" * 80)
        print("DATA INTEGRITY VERIFICATION")
        print("=" * 80)
        
        with sqlite3.connect(self.db_path) as conn:
            # Check each file's data
            for i in range(8):  # Check first 8 files
                query = f"""
                    SELECT 
                        file_order,
                        source_file,
                        patient_name,
                        movement_type,
                        jacket_status,
                        side,
                        COUNT(*) as rows,
                        MIN(frame) as min_frame,
                        MAX(frame) as max_frame,
                        MIN(id) as first_id,
                        MAX(id) as last_id
                    FROM landmarks
                    WHERE file_order = {i}
                    GROUP BY file_order, source_file, patient_name, movement_type, jacket_status, side
                """
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    print(f"\nFile Order {i}:")
                    print(f"  Source: {df['source_file'].iloc[0]}")
                    print(f"  Jacket: {df['jacket_status'].iloc[0]}, Side: {df['side'].iloc[0]}")
                    print(f"  Rows: {df['rows'].iloc[0]}, IDs: {df['first_id'].iloc[0]}-{df['last_id'].iloc[0]}")
                    print(f"  Frames: {df['min_frame'].iloc[0]}-{df['max_frame'].iloc[0]}")
                else:
                    break
        
        print("\n" + "=" * 80)
    
    def query_data(self, limit: int = 10, order_by: str = "file_order, id"):
        """Query data from the database with proper ordering"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(f"""
                SELECT 
                    id, patient_name, frame, movement_type, jacket_status, 
                    side, model_name, timestamp_ms, landmark_id,
                    x_norm, y_norm, z_norm, visibility, x_px, y_px,
                    source_file, file_order
                FROM landmarks
                ORDER BY {order_by}
                LIMIT {limit}
            """, conn)
    
    def show_first_rows_by_file(self):
        """Show the first few rows from each file to verify order"""
        print("\n" + "=" * 80)
        print("FIRST ROWS FROM EACH FILE")
        print("=" * 80)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get the file_order values
            file_orders = pd.read_sql_query("""
                SELECT DISTINCT file_order, source_file, jacket_status, side
                FROM landmarks
                ORDER BY file_order
                LIMIT 8
            """, conn)
            
            for _, row in file_orders.iterrows():
                file_order = row['file_order']
                source_file = row['source_file']
                jacket_status = row['jacket_status']
                side = row['side']
                
                print(f"\nFile Order {file_order}: {source_file}")
                print(f"  Jacket: {jacket_status}, Side: {side}")
                
                # Get first 3 rows from this file
                first_rows = pd.read_sql_query("""
                    SELECT id, frame, landmark_id, x_norm, y_norm
                    FROM landmarks
                    WHERE file_order = ?
                    ORDER BY id
                    LIMIT 3
                """, conn, params=(file_order,))
                
                print("  First 3 rows:")
                for _, r in first_rows.iterrows():
                    print(f"    ID: {r['id']}, Frame: {r['frame']}, Landmark: {r['landmark_id']}, X: {r['x_norm']}, Y: {r['y_norm']}")
    
    def show_processing_order(self):
        """Show the order in which files were processed"""
        print("\n" + "=" * 80)
        print("FILE PROCESSING ORDER")
        print("=" * 80)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT file_order, filename, status, rows_processed
                FROM processing_log
                ORDER BY file_order
            """, conn)
            
            for _, row in df.iterrows():
                print(f"Order {row['file_order']}: {row['filename']} - {row['status']} ({row['rows_processed']} rows)")


# Usage Example
if __name__ == "__main__":
    processor = CSVDatabaseProcessor(
        db_path="landmark_database.db",
        log_path="processing.log"
    )
    
    # Test filename parsing first
    print("\n" + "=" * 80)
    print("STEP 1: Testing filename parsing...")
    print("=" * 80)
    processor.test_filename_parsing(
        directory="./semantic_segmentation",
        sample_size=10
    )
    
    input("\nPress Enter to continue with processing, or Ctrl+C to abort...")
    
    # Process files (diagnostics will run automatically after this)
    processor.process_directory(
        directory="",
        pattern="*.csv"
    )
    
    # Additional verification methods
    print("\n" + "=" * 80)
    print("ADDITIONAL VERIFICATION")
    print("=" * 80)
    
    # Show processing order
    processor.show_processing_order()
    
    # Verify insertion order
    order_check = processor.verify_insertion_order()
    print("\nInsertion Order Check:")
    print(order_check)
    
    # Query first 1000 rows with proper ordering
    print("\nFirst 1000 rows (correctly ordered):")
    first_rows = processor.query_data(limit=1000, order_by="file_order, id")
    print(first_rows[['id', 'file_order', 'jacket_status', 'side', 'frame']].head(20))
    
    # Stats
    stats = processor.get_database_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")