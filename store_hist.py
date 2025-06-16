import polars as pl
import re
import os
from typing import List, Tuple

# Define the schema for the cache to ensure consistency with the main script
CACHE_SCHEMA = {
    "dataset_name": pl.Utf8,
    "columns": pl.List(pl.Utf8),
    "num_unique": pl.Int64,
    "unique_percentage": pl.Float64,
}

def parse_history_log(log_file_path: str) -> List[Tuple[List[str], int, float]]:
    """
    Parses a log file to extract column combinations and their uniqueness stats.
    """
    pattern = re.compile(r"Testing \[(.*?)\][^\d]*(\d+) unique rows \((.*?)%\)")
    parsed_records = []
    print(f"Reading history file: {log_file_path}")

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    column_str = match.group(1)
                    columns = [col.strip().strip("'\"") for col in column_str.split(',')]
                    num_unique = int(match.group(2))
                    percentage = float(match.group(3))
                    parsed_records.append((sorted(columns), num_unique, percentage / 100.0))
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return []
        
    print(f"Successfully parsed {len(parsed_records)} records from the log file.")
    return parsed_records


def preload_cache_from_history(
    history_file_path: str,
    cache_path: str,
    historical_dataset_name: str
):
    """
    Parses a history log file and loads its data into the Parquet cache,
    handling duplicates with any existing cache data.
    """
    parsed_records = parse_history_log(history_file_path)
    if not parsed_records:
        print("No records to add to the cache. Exiting.")
        return

    # --- THIS IS THE CORRECTED SECTION ---
    # 2. Convert parsed records to a Polars DataFrame using explicit Series
    # This is more robust and guarantees the correct dtypes, preventing the 'Object' type error.
    
    # Unzip the list of tuples into separate lists for each column
    columns_list, num_unique_list, unique_percentage_list = zip(*parsed_records)

    # Create the DataFrame from explicitly typed Polars Series
    new_history_df = pl.DataFrame({
        "dataset_name": pl.Series(
            [historical_dataset_name] * len(columns_list), dtype=CACHE_SCHEMA["dataset_name"]
        ),
        "columns": pl.Series(
            columns_list, dtype=CACHE_SCHEMA["columns"]
        ),
        "num_unique": pl.Series(
            num_unique_list, dtype=CACHE_SCHEMA["num_unique"]
        ),
        "unique_percentage": pl.Series(
            unique_percentage_list, dtype=CACHE_SCHEMA["unique_percentage"]
        ),
    })
    # --- END OF CORRECTION ---

    if os.path.exists(cache_path):
        print(f"Loading existing cache from: {cache_path}")
        existing_cache_df = pl.read_parquet(cache_path)
        combined_df = pl.concat([existing_cache_df, new_history_df])
        
        # Remove duplicates, keeping the data already in the cache
        final_df = combined_df.unique(subset=["dataset_name", "columns"], keep="first")
        num_added = len(final_df) - len(existing_cache_df)
        print(f"Added {num_added} new records to the existing cache.")
    else:
        print("No existing cache found. Creating a new one.")
        final_df = new_history_df
        print(f"Added {len(final_df)} new records to the new cache.")

    final_df.write_parquet(cache_path)
    print(f"Cache successfully saved to: {cache_path}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define file paths and the name for the historical dataset
    HISTORY_LOG_FILE = "history.txt"
    CACHE_FILE = "demo_cache.parquet"
    DATASET_ID = "sample_data_v1"

    # 1. Create a dummy history.txt file for the demonstration
    history_content = """
    INFO:root:Starting process on old machine
    DEBUG:main:Testing ['id']...4 unique rows (66.67%)
    DEBUG:main:Testing ['name']...4 unique rows (66.67%)
    INFO:main:Testing ['city']...3 unique rows (50.0%)
    Another line that should be ignored by the parser.
    DEBUG:main:Testing ['age', 'id']...6 unique rows (100.0%)
    DEBUG:main:Testing ['zip', 'name', 'city']...5 unique rows (83.33%)
    INFO:root:Process finished.
    """
    with open(HISTORY_LOG_FILE, "w") as f:
        f.write(history_content)
    
    print(f"Created a dummy history file at '{HISTORY_LOG_FILE}'")

    # 2. Clean up any previous cache file for a clean run
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"Removed old cache file '{CACHE_FILE}'")

    # 3. Run the preloading function
    print("\n--- Running Cache Preloader ---")
    preload_cache_from_history(
        history_file_path=HISTORY_LOG_FILE,
        cache_path=CACHE_FILE,
        historical_dataset_name=DATASET_ID
    )

    # 4. Verify the contents of the newly created cache
    if os.path.exists(CACHE_FILE):
        print("\n--- Verifying Cache Contents ---")
        loaded_cache = pl.read_parquet(CACHE_FILE)
        print(loaded_cache)

        # 5. (Optional) Demonstrate that running it again doesn't add duplicates
        print("\n--- Running Preloader Again to Test Deduplication ---")
        preload_cache_from_history(
            history_file_path=HISTORY_LOG_FILE,
            cache_path=CACHE_FILE,
            historical_dataset_name=DATASET_ID
        )
