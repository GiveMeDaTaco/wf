import polars as pl
import itertools
from typing import List, Tuple, Dict
import os
import time

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

    columns_list, num_unique_list, unique_percentage_list = zip(*parsed_records)

    new_history_df = pl.DataFrame({
        "dataset_name": pl.Series([historical_dataset_name] * len(columns_list), dtype=CACHE_SCHEMA["dataset_name"]),
        "columns": pl.Series(columns_list, dtype=CACHE_SCHEMA["columns"]),
        "num_unique": pl.Series(num_unique_list, dtype=CACHE_SCHEMA["num_unique"]),
        "unique_percentage": pl.Series(unique_percentage_list, dtype=CACHE_SCHEMA["unique_percentage"]),
    })

    if os.path.exists(cache_path):
        print(f"Loading existing cache from: {cache_path}")
        existing_cache_df = pl.read_parquet(cache_path)
        combined_df = pl.concat([existing_cache_df, new_history_df])
        final_df = combined_df.unique(subset=["dataset_name", "columns"], keep="first")
        num_added = len(final_df) - len(existing_cache_df)
        print(f"Added {num_added} new records to the existing cache.")
    else:
        print("No existing cache found. Creating a new one.")
        final_df = new_history_df
        print(f"Added {len(final_df)} new records to the new cache.")

    # --- ROBUSTNESS FIX ---
    # Explicitly cast the 'columns' column to the correct type before saving.
    # This guards against any previous operation (like .unique()) changing the dtype to Object.
    print("\nSchema BEFORE final cast:")
    print(final_df.schema)
    
    final_df = final_df.with_columns(
        pl.col("columns").cast(CACHE_SCHEMA["columns"])
    )
    
    print("\nSchema AFTER final cast (this will be written to disk):")
    print(final_df.schema)
    # --- END OF FIX ---

    final_df.write_parquet(cache_path)
    print(f"\nCache successfully saved to: {cache_path}")
    return overall_best_combination, overall_max_unique_rows
