import polars as pl
import itertools
from typing import List, Tuple, Dict
import os
import time
import ast # Used for safely parsing string representations of lists

# New, robust log format:
# CACHE_ENTRY | DATASET: dataset_name | COLUMNS: ['col_a', 'col_b'] | UNIQUE_COUNT: 123 | UNIQUE_PCT: 0.85

def parse_log_file(log_path: str, dataset_name: str) -> Dict[tuple, int]:
    """
    Parses the history log file and loads results for the specified dataset
    into an in-memory dictionary for fast lookups.

    Args:
        log_path (str): The path to the log file.
        dataset_name (str): The specific dataset to load from the log.

    Returns:
        Dict[tuple, int]: A dictionary mapping a column combination (tuple) to its unique count.
    """
    existing_results = {}
    print(f"Reading history from '{log_path}' for dataset '{dataset_name}'...")
    if not os.path.exists(log_path):
        print("Log file not found. Starting fresh.")
        return existing_results

    with open(log_path, "r") as f:
        for line in f:
            if not line.startswith("CACHE_ENTRY"):
                continue
            
            try:
                parts = [p.strip() for p in line.split('|')]
                # ['CACHE_ENTRY', 'DATASET: ...', 'COLUMNS: ...', 'UNIQUE_COUNT: ...', ...]
                
                log_dataset = parts[1].split(':', 1)[1].strip()
                
                if log_dataset == dataset_name:
                    # Safely parse the string representation of the list
                    columns_str = parts[2].split(':', 1)[1].strip()
                    columns_list = ast.literal_eval(columns_str)
                    
                    # The key for our dictionary is a sorted tuple
                    combo_key = tuple(sorted(columns_list))
                    
                    # Get the unique count
                    unique_count = int(parts[3].split(':', 1)[1].strip())
                    
                    existing_results[combo_key] = unique_count
            except (IndexError, SyntaxError, ValueError) as e:
                # Ignore malformed lines
                # print(f"Warning: Skipping malformed log line: {line.strip()} - Error: {e}")
                pass
                
    print(f"Found {len(existing_results)} cached results in log file.")
    return existing_results


def find_most_unique_column_combination_with_log(
    df: pl.DataFrame,
    dataset_name: str,
    top_n_to_carry_over: int = 5,
    log_path: str = "history.log"
):
    """
    Identifies the most unique column combination using a single text log file
    for both reading history and appending new results.
    """
    total_rows = df.height
    all_columns = df.columns

    # 1. Load all historical results from the log file into memory
    existing_results = parse_log_file(log_path, dataset_name)
    
    # 2. Open the log file in append mode to write new results as they happen
    with open(log_path, "a") as log_file:
        stage_results: List[Tuple[int, frozenset]] = []
        overall_best_combination = []
        overall_max_unique_rows = 0

        for r in range(1, len(all_columns) + 1):
            print(f"\n--- Checking combinations of length {r} ---")
            current_stage_candidates: List[List[str]] = []
            checked_combinations_this_stage: set[frozenset] = set()

            if r == 1:
                current_stage_candidates = [[col] for col in all_columns]
            else:
                for _, prev_combo_frozenset in stage_results:
                    for col in all_columns:
                        if col not in prev_combo_frozenset:
                            new_combo = sorted(list(prev_combo_frozenset) + [col])
                            new_combo_frozenset = frozenset(new_combo)
                            if new_combo_frozenset not in checked_combinations_this_stage:
                                current_stage_candidates.append(new_combo)
                                checked_combinations_this_stage.add(new_combo_frozenset)
            
            current_stage_metrics = []
            hits_from_cache = 0
            for combo_names_list in current_stage_candidates:
                combo_key = tuple(sorted(combo_names_list))

                if combo_key in existing_results:
                    # --- CACHE HIT ---
                    num_unique = existing_results[combo_key]
                    hits_from_cache += 1
                else:
                    # --- CACHE MISS ---
                    # Perform the expensive calculation
                    num_unique = df.select(pl.struct(combo_key).n_unique()).item()
                    
                    # Add to in-memory cache for this run
                    existing_results[combo_key] = num_unique
                    
                    # Append the new result to the log file immediately
                    unique_percentage = (num_unique / total_rows * 100) if total_rows > 0 else 0.0
                    log_entry = (
                        f"CACHE_ENTRY | DATASET: {dataset_name} | COLUMNS: {sorted(list(combo_key))} | "
                        f"UNIQUE_COUNT: {num_unique} | UNIQUE_PCT: {unique_percentage:.2f}"
                    )
                    log_file.write(log_entry + "\n")

                current_stage_metrics.append((num_unique, frozenset(combo_key)))

                if num_unique > overall_max_unique_rows:
                    overall_max_unique_rows = num_unique
                    overall_best_combination = list(combo_key)

                if overall_max_unique_rows == total_rows:
                    break

            print(f"Processed {len(current_stage_candidates)} combinations. ({hits_from_cache} from log file)")

            if overall_max_unique_rows == total_rows:
                break
            
            current_stage_metrics.sort(key=lambda x: x[0], reverse=True)
            stage_results = current_stage_metrics[:top_n_to_carry_over]
            
            if not stage_results:
                break

    print("\n--- Process Finished ---")
    print(f"Most unique column combination: {overall_best_combination}")
    print(f"Number of unique rows with this combination: {overall_max_unique_rows}")
    return overall_best_combination, overall_max_unique_rows

# --- Example Usage ---
if __name__ == "__main__":
    data = {
        "id": [1, 1, 2, 3, 3, 4],
        "name": ["Alice", "Alice", "Bob", "Charlie", "Charlie", "David"],
        "city": ["NY", "LA", "NY", "SF", "SF", "LA"],
        "age": [25, 25, 30, 35, 36, 40],
    }
    df = pl.DataFrame(data)
    
    LOG_FILE = "history.log"
    DATASET_ID = "customer_data_v2"
    
    # --- RUN 1: FRESH RUN ---
    # To ensure a clean demonstration, remove the old log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    print("="*60)
    print("      RUN 1: POPULATING THE LOG FILE FROM SCRATCH")
    print("="*60)
    start_time_1 = time.time()
    find_most_unique_column_combination_with_log(
        df, dataset_name=DATASET_ID, top_n_to_carry_over=3, log_path=LOG_FILE
    )
    end_time_1 = time.time()
    print(f"\nRun 1 completed in: {end_time_1 - start_time_1:.4f} seconds.")

    # --- RUN 2: USING THE LOG FILE ---
    print("\n" + "="*60)
    print("      RUN 2: USING THE EXISTING LOG FILE FOR CACHING")
    print("="*60)
    start_time_2 = time.time()
    find_most_unique_column_combination_with_log(
        df, dataset_name=DATASET_ID, top_n_to_carry_over=3, log_path=LOG_FILE
    )
    end_time_2 = time.time()
    print(f"\nRun 2 completed in: {end_time_2 - start_time_2:.4f} seconds.")

    # --- Final Log Contents ---
    print("\n" + "="*60)
    print("      FINAL CONTENTS OF LOG FILE")
    print("="*60)
    with open(LOG_FILE, "r") as f:
        print(f.read())
