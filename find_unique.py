import polars as pl
import itertools
from typing import List, Tuple, Dict
import os
import time

# Define the schema for the cache to ensure consistency
CACHE_SCHEMA = {
    "dataset_name": pl.Utf8,
    "columns": pl.List(pl.Utf8),
    "num_unique": pl.Int64,
    "unique_percentage": pl.Float64,
}

def find_most_unique_column_combination_progressive(
    df: pl.DataFrame,
    dataset_name: str,
    top_n_to_carry_over: int = 5,
    cache_path: str = "uniqueness_cache.parquet"
) -> Tuple[List[str], int]:
    """
    Identifies the unique combination of columns that makes the DataFrame most unique,
    using a strict progressive filtering approach with a caching mechanism.

    Subsequent rounds will ONLY check combinations that contain one of the top N
    ranked combinations from the previous round. For example, if a top 2-column
    combo is ['col_A', 'col_B'], the 3-column checks will only explore combinations
    that include ['col_A', 'col_B'], such as ['col_A', 'col_B', 'col_C'].

    Args:
        df (pl.DataFrame): The input Polars DataFrame.
        dataset_name (str): A unique identifier for the dataset being analyzed.
        top_n_to_carry_over (int): The number of top unique column combinations
                                   (from the previous stage) to carry forward.
        cache_path (str): The file path for the permanent cache file.

    Returns:
        Tuple[List[str], int]: A tuple containing:
            - List[str]: The list of column names that form the most unique combination.
            - int: The number of unique rows for that combination.
    """
    total_rows = df.height
    all_columns = df.columns

    # --- Caching Logic: Load and Prepare Cache ---
    try:
        cache_df = pl.read_parquet(cache_path)
        # Ensure schema matches
        for col, dtype in CACHE_SCHEMA.items():
            if col not in cache_df.columns or cache_df[col].dtype != dtype:
                 raise ValueError(f"Cache file at {cache_path} has an invalid schema.")
    except FileNotFoundError:
        cache_df = pl.DataFrame(CACHE_SCHEMA)

    print(f"Loading cache for dataset '{dataset_name}'...")
    existing_results: Dict[tuple, Tuple[int, float]] = {
        tuple(row[0]): (row[1], row[2])
        for row in cache_df.filter(pl.col("dataset_name") == dataset_name)
        .select(["columns", "num_unique", "unique_percentage"])
        .iter_rows()
    }
    print(f"Found {len(existing_results)} cached results for this dataset.")
    # --- End Caching Logic ---

    stage_results: List[Tuple[int, frozenset]] = []
    overall_best_combination = []
    overall_max_unique_rows = 0

    for r in range(1, len(all_columns) + 1):
        print(f"\n--- Checking combinations of length {r} ---")
        current_stage_candidates: List[List[str]] = []
        checked_combinations_this_stage: set[frozenset] = set()
        new_cache_entries = []

        if r == 1:
            # For the first round, all columns are candidates
            current_stage_candidates = [[col] for col in all_columns]
        else:
            # **Strict Candidate Generation**
            # Build new combinations exclusively from the top performers of the previous round.
            # For each top combination, extend it by adding one new column.
            for _, prev_combo_frozenset in stage_results:
                for col in all_columns:
                    if col not in prev_combo_frozenset:
                        new_combo = sorted(list(prev_combo_frozenset) + [col])
                        new_combo_frozenset = frozenset(new_combo)
                        if new_combo_frozenset not in checked_combinations_this_stage:
                            current_stage_candidates.append(new_combo)
                            checked_combinations_this_stage.add(new_combo_frozenset)
        
        # If no candidates could be generated (e.g., if top_n pruned everything),
        # the loop will naturally stop as current_stage_candidates will be empty.

        current_stage_metrics = []
        hits_from_cache = 0
        for combo_names_list in current_stage_candidates:
            combo_key = tuple(sorted(combo_names_list))

            if combo_key in existing_results:
                num_unique, _ = existing_results[combo_key]
                hits_from_cache += 1
            else:
                num_unique = df.select(pl.struct(combo_key).n_unique()).item()
                unique_percentage = num_unique / total_rows if total_rows > 0 else 0.0
                new_cache_entries.append(
                    (dataset_name, list(combo_key), num_unique, unique_percentage)
                )
                existing_results[combo_key] = (num_unique, unique_percentage)

            current_stage_metrics.append((num_unique, frozenset(combo_key)))

            if num_unique > overall_max_unique_rows:
                overall_max_unique_rows = num_unique
                overall_best_combination = list(combo_key)

            if overall_max_unique_rows == total_rows:
                break

        print(f"Processed {len(current_stage_candidates)} combinations. ({hits_from_cache} from cache)")

        if new_cache_entries:
            print(f"Adding {len(new_cache_entries)} new results to cache...")
            new_rows_df = pl.DataFrame(new_cache_entries, schema=CACHE_SCHEMA)
            cache_df = pl.concat([cache_df, new_rows_df])
            cache_df.write_parquet(cache_path)
            print("Cache saved to disk.")

        if overall_max_unique_rows == total_rows:
            print("\nFound a combination that uniquely identifies all rows.")
            return overall_best_combination, overall_max_unique_rows

        current_stage_metrics.sort(key=lambda x: x[0], reverse=True)
        stage_results = current_stage_metrics[:top_n_to_carry_over]
        
        if not stage_results:
            # If no combinations were good enough to carry over, we can't build the next round.
            print("\nStopping early as no further improvements were found in the top N candidates.")
            break

    return overall_best_combination, overall_max_unique_rows

# --- Example Usage ---
if __name__ == "__main__":
    # Create a sample Polars DataFrame
    data = {
        "id": [1, 1, 2, 3, 3, 4],
        "name": ["Alice", "Alice", "Bob", "Charlie", "Charlie", "David"],
        "city": ["NY", "LA", "NY", "SF", "SF", "LA"],
        "age": [25, 25, 30, 35, 36, 40],
        "zip": ["10001", "90210", "10001", "94105", "94105", "90210"]
    }
    df = pl.DataFrame(data)
    
    # --- DEMONSTRATION OF CACHING AND STRICT PROGRESSION ---
    CACHE_FILE = "demo_cache.parquet"
    DATASET_ID = "sample_data_v1"
    
    # Clean up previous cache file for a clean demonstration
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        
    print("="*50)
    print("      RUNNING WITH STRICT PROGRESSIVE FILTERING")
    print("="*50)
    
    # Use top_n_to_carry_over = 2 to see the pruning in action
    most_unique_cols, unique_count = find_most_unique_column_combination_progressive(
        df, dataset_name=DATASET_ID, top_n_to_carry_over=2, cache_path=CACHE_FILE
    )
    
    print(f"\nMost unique column combination: {most_unique_cols}")
    print(f"Number of unique rows: {unique_count}")

    # Show the contents of the cache file
    if os.path.exists(CACHE_FILE):
        print("\n--- Final Cache Contents ---")
        cached_data = pl.read_parquet(CACHE_FILE)
        print(cached_data)
