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
    
    total_rows = df.height
    all_columns = df.columns

    try:
        cache_df = pl.read_parquet(cache_path)
    except FileNotFoundError:
        # If cache doesn't exist, create an empty DataFrame with the correct schema
        cache_df = pl.DataFrame(CACHE_SCHEMA)

    print(f"Loading cache for dataset '{dataset_name}'...")
    existing_results: Dict[tuple, Tuple[int, float]] = {
        tuple(row[0]): (row[1], row[2])
        for row in cache_df.filter(pl.col("dataset_name") == dataset_name)
        .select(["columns", "num_unique", "unique_percentage"])
        .iter_rows()
    }
    print(f"Found {len(existing_results)} cached results for this dataset.")
    
    stage_results: List[Tuple[int, frozenset]] = []
    overall_best_combination = []
    overall_max_unique_rows = 0

    for r in range(1, len(all_columns) + 1):
        print(f"\n--- Checking combinations of length {r} ---")
        current_stage_candidates: List[List[str]] = []
        checked_combinations_this_stage: set[frozenset] = set()
        new_cache_entries = []

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
                num_unique, _ = existing_results[combo_key]
                hits_from_cache += 1
            else:
                num_unique = df.select(pl.struct(combo_key).n_unique()).item()
                unique_percentage = num_unique / total_rows if total_rows > 0 else 0.0
                # Add a tuple of results to the list
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

        # --- THIS IS THE CORRECTED SECTION ---
        if new_cache_entries:
            print(f"Adding {len(new_cache_entries)} new results to cache...")
            
            # Unzip the list of tuples into separate lists for each future column
            dataset_names, columns_lists, nums_unique, pcts_unique = zip(*new_cache_entries)

            # Create a new DataFrame from a dictionary of explicit, typed Series.
            # This is the robust way to prevent the 'Object' dtype error.
            new_rows_df = pl.DataFrame({
                "dataset_name": pl.Series(dataset_names, dtype=CACHE_SCHEMA["dataset_name"]),
                "columns": pl.Series(columns_lists, dtype=CACHE_SCHEMA["columns"]),
                "num_unique": pl.Series(nums_unique, dtype=CACHE_SCHEMA["num_unique"]),
                "unique_percentage": pl.Series(pcts_unique, dtype=CACHE_SCHEMA["unique_percentage"]),
            })

            # Now concatenate and save
            cache_df = pl.concat([cache_df, new_rows_df])
            cache_df.write_parquet(cache_path)
            print("Cache saved to disk.")
        # --- END OF CORRECTION ---

        if overall_max_unique_rows == total_rows:
            print("\nFound a combination that uniquely identifies all rows.")
            return overall_best_combination, overall_max_unique_rows

        current_stage_metrics.sort(key=lambda x: x[0], reverse=True)
        stage_results = current_stage_metrics[:top_n_to_carry_over]
        
        if not stage_results:
            print("\nStopping early as no further improvements were found in the top N candidates.")
            break

    return overall_best_combination, overall_max_unique_rows
