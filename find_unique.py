import polars as pl
import itertools
from typing import List, Tuple, Dict

def find_most_unique_column_combination_progressive(
    df: pl.DataFrame,
    top_n_to_carry_over: int = 5
) -> Tuple[List[str], int]:
    """
    Identifies the unique combination of columns that makes the DataFrame most unique,
    using a progressive filtering approach.

    Args:
        df (pl.DataFrame): The input Polars DataFrame.
        top_n_to_carry_over (int): The number of top-performing combinations
                                   from the previous stage to carry over
                                   as a requirement for the next stage.

    Returns:
        Tuple[List[str], int]: A tuple containing:
            - List[str]: The list of column names that form the most unique combination.
            - int: The number of unique rows for that combination.
    """

    total_rows = df.height
    all_columns = df.columns
    
    best_combination_overall = []
    max_unique_rows_overall = 0

    # Stores combinations and their unique counts for the current iteration
    # { frozenset(col_names): unique_count }
    current_stage_results: Dict[frozenset, int] = {} 

    # Stores the top N combinations from the previous stage to filter the current stage
    # This will be a list of frozensets
    previous_stage_top_combinations: List[frozenset] = []

    print(f"Starting unique key search for {len(all_columns)} columns, total rows: {total_rows}")

    for r in range(1, len(all_columns) + 1):
        print(f"\n--- Checking combinations of size {r} ---")
        current_stage_results.clear() # Reset for the new stage

        # Determine the pool of columns to consider for this stage's combinations
        # If it's the first stage (r=1), use all columns.
        # Otherwise, use columns present in the top combinations from the previous stage.
        pool_for_next_combinations = set()
        if r == 1:
            pool_for_next_combinations = set(all_columns)
        elif previous_stage_top_combinations:
            for combo_set in previous_stage_top_combinations:
                pool_for_next_combinations.update(combo_set)
        
        # If the pool is empty (e.g., no top combos from previous stage), break
        if not pool_for_next_combinations and r > 1:
            print(f"No suitable columns from previous stage to form combinations of size {r}. Stopping.")
            break

        # Generate combinations for the current stage
        # We need to iterate over all_columns, not just pool_for_next_combinations
        # to ensure new columns can be introduced, but then filter based on the previous stage's best.
        
        # For r=1, we check all single columns.
        # For r>1, we generate combinations from `all_columns` and then apply the filter.
        
        # Iterate over ALL possible combinations of size `r` from `all_columns`
        # Then, apply the filtering logic.
        
        candidate_combinations = itertools.combinations(all_columns, r)
        
        # Filtering: For r > 1, ensure at least one column from a previous top combo is present
        # This is where the progressive pruning logic is applied.
        filtered_candidate_combinations = []
        if r == 1:
            filtered_candidate_combinations = [combo for combo in candidate_combinations]
        else:
            for combo in candidate_combinations:
                combo_set = frozenset(combo)
                # Check if this combination contains AT LEAST ONE column from ANY of the
                # top_n_to_carry_over combinations from the previous stage.
                # This is a less strict requirement than requiring the *entire* previous top combo.
                # A more strict requirement (e.g., must contain a *specific* previous top combo)
                # might prune too aggressively.
                
                # Check if any part of the current combo overlaps with ANY previous_stage_top_combination
                if any(col in prev_combo_set for col in combo_set for prev_combo_set in previous_stage_top_combinations):
                     filtered_candidate_combinations.append(combo)

        # If no candidates after filtering, we can stop
        if not filtered_candidate_combinations and r > 1:
            print(f"No filtered candidate combinations for size {r}. Stopping.")
            break

        current_stage_count = 0
        for combo_names in filtered_candidate_combinations:
            current_stage_count += 1
            combo_names_list = list(combo_names)
            
            # Efficiently calculate n_unique
            num_unique = df.select(pl.struct(combo_names_list).n_unique()).item()
            
            current_stage_results[frozenset(combo_names_list)] = num_unique

            if num_unique > max_unique_rows_overall:
                max_unique_rows_overall = num_unique
                best_combination_overall = combo_names_list
                print(f"  New best found: {best_combination_overall} -> {max_unique_rows_overall} unique rows")

            if max_unique_rows_overall == total_rows:
                print(f"  Found a perfect unique key: {best_combination_overall}. Stopping early.")
                return best_combination_overall, max_unique_rows_overall
        
        print(f"  Processed {current_stage_count} candidate combinations for size {r}.")

        # Identify top N for the next stage based on current stage results
        sorted_current_stage_results = sorted(
            current_stage_results.items(), key=lambda item: item[1], reverse=True
        )
        
        previous_stage_top_combinations = [
            combo_set for combo_set, _ in sorted_current_stage_results[:top_n_to_carry_over]
        ]
        
        if previous_stage_top_combinations:
            print(f"  Top {len(previous_stage_top_combinations)} combinations for next stage from size {r}:")
            for combo_set in previous_stage_top_combinations:
                print(f"    - {list(combo_set)} (Unique: {current_stage_results[combo_set]})")
        else:
            print(f"  No top combinations to carry over for size {r}. Next stage might be limited.")
            # If no top combinations, the loop will likely break in the next iteration due to empty pool

    return best_combination_overall, max_unique_rows_overall

# --- Example Usage ---
if __name__ == "__main__":
    # Create a sample Polars DataFrame with 25 columns and some redundancy
    data = {f"col_{i}": [j % (i + 1) for j in range(1000)] for i in range(1, 26)}
    # Add a column that makes some combinations more unique
    data["id_base"] = [i for i in range(1000)]
    data["name_base"] = [f"Name_{i}" for i in range(1000)]
    
    # Introduce some non-unique elements
    data["col_1"][0] = data["col_1"][1] # Duplicate first two values
    data["name_base"][500] = data["name_base"][501] # Duplicate
    
    # Create a scenario where id_base + name_base is unique, but individually not.
    # And where other combinations might be "almost" unique.
    data["combined_key_part1"] = [i // 10 for i in range(1000)]
    data["combined_key_part2"] = [i % 10 for i in range(1000)]

    df_large = pl.DataFrame(data)

    print("Original DataFrame (partial view):")
    print(df_large.head())
    print(f"Total rows: {df_large.height}\n")

    # Experiment with different top_n_to_carry_over values
    # A smaller number prunes more aggressively, but might miss the true key if it's
    # built from "less unique" initial columns.
    # A larger number retains more options, but reduces the pruning benefit.

    # Example 1: Strict pruning (top 2)
    print("\n--- Running with top_n_to_carry_over = 2 ---")
    most_unique_cols_1, unique_count_1 = find_most_unique_column_combination_progressive(
        df_large, top_n_to_carry_over=2
    )
    print(f"\nResult (top_n=2):")
    print(f"Most unique column combination: {most_unique_cols_1}")
    print(f"Number of unique rows with this combination: {unique_count_1}")

    # Example 2: More relaxed pruning (top 5)
    print("\n--- Running with top_n_to_carry_over = 5 ---")
    most_unique_cols_2, unique_count_2 = find_most_unique_column_combination_progressive(
        df_large, top_n_to_carry_over=5
    )
    print(f"\nResult (top_n=5):")
    print(f"Most unique column combination: {most_unique_cols_2}")
    print(f"Number of unique rows with this combination: {unique_count_2}")
    
    # Example 3: Finding a true unique key
    df_true_unique = pl.DataFrame({
        "A": [1,2,3,4,5],
        "B": ["x","y","z","a","b"],
        "C": [10,11,12,13,14]
    })
    print("\n--- Running on DataFrame with true unique key (top_n_to_carry_over = 3) ---")
    most_unique_cols_true, unique_count_true = find_most_unique_column_combination_progressive(
        df_true_unique, top_n_to_carry_over=3
    )
    print(f"\nResult (True Unique Key):")
    print(f"Most unique column combination: {most_unique_cols_true}")
    print(f"Number of unique rows with this combination: {unique_count_true}")
