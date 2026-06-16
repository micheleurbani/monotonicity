# /home/michele/evidence-theory/NSGA-II/evidence-theory-monotonicity/notebooks/summarize_monotonicity.py
import pandas as pd
from pathlib import Path
import re

def summarize_monotonicity_violations(results_root="results"):
    """
    Reads all monotonicity_recap_fod*.csv files, groups them by FOD size,
    and summarizes any entropy measures that show monotonicity inconsistencies.
    """
    results_path = Path(results_root)
    if not results_path.is_dir():
        print(f"Error: Results directory '{results_root}' not found.")
        return

    all_recap_files = sorted(results_path.glob("monotonicity_recap_fod*.csv"))

    if not all_recap_files:
        print(f"No 'monotonicity_recap_fod*.csv' files found in '{results_root}'.")
        return

    all_data = []
    for file_path in all_recap_files:
        try:
            df = pd.read_csv(file_path)
            # Extract FOD size from filename using regex
            match = re.search(r'fod(\d+)', file_path.name)
            if match:
                fod_size = int(match.group(1))
                df['FOD_Size'] = fod_size
                all_data.append(df)
            else:
                print(f"Warning: Could not extract FOD size from filename: {file_path.name}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not all_data:
        print("No valid monotonicity recap data found to process.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    print("\n--- Monotonicity Inconsistency Summary ---")
    
    # Group by FOD_Size and then check for violations
    for fod_size, fod_group in combined_df.groupby('FOD_Size'):
        print(f"\nFOD Size: {fod_size}")
        violations_found = False
        for measure_name, measure_group in fod_group.groupby('Entropy Measure'):
            if 'Is Monotone' in measure_group.columns and (measure_group['Is Monotone'] == 'N').any():
                print(f"  - Measure '{measure_name}' shows MONOTONICITY VIOLATIONS.")
                violations_found = True
        
        if not violations_found:
            print("  All tested measures showed monotonicity for this FOD size.")

    print("\n--- Detailed Monotonicity Violations ---")
    violations_df = combined_df[combined_df['Is Monotone'] == 'N']

    if violations_df.empty:
        print("No monotonicity violations found across all tested FOD sizes and measures.")
    else:
        # Sort for better readability
        violations_df = violations_df.sort_values(by=['FOD_Size', 'Entropy Measure'])
        
        for (fod_size, measure_name), group in violations_df.groupby(['FOD_Size', 'Entropy Measure']):
            print(f"\nFOD Size: {fod_size}, Measure: {measure_name}")
            print("  Violations found:")
            # Display relevant columns for violations
            print(group[['Label', 'Uncertainty1', 'Uncertainty2', 'F_Value', 'Is Monotone']].to_string(index=False))
            print("-" * 40)


if __name__ == "__main__":
    # Assuming the script is run from the notebooks/ directory
    summarize_monotonicity_violations()
