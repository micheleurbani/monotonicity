import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

results_dir = Path("results")
measure_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])

for measure_dir in measure_dirs:
    measure_name = measure_dir.name

    # Find all hv_convergence files for this measure
    hv_files = sorted(measure_dir.glob("hv_convergence_*.csv"))

    if not hv_files:
        print(f"No hv_convergence files found for {measure_name}")
        continue

    # Load and combine all files, extracting FOD from filename
    dfs = []
    for file in hv_files:
        df = pd.read_csv(file)
        # Parse FOD from filename: fod2, fod3, etc.
        match = re.search(r'fod(\d+)', file.name)
        if match:
            fod = int(match.group(1))
        else:
            fod = None
        df['fod'] = fod
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Get unique FOD values and sort them
    fod_values = sorted([fod for fod in combined['fod'].unique() if fod is not None])

    if not fod_values:
        print(f"No FOD values found for {measure_name}")
        continue

    # Create one plot per FOD
    for fod in fod_values:
        fod_data = combined[combined['fod'] == fod]

        # Group by generation and calculate mean and std
        grouped = fod_data.groupby("generation")["hypervolume"].agg(["mean", "std"])

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        generations = grouped.index
        means = grouped["mean"].values
        stds = grouped["std"].values

        # Plot mean line
        ax.plot(generations, means, linewidth=2, color="steelblue", label="Mean HV")

        # Plot standard deviation band
        ax.fill_between(
            generations,
            means - stds,
            means + stds,
            alpha=0.3,
            color="steelblue",
            label="±1 Std Dev"
        )

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Hypervolume", fontsize=12)
        ax.set_title(f"Hypervolume Convergence: {measure_name} (FOD={fod})", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_file = results_dir / measure_name / f"hv_convergence_plot_{measure_name}_fod{fod}.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_file}")

        plt.close()

print("\nAll convergence plots generated!")
