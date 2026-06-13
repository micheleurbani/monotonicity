# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the optimization (from the notebooks/ directory)
python main.py [--seed 1] [--n-gen 200] [--pop-size 200] [--fod-size 2]

# Quick smoke-test run
python main.py --n-gen 10 --pop-size 50 --seed 42

# Install/manage dependencies (uv, Python 3.10)
uv add <package>
uv sync
```

## Architecture

This is a research script (`main.py`) that uses **NSGA-II** (via `pymoo`) to test whether entropy/uncertainty measures for Dempster-Shafer belief functions are **monotone**: if bel1 ≤ bel2 pointwise, does U(bel1) ≤ U(bel2)?

**Key dependency:** `evidence_theory` package lives one level up (`../evidence_theory/core.py`). `main.py` adds `..` to `sys.path` to import it. `core.py` (~2000 lines) contains all entropy measure implementations as standalone functions that accept a pandas DataFrame.

**Optimization problem (`BeliefFunctionOptimization`):**
- Variables: belief values for two belief functions bel1 and bel2 over the powerset of a Frame of Discernment (FOD) of size `--fod-size`
- 3 objectives: maximize entropy difference (f1 = U(bel2)−U(bel1)), maximize belief difference (f2), minimize focal element count (f3)
- Constraints enforce valid belief functions: bel(∅)=0, bel(X)=1, superadditivity (disjoint-subset rule), non-negative masses, and bel1 ≤ bel2

**Repair operator (`BeliefRepair`):** Iteratively projects populations onto the feasible region (boundary fix → disjoint-subset fix → singleton sum → belief sum → monotonicity clamp → quantize to 2 decimals).

**Entropy measures to test** are listed at the top of `main.py` as `entropy_measures`. Comment/uncomment entries to change which measures run. Once a measure produces a monotonicity violation it is removed from `entropy_measures_to_check` and not retested at larger FOD sizes.

## Outputs

Results land in `results/` (gitignored). Run ID format: `{n_gen}gen_{pop_size}pop_seed{seed}_{YYYYMMDD}`.

| File | Contents |
|---|---|
| `results/all_results_fod{N}_{run_id}.csv` | Per-solution detail for all measures |
| `results/measures_results_fod{N}_{run_id}.csv` | Pareto front f1/f2/f3 for all measures |
| `results/monotonicity_recap_fod{N}_{run_id}.csv` | Y/N monotonicity summary per measure |
| `results/{measure_name}/hv_convergence_*_{run_id}.csv` | Hypervolume per generation |
| `results/{measure_name}/pareto_plot_*_{run_id}.pdf` | Pareto front scatter plot |
