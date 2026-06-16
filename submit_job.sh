#!/bin/bash

# SLURM job submission script for main.py
# Usage: ./submit_job.sh --fod-size 3 [--seed 1] [--n-gen 200] [--pop-size 200]

# Default values
SEED=1
N_GEN=200
POP_SIZE=200
FOD_SIZE=2

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fod-size)
            FOD_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --n-gen)
            N_GEN="$2"
            shift 2
            ;;
        --pop-size)
            POP_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --fod-size <size> [--seed <seed>] [--n-gen <gen>] [--pop-size <size>]"
            exit 1
            ;;
    esac
done

# Calculate time limit: 2 * FOD size in hours
TIME_LIMIT=$((2 * FOD_SIZE))

# Create the temporary SLURM script
SLURM_SCRIPT=$(mktemp)
cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=${TIME_LIMIT}:00:00
#SBATCH --job-name=monotonicity_fod${FOD_SIZE}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=high_cache

cd /home/murbani/monotonicity
uv sync
uv run python main.py --seed $SEED --n-gen $N_GEN --pop-size $POP_SIZE --fod-size $FOD_SIZE
EOF

# Submit the job
sbatch "$SLURM_SCRIPT"

# Clean up temporary script
rm "$SLURM_SCRIPT"
