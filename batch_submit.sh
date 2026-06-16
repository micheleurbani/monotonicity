#!/bin/bash

# Batch submission script for 30 simulations
# Parameters: FOD=4, n-gen=500, pop-size=300

FOD_SIZE=4
N_GEN=500
POP_SIZE=300
NUM_JOBS=30

echo "Submitting $NUM_JOBS jobs with FOD=$FOD_SIZE, n-gen=$N_GEN, pop-size=$POP_SIZE"
echo "Seeds: 1 to $NUM_JOBS"
echo ""

for seed in $(seq 1 $NUM_JOBS); do
    job_id=$(./submit_job.sh --fod-size $FOD_SIZE --seed $seed --n-gen $N_GEN --pop-size $POP_SIZE)
    echo "Submitted seed $seed: $job_id"
done

echo ""
echo "All $NUM_JOBS jobs submitted!"
echo "Monitor with: squeue -u murbani"
