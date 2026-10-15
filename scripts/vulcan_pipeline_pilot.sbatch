#!/bin/bash
#SBATCH --job-name=bwlab-pilot10
#SBATCH --account=aip-sven
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: sbatch [options] script repository pilot_directory' >&2; exit 2; }
repo=$(realpath "$1")
workspace=$(realpath "$2")
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/env/bin/activate
cd "$repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
nvidia-smi
git rev-parse HEAD
for condition in fence fragmentation; do
    echo "Starting $condition at $(date --iso-8601=seconds)"
    srun python -m experiments.sam3_fragmentation.experiment run \
        --config "$workspace/$condition.json" --output "$workspace/${condition}_sam3"
done
echo 'Both pilot arms completed. See timing.json and predictions/timings.csv for each arm.'
