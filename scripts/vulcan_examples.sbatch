#!/bin/bash
#SBATCH --job-name=bwlab-examples
#SBATCH --account=aip-sven
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: script repo full-run-root new-output-directory' >&2; exit 2; }
repo=$(realpath "$1")
runs=$(realpath "$2")
output=$(realpath -m "$3")
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/env/bin/activate
cd "$repo"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
git rev-parse HEAD
hostname
srun python scripts/export_condition_examples.py \
    --fence-run "$runs/run-000" --fragmentation-run "$runs/run-001" \
    --count 3 --output "$output"
