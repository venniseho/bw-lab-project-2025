#!/bin/bash
#SBATCH --job-name=bwlab-point-val
#SBATCH --account=aip-sven
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
[[ $# -ge 3 ]] || { echo 'Usage: script repo config output [benchmark options]' >&2; exit 2; }
repo=$(realpath "$1")
config=$(realpath "$2")
output=$3
shift 3
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
cd "$repo"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
git rev-parse HEAD
hostname
srun python scripts/benchmark_point_validation.py --config "$config" --output "$output" --wandb-mode "${WANDB_MODE:-online}" "$@"
