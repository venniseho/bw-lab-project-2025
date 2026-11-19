#!/bin/bash
#SBATCH --job-name=bwlab-transfer
#SBATCH --account=aip-sven
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
[[ $# -ge 4 ]] || { echo 'Usage: script repo source-run output target-config [target-config ...]' >&2; exit 2; }
repo=$(realpath "$1")
source_run=$(realpath "$2")
output=$3
shift 3
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
cd "$repo"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
git rev-parse HEAD
hostname
srun python scripts/evaluate_point_transfer.py \
  --source-run "$source_run" --output "$output" --target-config "$@" \
  --wandb-mode "${WANDB_MODE:-online}"
