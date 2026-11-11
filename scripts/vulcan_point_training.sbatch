#!/bin/bash
#SBATCH --job-name=bwlab-point-train
#SBATCH --account=aip-sven
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@300
set -euo pipefail
[[ $# -ge 4 ]] || { echo 'Usage: script repo config scope output [max-steps | --resume checkpoint]' >&2; exit 2; }
repo=$(realpath "$1")
config=$(realpath "$2")
scope=$3
output=$4
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
cd "$repo"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-online}
git rev-parse HEAD
hostname
extra=()
shift 4
if [[ $# -eq 1 && $1 =~ ^[0-9]+$ ]]; then
  extra=(--max-steps "$1")
else
  extra=("$@")
fi
srun python -m experiments.sam3_point_finetuning.experiment \
  --config "$config" --scope "$scope" --output "$output" \
  --wandb-mode "$WANDB_MODE" "${extra[@]}"
