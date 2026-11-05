#!/bin/bash
#SBATCH --job-name=bwlab-eval-monitor
#SBATCH --account=aip-sven
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: script repo config-directory result-directory' >&2; exit 2; }
repo=$(realpath "$1")
configs=$(realpath "$2")
results=$(realpath -m "$3")
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
cd "$repo"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-online}
mkdir -p "$results"
git rev-parse HEAD
hostname
for condition in fence fragmentation; do
    srun python -m experiments.sam3_fragmentation.experiment run \
      --config "$configs/$condition.json" --output "$results/$condition" \
      --wandb-mode "$WANDB_MODE"
done
