#!/bin/bash
#SBATCH --job-name=bwlab-native-train
#SBATCH --account=aip-sven
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: script bwlab-repo training-config.json' >&2; exit 2; }
repo=$(realpath "$1")
config=$(realpath "$2")
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
cd "$repo"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-offline}
git rev-parse HEAD
hostname
srun python -m experiments.sam3_finetuning.launch --config "$config"
