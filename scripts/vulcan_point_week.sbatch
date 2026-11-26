#!/bin/bash
#SBATCH --job-name=bwlab-prepare-week
#SBATCH --account=aip-sven
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: script repo existing-plan-root new-output-root' >&2; exit 2; }
repo=$(realpath "$1")
previous=$(realpath "$2")
output=$(realpath "$3")
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
cd "$repo"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_RUN_GROUP=$(basename "$output")
git rev-parse HEAD
hostname
srun python scripts/prepare_point_week.py \
  --base-config "$previous/prepared/fence-medium/config.json" \
  --output-dir "$output/prepared"
python scripts/submit_point_week.py \
  --plan "$output/prepared/jobs.json" --repo "$repo" --output "$output" \
  --replay-source "$previous/training-fragmentation" \
  --replay-config "$previous/prepared/fragmentation/config.json"
