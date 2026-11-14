#!/bin/bash
#SBATCH --job-name=bwlab-prepare-half
#SBATCH --account=aip-sven
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
[[ $# -ge 3 && $# -le 4 ]] || { echo 'Usage: script repo base-config output [full|visible]' >&2; exit 2; }
repo=$(realpath "$1")
config=$(realpath "$2")
output=$3
region=${4:-full}
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
cd "$repo"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
git rev-parse HEAD
hostname
srun python -u scripts/prepare_point_followup.py \
  --base-config "$config" --output-dir "$output" \
  --train-fraction 0.5 --validation-interval-steps 5000 \
  --fence-loss-region "$region"
