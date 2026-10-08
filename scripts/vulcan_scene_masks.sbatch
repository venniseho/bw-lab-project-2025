#!/bin/bash
#SBATCH --job-name=sam3-scene
#SBATCH --account=aip-sven
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/vho/sam3-whole-scene/logs/%x-%j.out
#SBATCH --error=/scratch/vho/sam3-whole-scene/logs/%x-%j.err
set -euo pipefail
# Arguments: repository, prepared native-SAM3 venv, native checkpoint, image directory.
if [[ $# -lt 4 ]]; then
    echo 'Usage: sbatch ... repository venv sam3.pt image_directory [extra Python options]' >&2
    exit 2
fi
repo=$(realpath "$1")
venv=$(realpath "$2")
checkpoint=$(realpath "$3")
images=$(realpath "$4")
shift 4
[[ -f "$venv/bin/activate" && -f "$checkpoint" && -d "$images" ]]
source "$venv/bin/activate"
cd "$repo"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
output="/scratch/vho/sam3-whole-scene/results/job_${SLURM_JOB_ID}"
mkdir -p "$(dirname "$output")"
srun python scripts/sam3_scene_masks.py --image-dir "$images" \
    --checkpoint "$checkpoint" --output-dir "$output" "$@"
