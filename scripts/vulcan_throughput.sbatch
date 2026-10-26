#!/bin/bash
#SBATCH --job-name=bwlab-throughput
#SBATCH --account=aip-sven
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
[[ $# -ge 4 && $# -le 6 ]] || { echo 'Usage: script equivalence|diagnose|benchmark repo configs output [fence|fragmentation|both] [max-inputs]' >&2; exit 2; }
mode=$1
arm=${5:-both}
max_inputs=${6:-64}
[[ "$max_inputs" =~ ^[1-9][0-9]*$ ]] || { echo "max-inputs must be positive" >&2; exit 2; }
case "$arm" in
    both) conditions=(fence fragmentation) ;;
    fence|fragmentation) conditions=("$arm") ;;
    *) echo "Invalid condition: $arm" >&2; exit 2 ;;
esac
[[ "$mode" == equivalence || "$mode" == diagnose || "$mode" == benchmark ]] || { echo 'Invalid mode' >&2; exit 2; }
repo=$(realpath "$2")
configs=$(realpath "$3")
output=$(realpath -m "$4")
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/env/bin/activate
cd "$repo"
mkdir -p "$output"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
git rev-parse HEAD
hostname
srun python -c "import cv2, torch; assert torch.cuda.is_available(), 'No CUDA GPU allocated'; print('OpenCV:', cv2.__version__); print('GPU:', torch.cuda.get_device_name(0))"
failed=0
for condition in "${conditions[@]}"; do
    if [[ "$mode" == diagnose ]]; then
        # Complete both small diagnostic arms even if strict equivalence fails.
        srun python scripts/check_batch_equivalence.py \
            --config "$configs/$condition.json" --count 4 --diagnose \
            --output "$output/diagnostic-$condition.csv" || failed=1
        continue
    fi
    if [[ "$mode" == equivalence ]]; then
        srun python scripts/check_batch_equivalence.py \
            --config "$configs/$condition.json" --count 4 \
            --output "$output/equivalence-$condition.csv"
    else
        srun python scripts/benchmark_pipeline.py \
            --config "$configs/$condition.json" --max-inputs "$max_inputs" --workers 2 \
            --batch-sizes 1 2 4 8 16 32 64 --output "$output/$condition"
    fi
done
exit "$failed"
