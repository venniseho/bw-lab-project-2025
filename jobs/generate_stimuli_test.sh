#!/bin/bash
# --------------------------------------------------------------
# SLURM JOB: Stimuli Generation Test
# --------------------------------------------------------------
#SBATCH --partition=hovennis
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=generate_stimuli_test
#SBATCH --output=jobs/logs/generate_stimuli_test_%j.log

# 1. Load your environment 
source venv/bin/activate

# 2. Define Paths
COCO_ANN="COCO/annotations/instances_val2014.json"
COCO_IMG="COCO/val2014"
OUT_ROOT="tests/generate_stimuli_test"

# 3. Run the generation
# Note: We use 200 noise segments as a standard baseline
python3 stimuli_generation/make_stimuli.py \
    --coco_ann "$COCO_ANN" \
    --coco_imgdir "$COCO_IMG" \
    --out_root "$OUT_ROOT" \
    --limit 20 \
    --noise_count 200

echo "Stimuli generation complete. Check $OUT_ROOT$ for outputs."