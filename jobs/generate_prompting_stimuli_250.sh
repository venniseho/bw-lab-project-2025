#!/bin/bash
#SBATCH --partition=hovennis
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00  
#SBATCH --job-name=generate_prompting_stimuli_250
#SBATCH --output=jobs/logs/generate_prompting_stimuli_250_%j.log

source venv/bin/activate

# Define the NEW destination
OUT_ROOT="data/stimuli_prompting_250"

python3 stimuli_generation/make_stimuli.py \
    --coco_ann "COCO/annotations/instances_val2014.json" \
    --coco_imgdir "COCO/val2014" \
    --out_root "$OUT_ROOT" \
    --limit 250 \
    --noise_count 200

echo "250 Stimuli generated in $OUT_ROOT"