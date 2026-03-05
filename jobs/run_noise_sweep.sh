#!/bin/bash
#SBATCH --partition=hovennis
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=noise_sweep
#SBATCH --output=jobs/logs/noise_sweep_%j.log

source venv/bin/activate

# Paths
COCO_ANN="COCO/annotations/instances_val2014.json"
COCO_IMG="COCO/val2014"
STIMULI_DIR="data/noise_sweep_stimuli"

MODELS=("sam1" "sam3")

for MODEL in "${MODELS[@]}"; do
    echo "Starting sweep for model: $MODEL"
    
    if [ "$MODEL" == "sam3" ]; then
        CKPT="facebook/sam3"
    else
        CKPT="checkpoints/sam_vit_h_4b8939.pth"
    fi
    
    python3 scripts/run_noise_experiment.py \
        --model_type "$MODEL" \
        --sam_ckpt "$CKPT" \
        --coco_ann "$COCO_ANN" \
        --coco_imgdir "$COCO_IMG" \
        --noise_levels 0 25 50 75 100 200 300 400 600 800 \
        --limit 250 \
        --stimuli_root "$STIMULI_DIR" \
        --prompt_mode "centroid"
done