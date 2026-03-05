#!/bin/bash
#SBATCH --partition=hovennis
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00  # Short time for 20 images
#SBATCH --job-name=prompting_suite_test
#SBATCH --output=jobs/logs/prompting_suite_test_%j.log

source venv/bin/activate

# Points to your 20-image test directory
STIM_ROOT="tests/generate_stimuli_test"
MANIFEST="$STIM_ROOT/indexes/manifest.jsonl"

# Models and specific prompts for the mini-test
MODELS=("sam1" "sam3")
PROMPTS=("centroid" "3" "box")

for MODEL in "${MODELS[@]}"; do
    echo "------------------------------------------------"
    echo "RUNNING BATCH: $MODEL for all prompt modes"
    echo "------------------------------------------------"
    
    # We pass all modes here. The Python script handles the rest.
    python3 scripts/run_sam_batch.py \
        --out_root "tests/prompting_suite_test" \
        --sam_model_type "$MODEL" \
        --sam_ckpt "checkpoints/sam_vit_h_4b8939.pth" \
        --manifest "$MANIFEST" \
        --prompt_modes centroid 3 box
done

echo "Mini test complete. Check tests/prompting_suite_test/ for CSVs with box_area and gt_area."