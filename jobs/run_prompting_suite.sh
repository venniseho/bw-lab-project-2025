#!/bin/bash
#SBATCH --partition=hovennis
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=prompting_suite
#SBATCH --output=jobs/logs/prompting_suite_%j.log

STIM_ROOT="data/stimuli_prompting_250"
MODELS=("sam1" "sam3")

# List all modes you want to test here
PROMPT_LIST="centroid box 1 2 3 5 7 10 15 20"

for MODEL in "${MODELS[@]}"; do
    echo "------------------------------------------------"
    echo "STARTING BATCH: $MODEL"
    echo "PROMPT MODES: $PROMPT_LIST"
    echo "------------------------------------------------"

    # We call the script ONCE per model. 
    # The Python script handles the sub-directories for each prompt.
    python3 scripts/run_sam_batch.py \
        --out_root "outputs/prompting_exp" \
        --sam_model_type "$MODEL" \
        --sam_ckpt "checkpoints/sam_vit_h_4b8939.pth" \
        --manifest "$STIM_ROOT/indexes/manifest.jsonl" \
        --prompt_modes $PROMPT_LIST
done