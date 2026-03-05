#!/bin/bash
#SBATCH --partition=hovennis
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00  
#SBATCH --job-name=generate_noise_sweep_stimuli_test
#SBATCH --output=jobs/logs/generate_noise_sweep_stimuli_test_%j.log

source venv/bin/activate

# --- Configuration ---
COCO_ANN="COCO/annotations/instances_val2014.json"
COCO_IMG="COCO/val2014"
STIM_ROOT="tests/noise_sweep_test/stimuli"
LIMIT=5 
NOISE_LEVELS=(0 25 50 75 100 200 400 800)

# Define the Seed path (Noise 0)
SEED_MANIFEST="$STIM_ROOT/n0/indexes/manifest.jsonl"

echo "Starting Multi-Level Noise Stimuli Generation..."
echo "Target Root: $STIM_ROOT"

for N in "${NOISE_LEVELS[@]}"; do
    echo "------------------------------------------------"
    echo "GENERATING: Noise Level $N"
    echo "------------------------------------------------"
    
    OUT_DIR="$STIM_ROOT/n$N"
    
    # Construct the command
    CMD="python3 stimuli_generation/make_stimuli.py \
        --coco_ann $COCO_ANN \
        --coco_imgdir $COCO_IMG \
        --out_root $OUT_DIR \
        --noise_count $N"

    # If this is the FIRST level (0), use the limit to pick images.
    # Otherwise, tell the script to use the seed manifest.
    if [ "$N" -eq 0 ]; then
        CMD="$CMD --limit $LIMIT"
    else
        CMD="$CMD --from_manifest $SEED_MANIFEST"
    fi

    eval $CMD
done

echo "Sweep generation complete. Folders created in $STIM_ROOT"