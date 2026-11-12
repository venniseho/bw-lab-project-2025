# Mask-decoder-only comparison

`mask_decoder_only` updates only parameters under
`inst_interactive_predictor.model.sam_mask_decoder`, including its feature
projections and quality head. The image encoder and prompt encoder are frozen
and remain in evaluation mode. The image encoder runs without autograd; decoder
feature projections retain gradients. The first-update audit rejects trainable
parameters or gradients outside the mask decoder and verifies a decoder weight
update. Trainable parameter counts are printed and recorded in metadata/W&B.

This is stricter than the existing `frozen_encoder` mode, which trains both the
prompt encoder and mask decoder. Existing modes and completed runs are unchanged.

Use the same original pretrained checkpoint, image order, object selection,
transforms, AdamW settings, loss, one-epoch schedule and validation set as the
completed full point-path runs. Do not resume their trained checkpoints. This
isolates which parameters may change, rather than giving one scope a fine-tuned
initialization. Validation at step zero should closely match its full-run
baseline; compare the recorded numbers before interpreting the final difference.

## Submit all four on Vulcan

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin main
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
export WANDB_ENTITY=venniseho14-university-of-toronto
export WANDB_PROJECT=SAM-finetuning
export WANDB_MODE=online

submit_decoder() {
  local label=$1
  local source_config=$2
  local root
  local job
  root=$(mktemp -d "/scratch/vho/sam3-whole-scene/decoder-only-${label}.XXXXXX")
  cp "$source_config" "$root/config.json" || return
  job=$(sbatch --parsable --time=24:00:00 \
    --job-name="bwlab-dec-$label" \
    --output="$root/slurm-%j.out" \
    scripts/vulcan_point_training.sbatch "$PWD" \
    "$root/config.json" mask_decoder_only "$root/training-decoder-only-$label") || return
  echo "$label: job=$job results=$root"
}

submit_decoder fragmentation /scratch/vho/sam3-whole-scene/fragmentation-ahead.SQFHHT/prepared/config.json
submit_decoder fence-low /scratch/vho/sam3-whole-scene/fence-low.A5hku1/config.json
submit_decoder fence-medium /scratch/vho/sam3-whole-scene/medium-fence.AZuiTE/config.json
submit_decoder fence-high /scratch/vho/sam3-whole-scene/fence-high.BhsrQl/config.json
squeue --me
```

Each job requests one GPU, 12 CPUs, 64 GB CPU RAM and 8 loader workers. Four jobs
may run concurrently if resources are available. No extra COCO download is
needed; the configs retain the prepared annotation paths on scratch.

Each run uses 117,266 updates, batch size 1, 1008×1008 preprocessing; baseline
validation plus checks every 15,000 updates and at the end on 4,952 targets.
Checkpoints save every 1,000 updates and after validation, with best/latest kept
separately. CSVs, previews and online W&B tracking work as before. Only the
trainable scope changes. Actual runtime and memory must be measured for this
scope rather than copied from full-path training. Existing `--resume` support
requires the same scope and prepared config.

CPU tests exercise backward/update/checkpoint behavior in all three scopes,
verify frozen weights remain unchanged and gradient-free, preserve decoder
projection gradients, and reject accidental prompt-encoder training. Native
CUDA behavior still needs confirmation from the submitted job audits and logs.
