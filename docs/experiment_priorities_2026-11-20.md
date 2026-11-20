# Experiment priorities — 20 November 2026

## Recommended order

1. **Clean reference and retained clean ability.** Train one clean decoder-only
   control using the current fixed LR 1e-5, batch size 1 and one-epoch half-data
   budget. Its step-0 evaluation gives pretrained clean IoU. In parallel, evaluate
   the seven saved fixed-rate/visible-loss final checkpoints on the same clean
   validation cohort. This tests loss of clean ability; the existing transfer
   results alone do not establish clean-image forgetting. Scheduled-LR and earlier
   full point-path models are separate optional extensions.
2. **Small fixed-LR pilot before full mixture training.** Compare 1e-5, 3e-5 and
   1e-4 on a fixed small training subset/order and common validation cohort, with
   the same number of image presentations. Medium fence is a useful starting
   condition because its earlier performance plateaued. Do not assume one pilot
   establishes the best LR for every condition. Keep the same loss region within
   this comparison. LR schedule tuning is not the same as fixed-LR tuning.
3. **Gradient accumulation pilot.** Not currently implemented: `train_step`
   clears gradients and calls `optimizer.step()` on every image. Implement and
   verify accumulation separately, including averaged losses, partial final
   windows, checkpoint/resume and interrupted windows. With batch size 1 and
   accumulation 16, the effective batch is 16 images; there are roughly 16 times
   fewer optimizer updates for the same image budget. It is not guaranteed to
   improve speed or accuracy. Track images seen and optimizer updates separately;
   define validation/checkpoint intervals in explicit units. Do not automatically
   multiply LR by 16. Compare accumulation 1 versus 16 with a fixed image budget,
   then recheck LR if it changes the outcome.
4. **Mixed training to test retention and transfer.** Start with clean+fragmentation
   because fragmentation-to-fence collapse is the clearest current failure.
   Then clean+medium-fence and clean+fragmentation+medium-fence. Sample conditions
   with explicit balanced probabilities (1/2 each for two-way, 1/3 for three-way)
   while keeping the total image-presentation budget fixed. Do not give mixtures
   two or three times more training by materializing every variant per source.
   Train only the decoder initially; preserve the existing same-condition runs.
   Use common eligible source targets across compared mixtures (drop targets
   hidden by the chosen fence in all matched controls), or explicitly report the
   cohort mismatch. Evaluate every model on clean, fragmented and all fence levels,
   reporting both IoUs. Start with the full-region objective to match the original
   controls; a mixed visible-loss experiment should be a separate comparison.
5. **Backbones, TokenCut/mlvbench, MobileSAM and resolution.** A useful next workstream
   after the retention/mixing result, not another variable in this week's mixture
   comparison. First define a reproducible pretrained baseline, compatible prompts
   and target metrics. Lower resolution is a separate intervention; keep 1008
   for current controls. Comparing different encoders/decoders needs compatible
   feature interfaces and cannot be treated as a drop-in change without checking.

If time is tight, finish clean evaluation, the clean control and a small
clean+fragmentation pilot before the larger hyperparameter grid. If multiple
hours-long mixture runs are planned, the small LR pilot is worth doing first.
Do not queue the whole LR × accumulation × mixture grid at once.

## Start the clean control on Vulcan

This prepares a config on CPU using the exact saved fragmentation manifests and
training order: 58,633 training images, 4,952 validation targets, one target per
image. Clean means identity transform; it does not remove natural occlusion in
COCO. Training remains mask-decoder-only, LR 1e-5, native 1008 resolution,
validation at baseline/every 5,000/final and periodic checkpointing every 1,000.
Annotation paths still reference the original scratch prepared manifests.

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin main
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
export WANDB_ENTITY=venniseho14-university-of-toronto
export WANDB_PROJECT=SAM-finetuning
export WANDB_MODE=online

PLAN_ROOT=/scratch/vho/sam3-whole-scene/point-staged.dUreDG
CLEAN_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/clean-control.XXXXXX)
echo "Keep this path: $CLEAN_ROOT"

python scripts/prepare_clean_point.py \
  --base-config "$PLAN_ROOT/prepared/fragmentation/config.json" \
  --output-dir "$CLEAN_ROOT/prepared/clean"

CLEAN_ID=$(sbatch --parsable --time=05:00:00 \
  --job-name=bwlab-clean-control \
  --output="$CLEAN_ROOT/slurm-%j.out" \
  scripts/vulcan_point_training.sbatch "$PWD" \
  "$CLEAN_ROOT/prepared/clean/config.json" \
  mask_decoder_only "$CLEAN_ROOT/training-clean")
echo "Clean training job: $CLEAN_ID"
```

The five-hour request provides headroom over prior roughly three-hour decoder-only
training runs. It is a limit, not an estimate or a requirement to use five hours.

## Evaluate existing fine-tuned models on original images

This can run concurrently with clean training: it needs only the clean config
and existing completed checkpoints. Seven jobs, no new training. Each reruns the
pretrained baseline on identical targets/prompts, then loads that source's final
checkpoint and evaluates clean images. One GPU/12 CPUs/64 GB per job, two-hour
limit; two validation passes were previously about ten minutes plus setup.

```bash
for CONDITION in fragmentation fence-low fence-medium fence-high \
  fence-low-visible-loss fence-medium-visible-loss fence-high-visible-loss; do
  JOB=$(sbatch --parsable \
    --job-name="bwlab-clean-eval-$CONDITION" \
    --output="$CLEAN_ROOT/$CONDITION-%j.out" \
    scripts/vulcan_point_transfer.sbatch "$PWD" \
    "$PLAN_ROOT/training-$CONDITION" "$CLEAN_ROOT/$CONDITION-to-clean" \
    "$CLEAN_ROOT/prepared/clean/config.json")
  printf '%s,%s\n' "$CONDITION" "$JOB" | tee -a "$CLEAN_ROOT/evaluation-jobs.csv"
done
squeue --me
```

Per evaluation job: `clean/summary.csv`, `clean/heldout.csv/jsonl`,
`clean/timing.csv`, `clean/previews/`, root `metadata.json` and `complete.json`.
For clean inputs, full/visible IoU coincide. These are still validation-set
measurements, not an independent test set. The existing transfer report script
expects the earlier nine-comparison layout and should not be pointed at this
new clean-evaluation layout without adaptation.
