# Half-data point training with visible-region reporting

This protocol starts from the existing one-object-per-image manifests and
original pretrained checkpoint. It selects a seeded random half of all training
image IDs (58,633 of 117,266), rather than stopping halfway through the former
cache-dependent order. All four conditions start with the same selected IDs.
Each fence condition then excludes its fully hidden targets, keeping the same
target object rather than substituting another object or a hidden prompt.
Validation starts with all 4,952 selected val2017 objects and applies the same
visibility rule. Counts can differ between conditions; do not label every new
validation mean as n=4,952 without checking its saved count.

Preparation checks annotation masks and fence geometry on CPU. It does not
download images or run SAM. Each condition receives its own prepared annotations,
hashes, exact training order, config and `excluded_targets.csv`. The shared
`requested_training_ids.json` preserves the pre-exclusion half, and
`sample_counts.csv` reports candidates, exclusions and retained images/masks.

Validation runs before training, every 5,000 **successful optimizer updates**,
and at the end of one epoch. Final steps equal the retained training image count
and may be slightly below 58,633. The checkpoint interval remains 1,000 updates.
Input size remains 1008×1008 and training batch size remains 1.

## Full and visible measurements

Every prediction is still selected by the highest model quality score. The same
prediction is scored against the original COCO target (`full_iou`) and in the
region outside all fence bars (`visible_iou`). The latter removes both predicted
and target pixels under bars; false positives on visible background still count.
`selected_iou`/`mean_iou` remain aliases for full-target IoU for compatibility.
With no fence, full and visible measurements coincide.

Both losses are also recorded: `full_loss` and `visible_loss`, with corresponding
focal, Dice and quality-MSE components. Visible loss masks focal and Dice to
unobscured pixels and trains quality scores against IoU in that same region.
Bars contribute no pixel-level loss or quality-target overlap penalty. This
does not constrain the model's prediction behind bars.

`--fence-loss-region full` (default) uses the existing training objective and
records visible loss as a diagnostic. `--fence-loss-region visible` uses visible
loss for training fences. Fragmentation retains full-target supervision. Both
choices report all metrics to per-image CSV/JSONL, validation summaries and W&B.
The best checkpoint uses full IoU for full loss, or visible IoU for visible loss;
`best_checkpoint_metric` records which metric was used. Loss and protocol changes
are rejected when resuming an incompatible checkpoint. Start new runs instead.

## Check the previous four decoder-only runs

```bash
squeue --me
for ROOT in /scratch/vho/sam3-whole-scene/decoder-only-*; do
  [ -d "$ROOT" ] || continue
  echo "$ROOT"
  for RUN in "$ROOT"/training-decoder-only-*; do
    [ -d "$RUN" ] || continue
    cat "$RUN/summary.csv" 2>/dev/null
    if [ -f "$RUN/complete.json" ]; then
      cat "$RUN/complete.json"
    elif [ -f "$RUN/interrupted.json" ]; then
      cat "$RUN/interrupted.json"
    else
      tail -n 1 "$RUN/train.csv" 2>/dev/null
    fi
  done
done
```

Previous runs cannot gain additional 5,000-step validation checkpoints
retroactively. Their saved selected masks can support some post-hoc measurements,
but the new interval and smaller training budget require new training runs.

## Prepare the next four conditions on Vulcan

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin main
HALF_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/point-half.XXXXXX)
BASE_CONFIG=/scratch/vho/sam3-whole-scene/fragmentation-ahead.SQFHHT/prepared/config.json

# full = original training loss, plus visible loss logged as a diagnostic
# visible = train fence models with loss that ignores bars
PREP_ID=$(sbatch --parsable --output="$HALF_ROOT/preparation-%j.out" \
  scripts/vulcan_point_followup.sbatch "$PWD" "$BASE_CONFIG" \
  "$HALF_ROOT/prepared" full)
echo "Preparation job: $PREP_ID"
echo "Keep this root: $HALF_ROOT"
```

After preparation completes, inspect counts:

```bash
sacct -j "$PREP_ID" --format=JobID,State,Elapsed,ExitCode
cat "$HALF_ROOT/prepared/sample_counts.csv"
```

Only submit training after preparation succeeds. The following uses decoder-only
training; `full` can be substituted as the scope for a matched full-path control.
Do not treat comparisons with the previous 117,266-step runs as isolating only
the training scope: training budgets and visibility cohorts now differ.

```bash
export WANDB_ENTITY=venniseho14-university-of-toronto
export WANDB_PROJECT=SAM-finetuning
export WANDB_MODE=online
for CONDITION in fragmentation fence-low fence-medium fence-high; do
  JOB=$(sbatch --parsable --time=24:00:00 \
    --job-name="bwlab-half-$CONDITION" \
    --output="$HALF_ROOT/$CONDITION-%j.out" \
    scripts/vulcan_point_training.sbatch "$PWD" \
    "$HALF_ROOT/prepared/$CONDITION/config.json" \
    mask_decoder_only "$HALF_ROOT/training-half-$CONDITION")
  echo "$CONDITION: job=$JOB results=$HALF_ROOT/training-half-$CONDITION"
done
```

Each training job retains the standard one GPU, 12 CPUs, 64 GB CPU RAM, eight
workers and online W&B logging. Preparation is a separate CPU-only job. Keep
the original results; no existing job is cancelled or resubmitted by preparation.

## Seven-run comparison: both fence objectives

Use `both` as the preparation launcher's final argument to create four original
loss conditions and three additional fence conditions with visible-region loss.
Each fence pair shares the exact annotation paths, hashes, selected targets,
training order and exclusions. Only the optimization loss region and best-model
selection metric differ. Compare final checkpoints at the same update count to
isolate the loss effect; best checkpoints use full IoU for the original objective
and visible IoU for the visible objective, and are labelled accordingly.

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin main
PLAN_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/point-seven.XXXXXX)
BASE_CONFIG=/scratch/vho/sam3-whole-scene/fragmentation-ahead.SQFHHT/prepared/config.json
PREP_ID=$(sbatch --parsable --output="$PLAN_ROOT/preparation-%j.out" \
  scripts/vulcan_point_followup.sbatch "$PWD" "$BASE_CONFIG" \
  "$PLAN_ROOT/prepared" both)
echo "Preparation job: $PREP_ID"
echo "Keep this path: $PLAN_ROOT"
```

After preparation completes successfully:

```bash
cat "$PLAN_ROOT/prepared/sample_counts.csv"
export WANDB_ENTITY=venniseho14-university-of-toronto
export WANDB_PROJECT=SAM-finetuning
export WANDB_MODE=online
for CONDITION in fragmentation fence-low fence-medium fence-high \
  fence-low-visible-loss fence-medium-visible-loss fence-high-visible-loss; do
  JOB=$(sbatch --parsable --time=24:00:00 \
    --job-name="bwlab-half-$CONDITION" \
    --output="$PLAN_ROOT/$CONDITION-%j.out" \
    scripts/vulcan_point_training.sbatch "$PWD" \
    "$PLAN_ROOT/prepared/$CONDITION/config.json" mask_decoder_only \
    "$PLAN_ROOT/training-$CONDITION")
  echo "$CONDITION: job=$JOB results=$PLAN_ROOT/training-$CONDITION"
done
```

Later fence-to-fragmentation and fragmentation-to-fence evaluations reuse saved
model checkpoints; they do not require another training epoch. They should use
matched eligible targets when paired comparisons are intended. A clean-image
training control and learning-rate warmup/cosine comparisons remain separate
experiments; neither is added by this seven-run preparation.

## Updated staged plan: eight runs, then three

The `staged` preparation mode now creates all eleven configs in one CPU job.
Stage 1 contains fragmentation and all three fence severities, each with either
constant learning rate or warmup followed by cosine decay. Stage 2 contains only
the three fences trained with visible-region loss, using the constant learning
rate so they can be compared against the matching Stage 1 controls. All runs use
the mask-decoder-only scope, half-data protocol, and original pretrained weights.

The schedule uses the existing peak learning rate of 1e-5. It starts at 1e-6,
warms up over the first 5% of the actual training update budget, then decays to
1e-6 at the last update. These are configurable starting values, not tuned
settings. Each update logs its actual learning rate to CSV/JSONL and W&B.
The rate is computed from the completed-update counter and full training budget,
so resuming continues the same schedule rather than restarting warmup. A smoke
run's `max_steps` does not shorten the configured schedule horizon.

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin main
PLAN_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/point-staged.XXXXXX)
BASE_CONFIG=/scratch/vho/sam3-whole-scene/fragmentation-ahead.SQFHHT/prepared/config.json
PREP_ID=$(sbatch --parsable --output="$PLAN_ROOT/preparation-%j.out" \
  scripts/vulcan_point_followup.sbatch "$PWD" "$BASE_CONFIG" \
  "$PLAN_ROOT/prepared" staged)
echo "Preparation job: $PREP_ID"
echo "Keep this path: $PLAN_ROOT"
```

After successful preparation, submit **only Stage 1**:

```bash
cat "$PLAN_ROOT/prepared/sample_counts.csv"
export WANDB_ENTITY=venniseho14-university-of-toronto
export WANDB_PROJECT=SAM-finetuning
export WANDB_MODE=online
for CONDITION in fragmentation fence-low fence-medium fence-high \
  fragmentation-warmup-cosine fence-low-warmup-cosine \
  fence-medium-warmup-cosine fence-high-warmup-cosine; do
  JOB=$(sbatch --parsable --time=24:00:00 \
    --job-name="bwlab-half-$CONDITION" \
    --output="$PLAN_ROOT/$CONDITION-%j.out" \
    scripts/vulcan_point_training.sbatch "$PWD" \
    "$PLAN_ROOT/prepared/$CONDITION/config.json" mask_decoder_only \
    "$PLAN_ROOT/training-$CONDITION")
  echo "$CONDITION: job=$JOB results=$PLAN_ROOT/training-$CONDITION"
done
```

Once Stage 1 completes and its results have been checked, submit Stage 2 with the
same loop using only `fence-low-visible-loss`, `fence-medium-visible-loss`, and
`fence-high-visible-loss` as CONDITION values. Do not rerun the Stage 1 loop in
the same output directories. Stage 2 is not automatically submitted by any of
these commands. Clean-image training remains an additional control to plan.
