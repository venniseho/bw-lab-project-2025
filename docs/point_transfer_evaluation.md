# Fence and fragmentation transfer evaluation

These jobs evaluate saved final checkpoints without training or an optimizer.
Use final checkpoints for a common end-of-budget comparison, rather than choosing
different best steps using each source's validation metric. This remains a
validation-set analysis, not an independent test-set result.

Each source is compared with the original pretrained checkpoint on exactly the
same target condition's saved validation cohort, random foreground prompt and
seeded transformation. Cohort sizes differ by target condition because fully
hidden fence targets were excluded: fragmentation/low 4,952; medium 4,910; high
4,739. The target condition determines eligibility, not the training condition.
The same source image can have a different prompt across conditions, since the
point must be visible; prompts match between pretrained and fine-tuned evaluations
within each target condition.

## Queue the comparisons on Vulcan

Seven jobs cover nine comparisons: the fixed-rate fragmentation model on all
three fences, plus six fence models (three full-loss and three visible-loss)
on fragmentation. Scheduled-learning-rate models and the earlier full point-path
models are not included in this particular comparison.

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin main

export WANDB_ENTITY=venniseho14-university-of-toronto
export WANDB_PROJECT=SAM-finetuning
export WANDB_MODE=online

PLAN_ROOT=/scratch/vho/sam3-whole-scene/point-staged.dUreDG
CROSS_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/cross-condition.XXXXXX)
echo "Keep this path: $CROSS_ROOT"

JOB=$(sbatch --parsable \
  --output="$CROSS_ROOT/fragmentation-%j.out" \
  scripts/vulcan_point_transfer.sbatch "$PWD" \
  "$PLAN_ROOT/training-fragmentation" "$CROSS_ROOT/fragmentation-to-fences" \
  "$PLAN_ROOT/prepared/fence-low/config.json" \
  "$PLAN_ROOT/prepared/fence-medium/config.json" \
  "$PLAN_ROOT/prepared/fence-high/config.json")
printf 'fragmentation-to-fences,%s\n' "$JOB" | tee "$CROSS_ROOT/jobs.csv"

for CONDITION in fence-low fence-medium fence-high \
  fence-low-visible-loss fence-medium-visible-loss fence-high-visible-loss; do
  JOB=$(sbatch --parsable \
    --output="$CROSS_ROOT/$CONDITION-%j.out" \
    scripts/vulcan_point_transfer.sbatch "$PWD" \
    "$PLAN_ROOT/training-$CONDITION" "$CROSS_ROOT/$CONDITION-to-fragmentation" \
    "$PLAN_ROOT/prepared/fragmentation/config.json")
  printf '%s-to-fragmentation,%s\n' "$CONDITION" "$JOB" | tee -a "$CROSS_ROOT/jobs.csv"
done

squeue --me
```

Each job requests one GPU, 12 CPU cores, 64 GB RAM and two hours, using the
prepared worker count (eight). It checks Slurm allocation, native SAM3 revision,
annotation hashes and completed source checkpoint provenance. The existing
training runs and their checkpoints are read only. If a job fails, resubmit only
that job with a fresh output directory; evaluations do not resume automatically.

## Read the results

Under each job's output directory, every target condition gets:

- `summary.csv`: pretrained (step 0) and fine-tuned (source final step) mean
  full-target IoU, visible-region IoU, full loss and visible loss.
- `heldout.csv/jsonl`: one record per target and phase, including target IDs,
  prompt coordinates, candidate scores/IoUs and the selected mask as COCO RLE.
- `timing.csv/jsonl`: wall time and images/second for each validation pass.
- `previews/`: before/after sheets for the first four validation targets.

The job root contains source/target metadata, checkpoint SHA256, and
`complete.json` with all results. W&B records hardware telemetry, per-target
IoU/loss under `cross/`, and before/after images under `cross_examples/`.
CSV timing includes loading, transforms, inference, metrics, per-image record
writing and the first four preview sheets, but excludes model loading and
checkpoint restoration. Root wall time covers the whole evaluation, including setup.

Both losses are logged using full-region loss as the conventional `mean_loss`
alias; no loss is backpropagated. Full and visible IoU are identical for the
current fragmentation transform. Report differences against each paired
pretrained baseline, rather than comparing raw means across different cohorts.
Clean-image evaluation/training controls remain a separate experiment.
