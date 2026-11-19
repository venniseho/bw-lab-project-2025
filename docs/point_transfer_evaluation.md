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

## Generate charts after all seven jobs finish

This reads CSV/JSON and saved previews on the CPU; no GPU allocation or new
inference is needed. It requires every expected completion marker and checks
target/prompt pairing and agreement between per-image and aggregate IoUs.

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin main
module load python/3.12.4
source /scratch/vho/sam3-whole-scene/train-env/bin/activate

# Or set CROSS_ROOT to the exact path printed when you submitted the jobs.
CROSS_ROOT=$(ls -td /scratch/vho/sam3-whole-scene/cross-condition.* | head -n 1)
echo "Reading: $CROSS_ROOT"
REPORT_ROOT="$CROSS_ROOT/cross-condition-results"

python scripts/report_point_transfer.py \
  --cross-root "$CROSS_ROOT" --output-dir "$REPORT_ROOT"
echo "Report: $REPORT_ROOT/index.html"
```

The report includes PNG/SVG charts of mean IoUs and their changes, plus paired
scatterplots for both metrics. Scatter x = original pretrained model IoU,
y = fine-tuned model IoU, not clean-versus-occluded IoU. Above diagonal means
improvement after fine-tuning on the source condition. (0,0) means no overlap
with the labelled target from either model; overlapping points are counted in
each panel. The dots are darker and have no coordinate jitter.

`run_summary.csv` includes before/after means, deltas and counts of improved,
equal and worse targets. `paired_image_results.csv` contains each target's IoUs
and changes. `timing.csv` combines recorded validation timings. `index.html`
also displays the copied before/after contact sheets. Download this report folder
with `scp -r` to view it locally; its links are relative. Choose a fresh output
directory to generate a second report without overwriting the first.
