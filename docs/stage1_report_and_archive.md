# Report and archive the eight completed learning-rate comparisons

`scripts/report_point_staged.py` reads existing CSV/JSON and PNG files. It needs
no model checkpoint loading, CUDA or new SAM inference. It requires eight
completed one-epoch runs and verifies matching data, order, objectives, validation
steps, scope and settings within each learning-rate pair. Cohorts may differ
across conditions because fully hidden targets were excluded.

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin main
module load python/3.12.4
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
PLAN_ROOT=/scratch/vho/sam3-whole-scene/point-staged.dUreDG

python scripts/report_point_staged.py \
  --plan-root "$PLAN_ROOT" --output-dir "$PLAN_ROOT/report-stage1"
```

Choose a fresh output directory if the report already exists. The outputs are:

* PNG/SVG comparisons for full-target IoU, visible-region IoU, both losses and
  recorded learning rates. Each figure has one panel per condition.
* `run_summary.csv`: baseline/final/best IoU, image counts, updates, phase timings,
  training images/second, checkpoint time, fit duration and peak GPU memory.
* `paired_differences.csv`: final warmup/cosine minus fixed-rate IoUs. No best-step
  selection is used for this comparison.
* `validation_curves.csv` and `learning_rates.csv`: values used in the figures.
  Learning-rate traces are sampled every 500 updates, with first and last included.
* `index.html` and `examples/`: an offline report and up to 64 copied before/after
  sheets (best and final snapshots, first four validation targets per run).
* `sources.json`: source run paths and recorded metadata for traceability.

Training throughput uses training phase time, which excludes validation and
checkpoint pauses. Fit duration includes those pauses but excludes model
initialization and Slurm queue time. Four preview samples are not representative
performance estimates. Best models were selected on the same validation set.

## Archive completed runs without copying active Stage 2 outputs

The following copies prepared annotations/configs, all outputs from the eight
completed Stage 1 runs, their Slurm logs, and `report-stage1` if present:

```bash
bash scripts/archive_point_stage1.sh \
  "$PLAN_ROOT" /project/aip-sven/vho/bwlab-results/point-staged.dUreDG
```

It checks for completion markers and both checkpoint files before copying,
retains scratch originals, and ignores the three visible-loss run directories.
Rerunning the rsync-based script can finish an interrupted copy. Checkpoint files
alone total approximately 56 GB. This does not archive the shared COCO JPEGs;
prepared records still contain their absolute scratch paths. Keep the originals
available for the upcoming evaluations and rebase paths if restoring elsewhere.

## Download the report to the laptop

From the laptop's WSL terminal:

```bash
mkdir -p /mnt/c/Users/venni/Downloads/bwlab-results
scp -r vulcan:/scratch/vho/sam3-whole-scene/point-staged.dUreDG/report-stage1 \
  /mnt/c/Users/venni/Downloads/bwlab-results/
explorer.exe 'C:\Users\venni\Downloads\bwlab-results\report-stage1'
```

Open `index.html`; its image paths remain relative so the copied report works
offline. Download the small report folder, rather than all checkpoint files.
