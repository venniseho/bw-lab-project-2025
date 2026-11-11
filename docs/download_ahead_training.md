# Start training before all COCO downloads finish

Leave the existing download process running. This mode does not launch a second
downloader: two producers must not write the same `.jpg.part` temporary file.
Validation JPEGs must be available before training. `--download` in this mode
downloads missing validation files/annotations, not training JPEGs.

Preparation saves a fixed order: seeded shuffle of the cached training images,
then missing images in ascending ID order, matching the existing downloader.
Each eligible image still appears once, with one selected object. This changes
the ordering from a globally shuffled epoch; the cache snapshot and order are
recorded in `config.json`. Reuse that config when resuming and for matched
experiments. Do not prepare it again to resume, since cache availability changes.

## Start on Vulcan

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin main
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate

export WANDB_ENTITY=venniseho14-university-of-toronto
export WANDB_PROJECT=SAM-finetuning
export WANDB_MODE=online

START_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/fragmentation-ahead.XXXXXX)
echo "Keep this path: $START_ROOT"

python -u scripts/prepare_point_training.py \
  --native-root /project/aip-sven/vho/sam3 \
  --coco-root /scratch/vho/datasets/coco \
  --output-dir "$START_ROOT/prepared" \
  --official-splits --train-images 0 --heldout-images 0 \
  --transform fragmentation --epochs 1 --workers 8 \
  --validation-interval-steps 15000 \
  --checkpoint-interval-steps 1000 \
  --download-ahead --download
```

This scans labels and writes manifests; it does not wait for all training JPEGs.
After it prints the config and cached-first buffer, submit:

```bash
TRAIN_ID=$(sbatch --parsable --time=24:00:00 \
  --output="$START_ROOT/slurm-%j.out" \
  scripts/vulcan_point_training.sbatch "$PWD" \
  "$START_ROOT/prepared/config.json" full "$START_ROOT/training")
echo "Job: $TRAIN_ID"
echo "Results: $START_ROOT"
squeue -j "$TRAIN_ID"
```

This is full point-path fine-tuning on fragmented train2017, at 1008×1008,
batch size 1. All eligible val2017 targets are evaluated before training, every
15,000 updates and at the end. The pilot suggests approximately 90 minutes per
15,000 updates and five minutes per validation; checkpoint I/O adds time.

## What survives an interruption

* Latest model, optimizer, completed sample cursor, global step, epoch and random
  states are saved every 1,000 updates and after validation. Latest and best
  checkpoints are replaced atomically; old snapshots do not accumulate.
* CSV/JSONL rows are written per sample. Timing excludes validation/checkpoint
  pauses, includes data waiting, and records completed partial training phases.
  `cache_wait_seconds` records worker waiting for missing JPEGs; main-process
  waiting can overlap worker prefetch and is not equal to its sum.
* Slurm sends USR1 about five minutes before the limit. The trainer stops at a
  safe update boundary and attempts an emergency checkpoint. Errors also trigger
  a save attempt. `interrupted.json` identifies a saved interrupted run;
  `complete.json` identifies a completed run. A cooperative stop exits with an
  error status, so Slurm/W&B may label it failed despite a saved resumable
  checkpoint; inspect `interrupted.json`.
* If a missing JPEG is still absent after 120 seconds, the worker raises an error
  and the trainer attempts a checkpoint. Check/restart the downloader before
  resubmitting. No images are skipped silently, and no network requests happen
  inside training workers.
* SIGKILL, a node failure, disk failure or an exhausted grace period can prevent
  an emergency save. The previous completed atomic checkpoint remains the
  recovery point: normally at most 999 completed updates are newer than it.

SSH disconnection does not cancel an sbatch job. This does not automatically
resubmit a stopped job; use the command below. W&B records each submission as a
separate run with `resumed_from` provenance and continued global step. Weights
are not uploaded to W&B; retain the previous run directory, including its best
checkpoint. Checkpoints on scratch still need archival after the experiment.

## Resume

Set START_ROOT to the saved path after reconnecting. Keep the same prepared config.
Choose the latest run directory if training has already been resumed once.

```bash
PREVIOUS_RUN="$START_ROOT/training"
RESUME_DIR=$(mktemp -d "$START_ROOT/resumed.XXXXXX")
rmdir "$RESUME_DIR"
RESUME_ID=$(sbatch --parsable --time=24:00:00 \
  --output="$START_ROOT/slurm-%j.out" \
  scripts/vulcan_point_training.sbatch "$PWD" \
  "$START_ROOT/prepared/config.json" full "$RESUME_DIR" \
  --resume "$PREVIOUS_RUN/checkpoint.pt")
echo "Job: $RESUME_ID"
echo "Results: $RESUME_DIR"
```

The loader resumes after completed samples, not after prefetched samples.
Configuration/scope changes are rejected. Model, optimizer and random state
restoration are tested with a CPU toy model, including stochastic updates.
This does not guarantee bitwise-identical CUDA results or recovery from every
hardware failure. GPU and Slurm signal behavior must still be checked on Vulcan.
