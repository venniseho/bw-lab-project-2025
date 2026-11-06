# Point-prompt SAM3 training and validation

The active protocol is 1008x1008 preprocessing, batch size 1, one seeded random
noncrowd target per source image, and one training epoch. Full point-path training
updates the shared vision encoder, point encoder and mask decoder. The frozen
scope updates only the point encoder and mask decoder. Language/concept/video
branches are unused. BF16 forward, float32 weights/loss, AdamW lr=1e-5 and weight
decay=.01, gradient clipping=1.0. This is our small PyTorch loop using Meta's native
interactive modules, not its concept-segmentation Trainer.

## Four controlled runs, after benchmarking

| Scope | Training condition | Validation condition |
|---|---|---|
| full | fragmentation | fragmentation |
| full | fence | fence |
| frozen_encoder | fragmentation | fragmentation |
| frozen_encoder | fence | fence |

Training order is seeded and shuffled; validation order is fixed.
Each starts from the same original checkpoint, uses the same source images,
selected object IDs, seeded points and transform settings, and runs independently.
Run fragmentation first. Interval frequency and full-versus-subset validation
must be selected after measuring validation cost; no full runs are submitted by
preparation. Fence width is an explicit experimental parameter; the CLI's example
starting value is 12 original-image pixels, spacing 40, angle 90 degrees.

## Data and transforms

New preparation selects one random noncrowd annotation per image. Thus training
steps equal eligible training images at batch size 1, rather than total objects.
Images without a noncrowd target are excluded and counted explicitly. A fixed
random target and foreground click are used throughout the run for reproducible
comparisons. A fence click is selected on visible target support; if none remains,
it falls back to original target support and records that it is not visible.
For fragmentation, the point is inside the full target and may be on black canvas;
`prompt_on_rendered_signal` records this. Labels remain the original binary mask.

`--official-splits` uses train2017 for training and val2017 for validation. A count
of 0 selects all eligible images in that official split. These are validation
results, not a new blind test set. The earlier exploratory val2017 work still
exists. Without that flag, small disjoint train2017 subsets remain available.
Only missing JPEGs are downloaded and cached on Vulcan scratch. Transformations
run in RAM in CPU loader workers; transformed JPEGs are not precomputed.

`--validation-image-count N` fixes a seeded image subset for baseline and interval
checks. Without it, every prepared validation image is evaluated. If a subset is
used, the baseline and end of training additionally evaluate all prepared validation images
under `heldout_full/*`; the best checkpoint is selected only using interval data.
The subset is fixed across intervals and runs, not resampled at every check.

The fragmentation stimulus depends on the target mask. Different target objects
can produce different input pixels, so their features cannot be shared. With one
target and one point per image, one encoder pass is already sufficient. Several
independent points on the same stimulus reuse a single encoder result. Features
are scoped to a validation call and never retained across optimizer updates.

## Benchmark validation first

On Vulcan (no running job should be using the checkout while pulling):

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin main
module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
export WANDB_ENTITY=venniseho14-university-of-toronto
export WANDB_PROJECT=SAM-finetuning
export WANDB_MODE=online

BENCH_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/point-validation.XXXXXX)
python scripts/prepare_point_training.py \
  --native-root /project/aip-sven/vho/sam3 \
  --coco-root /scratch/vho/datasets/coco \
  --output-dir "$BENCH_ROOT/prepared" \
  --official-splits --train-images 8 --heldout-images 512 \
  --transform fragmentation --epochs 1 --workers 8 --download

BENCH_ID=$(sbatch --parsable --output="$BENCH_ROOT/slurm-%j.out" \
  scripts/vulcan_point_validation.sbatch "$PWD" \
  "$BENCH_ROOT/prepared/config.json" "$BENCH_ROOT/benchmark")
echo "Job: $BENCH_ID"
echo "Results: $BENCH_ROOT"
```

This is evaluation only: no backward, weight updates or checkpoints. One warmup
image is excluded. Two repetitions alternate reference/shared order. It measures
1 and 3 independent points per image; the 3-point case is a performance diagnostic,
not a change to the one-point training protocol. Full before/after per-object
metrics and compact selected masks are saved. Numerical differences are reported,
not rejected for failing exact pixel equality. Repeat with `--transform fence`
and explicit fence parameters in a new preparation/output directory.

Inspect `benchmark.csv`, `benchmark_samples.csv`, `benchmark-results.json`, W&B
and the Slurm log. `encoder_forwards` should drop from three per image to one in
the 3-point shared case. Actual GPU savings and full/subset duration must be
measured; CPU toy-model tests cannot establish those numbers.

## Interval planning and training

After benchmarking, prepare an official full-data configuration with
`--train-images 0 --heldout-images 0 --epochs 1`, the selected transform, and an
explicit `--validation-interval-steps N`. Optionally set
`--validation-image-count M`. Decide N and M from measured cost, not arbitrary
constants. Rough evaluation overhead is T_eval / (N*T_step + T_eval); checkpoint
I/O adds to this. The checkpoint log reports its own cost.

The training launcher accepts `repo config scope output [max-steps]`. The optional
last argument is only for smoke tests. Omitting it trains a complete epoch.
Every output directory must be new. Each job requests one GPU, 12 CPUs, 64 GB host
RAM and uses 8 loader workers. GPU execution requires a Slurm step; the code
refuses inference/training on the login node. No four-job launch is automatic.

## Saved outputs

* Prepared `samples.csv`: split, image/annotation IDs, source path, target area,
  seed. Prepared annotations, hashes and config retain labels and transform settings.
* `train.csv` and `train.jsonl`: one row per update; source/target IDs and path,
  dimensions, target/visible area, transform parameters/seed, point coordinates,
  all candidate scores/IoUs, selected candidate, losses, gradient norm, compute
  step time, epoch and global step.
* `heldout.csv` and `heldout.jsonl`: the same per-object provenance and scores at
  each validation, plus lossless selected-mask COCO RLE. The mask can be decoded
  without rerunning SAM3. Candidate masks themselves are not all retained.
* `summary.csv/jsonl`: baseline, interval/end and optional full-validation mean IoU
  and loss. W&B uses global step, so several checks in one epoch stay distinct.
* `timing.csv/jsonl`: end-to-end train/validation phase wall times, instance/image
  presentation rates, unique-image coverage and encoder forward counts. Train time
  excludes pauses for interval validation/checkpointing. Model initialization,
  warmup and checkpoint I/O are outside phase rates; total run time includes
  baseline, training, validation and checkpointing. CUDA synchronizes at phase
  boundaries, not at every step. Images/sec counts presentations; one target per
  image makes it equal source-image coverage for normal training/validation.
* `checkpoints.csv/jsonl`: save time, path, size, epoch, step and validation IoU.
  `checkpoint.pt` is latest; `checkpoint-best.pt` is best on the fixed interval
  set, including baseline. Saves are atomic, include full model and optimizer
  state, and are read back. Old latest snapshots are replaced, not accumulated.
* `previews/`: four before/after held-out examples per check, named by epoch and
  global step so interval images are not overwritten. W&B also receives previews.
* `metadata.json`, `audit.json`, `complete.json`: code/checkpoint provenance,
  hardware, split counts, gradient/weight-change checks and final summaries.

Checkpoints remain on scratch and are not automatically uploaded to W&B. To reuse
one, initialize the same native model from the original checkpoint, then load
`torch.load(path, map_location='cpu', weights_only=True)['model']` with
`model.load_state_dict(..., strict=True)`. This is BWLab checkpoint format, not
Meta's checkpoint-prefix format. Exact interrupted-run resume is not implemented;
the weights can be reused for evaluation or a new training experiment.

## Loss

One positive point produces three masks. The smallest `20*focal + Dice` candidate
receives mask supervision; all three quality scores receive IoU MSE supervision.
Evaluation selects highest predicted quality, not ground-truth-best IoU.
Total loss is `20*focal(alpha=.25,gamma=2) + soft_Dice + quality_MSE`.
This follows [SAM Appendix A](https://arxiv.org/html/2304.02643v1) as a starting
loss recipe, not the full original iterative-click training procedure.
