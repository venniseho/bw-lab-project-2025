# PyTorch throughput work

The optional `runtime` config enables a tensor-batched path and numerical
evaluation. Existing configs without it retain the archive workflow. This work
implements the meeting's optimization checklist; speedups require GPU measurement.

## What changed

- HWC RGB arrays become CHW uint8 Torch tensors. They transfer to the GPU,
  pass through the installed native processor's resize/normalization and stack
  into a real image-encoder batch. Native interactive feature projections are
  preserved. Variable native image sizes are retained for mask/prompt mapping.
- Independent point queries are also batched: `(Q, 1, 2)` means Q separate
  one-point queries, not one query containing Q positive points. The native mask
  decoder still loops over images internally; only claim the implemented image
  encoder batching and within-image prompt batching.
- Runner evaluation and the batched adapter use `@torch.inference_mode()`.
  The loader already sets `.eval()`. Warm-up repeats the first batch without
  exporting or scoring it, then evaluates every input once for recorded results.
- Optional cuDNN benchmarking and encoder compilation are recorded. Compilation
  targets `backbone.forward_image`; it is disabled by default. Keep native
  checkpoint dtypes (including complex RoPE buffers) and existing TF32 settings.
- Resource logs include parameter counts/dtypes/bytes, buffer bytes, total VRAM,
  peak allocated/reserved VRAM and sampled GPU utilization. Parameter bytes are
  not an estimate of activation/workspace VRAM. `gpu_utilization.csv` contains
  node GPU index/UUID data; match the allocated device rather than assuming
  physical GPU index zero is the assigned GPU. Shared GPUs can have other work.
- A streaming CSV evaluator computes measured full/visible IoU separately from
  model-predicted quality. It renders only `preview_batches` and writes aggregate
  JSON once at the end. Failed runs close/flush their CSV.
- Batch-size profiling runs each candidate in a fresh process, logs OOM and
  stops the larger-size sweep. Other failures stop immediately. Pick the best
  measured throughput, not automatically the largest fitting batch.
- Best/worst selection uses measured IoU within a condition and reconstructs
  original batch membership when replaying, including the short final batch.
- Submitit `launch.py` submits configurations and optional `afterok` dependencies.
  `experiment.py` owns the actual run. `--local` directly calls run on a personal
  GPU or allocated compute node. Cluster login nodes remain unsuitable for runs.

## Configuration

Add to an existing experiment config (paths remain those of your existing run):

```json
{
  "batch_size": 4,
  "num_workers": 2,
  "prefetch_factor": 2,
  "runtime": {
    "batched": true,
    "prompt_batch_size": 9,
    "pin_memory": true,
    "cudnn_benchmark": true,
    "compile_encoder": false,
    "warmup_batches": 1,
    "preview_batches": 1,
    "monitor_gpu": true,
    "allow_local": false
  }
}
```

`runtime.batched=false` is a useful reference: numerical output with the
existing PIL/single-image model protocol. Increasing loader batch size in that
reference does not batch inference. `preview_batches=0` disables all images for
fair batch-size profiling. `max_inputs` can limit a representative profiling run.

## Vulcan validation sequence

Run these yourself on Vulcan after fetching this branch. Use the existing native
SAM3 environment/checkpoint; no model download or environment replacement is needed.
Paths confirmed from the user's Vulcan terminal on 21 October 2026:

- Project Git checkout: `/project/aip-sven/vho/bw-lab-project-2025`.
- Native Meta SAM3 source: `/project/aip-sven/vho/sam3`.
- Scratch workspace: `/scratch/vho/sam3-whole-scene` (not a Git checkout).
- Existing virtual environment: `/scratch/vho/sam3-whole-scene/env`.
- Existing checkpoint directory: `/scratch/vho/sam3-whole-scene/checkpoints`.

Start in the project checkout and activate the existing environment:

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git fetch origin --prune
git switch performance
git pull --ff-only origin performance

module load python/3.12.4
module load opencv/4.12.0
source /scratch/vho/sam3-whole-scene/env/bin/activate
```

The scratch workspace contains environments, checkpoints, logs and run outputs;
do not run project Git commands there. From the project checkout, prepare new
configs without inference:

```bash
python scripts/prepare_pipeline_pilot.py \
  --count 64 --optimized --output-dir "$SCRATCH/bwlab-throughput-inputs"
```

This selects 64 archived source images, producing 256 fence inputs and 64
contour inputs. It is a small software/performance pilot, not the 5,000-image
COCO validation benchmark. Preparation performs no model inference. The first
equivalence checks use only four inputs per arm, comparing old and new paths.

From the login node, the recommended route is the batch wrapper, which requests
one GPU and loads Python/OpenCV plus the environment inside the compute job:

```bash
CHECK_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/throughput-check.XXXXXX)
JOB_ID=$(sbatch --parsable --output="$CHECK_ROOT/slurm-%j.out" \
  scripts/vulcan_throughput.sbatch equivalence "$PWD" \
  "$SCRATCH/bwlab-throughput-inputs" "$CHECK_ROOT")
echo "$JOB_ID"
```

Inspect the job output and equivalence CSVs before submitting a batch-size sweep.
Replace `equivalence` with `benchmark` and use a fresh directory for that sweep.
Both modes stop on failed preflight/checks. No job is submitted by preparation.

Bare `srun python ...` from a login node may allocate a **CPU-only** compute node.
It does not automatically request a GPU. For manual commands below, first obtain
a GPU allocation with the required account/partition/resources and load the same
modules/environment on the compute node. Alternatively, use the wrapper above.
Within that GPU allocation, compare the reference and tensor-batched paths:

```bash
srun python scripts/check_batch_equivalence.py \
  --config "$SCRATCH/bwlab-throughput-inputs/fence.json" \
  --count 4 --output "$SCRATCH/bwlab-equivalence-fence.csv"
srun python scripts/check_batch_equivalence.py \
  --config "$SCRATCH/bwlab-throughput-inputs/fragmentation.json" \
  --count 4 --output "$SCRATCH/bwlab-equivalence-contour.csv"
```

The check requires identical binary masks and score differences <= 1e-5. A
failure is evidence to investigate numerical/preprocessing differences before a
large run, not a reason to silently loosen the comparison. CPU tests validate
tensor/PIL preprocessing and prompt/candidate correspondence, not actual SAM3
GPU numerics. The full GPU checks have not been run by the coding agent.

### Investigate numerical differences before benchmarking

The user's job `1889951` passed the L40S/OpenCV preflight, then reported 108
strict differences in the four-input fence check. The first 11 pasted rows had
5–428 changed pixels and score differences from 0 to 0.005859375. This does not
establish that 108 masks are entirely wrong or identify which optimization is
responsible. PyTorch does not guarantee bitwise equality between batched and
individual computations; see [numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
Small image-wide pixel changes can still substantially affect a tiny mask.

Use the small diagnostic mode, which preserves the strict tolerance but runs
both arms and isolates each change:

```bash
DIAG_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/throughput-diagnostic.XXXXXX)
JOB_ID=$(sbatch --parsable --output="$DIAG_ROOT/slurm-%j.out" \
  scripts/vulcan_throughput.sbatch diagnose "$PWD" \
  "$SCRATCH/bwlab-throughput-inputs" "$DIAG_ROOT")
echo "$JOB_ID"
echo "$DIAG_ROOT"
```

Each arm saves `diagnostic-CONDITION.csv` and `diagnostic-CONDITION.summary.json`.
The summary records GPU preprocessing equality, repeated-reference stability,
tensor path with single images/prompts, image batching alone, prompt batching
alone, their combination with cuDNN benchmarking disabled, and the requested
runtime settings. CSVs include mask-to-mask IoU, changed-pixel fraction, mask
areas, score delta and target-IoU delta; summaries count changed selected
candidates. The diagnostic job can finish with a failed state because strict
differences remain, while still writing both arms' evidence. No mask/score
tolerance is automatically relaxed and no throughput sweep is launched.

Then profile the two arms separately, starting with compilation off:

```bash
srun python scripts/benchmark_pipeline.py \
  --config "$SCRATCH/bwlab-throughput-inputs/fence.json" \
  --max-inputs 64 --workers 2 --batch-sizes 1 2 4 8 16 32 64 \
  --output "$SCRATCH/bwlab-fence-batches"
srun python scripts/benchmark_pipeline.py \
  --config "$SCRATCH/bwlab-throughput-inputs/fragmentation.json" \
  --max-inputs 64 --workers 2 --batch-sizes 1 2 4 8 16 32 64 \
  --output "$SCRATCH/bwlab-contour-batches"
```

Repeat the promising sizes with workers 0/2/4, then with
`--no-cudnn-benchmark`, and finally `--compile` in new output directories.
`max-inputs` must cover the largest requested batch and the dataset must actually
have enough inputs. The 64-input sweep has limited steady-state evidence at a
batch size of 64; use more representative inputs before choosing production
settings. The script records observed throughput rather than guaranteeing that
32 or 64 fits the GPU.

The preprocessor transfers and normalizes each source tensor before stacking;
it does not claim that all host transfer/preprocessing has been vectorized. The
fence and historical contour transforms still run on CPU inside DataLoader
workers. Data augmentation stays in memory. Profile before choosing a Torch
rewrite of contour geometry; a GPU rewrite is not part of this patch.

## Results and replay

- `predictions/image_results.csv`: one row per transformed dataset sample,
  including stable source key, batch/index, provenance, measured full/visible
  IoU, oracle diagnostics and stage times. Dataset samples can be instances;
  this file does not establish one row per unique COCO source image.
- `timing.json`: warm-up, model load, evaluation throughput and model/VRAM info.
- `gpu_utilization.csv`: one-second node utilization/memory samples when
  `nvidia-smi` is available; monitor status is recorded.
- `predictions/previews/`: first configured number of batches only.

Numerical scores currently retain the exploratory independent-grid protocol:
the highest predicted score across all grid proposals is the model choice;
best-overlap among candidates is a separately labeled oracle. This is not an
object-directed COCO evaluation protocol or a claim of autonomous object matching.
Empty model-selected masks score zero against a nonempty target. Empty targets
are reported explicitly, not included as artificial perfect IoUs.

Select cases on the CPU without loading SAM3:

```bash
python scripts/replay_ranked_examples.py \
  --run-dir "$SCRATCH/bwlab-fence-batches/batch-4" \
  --condition heavy --count 10 --list-only \
  --output "$SCRATCH/bwlab-ranked-list"
```

Omit `--list-only` inside `srun` with a fresh output directory to rerun those
image batches and render only selected examples. The script restores the saved
config and compares sample keys with the recorded rows. Sources/checkpoint must
still be present and unchanged. First use `--condition clean`, `light`, `medium`
or `heavy` for fences, and `fragmentation` for contours. Choose `--metric
visible_iou` when appropriate. Ranking never uses the predicted quality score.

## Cluster launcher

Submitit is an optional `cluster` dependency. Once available in the existing
environment, run the launcher from the repository with the same modules/env
used for the prior GPU jobs. It uses the active Python and inherited environment.

```bash
python -m experiments.sam3_fragmentation.launch \
  --configs "$SCRATCH/bwlab-throughput-inputs/fence.json" \
            "$SCRATCH/bwlab-throughput-inputs/fragmentation.json" \
  --output "$SCRATCH/bwlab-parallel-runs"
```

This submits one GPU job per config and records job IDs/output paths. Pass
`--after-job JOB_ID` for dependencies. GPU compilation/warm-up occurs in each
worker, where its process-local caches and allocations can actually be reused.
`--local` calls the same run logic directly for debugging. Do not use that flag
to bypass the cluster's login-node policy.

## Runtime targets and remaining work

Use measured end-to-end throughput, not inference-only time, for projections.
At 300 ms/input, 5,000 inputs take 25 minutes; four fence conditions (20,000
inputs) take 100 minutes before setup. The whole four-condition sweep needs an
average under 180 ms/input to fit an hour on one GPU. Report per-condition and
whole-sweep runtime separately.

Direct/lazy COCO loading, per-image reuse across instances, a fixed object
prompting protocol and full-validation plots are subsequent benchmark work.
Smaller SAM/SAM2 checkpoints can be separate comparison models; this patch does
not invent a smaller SAM3 checkpoint or silently swap model families. Native
GPU speed, compile compatibility and optimal batch/worker sizes remain to be
established on Vulcan. No GPU jobs have been submitted by this change.

Reference API: native SAM3 revision
`0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`, processor preprocessing and
`predict_inst_batch`. PyTorch references:
[inference mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html),
[cuDNN settings](https://docs.pytorch.org/docs/2.14/backends.html#torch.backends.cudnn.benchmark),
[compilation](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html),
[Submitit](https://github.com/facebookincubator/submitit).
