# SAM3 single-image point smoke test

Status: GPU-ready, verified against official API source and CPU mocks.
Actual checkpoint loading and CUDA inference have not been executed here:
the local Python environment has neither PyTorch nor Transformers/SAM3.

## API choice and historical compatibility

Use Hugging Face `Sam3TrackerModel` with `Sam3TrackerProcessor` for promptable
visual segmentation (PVS). These are the classes used by `sam_runner.py`.
The historical SAM3 point nesting, `pred_masks`/`iou_scores` outputs and
`post_process_masks(..., original_sizes)` match the current public API.
This script keeps that API choice without importing the old runner's SAM1,
OpenCV, sklearn, evaluation or plotting dependencies.

The old SAM3 branch loads the moving `facebook/sam3` model name, ignores its
checkpoint argument, permits CPU fallback and does not explicitly use eval
mode. Those behaviors are not carried into this smoke test. No historical
code, fence generation or evaluation preparation has been changed.

The checked reference is Transformers **5.17.0**:

- [Tracker API and single-point example](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/sam3_tracker)
- [Processor implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/sam3_tracker/processing_sam3_tracker.py)
- [Tracker implementation and output shapes](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/sam3_tracker/modeling_sam3_tracker.py)
- [Mask resizing and binarization](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/sam3/image_processing_sam3.py)
- [Checkpoint loading diagnostics](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/utils/loading_report.py)

Only a positive point is passed, shaped `[image, object, point, xy]` with
labels `[image, object, point]`. No text, exemplars or concept-detection
processor is used. The installed version is recorded at runtime; the first
cluster run still needs to confirm actual weights and environment compatibility.

## Environment and checkpoint prerequisites

Use an environment with CUDA-enabled PyTorch and matching torchvision, NumPy,
Pillow, safetensors and `transformers==5.17.0`. Use a Python version supported
by that environment (Python 3.11 or 3.12 is a reasonable starting point).
Cluster CUDA/PyTorch module names depend on the cluster; use its supported
environment rather than running the old `setup_env.sh`. Nothing is installed
by this script. If the GPU environment is missing, establish it separately
before inference; do not execute GPU work on the login node.

For an already CUDA-ready virtual environment, the additional package setup
command is:

```bash
python -m pip install 'transformers==5.17.0' numpy pillow safetensors
```

This command does not establish a CUDA PyTorch/torchvision installation.
The dependencies must resolve successfully before proceeding.

The checkpoint must be a **local Hugging Face snapshot** of `facebook/sam3`,
including its JSON configuration/processor files and all safetensors shards.
A Meta `.pt` checkpoint is not the format accepted by this workflow. Obtain
access to the gated model through its model page first. If a snapshot is
already available, reuse it. Optional staging on a host with network access:

```bash
# Set these to an immutable HF commit and a persistent local directory.
export SAM3_REVISION=REPLACE_WITH_HF_COMMIT_HASH
export SAM3_MODEL_DIR=/absolute/path/to/sam3-snapshot
hf auth login
hf download facebook/sam3 --revision "$SAM3_REVISION" --local-dir "$SAM3_MODEL_DIR"
```

This is a manual preparation command, not something the inference script
runs. See [HF CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli).
Inference uses `local_files_only=True`, disables remote code and requires
safetensors. Missing/mismatched model keys fail rather than silently running
newly initialized parameters. Unexpected keys are retained in the report:
a full SAM3 snapshot can contain components unused by the Tracker path.

## Run exactly one image on an allocated GPU node

Replace the account, environment, checkout and checkpoint paths below.
Use the cluster's GPU partition/type options if its scheduler requires them.
The resource request is an initial smoke-test allocation, not a measured
memory requirement; float32 peak allocation will be recorded by the script.

```bash
# Request allocation; this command may wait for resources.
salloc --account=YOUR_ACCOUNT --time=00:15:00 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=32G --gres=gpu:1

# Enter a job step on the allocated compute node. salloc alone is insufficient.
srun --ntasks=1 --pty bash
hostname
nvidia-smi
source /absolute/path/to/sam3-environment/bin/activate
cd /absolute/path/to/bw-lab-project-2025

python -c 'import torch, torchvision, transformers; from transformers import Sam3TrackerModel, Sam3TrackerProcessor; print(torch.__version__, torchvision.__version__, transformers.__version__); assert torch.cuda.is_available()'

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SAM3_MODEL_DIR=/absolute/path/to/sam3-snapshot
export SAM3_REVISION=REPLACE_WITH_HF_COMMIT_HASH
python -B scripts/sam3_smoke_test.py \
  --require-slurm \
  --image outputs/fence_demo/source_rgb.png \
  --point 415 241 \
  --checkpoint-dir "$SAM3_MODEL_DIR" \
  --checkpoint-revision "$SAM3_REVISION" \
  --output-dir "outputs/sam3_smoke_${SLURM_JOB_ID}_${SLURM_STEP_ID}" \
  --seed 0
```

The repository example is 640 x 360 pixels. `(415, 241)` was checked with the
existing preparation module to lie inside `outputs/fence_demo/source_mask.png`
and remain visible with widths 0/4/12/24, spacing 40, angle 90 and fence seed
123. This command runs **one clean image**, not the sweep. For another image,
supply its own prepared point in original pixel coordinates. The CLI does
not recalculate prompts or check GT membership; it validates finite bounds.

`--require-slurm` checks both job and step identifiers before importing CUDA
packages. Always use it on the cluster. `srun` launches a job step on allocated
resources; see [Slurm's documentation](https://slurm.schedmd.com/srun.html).
Exit the job-step shell and allocation shell when finished. No job has been
submitted or allocation requested by this implementation task.

## Outputs and interpretation

The output directory must be new. Files are:

- `prediction.png`: selected binary mask, grayscale 0/255, original resolution.
- `overlay.png`: original RGB image with selected foreground tinted cyan.
- `candidate_00.png`, etc.: every binary candidate in API output order.
- `metadata.json`: each predicted IoU score and selection flag; positive point;
  optional object-presence logits; image/checkpoint hashes; source revision;
  loading diagnostics; model/processor settings; package versions; device;
  seed; GPU memory and timing measurements.

"Model-selected" is explicitly the highest predicted IoU among candidates,
with the first candidate winning ties. Scores are the model's quality estimates,
not measured GT IoU or calibrated probabilities. The decoder's separate
single-mask/stability fallback is not invoked. No GT enters mask selection.
Object-presence logits are recorded without introducing a filtering threshold.

Pillow decodes to RGB; there is no BGR conversion. EXIF rotation is not applied
because it would change point coordinates. Prepare image and point in the same
decoded coordinate frame. The processor transforms the point when it resizes
the image; predictions are resized back to the original dimensions before
thresholding logits at zero. Hole/sprinkle cleanup and overlap constraints are
disabled. Precision is float32 with TF32 disabled and SDPA attention; no
autocast, compilation or precision comparison is attempted. The seed is
recorded, but bitwise GPU reproducibility is not asserted.

`cuda_forward_seconds` synchronizes before and after the first model forward.
`inference_seconds` includes processor preprocessing, transfer, forward,
CPU postprocessing and candidate selection. Both exclude model loading and
output I/O. Checkpoint hashing reads the weights once more and is timed
separately. These are first-call smoke-test latencies, not warmed throughput.
`peak_cuda_allocated_bytes` includes model-resident memory during inference;
it is not total device usage or load-time peak memory.

## CPU verification

```bash
python -B -m unittest discover -s tests -p test_sam3_smoke.py -v
python -B -m unittest discover -s tests -p test_probe_preparation.py -v
python -B -m unittest discover -s tests -p test_fence.py -v
python -B scripts/sam3_smoke_test.py --help
```

Mock tests verify the complete CLI loading/saving workflow, API argument
dimensions, RGB order, original-resolution masks, candidate selection/ties,
single-mask handling, malformed outputs, CUDA failure, checkpoint loading
diagnostics (including the current API's sets), and the SLURM launch check.
Mock outputs are temporary test artifacts, not actual SAM3 predictions.
