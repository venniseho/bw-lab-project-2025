# Automatic whole-scene SAM3 visual proposals

For the current 3x3 one-image setup, use [Vulcan setup instructions](vulcan_scene_setup.md). The cluster commands below are historical examples with unverified paths; the linked setup replaces them.

This is automatic **point-grid prompting**, not inherently prompt-free SAM3 inference. No text/concept queries are used. No turnkey class-agnostic automatic mask generator was found in the inspected [Meta SAM3 API](https://github.com/facebookresearch/sam3). The [processor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py) computes features once with `Sam3Processor(model).set_image(PIL_RGB_image)`. Each independent grid point is queried through [`model.predict_inst(inference_state, ...)`](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image.py) with `multimask_output=True`. The integrated tracker intentionally has no backbone, so calling its standalone predictor's `set_image()` is invalid. The grid is never passed as multiple positive points for one object.

The native [image builder](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py) requires `enable_inst_interactivity=True`. This script loads an existing native SAM3 `sam3.pt`, disables downloads, and rejects missing checkpoint keys rather than accepting randomly initialized components. It uses Meta's native implementation, whereas the previous single-point smoke script uses Hugging Face's tracker. Numerical equivalence between these implementations has not been established. SAM3.1 and HF safetensors are not interchangeable checkpoint inputs here.

The processor/predict_inst integration was checked against Meta revision [`0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`](https://github.com/facebookresearch/sam3/tree/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc), matching the user-reported Vulcan installation. This verified API revision is recorded separately from runtime source fingerprints; it does not claim a future installation has identical source.

The public visual API returns binary candidate masks and predicted-IoU quality scores, not text class labels or calibrated object-presence probabilities. Every raw candidate, including empty and duplicate masks, is saved by default. Low-resolution logits are not archived. Default visualization selection: exclude empty masks, no score threshold, greedy exact binary-mask IoU deduplication at >=0.95, highest predicted IoU first (raw ID breaks ties), then retain at most 100. Every candidate has an explicit decision in `metadata.json`. `--no-dedup`, `--min-score`, and `--proposal-limit` configure these rules. `--save-candidate-limit N` optionally caps the raw archive in grid order; retained masks are still saved even outside this cap and omissions are recorded explicitly. No ground truth participates in selection.

Overlapping retained masks remain overlapping proposals, not a panoptic scene partition. Overlay overlap colors are averaged. The contact sheet shows the original, retained-proposal overlay, and every retained individual mask. A finite grid can miss small objects; background, object parts and nested masks may remain. These proposals do not establish exhaustive scene understanding. Fence and evaluation-preparation modules are untouched.

Per-image metadata includes source/checkpoint/code/tokenizer SHA256, grid coordinates, all scores/decisions, model-load and embedding/query/selection/save runtimes, and peak PyTorch CUDA allocated/reserved bytes (including resident model, not total system GPU usage). No autocast or TF32 is used; seed 0 is recorded but GPU bitwise determinism is not guaranteed. Model transfer uses `.to(device="cuda")` without a dtype cast, preserving complex rotary-position buffers and their imaginary components. PIL converts to RGB without EXIF rotation. Hole/sprinkle cleanup on the existing integrated predictor is disabled explicitly. A failed image fails the run and updates `run.json`; no alternate model or inference path is substituted.

## Local checks

No SAM3, torch, GPU or weights are needed for input listing or mocked tests:

```bash
python scripts/sam3_scene_masks.py --image-dir data/examples/natural_occlusions --limit-images 7 --list-images
python -m unittest discover -s tests -p test_scene_proposals.py -v
```

## Vulcan prerequisites and live resource checks

User-confirmed: `vho@vulcan.alliancecan.ca`, account `aip-sven`, partition `gpubase_bygpu_b1`, L40S GPU. A prepared **native Meta SAM3** environment and authorized local `sam3.pt` must already exist. Meta documents Python >=3.12, PyTorch >=2.7 and CUDA >=12.6; consult the installed source's requirements. No installation, checkpoint download, GPU inference or submission was performed locally.

The batch requests 1 GPU, 4 CPUs, 32G and 30 minutes. Live partition limits cannot be established from repository files or public documentation. **Run the read-only checks below first**, inspect `MaxTime`, resource availability and account/QOS restrictions; reduce or override batch resource options if required. `--test-only` validates the exact request without submitting. It is not a guarantee of immediate GPU availability.

Outputs use `/scratch/vho/sam3-whole-scene/results/job_JOBID`, as requested. [Vulcan documentation](https://ualberta-rcg.github.io/ragflow-wiki-data/en/clusters/vulcan/) identifies scratch as purgeable, not permanent archival storage. Copy results back promptly. No `$PROJECT` path is assumed.

From WSL (replace repository/environment/checkpoint paths with existing cluster locations):

```bash
ssh vho@vulcan.alliancecan.ca
# On the login node: configuration checks and submission only, never inference.
REPO=/scratch/vho/bw-lab-project-2025
VENV=/scratch/vho/venvs/sam3
CHECKPOINT=/scratch/vho/checkpoints/sam3.pt
IMAGES=/scratch/vho/sam3-whole-scene/inputs
bash "$REPO/scripts/vulcan_scene_preflight.sh"
mkdir -p /scratch/vho/sam3-whole-scene/logs "$IMAGES"
sbatch --test-only "$REPO/scripts/vulcan_scene_masks.sbatch" "$REPO" "$VENV" "$CHECKPOINT" "$IMAGES"
```

Transfer the seven ignored images separately from WSL; the repository commit does not contain them:

```bash
rsync -av --partial --protect-args /mnt/c/Users/venni/Code/bw-lab-project-2025/data/examples/natural_occlusions/ vho@vulcan.alliancecan.ca:/scratch/vho/sam3-whole-scene/inputs/
```

Ensure the cluster checkout includes this commit (normal Git fetch/checkout on your development branch, without overwriting uncommitted work). Then, on Vulcan, submit the single-image smoke test manually:

```bash
sbatch "$REPO/scripts/vulcan_scene_masks.sbatch" "$REPO" "$VENV" "$CHECKPOINT" "$IMAGES"
# Default: first image in filename order, 3x3 independent grid, all raw masks archived.
squeue -u vho
# Replace JOBID with sbatch's returned job number:
tail -f /scratch/vho/sam3-whole-scene/logs/sam3-scene-JOBID.out
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,AllocTRES
```

Inspect that smoke result before manually requesting all seven. Grid density is configurable; more points increase runtime and may require a longer time limit after observing the smoke test. Verify limits again before any resource override:

```bash
sbatch --test-only "$REPO/scripts/vulcan_scene_masks.sbatch" "$REPO" "$VENV" "$CHECKPOINT" "$IMAGES" --limit-images 7 --points-per-side 8
sbatch "$REPO/scripts/vulcan_scene_masks.sbatch" "$REPO" "$VENV" "$CHECKPOINT" "$IMAGES" --limit-images 7 --points-per-side 8
```

Download from WSL (after completion; no `--delete`):

```bash
mkdir -p ~/sam3-whole-scene-results
rsync -av --partial --protect-args vho@vulcan.alliancecan.ca:/scratch/vho/sam3-whole-scene/results/job_JOBID/ ~/sam3-whole-scene-results/job_JOBID/
rsync -av --partial --protect-args vho@vulcan.alliancecan.ca:/scratch/vho/sam3-whole-scene/logs/ ~/sam3-whole-scene-results/logs/
explorer.exe "$(wslpath -w ~/sam3-whole-scene-results)"
```

Open each image directory's `contact_sheet.png`, `all_proposals_overlay.png`, and `metadata.json`. Scheduler logs and `run.json` should be retained with the masks for reproducibility. These commands are instructions only; no job is automatically submitted.
