# Vulcan one-image SAM3 setup: automatic 3x3 visual grid

Nine positive points are placed automatically; each is an independent visual query with `multimask_output=True`. Typically this yields 27 raw masks; actual output count is recorded. No clicks, text prompts, model replacement or GT-based mask selection. These are qualitative proposals: small objects can be missed, and object parts/background can be proposed.

All raw binary masks and individual RGB overlays are saved, including empty/duplicate/low-score candidates. `retained/` contains masks retained after filtering/deduplication. `raw_all_proposals_overlay.png` shows all raw candidates, `all_proposals_overlay.png` shows retained candidates, and `contact_sheet.png` shows original, retained overlay, and each retained mask. Metadata records original path/checksum, raw/retained counts, all scores/decisions, timings, peak CUDA allocated/reserved bytes, and checkpoint/implementation information. Default retention rules remain unchanged: empty-mask exclusion from retention only, no score threshold, exact binary-mask IoU deduplication >=0.95, descending predicted IoU with raw ID tie-break, retained limit 100. No raw archive limit is used below. Predicted IoU is a quality score, not calibrated object-existence confidence.

## Verification and proposed layout

WSL `ssh vulcan` reached Vulcan with host verification enabled, but non-interactive authentication failed (`keyboard-interactive`). Therefore **existing repo/environment/checkpoint paths and live partition limits were not verified**. Use your working interactive session below. No Windows SSH, remote writes, installation, download, GPU inference or submission was performed by this task.

These paths are **created by the commands**, not assumed to exist:

```text
~/research/bw-lab-project-2025/       project checkout
~/research/sam3/                    official Meta source, revision recorded
/scratch/vho/sam3-whole-scene/
  inputs/                          seven original image files
  env/                             Python venv, if a suitable existing one is absent
  checkpoints/sam3.pt               authorized native SAM3 checkpoint
  setup/                           revisions, package list, checkpoint checksum
  logs/                            SLURM logs
  results/job_JOBID/                one-image smoke results
```

Scratch is purgeable; copy results back promptly. No `$PROJECT` is assumed.

## 1. Inspect existing resources, through WSL

```bash
ssh vulcan
# On Vulcan: read-only checks, no inference.
scontrol show partition gpubase_bygpu_b1
sinfo -N -p gpubase_bygpu_b1 -o '%N %G %c %m %a %l'
sacctmgr -nP show assoc where user="$USER" account=aip-sven format=Cluster,Account,Partition,QOS
python3 --version
module list
find "$HOME" /scratch/vho -maxdepth 5 \
  \( -name .cache -o -name results -o -name outputs -o -name node_modules \) -prune -o \
  \( -name sam3.pt -o -path '*/bin/activate' -o -name .git \) -print 2>/dev/null
exit
```

The inventory is scoped, not exhaustive. If suitable paths are found, reuse them by changing the variables below. Confirm that one L40S GPU, four CPUs, 32 GB and 30 minutes are permitted by partition/account/QOS policy. Resource requests are provisional until checked.

## 2. Copy the exact committed project branch and seven images

From WSL. A Git bundle avoids assuming the latest local commit was pushed to GitHub; it preserves history and requires no cluster GitHub authentication. Ignored images are transferred separately.

```bash
LOCAL=/mnt/c/Users/venni/Code/bw-lab-project-2025
git -C "$LOCAL" status --short
git -C "$LOCAL" bundle create /tmp/sam3-fragmentation-v2.bundle sam3-fragmentation-v2
ssh vulcan 'mkdir -p ~/research /scratch/vho/sam3-whole-scene/{inputs,checkpoints,setup,logs,results}'
rsync -av --partial /tmp/sam3-fragmentation-v2.bundle vulcan:research/
rsync -av --partial --protect-args "$LOCAL/data/examples/natural_occlusions/" vulcan:/scratch/vho/sam3-whole-scene/inputs/
ssh vulcan
```

On Vulcan, clone or update without overwriting existing work:

```bash
REPO="$HOME/research/bw-lab-project-2025"
if [ ! -e "$REPO" ]; then
  git clone --branch sam3-fragmentation-v2 "$HOME/research/sam3-fragmentation-v2.bundle" "$REPO"
else
  git -C "$REPO" status --short
  git -C "$REPO" branch --show-current
  test -z "$(git -C "$REPO" status --porcelain)" && \
  test "$(git -C "$REPO" branch --show-current)" = sam3-fragmentation-v2 && \
  git -C "$REPO" fetch "$HOME/research/sam3-fragmentation-v2.bundle" sam3-fragmentation-v2 && \
  git -C "$REPO" merge --ff-only FETCH_HEAD
fi
git -C "$REPO" log -1 --oneline
ROOT=/scratch/vho/sam3-whole-scene
VENV="$ROOT/env"
CHECKPOINT="$ROOT/checkpoints/sam3.pt"
IMAGES="$ROOT/inputs"
```

Stop if the clone/update fails, the branch differs, or local changes/divergence are reported. If your usable existing checkout is elsewhere, set `REPO` to its real path. No GitHub push is performed here.

## 3. Activate or prepare native SAM3

If a suitable native Meta SAM3 venv exists, set `VENV` to it and activate. The batch expects `bin/activate`; do not pass a Conda directory as if it were a venv. Otherwise these are **manual installation commands**, not actions already performed. First verify Python >=3.12. If it is unavailable, use `module spider python` to identify and load an available >=3.12 module; its actual name is not verified from this environment.

```bash
python3 -c 'import sys; assert sys.version_info >= (3,12), "Load Python >=3.12 first"'
# Continue only after the above succeeds; do not overwrite an existing environment.
test ! -e "$VENV" && python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
test -e "$HOME/research/sam3" || git clone https://github.com/facebookresearch/sam3.git "$HOME/research/sam3"
python -m pip install -e "$HOME/research/sam3"
```

This follows [Meta's current installation guidance](https://github.com/facebookresearch/sam3#installation). Driver/CUDA compatibility still needs verification on the allocated GPU. Do not install training/notebook extras or optional compiled accelerators for this step. If installation requires heavy compilation or GPU hardware, stop and handle it separately on an allocated node.

CPU-side checks only, no model construction:

```bash
source "$VENV/bin/activate"
python - <<'PY'
import sys, pathlib, torch, sam3
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
assert sys.version_info >= (3,12)
asset = pathlib.Path(sam3.__file__).parent / 'assets/bpe_simple_vocab_16e6.txt.gz'
assert asset.is_file(), asset
print('Python:', sys.version.split()[0], 'torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('Native SAM3:', sam3.__file__, 'tokenizer:', asset)
PY
python -m pip check
python -m pip freeze > "$ROOT/setup/packages.txt"
git -C "$HOME/research/sam3" rev-parse HEAD > "$ROOT/setup/meta-sam3-revision.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/setup/project-revision.txt"
python "$REPO/scripts/sam3_scene_masks.py" --image-dir "$IMAGES" --limit-images 7 --list-images
```

For an existing SAM3 environment, record its real source revision instead of assuming `~/research/sam3` exists. CUDA availability on the login node is not required.

## 4. Native SAM3 checkpoint availability

If an existing checkpoint was found, set `CHECKPOINT` to it. Confirm its provenance is **facebook/sam3 native sam3.pt**, not SAM3.1 or HF safetensors. File existence alone does not prove compatibility; allocated-node loading checks state keys and rejects partial loads.

```bash
test -s "$CHECKPOINT"
ls -lh "$CHECKPOINT"
sha256sum "$CHECKPOINT" > "$ROOT/setup/checkpoint.sha256"
```

If absent, checkpoint access is a blocker. [Meta requires approved Hugging Face access](https://github.com/facebookresearch/sam3#getting-started). Request access to [facebook/sam3](https://huggingface.co/facebook/sam3). **Only after access approval and your authorization to download**, run:

```bash
hf auth login
hf download facebook/sam3 sam3.pt --local-dir "$ROOT/checkpoints"
test -s "$CHECKPOINT"
sha256sum "$CHECKPOINT" > "$ROOT/setup/checkpoint.sha256"
```

No alternative checkpoint is supplied. Inference disables automatic checkpoint downloads.

## 5. Preflight and exact one-image submission

On Vulcan, with `REPO`, `VENV`, `CHECKPOINT`, `ROOT`, `IMAGES` set to verified paths:

```bash
bash "$REPO/scripts/vulcan_scene_preflight.sh"
mkdir -p "$ROOT/logs"
# Scheduler dry run: does not submit.
sbatch --test-only "$REPO/scripts/vulcan_scene_masks.sbatch" "$REPO" "$VENV" "$CHECKPOINT" "$IMAGES" --points-per-side 3 --limit-images 1
# Submit manually only after reviewing preflight and dry-run results:
sbatch "$REPO/scripts/vulcan_scene_masks.sbatch" "$REPO" "$VENV" "$CHECKPOINT" "$IMAGES" --points-per-side 3 --limit-images 1
```

If partition/QOS limits require changing memory/time, supply the same `sbatch --mem`/`--time` overrides to dry run and submission. Default resources: account `aip-sven`, partition `gpubase_bygpu_b1`, one GPU, four CPUs, 32 GB, 30 minutes. The batch uses `srun`; Python independently requires SLURM job AND step IDs. No login-node inference.

The one selected image is the first filename in order, `cat-behind-fence-iStock.jpg`. All seven are transferred; the other six remain ready. No seven-image job is submitted or suggested for immediate execution in this step.

## 6. Monitor and download with WSL rsync

On Vulcan, replace `JOBID` with the submitted number:

```bash
squeue -u vho
tail -f /scratch/vho/sam3-whole-scene/logs/sam3-scene-JOBID.out
cat /scratch/vho/sam3-whole-scene/logs/sam3-scene-JOBID.err
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,AllocTRES
```

After completion, from WSL:

```bash
JOBID=123456  # replace with your actual job number
LOCAL=/mnt/c/Users/venni/Code/bw-lab-project-2025
DEST="$LOCAL/outputs/vulcan_scene/job_$JOBID"
mkdir -p "$DEST"
rsync -av --partial --protect-args "vulcan:/scratch/vho/sam3-whole-scene/results/job_$JOBID/" "$DEST/"
rsync -av --partial "vulcan:/scratch/vho/sam3-whole-scene/logs/sam3-scene-$JOBID.*" "$DEST/"
rsync -av --partial vulcan:/scratch/vho/sam3-whole-scene/setup/ "$DEST/setup/"
explorer.exe "$(wslpath -w "$DEST")"
```

`outputs/` is Git-ignored. No `rsync --delete`. Keep logs and setup manifests with the images, inspect the smoke output, then decide separately whether to process the remaining six.
