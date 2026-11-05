# Point-prompt fine-tuning: start here

This is the active fine-tuning experiment for the perceptual-grouping project. It uses Meta's native **interactive modules**, with
a small explicit PyTorch loop. It does not use Meta's concept-segmentation Trainer.

## What one training step does

1. Load an RGB image and one COCO instance's binary target mask.
2. Choose one positive point: the foreground pixel nearest the mask centroid.
   This always lands on the target, even if its centroid is outside the mask.
3. Apply native 1008-pixel preprocessing, image encoder, prompt encoder and
   interactive mask decoder. Three candidate masks and quality scores result.
4. Compare candidates with the target and backpropagate
   `20*focal(alpha=.25,gamma=2) + soft_Dice + quality_MSE`.
5. Clip gradients and update trainable parameters with AdamW, learning rate 1e-5.

An **instance mask** is simply the binary mask of one object: 1 for that object,
0 elsewhere. This loss does not imply a text query or concept task. Focal loss
penalizes incorrect pixel predictions; Dice measures foreground overlap; MSE
teaches predicted quality scores to match measured mask IoU. Only the candidate
with the smallest focal+Dice error receives mask supervision; quality supervision
covers all three candidates. Evaluation selects the highest *predicted* quality,
without using ground truth to select the mask.

This is inspired by [SAM, Appendix A](https://arxiv.org/html/2304.02643v1), which
describes the 20:1 mask losses, minimum-loss candidate and quality-score MSE.
It is a small single-point sanity experiment, not reproduction of the full
iterative-click training recipe or a published fine-tuning result.

## Controlled comparison

| Scope | Image encoder | Prompt encoder + mask decoder |
|---|---|---|
| `full` | Train shared vision trunk and interactive feature neck | Train |
| `frozen_encoder` | Frozen, with no encoder autograd graph | Train |

Unused concept/language/video modules stay frozen in both runs. Decoder feature
projections stay trainable even when the image encoder is frozen. Full means the
active point-prompt model, not every unrelated SAM3 branch. The full run uses
native activation checkpointing to reduce activation memory, at a compute cost.
Both use identical initial weights, data, order, point rule, loss and learning rate.
Encoder train/eval behavior follows its trainable/frozen status. BF16 is used for
forward computation; loss and stored parameters remain float32.

The initial subset is eight training source images and four disjoint held-out
source images, all from **train2017**. Every noncrowd instance in those images is
one sample, so there may be more than eight training steps per epoch. **val2017
is untouched.** Clean images only: `datasets.point_training.dataset(..., transform=...)` is the
in-memory hook for later fragmentation/fence experiments; the launcher currently
rejects any non-identity transform configuration.

This target-directed prompt differs from the earlier scene-wide 3x3 grid. The
held-out results should be compared to this run's own pre-training baseline,
not directly to the older grid-selected IoU figures.

Similar improvements in both scopes would suggest encoder updates were not
needed under these experimental conditions. A tiny run cannot establish that
the decoder is sufficient in general. Failure to improve calls for checks of
data, prompts, loss and optimization before concluding an architecture limitation.

## Vulcan: first prove that two updates work

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin performance
module load python/3.12.4
module load opencv/4.12.0
bash scripts/setup_native_training_env.sh
source /scratch/vho/sam3-whole-scene/train-env/bin/activate

POINT_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/point-training.XXXXXX)
echo "Keep this path: $POINT_ROOT"

python scripts/prepare_point_training.py \
  --native-root /project/aip-sven/vho/sam3 \
  --coco-root /scratch/vho/datasets/coco \
  --output-dir "$POINT_ROOT/prepared" --download

SMOKE_ID=$(sbatch --parsable --output="$POINT_ROOT/smoke-%j.out" \
  scripts/vulcan_point_training.sbatch "$PWD" \
  "$POINT_ROOT/prepared/config.json" frozen_encoder "$POINT_ROOT/smoke" 2)
echo "Smoke job: $SMOKE_ID"
echo "Results: $POINT_ROOT"
```

Preparation downloads only missing selected train2017 images and caches them;
the script does not download all training images or run a model on the login node.
The annotation JSON may already be cached in the train/validation annotation ZIP.
Training requires a Slurm GPU allocation and refuses the login node. Each job
requests one GPU, 12 CPU cores, 64 GB host memory and eight loader workers.

```bash
sacct -j "$SMOKE_ID" --format=JobID,State,Elapsed,ExitCode
tail -n 35 "$POINT_ROOT/smoke-${SMOKE_ID}.out"
cat "$POINT_ROOT/smoke/complete.json"
```

After a successful smoke run, launch both independent two-epoch runs:

```bash
for SCOPE in frozen_encoder full; do
  sbatch --output="$POINT_ROOT/${SCOPE}-%j.out" \
    scripts/vulcan_point_training.sbatch "$PWD" \
    "$POINT_ROOT/prepared/config.json" "$SCOPE" "$POINT_ROOT/$SCOPE"
done
```

These are separate single-GPU jobs, not one model distributed across two GPUs.
If a run fails or exhausts memory, inspect its log rather than silently changing
the scope, target, preprocessing or precision. Every output directory must be new.

## What to inspect and what has actually been tested

* `metadata.json`: native source/checkpoint provenance, split configuration,
  parameter counts and pre-update agreement with native single-point inference.
* `audit.json`: actual nonzero encoder/decoder gradients as required by scope,
  and measured changes to checked weights after the first optimizer step.
* `train.jsonl`: per-instance loss components, selected-mask IoU and gradient norm.
* `heldout.jsonl`, `summary.jsonl`: baseline at epoch 0, then held-out scores after
  each epoch. The lowest-loss training candidate is not used for evaluation.
* `previews/epoch-*.png`: the first four held-out instances, with input point,
  COCO target, baseline mask and updated mask, each labelled with IoU.
* `checkpoint.pt`: full model/optimizer state, scope and configuration. The file
  is read back to verify its metadata; `complete.json` appears only on success.

The checkpoint uses BWLab format `bwlab_point_finetune_v1`, not Meta's original
`detector.`/`tracker.` prefixes. To restore it, initialize the same native model
using `load_scene_model` and the original checkpoint, then call
`model.load_state_dict(torch.load(path, weights_only=True)['model'], strict=True)`.
Do not pass this checkpoint directly to the existing evaluation CLI yet.

CPU tests exercise gradient flow, frozen weights, trainable decoder projections,
candidate selection, decreasing toy-model loss and checkpoint restoration. A local
component check also passed backward through the pinned native prompt encoder,
two-way transformer and mask decoder at reduced dimensions. Actual
SAM3 GPU forward/backward, memory fit and learning remain to be tested on Vulcan.
The first real job checks native-forward agreement before attempting updates.
That is an implementation check, not a requirement that batched predictions match
pixel-for-pixel, and not evidence of improved segmentation accuracy.

W&B defaults to **offline** and stores the run inside the output directory.
Nothing is uploaded automatically. For online logging, run `wandb login` yourself
and submit with `WANDB_MODE=online`; this also requires network access on the
compute node. GPU/CPU system metrics are sampled by W&B where supported.

## Implementation map

`experiment.py` constructs the components. `datasets/point_training.py` loads
instances; `core/models/point_sam3.py` owns the native differentiable point path;
`core/tasks/point_segmentation.py` computes the loss; `core/trainer.py` runs the
optimization/evaluation loop; `core/training_output.py` stores logs and previews.
