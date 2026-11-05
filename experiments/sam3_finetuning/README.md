# Superseded for the perceptual-grouping experiment

This folder trains the concept-segmentation branch. The requested point-prompt
experiment now lives in [sam3_point_finetuning](../sam3_point_finetuning/README.md).
Use that folder's commands; the older commands below target a different task.

# Native SAM3 segmentation training smoke check

The first objective is verified mask-loss backpropagation and a saved checkpoint,
not a published metric or a claim that contour grouping improved. This uses
Meta's real Trainer, dataset, matching, mask loss, AMP, optimizer loop and checkpoint
writer at revision `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`.

The native trainer runs **concept/category-prompted instance segmentation**.
`mask_head` trains `segmentation_head.*` only; it does **not** train the interactive
`inst_interactive_predictor.model.sam_mask_decoder` used by the point-grid baseline.
Do not evaluate this checkpoint as an updated point decoder. `full` trains the
whole concept model, and `encoder` trains only `backbone.vision_backbone.*`.
No tracker or interactive predictor is instantiated in any scope. These scopes
are recorded with the exact parameter names and counts.

## First run on Vulcan

Start in the project checkout. All commands are executed by the user; setup and
preflight do not run a model. The final sbatch command requests a compute GPU.

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git pull --ff-only origin performance
module load python/3.12.4
module load opencv/4.12.0
bash scripts/setup_native_training_env.sh
source /scratch/vho/sam3-whole-scene/train-env/bin/activate

TRAIN_ROOT=$(mktemp -d /scratch/vho/sam3-whole-scene/native-training.XXXXXX)
echo "Keep this path: $TRAIN_ROOT"

python scripts/prepare_native_training.py \
  --native-root /project/aip-sven/vho/sam3 \
  --coco-root /scratch/vho/datasets/coco \
  --output-dir "$TRAIN_ROOT/prepared" \
  --download --train-images 8 --heldout-images 4 \
  --epochs 2 --workers 8 --scope mask_head

python -m experiments.sam3_finetuning.launch \
  --config "$TRAIN_ROOT/prepared/training.json" --preflight

JOB_ID=$(sbatch --parsable --output="$TRAIN_ROOT/slurm-%j.out" \
  scripts/vulcan_native_training.sbatch "$PWD" \
  "$TRAIN_ROOT/prepared/training.json")
echo "Job: $JOB_ID"
echo "Results: $TRAIN_ROOT/prepared/run"
```

The training environment adds packages in a separate venv, reusing the existing
SAM3 environment through `site.addsitedir`. Torch/torchvision/numpy/pillow/timm
versions are constrained to the working stack. It does not overwrite that base
environment, but depends on it remaining present. Setup can resume after a failed
package download; a conflicting preexisting environment is not overwritten.

The shared COCO annotation archive is reused, extracting **train2017** annotations.
Only twelve source JPEGs are downloaded. The JSON is large and may take time to
read. Standard COCO instance masks and category names are retained; crowd instances
are excluded. All noncrowd instances in each selected source are retained. Selection
is seeded, and training/held-out source IDs are disjoint. `test` in the prepared
folder means the four held-out **train2017** images, not official COCO test/val.
No validation-set training or precomputed transformed JPEGs are used.

The dataset config uses native transforms and 1008 resolution; no fence,
fragmentation or crop ablations are added yet. Mask decoding is enabled in both
training and held-out pipelines. Box/classification losses from the official example
are retained and instance-mask focal/dice loss is added. The optional semantic
mask criterion is not used. All weights load from the existing SAM3 checkpoint;
there is no Hugging Face download. The short run uses fixed AdamW lr=1e-5,
batch size 1, bfloat16 AMP, two epochs, eight workers, and one GPU with twelve
requested CPU cores. Held-out evaluation measures loss only, not AP/IoU.

The native forward computes assignment only during training. A read-only forward
hook calls the same native matcher on eval outputs so held-out mask loss is valid;
it adds assignment indices and does not change prediction tensors. The native
Trainer's training forward/backward/optimizer loop is not replaced or monkeypatched.

## Evidence and monitoring

```bash
squeue -j "$JOB_ID"
tail -n 80 "$TRAIN_ROOT/slurm-${JOB_ID}.out"
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,ExitCode
cat "$TRAIN_ROOT/prepared/run/training_audit.json"
```

Expected evidence:

- `loss_steps.jsonl`: native mask/dice and total losses for train and held-out steps.
- `optimizer_steps.jsonl`: actual optimizer updates, changed weight probes,
  a frozen-weight probe, step wall time and allocated GPU bytes.
- `trainable_parameters.json`: exact parameter names and trainable/total counts.
- `training_audit.json`: completion status and verified optimizer-update count.
- `checkpoints/checkpoint.pt`: latest checkpoint, preserving the input checkpoint.
- `tensorboard/`, `logs/train_stats.json`, `logs/val_stats.json`: native logging.
- `gpu_utilization.csv`: sampled GPU utilization/memory, including UUID/index
  so the allocated GPU can be identified. Audit timings add synchronization and
  parameter checks; this is not a fair throughput benchmark against inference.
- `provenance.json`, `resolved_config.yaml`, `selection.json` in preparation root:
  sources, revisions and fingerprints. Scratch results should be preserved later.

W&B is **offline by default**: it records losses, audit metrics and the SDK's
available system statistics without login or compute-node internet access. SDK
hardware monitoring availability depends on the node; native logs remain available.
To disable it, prepare with `--wandb-mode disabled`. To upload an offline run
later, explicitly use `wandb sync` from a network-enabled environment after login.
Do not assume setting the native config's unused `wandb_writer` enables tracking;
this integration logs through the audit subclass instead.

## Next decisions after a successful smoke check

1. Plot the loss log and inspect finite mask losses, nonzero gradients, changed
   trainable weights and saved checkpoint. Do not expect eight images to establish
   generalization or promise monotonic training loss under randomized transforms.
2. Choose the actual research path: concept segmentation adaptation versus a
   separate supported training loop for the interactive point decoder.
3. Establish a fixed target-directed held-out evaluation before reporting grouping
   improvements. Highest-score masks anywhere in the image confound the old grid
   baseline's target IoU.
4. Add fragmentation/fence transforms during training as a separate controlled
   change, then compare mask-head-only, full-model and explicit frozen-head scopes.
5. After basic training is working, extend duration, data and W&B online monitoring.
   A full reproduction needs an identified metric, dataset and recipe; the smoke
   check is not a reproduction of the third-party template's runtime claim.
