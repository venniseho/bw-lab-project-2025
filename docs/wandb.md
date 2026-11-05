# Online Weights & Biases on Vulcan

Use project `bwlab-point-finetuning`. An **entity** is the personal/team workspace
that owns a project; it is the first part of its URL:
`https://wandb.ai/ENTITY/bwlab-point-finetuning`. It may differ from your username.
The project can be created by the first successful upload if you have permission.

## Authenticate and upload the completed smoke run

Run on the Vulcan login node. This uploads existing logs; it does not run SAM3:

```bash
module load python/3.12.4
source /scratch/vho/sam3-whole-scene/train-env/bin/activate
wandb login
```

Follow the CLI's authorization link and enter the API key at its terminal prompt.
Do not put the key in a script, Git commit or chat message. Login is saved under
your account so batch jobs can authenticate without an interactive prompt.

Set the workspace name using the workspace shown in your W&B browser URL:

```bash
export WANDB_ENTITY="YOUR_WANDB_WORKSPACE"
export WANDB_PROJECT=bwlab-point-finetuning
export WANDB_MODE=online

wandb sync --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" \
  /scratch/vho/sam3-whole-scene/point-training.DAHWRv/smoke/wandb/offline-run-20261105_104022-3y9dst31
```

Replace `YOUR_WANDB_WORKSPACE` before running. `wandb sync` prints a link to the
uploaded run. Existing smoke logs retain their original metric names/axes and
contain only what that run originally logged; new image logging is not retroactive.

## Use the updated online integration

```bash
cd /project/aip-sven/vho/bw-lab-project-2025
git fetch origin
git switch refactor/project-cleanup
git pull --ff-only origin refactor/project-cleanup
python -m pip install -e . --no-deps

POINT_ROOT=/scratch/vho/sam3-whole-scene/point-training.DAHWRv
JOB_ID=$(sbatch --parsable --output="$POINT_ROOT/online-smoke-%j.out" \
  scripts/vulcan_point_training.sbatch "$PWD" \
  "$POINT_ROOT/prepared/config.json" frozen_encoder "$POINT_ROOT/online-smoke" 2)
echo "Job: $JOB_ID"
```

The entity/project/mode exports must be present in the shell submitting `sbatch`;
Slurm normally exports that environment to the job. New jobs default to online,
and the log prints `W&B dashboard: ...` after successful W&B initialization. This
first online smoke run also validates the refactored code on the GPU. Use a fresh
output directory each time; the existing offline smoke results remain available.

If staying on `performance`, exporting `WANDB_MODE=online` also enables its
existing scalar logging; the new axes, grouping and preview uploads require the
updated cleanup branch. Do not switch branches while a job still uses that checkout.

## What the dashboard contains

* Training loss, focal/Dice/quality components, selected-mask IoU and gradient norm,
  plotted against actual optimizer step (`train/step`).
* Held-out mean IoU **and loss**, instance count and epoch. Epoch 0 is the
  pre-training baseline; a max-steps smoke run can stop during its first epoch.
* Up to four held-out before/after contact sheets per evaluation epoch, under
  `examples`. Their captions identify the target and before/after IoU.
* Native/checkpoint/data configuration, trainable parameter counts and precision.
* Final optimizer-step count, measured run duration and peak Torch-allocated VRAM.
* W&B's sampled GPU/CPU/RAM system metrics where supported by the compute node.

Runs from the same output parent are grouped together; `job_type` identifies
`full` versus `frozen_encoder`. Override the group with `WANDB_RUN_GROUP` if needed.
Model checkpoint files stay on scratch; this integration does not upload them.
Local JSONL logs, previews and checkpoints are still saved alongside cloud logs.

If cloud connectivity is unavailable on a compute node, explicitly submit with
`WANDB_MODE=offline` and sync afterward. Online initialization errors are not
silently converted to offline runs. Run `wandb login --relogin` if an expired or
incorrect saved login needs replacement.

References: [W&B custom axes](https://docs.wandb.ai/models/track/log/customize-logging-axes),
[official CLI implementation](https://github.com/wandb/wandb/blob/main/wandb/cli/cli.py),
[automatic system metrics](https://wandb.ai/site/sdk/).
