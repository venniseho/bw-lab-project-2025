# Dataset → transforms → SAM3

Use the worktree `.worktrees/stimulus-refactor` for this branch. Python 3.12 is
the deployment target; no Python 3.14 syntax or mlvbench dependency is introduced.
The minimal `pyproject.toml` allows editable installation with
`python -m pip install --no-deps --no-build-isolation -e .` in the existing environment;
this does not reinstall SAM3, PyTorch or checkpoints. CPU dependencies are NumPy, Pillow (>=10.1 for sized default fonts), OpenCV,
Matplotlib, PyTorch and torchvision. COCO file loading additionally needs
pycocotools; an existing COCO API is injected rather than downloaded.

```python
from torchvision.transforms import Compose
from bwlab.datasets import ImageMaskDataset, CocoInstanceDataset, build_loader
from bwlab.stimuli import FenceOcclusion, ContourFragmentation
from bwlab.stimuli import FragmentationConfig

fence = FenceOcclusion(bar_width=12, bar_spacing=40, bar_angle=90, seed=42)
contour = ContourFragmentation(FragmentationConfig(noise_count=300), seed=42)
dataset = ImageMaskDataset(records, transform=Compose([fence]))
sample = dataset[0]
# Or: CocoInstanceDataset(coco, image_root, explicit_image_ids, transform=fence)
loader = build_loader(dataset, split="test", num_workers=0)
```

Choose fence and contour as separate experimental arms, each applied to the
same original sample. Compose also supports vertical/horizontal fences together.
Applying fragmentation after a fence is rejected because it regenerates the
boundary from the original target and would discard occlusion semantics.

| Sample field | Contract |
| --- | --- |
| `__key__` | Stable nonempty string; COCO uses `coco/image_id/annotation_id`. |
| `image` | Nonempty uint8 RGB `(H,W,3)`; original native coordinates. |
| `segmentation` | Original full target, binary HW, preserved including encoding. |
| `visible_segmentation` | Fence arm only: boolean `segmentation & ~occlusion_mask`. |
| `occlusion_mask` | Fence arm only: boolean, True means hidden by synthetic bars. |
| `metadata` | Original source/selection metadata, copied and retained. |
| `transforms` | Ordered parameter and seed provenance. |
| `fragmentation` | Outline/noise segments, effective edge/gap, RNG protocol. |

Datasets return fresh arrays. Transforms do not mutate inputs. Seeded transforms
are repeatable when image geometry/configuration is fixed. The same fence seed
uses the same pixel phase across images; choose per-image seeds explicitly if
desired, and reuse each seed across that image's severities. Train/validation/test
records must be split by source image before expanding instances or severities;
the loader does not decide scientific splits. It shuffles training only and
keeps variable-size samples in lists. Worker prefetch is set only for workers>0.
COCO selection decisions are inspectable as `dataset.selection_records`.

## Prepare local inputs on CPU

Create a manifest such as `records.json` (paths relative to that file):

```json
[{"__key__":"example/1","image_path":"image.png","mask_path":"mask.png"}]
```

```sh
python scripts/prepare_stimuli.py --manifest records.json --output-dir outputs/fence_run --transform fence --widths 0 4 12 24 --spacing 40 --angle 90 --seed 42
python scripts/prepare_stimuli.py --manifest records.json --output-dir outputs/contour_run --transform fragmentation --seed 42
```

Optional `--fragmentation-config config.json` accepts the fields of
`FragmentationConfig`; e.g. `{"noise_count":600,"outline_mode":"random"}`.
Omit `--seed` for historical unseeded contour RNG behavior. Fence `--phase`
controls the offset directly; omit `--seed` when supplying it. Each output
directory must be new. A partial failure leaves a `preparing` manifest for
inspection, not a misleading complete run.

Only `inputs/` contains model-ready RGB PNGs. `targets/` holds NPZ archives with
the original mask and, for fences, separate visible/occlusion masks. Contour NPZs
also preserve the generated segment coordinates. `manifest.json` maps stable
keys and conditions to files and transform parameters. These are preparation
artifacts, not inference or evaluation results.

## Run the existing SAM3 path after GPU approval

The following is a command template for an approved SLURM GPU allocation;
this refactor has not run it or submitted a job:

```sh
python scripts/sam3_scene_masks.py --image-dir outputs/fence_run/inputs --limit-images 4 --checkpoint /scratch/vho/sam3-whole-scene/checkpoints/sam3.pt --output-dir /scratch/vho/sam3-whole-scene/results/fence_run --points-per-side 3 --no-dedup
```

Set `--limit-images` explicitly to the prepared input count, since the existing
CLI defaults to one image. Refer to [Vulcan setup](vulcan_scene_setup.md) for the
existing allocation/environment workflow. Keep Meta source and checkpoint fixed.
This is automatic grid prompting, not prompt-free segmentation. Existing
deduplication, score threshold and proposal-limit options are recorded selection
policies; `--no-dedup` keeps raw proposal distinctions, with the archive retaining
all candidates by default. Contact sheets display retained proposals only.

Rebuild an existing complete scene archive with improved labels, on CPU:

```sh
python scripts/relabel_scene.py --scene-dir /path/to/saved/scene --output /path/to/saved/scene/contact_sheet_labeled.png
```

This requires the scene's `original.png`, `metadata.json`, and saved masks; it
refuses to overwrite an existing output. The seven local natural-example
directories currently contain contact sheets but lack the original RGB/metadata
ledger and usable mask exports, so they cannot be faithfully rebuilt locally.
Use the complete Vulcan result directory; do not invent candidate scores or
coordinates from a flattened image. New inference exports use the new labels.

The primary configured entrypoint is now
[`experiments/sam3_fragmentation/experiment.py`](../experiments/sam3_fragmentation/experiment.py),
which wires the DataModule, transforms, Sam3GridAdapter, Runner and concrete
SceneProposalTask together. See the [experiment guide](../experiments/sam3_fragmentation/README.md)
for JSON configuration and `prepare`/`run` commands. New code imports the reusable
`bwlab` package; previous paths remain compatibility imports.

For in-memory integration, `bwlab.models.sam3.Sam3GridAdapter.predict()` reuses
`collect_proposals` on an already loaded model/processor. `bwlab.runner.Runner`
reads explicit evaluation loaders and passes only RGB to the model; its task's
evaluator receives sample dictionaries and archives targets separately through
`update()` / `finalize()`. `SceneProposalTask` saves the existing proposal archive
and reports artifact locations/counts, without introducing quantitative scores.
The old `run_pipeline` callable interface remains for compatibility.

```python
from bwlab.datasets import ManifestDataModule
from bwlab.models import Sam3GridAdapter
from bwlab.runner import Runner
from bwlab.tasks.scene_proposals import SceneProposalTask

module = ManifestDataModule({"test": manifest_path}, transform=fence)
runner = Runner(module, SceneProposalTask())
# Inside an approved inference context, with an already loaded native model:
# results = runner.evaluate(Sam3GridAdapter(model, processor), output_path)
```

The full structure and reference correspondence are documented in
[architecture.md](architecture.md). Ordinary CPU preparation does not load SAM3.

## Interpretation and validation

Synthetic full-object targets refer to the source annotation before added bars.
Ordinary COCO masks do not establish completion of regions already naturally
occluded in the source image. A future genuine amodal test needs suitable amodal
annotations. Report visible segmentation separately from full-target recovery;
inspect hidden-region behavior before claiming completion. Oracle best-mask IoU
must be labeled an upper bound, separately from model-score selection.

Regression tests check code behavior, not human-like grouping or causal failure
localization. Current CPU tests use synthetic masks and mocked SAM3; the merged baseline has now passed the 93-test suite on Vulcan Python 3.12 and
a real five-condition GPU smoke test (job 1762018). See the validation record
for the separately tested rendering follow-up.
After explicit merge approval, branch from updated main to extend quantitative
protocols, paired prompts, selection policies and result aggregation.

Run focused tests (historical root experiments require external data/models and
are not a self-contained discovery suite):

```sh
python -B -m unittest discover -s tests -p test_fence.py -v
python -B -m unittest discover -s tests -p test_coco_selection.py -v
python -B -m unittest discover -s tests -p test_probe_preparation.py -v
python -B -m unittest discover -s tests -p test_sam3_smoke.py -v
python -B -m unittest discover -s tests -p test_scene_proposals.py -v
python -B -m unittest discover -s tests -p 'test_stimulus_*.py' -v
python -B -m unittest discover -s tests -p test_visualization.py -v
```

Prompt overlays now label every queried point as G0, G1, etc. on both header
previews. Individual candidates show their own query over RGB plus the colored
mask. These are display-only markers; saved raw masks and original RGB are
unchanged. See [supervisor_update.md](supervisor_update.md) for a project update.
