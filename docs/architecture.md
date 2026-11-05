# Code organization

The layout follows [mlvbench](https://github.com/mtangemann/mlvbench): reusable
components, small experiment assembly functions, and separate cluster launchers.
Dataset loading is a top-level package at the project's author's request.

```text
datasets/
  coco.py, coco_manifest.py       Lazy COCO image/instance loading
  image_mask.py, manifest.py      Explicit image/mask manifests
  selection.py                   Instance validity and exclusion reasons
  splits.py                      Disjoint source-image training splits
  download.py                    Atomic, cached official COCO downloads
  conditions.py                  Apply each condition to the original sample
  point_training.py              Tiny point-training dataset and loader

core/
  models/
    sam3.py                      Native model loading and grid adapter
    batched_sam3.py               Batched evaluation adapter
    point_sam3.py                 Differentiable point path and trainable scope
    proposals.py                 Candidate generation and selection
  stimuli/
    transforms.py                Callable fence/fragmentation transforms
    fence.py                     Fence geometry and coverage
    fragmentation.py             Fragmentation configuration and RNG ownership
    _contours.py                 Contour/noise geometry; no file I/O
    export.py                    Optional stimulus export
  tasks/
    scene_proposals.py            Evaluation and proposal export
    point_segmentation.py         Point target construction and mask loss
    preparation.py               Common-visible prompts across conditions
  runner.py                      Inference loop with injected model and evaluator
  trainer.py                     Training loop with injected segmenter/task/output
  training_output.py             Training logs and before/after output
  visualization.py               Rendering only
  samples.py                     Shared image/mask contracts
  infrastructure.py              File fingerprints and Slurm guard
  runtime.py, monitoring.py       Evaluation performance configuration/reporting

experiments/
  sam3_fragmentation/             Evaluation configuration, run and cluster launch
  sam3_point_finetuning/          Fine-tuning assembly and instructions
```

## What the design rules mean here

* **Single responsibility:** loaders decode samples; transforms modify samples;
  model adapters call SAM3; tasks calculate loss/metrics; outputs persist results.
  `experiment.py` assembles these pieces rather than implementing them.
* **Open for extension:** new transforms are callables over the same sample
  contract. Runners receive their models/tasks; no condition-specific switches
  belong in the training loop.
* **Substitution:** dataset implementations share PyTorch's indexing contract;
  model/task implementations must preserve their documented input/output contract.
  Tests exercise the actual composition with small model doubles.
* **Small interfaces:** evaluation uses `Model`, `Task` and `Evaluator` protocols.
  Training uses a separate `TrainableSegmenter` protocol; an inference adapter
  does not need dummy optimization methods.
* **Dependency injection:** `Runner` and `PointTrainer` receive their dependencies.
  They do not download COCO, construct SAM3, log into W&B or submit cluster jobs.

Classes are used for components that own state or implement a callable contract.
Pure geometry and loss calculations remain functions. SOLID is a design guide,
not a reason to turn every helper into a class or add inheritance hierarchies.

## Scientific behavior preserved

This cleanup does not change fence formulas, fragmentation geometry, source
selection policies, point coordinates, candidate selection, mask losses or
training scopes. Contour RNG streams are now passed explicitly rather than
installed into dynamically loaded legacy modules; their order is preserved.
Unseeded contour output still uses entropy-based random generators.

`full` fine-tuning updates the image encoder and interactive prompt/mask modules.
`frozen_encoder` updates the interactive prompt/mask modules only. Unused text,
concept and video modules are frozen in both runs. This remains a clean-image
training sanity experiment, with the existing transform hook reserved for later.

The superseded concept-segmentation training experiment was archived. Native
point-prompt training does not import its configuration, trainer or loss.

## Local cleanup and compatibility

The former `bwlab` package is now `core`; its dataset modules are now `datasets`.
Update external Python imports accordingly and refresh the editable install with
`python -m pip install -e . --no-deps`. Repository/Vulcan paths and active command
filenames remain unchanged. There is no compatibility package hiding the rename.

Historical assets and generated outputs were archived locally, verified by hash,
and removed from the active tracked tree. Ten original source/mask pairs remain
as offline fixtures. Plot/contact-sheet downloads were relocated to
`outputs/figures`. No model checkpoints or real datasets were removed.

Run checks with `python -B -m unittest discover -s tests -t . -v`. The frozen
contour reference in tests is deliberately excluded from production style rules:
its exact bytes are the independent historical regression baseline.
