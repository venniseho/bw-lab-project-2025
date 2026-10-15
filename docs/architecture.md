# Structure following the pinned mlvbench revision

Reference: https://github.com/mtangemann/mlvbench/tree/c204a8185fb662b3b1edacfb7a98f7b36fbd9a70

The reusable library and experiment wiring are separate, as in Matthias's repo.
The initial PR exposed similar interfaces within the old directories; the
revised implementation makes that organization the actual source of the code.

```text
bwlab/                          reusable library, analogous to mlvbench/
  samples.py                    shared dictionary and binary-mask contracts
  datasets/
    _base.py                    DataModule: explicit train/val/test loaders
    _loading.py                 native-resolution list collation
    image_mask.py               ImageMaskDataset
    coco.py                     CocoInstanceDataset
    selection.py                reused inclusive COCO validation policy
    manifest.py                 manifests, source-image split checks
    conditions.py               independently transform each original sample
  stimuli/
    functional.py               draw_fence / fragment_contours array API
    fence.py                    existing fixed-center fence algorithm
    fragmentation.py            adapter calling the frozen archive helpers
    transforms.py               Compose-compatible dictionary transforms
    export.py                   CPU input/target export
  models/
    _base.py                    image-only Model contract
    sam3.py                     native loader, API checks and Sam3GridAdapter
    proposals.py                grid queries and explicit selection policy
  tasks/
    _base.py                    Task.get_evaluator; update/finalize contracts
    scene_proposals.py          concrete proposal-export task/evaluator
    _scene_export.py            existing masks/metadata archive format
    preparation.py              separate common-visible prompt protocol
  runner.py                     shared DataLoader -> model -> evaluator loop
  visualization.py              contact sheets and overlays
  infrastructure.py             shared I/O, fingerprints and SLURM guard
experiments/
  sam3_fragmentation/
    experiment.py               Config, build_data_module, prepare, run
    configs/                    fence and contour JSON examples
    README.md                   experiment commands and output layout
tests/{datasets,models,stimuli,tasks}/  tests mirror library responsibilities
tests/helpers.py                shared native SAM3 test doubles
legacy/contour_fragmentation/   exact archive source, unchanged
```

| Pinned mlvbench responsibility | BWLab implementation |
| --- | --- |
| `mlvbench/datasets/_base.py::DataModule` | `bwlab/datasets/_base.py::DataModule` owns named splits and loading; only train shuffles. |
| `mlvbench/datasets/msra10k.py::MSRA10KDataset` | Image/mask and COCO Datasets implement `__len__` and `__getitem__`; each returns a sample dictionary and owns its optional transform. |
| `mlvbench/stimuli/transforms.py` | `bwlab/stimuli/transforms.py` provides small callable transforms over `__key__`, `image`, `segmentation` and retained metadata. |
| `mlvbench/models/_base.py` and model implementations | `bwlab/models/_base.py` and `sam3.py` separate the image-only interface and native implementation from experiments. |
| `mlvbench/tasks/_base.py` | `bwlab/tasks/_base.py` separates the task from its incremental evaluator. |
| `mlvbench/tasks/binary_segmentation.py` | `SceneProposalTask` supplies a concrete evaluator for the existing proposal archive protocol; new quantitative metrics remain deferred. |
| `mlvbench/trainer.py` | `bwlab/runner.py` owns the reusable inference loop. No trainer is added before training is needed. |
| `experiments/figure_ground_shape_cues/experiment.py` | `experiments/sam3_fragmentation/experiment.py` holds Config and wires components in `run()`. |
| Experiment `launch.py` and config files | JSON config examples plus explicit `prepare`/`run` commands. Existing SLURM allocation scripts are retained; no new automatic job launcher is introduced. |

`bwlab` never imports experiments, scripts or compatibility packages. The model
receives only RGB arrays; the evaluator receives the sample and can archive
targets. Rendering and filesystem export do not live in the native model adapter.
Tests enforce the dependency direction and exercise the complete composition
with the existing native SAM3 fake, not a replacement segmentation algorithm.

`ManifestDataModule` rejects a source-image path shared across supplied splits.
`DataModule` can also wrap explicit COCO instance datasets: callers must split
image IDs before expanding instances, and can supply just an evaluation split.
Condition expansion preserves `source_key` and appends the condition name to
`__key__`, giving each transformed sample a stable unique identifier. Every
severity starts from the same original sample, preserving the full target.

The old import-only compatibility folders have been removed. All callers use
`bwlab` directly. Historical root scripts and their assets now live together
under `legacy/main_2025/`, with their contents preserved. The frozen contour
baseline remains under `legacy/contour_fragmentation/`.

The adaptation keeps native NumPy RGB arrays, the existing Meta API and Python
3.12 compatibility. It does not copy the reference's feature/probe trainer,
caching layer, multi-model dependencies, Python 3.14 requirement, or full
scheduler launcher. Those components serve reference experiments not yet part
of the authorized BWLab refactor. The pinned README, library layout, model
interface, trainer, experiment and launcher were inspected in addition to the
six originally requested files.

The first validated refactor was merged via PR #3. Follow-up rendering and
function cleanup is on a new branch based on that updated main. The follow-up also removes compatibility
folders and relocates historical files without rewriting them.
