# Preservation inventory and merge review

Inventory date: 2026-10-15. Remote heads were refreshed over HTTPS before branching.
The original checkout remains on `sam3-fragmentation-v2`; its pre-existing deletion
of `outputs/fence_preview.png` is untouched. Work is isolated in
`.worktrees/stimulus-refactor`, branch `refactor/stimulus-transforms-preserve-legacy`.

| Branch | Inspected tip | Contents and implications |
| --- | --- | --- |
| `main` | `154e295` | Older root fragmenter, `coco_pipeline.py`, `coco_pipeline_revised.py`, original Hii/Pizlo stimulus sources, images, masks and experimental outputs. All retained in this proposal. |
| `archive/legacy-contour-fragmentation` | `27f9bd6` | Later per-instance fragmenter, `make_stimuli.py`, `run_sam_batch.py`, `sam_runner.py`, historical analyses and prompting experiments. Source of frozen baseline. |
| `refactor_pipeline` | `8a87159` | Separate rewrite: `stimuli_generation/mask_fragmenter.py`, `coco_utils.py`, new `core/` helpers and CLI/analysis modules. Not the frozen baseline and not imported. |
| `sam3-fragmentation-v2` | `b8aa1e8` | Same fragmenter plus fence generation, inclusive COCO validation, common-visible prompt preparation and working native SAM3 grid proposals. |
| `test_outputs` | `c245cb5` | Earlier experiment-output history, retained remotely and locally. |
| `test_pipeline` | `2288a7b` | Earlier pipeline/plot experiments, retained remotely and locally. |

`main...sam3-fragmentation-v2` reports 4,324 changed files, including historical
deletions. A direct merge would remove code and assets. This refactor instead
starts from `main` and imports only the working modules, their docs/tests, and
explicitly named frozen archive sources. It does not merge the old development
branch or rewrite its history. Review `git diff --name-status origin/main...HEAD`
and `git diff --diff-filter=D --name-only origin/main...HEAD`; the latter should
be empty. The proposal retains main's tracked assets even when they are large.

## Both contour baselines remain identifiable

- `mask_fragmenter_clean.py`: main's original older version, blob
  `9916f39d599d6c606a28aa519bf737d0e224c555`, untouched.
- `legacy/contour_fragmentation/mask_fragmenter_clean.py`: exact archive baseline,
  blob `4623aa609d7007e8a9157fde9cd141379b366520`, also present unchanged at
  `b8aa1e8`. This is the canonical later baseline for the
  new adapter, rather than claiming every historical experiment used one version.
- The archive's `make_stimuli.py`, `run_sam_batch.py`, `sam_runner.py`, and
  `setup_env.sh` are copied byte-for-byte alongside it. Run historical scripts
  from that directory so their original absolute imports resolve locally.
- Root `coco_pipeline.py` and `coco_pipeline_revised.py` remain untouched. They
  represent earlier union/per-instance workflows and retain their own defaults.

Relevant history inspected includes `5672d1c`, `8f563b8`, `2731cd1`, `46b0c52`,
`a764daf`, `27f9bd6`, and `8a87159`. The main/archive fragmenter diff contains
885 additions and 259 removals; replacing main's file would silently change an
earlier baseline, so this proposal deliberately keeps both files.

The existing `refactor_pipeline` was inspected separately: it replaces the
1,180-line archive fragmenter with a 210-line rewrite, changes default gap factor
from 0.4 to 0.35 and default noise count from 300 to 400, and narrows the entrypoint
options. It is not a behavior-preserving drop-in replacement. That branch is
retained unchanged for reference; its reorganized framework is not silently
adopted as the original algorithm.
Its noise generator also samples variable 4–12 px lengths and adds orientation
jitter, while `compute_and_save_metrics` is a `pass` placeholder. Those changes
are further reasons not to use it as the reproducible historical baseline.

## Preserved behavior and limitations

The adapter in `bwlab/stimuli/fragmentation.py` calls the frozen outline,
noise and drawing helpers with `fragment_one` defaults: automatic edge length
bounded to 4–12 px, gap factor 0.4, scan outline, zero jitter, 1 px thickness,
300 uniform noise attempts, separation pad 1, target density 6 per 100 px.
Random outlines, grid noise, split inside/outside counts and lengths remain
available. Black canvas and identical white outline/noise rendering are retained.
The safe adapter omits disk writes, debug panels and histograms; use the frozen
entrypoint for those original outputs. Pixel regressions use the original
entrypoint with controlled RNGs, rather than a second implementation of geometry.

The historical algorithm is unseeded and uses the largest external contour,
discarding holes and smaller disconnected components. Grid noise rejects
midpoints inside the object; uniform `region="any"` does not. Collision rejection
and finite retries mean requested and realized noise counts differ. Split noise
placement may create density cues. These properties are preserved, not validated
as scientifically neutral. The old `make_stimuli.py` also filters connectivity
and centroid containment; centroid containment is not a convexity test.
Earlier COCO code includes area selection and bounding-box fallbacks. Those
policies are historical, not defaults of the new dataset wrapper.

The new array contract rejects soft, nonbinary and misaligned masks rather than
using the historical loader's resizing, Otsu thresholding or inversion heuristic.
For already aligned binary masks, geometry is unchanged. Empty contour masks
still raise; tiny contours can yield no usable segments. Seeded generation is
an explicitly separate RNG protocol (`spawned_streams_v1`), using private local
module namespaces, not global monkey patches. It cannot recover past entropy
draws, and increasing contour fragmentation settings does not guarantee nesting.
The original unseeded file-based entrypoint remains available unchanged.

`bwlab/stimuli/fence.py` (also available through `stimuli_generation/fence.py`) retains its seeded fixed-center rasterization,
pixel-center coordinates and half-open bars. Explicit `phase` is an additive
keyword-only option; phase and seed are mutually exclusive. Fixed spacing,
angle and seed/phase give nested masks as width increases. Multiple fence
transforms compose by unioning occlusion and retaining the full-object mask.

## Reference decisions

Read the project log and previous report before implementation:

- https://docs.google.com/document/d/1Jh1bZ4EN473fMY0mR8LZzR4dBOX_XixAOIBk6ac92lc
- https://docs.google.com/document/d/1mQrYGKPKZ44tVxhy-GtNIQdZE1FF7ZA91ec0WD_aSHA

The report motivates matched-noise contours and separate model/oracle ranking;
its causal claims about human grouping and architecture are not established by
this software refactor. The log motivates natural fences, controlled severity,
and later encoder/decoder investigation. No private document content is copied.

Inspected all six requested reference files at exactly:
https://github.com/mtangemann/mlvbench/tree/c204a8185fb662b3b1edacfb7a98f7b36fbd9a70

| Reference | Adopted pattern |
| --- | --- |
| `mlvbench/stimuli/transforms.py` | Callable sample dictionaries compatible with Compose; preserve metadata rather than dropping extra keys. |
| `mlvbench/datasets/_base.py` | Explicit train/evaluation loading; avoid its caching/training framework. |
| `mlvbench/datasets/msra10k.py` | Map-style Dataset, stable `__key__`, `__len__`, `__getitem__`, transforms owned by dataset. |
| `mlvbench/tasks/_base.py` | Evaluator `update()` / `finalize()` boundary. |
| `mlvbench/tasks/binary_segmentation.py` | Separate predictions, targets and aggregation; no copied patch classifier metrics. |
| `experiments/figure_ground_shape_cues/experiment.py` | Configuration separated from implementation. |

Use native-resolution NumPy HWC RGB arrays and HW binary masks for the existing
SAM3 path. No crop/resize is introduced. Native Meta processor source was checked
at `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`; the existing loader passes PIL RGB
to avoid the processor's tensor-layout assumptions for NumPy dimensions.
Neither Meta source, checkpoint nor Vulcan environment is modified.

## Stopping boundary

This proposal adds Dataset/transform/model/evaluator interfaces and CPU stimulus
preparation. It does not implement a new quantitative benchmark, training loop,
encoder/decoder probe, or oracle mask-selection policy. The existing
`bwlab/tasks/preparation.py` common-visible positive prompt protocol
(re-exported at `evaluation/preparation.py`) remains
separate from automatic independent grid queries. Merge requires the owner's
explicit approval; subsequent quantitative development must branch from updated
main after that merge.


## Structural revision after review

The reusable implementation now lives under `bwlab/`, matching the pinned
reference's division into datasets, stimuli, models and tasks. The experiment
has its own Config and `run()` under `experiments/sam3_fragmentation/` and uses
the shared Runner. The initial directories are compatibility wrappers, not
parallel implementations. See [architecture.md](architecture.md) for the exact
correspondence and [refactor_validation.md](refactor_validation.md) for current
test evidence. Frozen sources and existing main files remain unchanged.
