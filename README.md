# Perceptual grouping with SAM3

Test how segmentation changes under contour fragmentation and fence occlusion,
then fine-tune the point-prompted model using controlled comparisons.

## Project map

```text
core/           Data loading, models, transforms, evaluation and training
experiments/    Configuration and the code that assembles an experiment
scripts/        Commands, figures and Vulcan job submission
tests/          Regression tests and small offline fixtures
docs/           Usage guides and research notes
data/           Local datasets; ignored by Git
outputs/        Runs, figures and local archives; ignored by Git
```

There are two active experiments:

* [Evaluate clean, fenced and fragmented images](experiments/sam3_fragmentation/README.md).
* [Fine-tune the point-prompted model](experiments/sam3_point_finetuning/README.md),
  comparing the full point path against a frozen image encoder.

Each experiment's `experiment.py` puts together datasets, transforms, a model and
a runner/trainer. Reusable implementations live in `core/`; dataset loaders
are in `core/data_loading/`, and actual images/annotations belong in `data/`.
This follows the library/experiment separation in
[Matthias's mlvbench](https://github.com/mtangemann/mlvbench).

## Start here

* [Architecture and design rules](docs/architecture.md)
* [Vulcan environment and native SAM3 setup](docs/vulcan_scene_setup.md)
* [Throughput benchmarks and profiling](docs/pytorch_throughput.md)
* [Online W&B setup and dashboard](docs/wandb.md)
* [Dataset/transform sample contract](docs/stimulus_pipeline.md)
* [COCO instance-selection policy](docs/coco_selection.md)
* [Paired target-point preparation](docs/prompt_preparation.md)

After switching to this layout, refresh the editable install in your existing
environment. This does not replace PyTorch or download model weights:

```bash
python -m pip install -e . --no-deps
python -B -m unittest discover -s tests -t . -v
```

The explicit test root (`-t .`) keeps test imports consistent. GPU commands require
a Slurm GPU job step; CPU tests do not
establish SAM3 GPU training success or improved segmentation accuracy.

## Results and historical material

Downloaded contact sheets and plots live in `outputs/figures/`. New jobs continue
to save to their configured output directories, including scratch on Vulcan.

The local cleanup archive is `outputs/archive/pre-cleanup-cb240b7.zip`, with a
SHA-256 manifest beside it. It preserves the historical 2025 tree and superseded
concept-training implementation. It is ignored by Git and is not uploaded.
Those tracked sources also remain recoverable from commit `cb240b7`.

The offline ten-image pilot uses `tests/fixtures/pilot/`. Larger runs should use
the COCO preparation command in the throughput guide. The original contour
implementation is retained only as `tests/fixtures/contour_reference.py` for
scientific regression checks; production transforms do not import it.
