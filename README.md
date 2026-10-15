# BWLab perceptual grouping pipeline

Two separately usable transforms preserve historical contour fragmentation and
synthetic fence occlusion. Full and visible targets remain separate.

Reusable code lives under `bwlab/{datasets,stimuli,models,tasks}`. Experiments
own their configuration and wiring under `experiments/`, following the pinned
mlvbench structure. `Runner` connects DataLoader → model → evaluator; a concrete
proposal-export task reuses the working native SAM3 workflow.

- [Architecture map and reference correspondence](docs/architecture.md)
- [Supervisor update draft and next steps](docs/supervisor_update.md)
- [Timed ten-image Vulcan pilot](docs/timed_pipeline_pilot.md)
- [Configured fence/contour experiment](experiments/sam3_fragmentation/README.md)
- [Pipeline usage and sample contract](docs/stimulus_pipeline.md)
- [Branch inventory, preserved algorithms, and merge review](docs/refactor_inventory.md)
- [Presentation outline](docs/presentation_outline.md)
- [Lab wiki pitch draft](docs/lab_wiki_pitch_draft.md)
- [Existing native SAM3 workflow](docs/sam3_scene_masks.md)

## Where things live

| Folder | Purpose |
| --- | --- |
| `bwlab/` | Reusable datasets, transforms, model adapters and tasks |
| `experiments/` | Experiment entry points and configurations |
| `scripts/` | Command-line tools and Vulcan job scripts |
| `tests/` | Tests grouped like the library |
| `docs/` | Usage, architecture and research notes |
| `legacy/contour_fragmentation/` | Frozen contour algorithm used by the adapter |
| `legacy/main_2025/` | Historical scripts, source assets and old outputs |
| `data/` | Local datasets (ignored by Git) |
| `outputs/` | Current runs and downloaded Vulcan results (ignored by Git) |

The old import-only folders have been removed; use `bwlab` imports.
Historical files were relocated without changing their contents. See
[the archive guide](legacy/main_2025/README.md) before running old scripts.
The ten-image pilot automatically finds the archived image/mask pairs.

Run the CPU checks with `python -B -m unittest discover -s tests -v`.
Use the [Vulcan pilot guide](docs/timed_pipeline_pilot.md) for GPU runs.
