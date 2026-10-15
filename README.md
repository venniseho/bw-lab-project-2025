# BWLab perceptual grouping pipeline

Two separately usable transforms preserve historical contour fragmentation and
synthetic fence occlusion. Full and visible targets remain separate.

Reusable code lives under `bwlab/{datasets,stimuli,models,tasks}`. Experiments
own their configuration and wiring under `experiments/`, following the pinned
mlvbench structure. `Runner` connects DataLoader → model → evaluator; a concrete
proposal-export task reuses the working native SAM3 workflow.

- [Architecture map and reference correspondence](docs/architecture.md)
- [Supervisor update draft and next steps](docs/supervisor_update.md)
- [Configured fence/contour experiment](experiments/sam3_fragmentation/README.md)
- [Pipeline usage and sample contract](docs/stimulus_pipeline.md)
- [Branch inventory, preserved algorithms, and merge review](docs/refactor_inventory.md)
- [Presentation outline](docs/presentation_outline.md)
- [Lab wiki pitch draft](docs/lab_wiki_pitch_draft.md)
- [Existing native SAM3 workflow](docs/sam3_scene_masks.md)

The root historical scripts and tracked assets remain intact. New contour
adapters use the frozen archive copy under `legacy/contour_fragmentation/`.
Previous import paths remain available as compatibility wrappers.
No quantitative extension or GPU experiment is part of this refactor.
