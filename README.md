# BWLab perceptual grouping pipeline

Two separately usable transforms preserve historical contour fragmentation and
synthetic fence occlusion. Full and visible targets remain separate.

- [Pipeline usage and sample contract](docs/stimulus_pipeline.md)
- [Branch inventory, preserved algorithms, and merge review](docs/refactor_inventory.md)
- [Presentation outline](docs/presentation_outline.md)
- [Lab wiki pitch draft](docs/lab_wiki_pitch_draft.md)
- [Existing native SAM3 workflow](docs/sam3_scene_masks.md)

The root historical scripts and tracked assets remain intact. New contour
adapters use the frozen archive copy under `legacy/contour_fragmentation/`.
No quantitative extension or GPU experiment is part of this refactor.
