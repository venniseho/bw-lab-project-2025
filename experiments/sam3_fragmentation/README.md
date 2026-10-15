# SAM3 fragmentation and fence experiment

This directory owns configuration and component wiring, following the role of
`experiments/figure_ground_shape_cues/` in the pinned mlvbench repository.
Reusable implementations live in `bwlab/`.

Create `records.json` in the repository root with explicit image/mask records:

```json
[{"__key__":"example/1","image_path":"image.png","mask_path":"mask.png"}]
```

Image/mask paths are relative to the manifest; the manifest/checkpoint paths
in a configuration are relative to the configuration file. An optional metadata
object on each record is retained. Masks must be aligned binary HW arrays.

From the repository root, prepare either arm on CPU:

```sh
python -m experiments.sam3_fragmentation.experiment prepare --config experiments/sam3_fragmentation/configs/fence.json --output outputs/fence_prepared
python -m experiments.sam3_fragmentation.experiment prepare --config experiments/sam3_fragmentation/configs/fragmentation.json --output outputs/contours_prepared
```

Outputs must be new directories. Preparation writes `config.json`, an incremental
`manifest.json`, RGB inputs under `inputs/` and masks/segment arrays under
`targets/`. It loads no SAM3 model. Fence defaults expand each source into clean,
light, medium and heavy conditions. Contour settings remain a separate arm.
Use JSON `"seed": null` for historical unseeded contour behavior; use
`"seed": null` when supplying an explicit `fence_phase`.

The experiment's central inference flow is:

```python
data_module = build_data_module(config)
model, processor, info = load_scene_model(config.checkpoint, config.bpe_path, torch)
task = SceneProposalTask(model_metadata=info)
runner = Runner(data_module, task)
runner.evaluate(Sam3GridAdapter(model, processor), output_path, subset=config.subset)
```

The actual `run()` adds SLURM/CUDA checks and uses all selection/loader settings
from Config. It records config and native model provenance. Dataset transforms
run lazily; the model receives only image pixels and the evaluator separately
archives targets. There is no intermediate manual handoff of prepared files.

GPU command template, **only after approval and inside an existing SLURM step**:

```sh
python -m experiments.sam3_fragmentation.experiment run --config /path/to/reviewed-run-config.json --output /scratch/vho/sam3-whole-scene/results/new_run
```

Set `checkpoint` in that reviewed config to the existing native checkpoint,
`/scratch/vho/sam3-whole-scene/checkpoints/sam3.pt`. Use the existing SAM3
environment; do not reinstall it. The command submits no job and refuses a
login-node run. The included configs leave checkpoint null to prevent an
accidental inference run. No GPU run has been performed for this refactor.

Inference creates `config.json` and `predictions/results.json`. Each numbered
scene archive contains RGB, raw candidate masks, retained masks, clearer contact
sheets, proposal metadata and separate `targets.npz`. The evaluator's final
report gives artifact locations and proposal counts, not segmentation accuracy.
No ground-truth mask participates in proposal generation or selection.

The [architecture map](../../docs/architecture.md) explains each layer. Extend
quantitative metrics and the separate common-visible prompting protocol only
after approved merge into main. The old natural-image CLI remains available
for qualitative images without masks.
