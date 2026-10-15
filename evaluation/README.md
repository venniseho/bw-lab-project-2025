# Evaluation preparation

This CPU-only module accepts aligned binary masks (bool, 0/1, or 0/255).
It returns independent boolean masks and metadata without saving files or
loading a model. Full-object support remains the original annotation;
visible-object support is `original & ~occlusion`. Inputs are never modified.

```python
import numpy as np
from evaluation.preparation import OcclusionCondition, prepare_probe
from stimuli_generation.fence import generate_fence

# image_rgb and original_mask are arrays at the same original resolution.
conditions = []
for name, width in zip(("clean", "light", "medium", "heavy"), (0, 4, 12, 24)):
    params = {"bar_width": width, "bar_spacing": 40, "bar_angle": 90}
    fence = generate_fence(image_rgb, **params, seed=123, gt_mask=original_mask)
    conditions.append(OcclusionCondition(name, fence.occlusion_mask, params, 123))

probe = prepare_probe(original_mask, conditions, image_id="image-1", object_id=1)
if probe.valid_for_evaluation:
    point_xy = probe.prompt_xy  # Reuse unchanged for every condition.
    full_target = probe.targets["heavy"].full_object
    visible_target = probe.targets["heavy"].visible_object

# probe.metadata is a JSON-serializable list, one row per condition,
# including invalid examples. Saving a manifest is a separate caller action.
```

Prepare the complete severity set together. A candidate must belong to the
original object and remain visible in every supplied condition. This uses all
masks, including nonnested masks. Selection is deterministic: nearest pixel to
the original object's centroid, then smallest y, then smallest x on ties.
Disconnected/concave masks are safe because only actual foreground pixels
are candidates. No interior-distance margin is imposed, so a prompt may lie
close to an object or occluder boundary. Coordinates are zero-based `(x, y)`
pixel indices at original resolution; future model preprocessing must transform
image, point, and predictions consistently.

An empty object gives `empty_object`; no shared visible pixel gives
`no_common_visible_pixel`. Both invalidate the whole paired example and return
no prompt, while retaining targets and provenance. Each row records IDs,
parameters, seed, actual object occlusion percentage (0–100), point/positive
label, selection rule, complete condition names, common visible pixel count,
validity, and exclusion reason. Empty-object percentages are `None` because
the denominator is zero. Supplied parameters are recorded rather than checked
against a regenerated mask; callers must retain the actual generation values.

These targets answer different questions: full-object evaluation measures
recovery of the annotated extent behind the fence, while visible-object
evaluation measures remaining annotated support. The original annotation is
not assumed to recover any real occlusions already present in the source image.
This step defines neither metric aggregation nor model mask selection.

The point uses ground truth and knowledge of the full severity set, including
the strongest occlusion. It controls prompt location across severities but is
an assisted evaluation, rather than realistic user interaction. Changing the
severity set can change the selected point or validity. Report exclusions and
use the same valid examples across conditions to avoid severity-dependent
cohort changes; the retained cohort favors objects with surviving visible
pixels. Severity-specific point selection would confound the comparison.

Run the focused tests and the existing fence regression tests:

```powershell
python -B -m unittest discover -s tests -p test_probe_preparation.py -v
python -B -m unittest discover -s tests -p test_fence.py -v
```
