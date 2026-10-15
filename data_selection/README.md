# Natural-image COCO selection

This policy is separate from `make_stimuli.py`. It validates original masks
for fence experiments and never extracts a contour. It needs only NumPy and
the existing preparation module; decoding accepts an already loaded COCO API
object. No dataset downloads, model imports, file writes or image resizing
occur inside the module.

## What was excluded historically

`make_stimuli.get_instance_masks` silently drops empty masks and resizes
misaligned masks, then converts values with `astype(uint8) & 1`.
`make_stimuli.main` rejects areas below 5,000 pixels or 1% of image pixels.
`make_stimuli.is_mask_valid` requires one 4-connected component and a
foreground pixel at the truncated geometric centroid. Centroid containment
does not establish convexity, despite the old comment calling it a convexity
check. `mask_fragmenter_clean.largest_external_contour` uses external contours and
selects the largest by contour area. These historical functions are unchanged.

## New policy

`validate_instance` accepts every nonempty, aligned binary mask, with no
minimum object area/fraction, connectivity, convexity, centroid, perimeter or
category filter. All foreground components and holes are preserved. Annotations
and decoded masks are copied without changing values, dtype or annotation ID.
Original polygon/RLE data remains in `InstanceSelection.annotation`.

Remaining validation requirements and primary exclusion reasons:

| Requirement | Recorded reason |
|---|---|
| Original annotation and image IDs are nonnegative integers | `invalid_annotation_id`, `invalid_image_id` |
| Annotation refers to the requested source image | `annotation_image_id_mismatch` |
| `iscrowd` is present and 0/1 (or bool) | `invalid_iscrowd` |
| Annotation segmentation decodes successfully | `mask_decode_error` |
| Mask is a NumPy 2-D array | `malformed_mask` |
| Mask is binary: bool, 0/1 or 0/255 | `nonbinary_mask` |
| Mask dimensions equal the actual decoded RGB image | `mask_shape_mismatch` |
| At least one object pixel exists | `empty_mask` |
| Noncrowd under the default crowd policy | `crowd_annotation` |

These checks ensure identifiable annotations, defined foreground support and
aligned pixel coordinates. Empty masks have no object target/area denominator;
soft or malformed masks must not be silently thresholded. Misalignment must
be investigated rather than repaired by an unrecorded geometric transformation.
One primary reason per annotation makes reason totals equal excluded counts;
mask shape, areas, crowd status and decoder error detail remain in the record.

Default `crowd_policy="exclude"` retains crowd records but excludes them from
single-instance evaluation. A point may identify only a subregion of a crowd
target, so treating that region like a single ordinary instance makes the
task ambiguous. The [COCO API documents separate crowd matching semantics](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py).
Explicit `crowd_policy="include"` accepts valid crowd masks for a deliberately
defined grouped-region experiment. It does not split crowd regions into
individual instances or change any evaluation metric. Missing crowd status
is flagged, rather than assumed to be zero.

Area statistics use actual decoded foreground pixel counts, not COCO's
annotation `area`. Fraction is foreground count divided by actual image area
(0..1). Empty aligned masks record zero; malformed/undecodable masks record
unknown area as `None`. A binary but misaligned mask retains its measured area
while its image-area fraction is `None`. Crowd exclusions still have areas.
Small objects increase sensitivity to pixel rasterization and discrete
occlusion steps; retain them and use these area statistics when interpreting
results, rather than imposing the historical threshold.

## Use with the existing fence pipeline

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from data_selection.coco import (
    load_coco_image_instances, prepare_selected_probe, summarize_selection,
)
from evaluation.preparation import OcclusionCondition
from stimuli_generation.fence import generate_fence

coco = COCO("/path/to/existing/instances.json")
image_id = sorted(coco.getImgIds())[0]  # Explicit source selection, not a filter.
info = coco.loadImgs([image_id])[0]
with Image.open(Path("/path/to/images") / info["file_name"]) as source:
    rgb = np.array(source.convert("RGB"))
records = load_coco_image_instances(coco, image_id, rgb.shape[:2])

# Generate each image-level condition once, then reuse it across instances.
conditions = []
for name, width in zip(("clean", "light", "medium", "heavy"), (0, 4, 12, 24)):
    params = {"bar_width": width, "bar_spacing": 40, "bar_angle": 90}
    fence = generate_fence(rgb, **params, seed=123)
    conditions.append(OcclusionCondition(name, fence.occlusion_mask, params, 123))

probes = [prepare_selected_probe(record, conditions)
          for record in records if record.accepted]
condition_rows = [row for probe in probes for row in probe.metadata]
# Keep ALL condition_rows, including promptable=False. Only valid probes
# can enter point-prompt inference; selection records must not be discarded.

stats = summarize_selection(records, source_image_ids=[image_id])
stats_json = json.dumps(stats, indent=2, allow_nan=False)
condition_rows_json = json.dumps(condition_rows, indent=2, allow_nan=False)
# Caller may save these JSON strings to its experiment manifest.
```

The loader calls `getAnnIds(imgIds=[image_id])` without area, category or crowd
prefilters and sorts annotation IDs. It records expected annotation decoding
errors per instance; unexpected failures propagate. Pass the actual RGB array
dimensions so the module can flag mismatches. Source image paths/loading errors
remain the caller's responsibility and must be reported if an image cannot be
loaded; do not replace its dimensions with a resized proxy.

`summarize_selection` accepts records or a streaming iterator, retaining only
metadata rows in the report. Supply the complete assessed source image ID set,
including images with zero annotations. Counts describe that assessed set,
not an unprocessed whole dataset. `object_instance_count` counts annotation
records, including crowd regions once; `crowd_annotation_count` makes that
distinction explicit. `accepted_count + excluded_count = object_instance_count`.
The report includes reason counts and every instance's ID, area/fraction and
decision. Duplicate annotation IDs raise to prevent accidental double counting.

Selection acceptance and experiment promptability are separate. The bridge
`prepare_selected_probe` reuses the existing preparation rule and adds selection
provenance and `promptable` to every condition row. An object with no pixel
visible across the entire severity set stays `accepted=True`, while its probe
is `valid_for_evaluation=False`, `promptable=False` and has no point, with
`invalid_reason="no_common_visible_pixel"`. Full/visible targets and all IDs
remain available. This cannot be rescued by dropping its most severe condition
or silently selecting different points. Report promptability counts separately
for each experiment's severity set; dataset exclusion statistics do not change.

## Tests

```bash
python -B -m unittest discover -s tests -p test_coco_selection.py -v
python -B -m unittest discover -s tests -p test_fence.py -v
python -B -m unittest discover -s tests -p test_probe_preparation.py -v
python -B -m unittest discover -s tests -p test_sam3_smoke.py -v
```

Synthetic tests cover one-pixel/very small masks, concavity, disconnected
components, holes, empty/soft/malformed/misaligned masks, crowd policies,
statistics and promptability retention. An optional installed-pycocotools test
decodes multipart polygons and RLE masks from a tiny in-memory COCO dataset.
No real dataset, checkpoint or model inference is needed.
