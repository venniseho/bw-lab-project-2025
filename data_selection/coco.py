"""COCO instance validation without size, shape or promptability filtering.

No dataset, model or pycocotools import is needed to validate masks. The loader
accepts an existing COCO API object and decodes every annotation for an image.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluation.preparation import OcclusionCondition, PreparedProbe, prepare_probe


@dataclass
class InstanceSelection:
    annotation: dict[str, Any]
    mask: np.ndarray | None
    metadata: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return self.metadata["accepted"]


def _identifier(value) -> int | None:
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0:
        return int(value)
    return None


def validate_instance(
    annotation: dict,
    mask: np.ndarray | None,
    image_shape: tuple[int, int],
    *,
    crowd_policy: str = "exclude",
    expected_image_id: int | None = None,
    decode_error: str | None = None,
) -> InstanceSelection:
    """Keep original annotation and decoded mask copies, even when excluded.

    image_shape is the ACTUAL decoded RGB image's (H, W), not a resized target.
    No resizing, thresholding, component selection or hole filling occurs.
    Binary encodings bool, 0/1 and 0/255 are accepted, matching preparation.
    One primary exclusion reason is recorded so reason counts sum to exclusions.
    Unknown/malformed iscrowd is excluded rather than assumed noncrowd.
    """
    if crowd_policy not in ("exclude", "include"):
        raise ValueError("crowd_policy must be 'exclude' or 'include'")
    if (len(image_shape) != 2
            or any(_identifier(size) is None or size == 0 for size in image_shape)):
        raise ValueError("image_shape must be positive integer (H, W) dimensions")
    shape = tuple(int(size) for size in image_shape)
    annotation_id = _identifier(annotation.get("id"))
    image_id = _identifier(annotation.get("image_id"))
    crowd = annotation.get("iscrowd")
    crowd_valid = isinstance(crowd, (bool, np.bool_, int, np.integer)) and crowd in (0, 1)
    iscrowd = int(crowd) if crowd_valid else None
    reason = None
    if annotation_id is None:
        reason = "invalid_annotation_id"
    elif image_id is None:
        reason = "invalid_image_id"
    elif expected_image_id is not None and image_id != expected_image_id:
        reason = "annotation_image_id_mismatch"
    elif not crowd_valid:
        reason = "invalid_iscrowd"

    area = fraction = None
    mask_reason = None
    mask_shape = list(mask.shape) if isinstance(mask, np.ndarray) else None
    if decode_error is not None:
        mask_reason = "mask_decode_error"
    elif not isinstance(mask, np.ndarray):
        mask_reason = "malformed_mask"
    elif mask.ndim != 2:
        mask_reason = "malformed_mask"
    elif mask.dtype.kind not in "buif" or not (
        np.all((mask == 0) | (mask == 1)) or np.all((mask == 0) | (mask == 255))
    ):
        mask_reason = "nonbinary_mask"
    else:
        area = int(np.count_nonzero(mask))
        if mask.shape != shape:
            mask_reason = "mask_shape_mismatch"
        else:
            fraction = area / (shape[0] * shape[1])
            if area == 0:
                mask_reason = "empty_mask"
    reason = reason or mask_reason
    if reason is None and iscrowd == 1 and crowd_policy == "exclude":
        reason = "crowd_annotation"
    metadata = {
        "image_id": image_id,
        "source_image_id": expected_image_id if expected_image_id is not None else image_id,
        "annotation_id": annotation_id,
        "category_id": annotation.get("category_id"),
        "iscrowd": iscrowd,
        "crowd_policy": crowd_policy,
        "image_shape_hw": list(shape),
        "mask_shape": mask_shape,
        "object_area_pixels": area,
        "object_area_fraction": fraction,
        "accepted": reason is None,
        "exclusion_reason": reason,
        "decode_error": decode_error,
    }
    return InstanceSelection(deepcopy(annotation),
                             mask.copy() if isinstance(mask, np.ndarray) else None,
                             metadata)


def load_coco_image_instances(
    coco, image_id: int, image_shape: tuple[int, int], *, crowd_policy: str = "exclude"
) -> list[InstanceSelection]:
    """Decode all annotations, including crowds; retain every decision record.

    IDs are sorted for reproducible order. Deliberately do not use areaRng,
    category filters or iscrowd filters in getAnnIds. Known annotation decoding
    failures are recorded per object; unexpected programming/runtime failures
    propagate. No images or masks are written to disk.
    """
    if crowd_policy not in ("exclude", "include"):
        raise ValueError("crowd_policy must be 'exclude' or 'include'")
    records = []
    for annotation in coco.loadAnns(sorted(coco.getAnnIds(imgIds=[image_id]))):
        error = None
        try:
            mask = coco.annToMask(deepcopy(annotation))
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exception:
            mask = None
            error = f"{type(exception).__name__}: {exception}"
        records.append(validate_instance(
            annotation, mask, image_shape, crowd_policy=crowd_policy,
            expected_image_id=image_id, decode_error=error,
        ))
    return records


def summarize_selection(
    records: Iterable[InstanceSelection], *, source_image_ids: Iterable[int]
) -> dict[str, Any]:
    """JSON-ready counts and per-annotation areas for the assessed source set.

    Include images with zero annotations in source_image_ids. Counts refer to
    supplied records, not the whole COCO dataset. Duplicate annotation IDs or
    a record outside the source set raise rather than corrupting statistics.
    All annotations, including crowds and failures, enter accepted/excluded
    counts; crowd regions are not counted as multiple individual objects.
    """
    image_ids = set(source_image_ids)
    seen = set()
    rows = []
    for record in records:
        row = record.metadata
        if row["source_image_id"] is not None and row["source_image_id"] not in image_ids:
            raise ValueError("record source_image_id is outside source_image_ids")
        annotation_id = row["annotation_id"]
        if annotation_id is not None:
            if annotation_id in seen:
                raise ValueError("duplicate COCO annotation ID in selection statistics")
            seen.add(annotation_id)
        rows.append(deepcopy(row))
    reasons = Counter(row["exclusion_reason"] for row in rows if not row["accepted"])
    accepted = sum(row["accepted"] for row in rows)
    return {
        "source_image_count": len(image_ids),
        "object_instance_count": len(rows),
        "counting_unit": "COCO annotation (crowd region counts once)",
        "crowd_annotation_count": sum(row["iscrowd"] == 1 for row in rows),
        "accepted_count": accepted,
        "excluded_count": len(rows) - accepted,
        "exclusion_reasons": dict(sorted(reasons.items())),
        "instances": rows,
    }


def prepare_selected_probe(
    instance: InstanceSelection, conditions: list[OcclusionCondition]
) -> PreparedProbe:
    """Bridge an accepted original mask to the existing fence preparation API.

    Unpromptable objects remain accepted by selection. Return all their target
    masks and condition metadata, with promptable=False and the preparation
    reason. Never drop them or substitute a different prompt per condition.
    """
    if not instance.accepted:
        raise ValueError("excluded instances cannot be prepared for evaluation")
    probe = prepare_probe(instance.mask, conditions,
                          image_id=instance.metadata["image_id"],
                          object_id=instance.metadata["annotation_id"])
    for row in probe.metadata:
        row.update(deepcopy(instance.metadata))
        row["promptable"] = probe.valid_for_evaluation
    return probe
