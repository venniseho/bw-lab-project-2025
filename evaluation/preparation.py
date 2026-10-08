"""Prepare full/visible targets and one positive prompt for a severity sweep."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np


class EvaluationTargets(NamedTuple):
    full_object: np.ndarray
    visible_object: np.ndarray


@dataclass(frozen=True)
class OcclusionCondition:
    """An already generated condition, with its generation provenance.

    True/nonzero occlusion pixels mean hidden. Parameters must be JSON
    compatible, e.g. bar_width, bar_spacing, bar_angle. The seed controls
    corruption generation; point selection uses no randomness. The caller
    supplies parameters corresponding to this mask, without regeneration.
    """

    name: str
    occlusion_mask: np.ndarray
    corruption_parameters: Mapping[str, Any]
    seed: int | None = None


@dataclass
class PreparedProbe:
    targets: dict[str, EvaluationTargets]
    metadata: list[dict[str, Any]]
    prompt_xy: tuple[int, int] | None
    valid_for_evaluation: bool
    invalid_reason: str | None


def _binary_mask(mask: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if mask.ndim != 2 or 0 in mask.shape:
        raise ValueError(f"{name} must have nonempty (H, W) dimensions")
    if mask.dtype.kind not in "buif":
        raise TypeError(f"{name} must have a boolean or numeric dtype")
    # Reject soft masks and malformed values rather than silently thresholding.
    if not (np.all((mask == 0) | (mask == 1))
            or np.all((mask == 0) | (mask == 255))):
        raise ValueError(f"{name} must be binary: bool, 0/1, or 0/255")
    return mask != 0


def build_evaluation_targets(
    original_mask: np.ndarray, occlusion_mask: np.ndarray
) -> EvaluationTargets:
    """Return independent boolean masks in the original image coordinates.

    Full-object support is unchanged; visible support excludes hidden pixels.
    No resizing, interpolation, morphological operations, or input mutation
    occur. Both inputs must be aligned binary (H, W) arrays.
    """
    full = _binary_mask(original_mask, "original_mask")
    occlusion = _binary_mask(occlusion_mask, "occlusion_mask")
    if occlusion.shape != full.shape:
        raise ValueError("occlusion_mask must match original_mask dimensions")
    return EvaluationTargets(full, full & ~occlusion)


def prepare_probe(
    original_mask: np.ndarray,
    conditions: Sequence[OcclusionCondition],
    *,
    image_id: str | int,
    object_id: str | int | None = None,
) -> PreparedProbe:
    """Prepare an entire condition set together, never one severity at a time.

    Include clean (an all-false mask) and every intended severity. The chosen
    positive pixel is in original_mask and unoccluded in ALL supplied masks,
    including nonnested conditions. Among candidates, select the pixel nearest
    the original object's centroid (squared Euclidean distance). Ties select
    the smallest y, then x. Coordinates are zero-based (x, y) pixel indices,
    not (row, column), normalized coordinates, or resized model coordinates.

    An empty object or empty common-visible set invalidates the whole paired
    example: prompt_xy is None, but targets and metadata are still returned.
    Metadata is JSON serializable and records actual object occlusion in
    percent (0..100); an empty object's percentage is None, not zero.
    """
    full = _binary_mask(original_mask, "original_mask")
    if not conditions:
        raise ValueError("at least one condition is required")

    targets = {}
    metadata = []
    common_visible = full.copy()
    object_pixels = int(np.count_nonzero(full))
    for condition in conditions:
        if not isinstance(condition.name, str) or not condition.name:
            raise ValueError("condition names must be nonempty strings")
        if condition.name in targets:
            raise ValueError("condition names must be unique")
        if condition.seed is not None and (
            isinstance(condition.seed, (bool, np.bool_))
            or not isinstance(condition.seed, (int, np.integer))
            or condition.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None")
        if not isinstance(condition.corruption_parameters, Mapping):
            raise TypeError("corruption_parameters must be a mapping")
        # Copy provenance into plain JSON data, without retaining caller-owned
        # nested containers or allowing NaN/Infinity in future manifests.
        parameters = json.loads(json.dumps(
            dict(condition.corruption_parameters), allow_nan=False
        ))
        target = build_evaluation_targets(full, condition.occlusion_mask)
        targets[condition.name] = target
        common_visible &= target.visible_object
        visible_pixels = int(np.count_nonzero(target.visible_object))
        metadata.append({
            "image_id": image_id,
            "object_id": object_id,
            "condition": condition.name,
            "corruption_parameters": parameters,
            "seed": None if condition.seed is None else int(condition.seed),
            "object_occluded_percent": (
                100.0 * (object_pixels - visible_pixels) / object_pixels
                if object_pixels else None
            ),
        })

    point = None
    reason = None
    if object_pixels == 0:
        reason = "empty_object"
    elif not np.any(common_visible):
        reason = "no_common_visible_pixel"
    else:
        object_y, object_x = np.nonzero(full)
        candidate_y, candidate_x = np.nonzero(common_visible)
        distances = ((candidate_x - object_x.mean()) ** 2
                     + (candidate_y - object_y.mean()) ** 2)
        # np.nonzero returns row-major order; argmin takes the first tie.
        index = int(np.argmin(distances))
        point = (int(candidate_x[index]), int(candidate_y[index]))

    valid = point is not None
    names = list(targets)
    for row in metadata:
        row.update({
            "prompt_xy": list(point) if valid else None,
            "prompt_label": 1 if valid else None,
            "prompt_coordinate_system": "xy_zero_based_pixel_indices",
            "prompt_selection": "nearest_original_centroid_then_y_then_x",
            "prompt_condition_names": names.copy(),
            "common_visible_object_pixels": int(np.count_nonzero(common_visible)),
            "valid_for_evaluation": valid,
            "invalid_reason": reason,
        })
    return PreparedProbe(targets, metadata, point, valid, reason)
