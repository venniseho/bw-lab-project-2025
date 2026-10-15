"""Shared dictionary and aligned binary-mask contracts."""
from typing import Any, NamedTuple
import numpy as np

Sample = dict[str, Any]


class EvaluationTargets(NamedTuple):
    full_object: np.ndarray
    visible_object: np.ndarray

def binary_mask(mask: np.ndarray, name: str) -> np.ndarray:
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
    full = binary_mask(original_mask, "original_mask")
    occlusion = binary_mask(occlusion_mask, "occlusion_mask")
    if occlusion.shape != full.shape:
        raise ValueError("occlusion_mask must match original_mask dimensions")
    return EvaluationTargets(full, full & ~occlusion)


def validate_sample(sample: Sample) -> None:
    if not isinstance(sample.get("__key__"), str) or not sample["__key__"]:
        raise ValueError("__key__ must be a nonempty stable string")
    image = sample["image"]
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape):
        raise ValueError("image must be nonempty uint8 (H,W,3) RGB")
    mask = binary_mask(sample["segmentation"], "segmentation")
    if mask.shape != image.shape[:2]:
        raise ValueError("segmentation must align with image")
