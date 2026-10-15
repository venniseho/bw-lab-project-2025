"""Compose-compatible dictionary transforms, at native image coordinates."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluation.preparation import _binary_mask, build_evaluation_targets
from stimuli_generation.fence import generate_fence
from stimuli_generation.fragmentation import FragmentationConfig, generate_fragmentation

Sample = dict[str, Any]


def validate_sample(sample: Sample) -> None:
    if not isinstance(sample.get("__key__"), str) or not sample["__key__"]:
        raise ValueError("__key__ must be a nonempty stable string")
    image = sample["image"]
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape):
        raise ValueError("image must be nonempty uint8 (H,W,3) RGB")
    mask = _binary_mask(sample["segmentation"], "segmentation")
    if mask.shape != image.shape[:2]:
        raise ValueError("segmentation must align with image")


@dataclass(frozen=True)
class FenceOcclusion:
    bar_width: float
    bar_spacing: float
    bar_angle: float = 90.0
    seed: int | None = None
    phase: float | None = None

    def __call__(self, sample: Sample) -> Sample:
        validate_sample(sample)
        result = generate_fence(sample["image"], self.bar_width, self.bar_spacing,
                                self.bar_angle, self.seed, sample["segmentation"],
                                phase=self.phase)
        occlusion = result.occlusion_mask
        if "occlusion_mask" in sample:
            previous = _binary_mask(sample["occlusion_mask"], "occlusion_mask")
            if previous.shape != occlusion.shape:
                raise ValueError("previous occlusion must align with image")
            occlusion = occlusion | previous
        targets = build_evaluation_targets(sample["segmentation"], occlusion)
        output = deepcopy(sample)
        output.update(image=result.corrupted_rgb, occlusion_mask=occlusion,
                      visible_segmentation=targets.visible_object)
        output.setdefault("transforms", []).append({
            "name": "fence_occlusion", "bar_width": self.bar_width,
            "bar_spacing": self.bar_spacing, "bar_angle": self.bar_angle,
            "phase": self.phase, "seed": self.seed,
            "cumulative_image_occluded_fraction": float(occlusion.mean()),
            "cumulative_object_occluded_fraction": (
                float(np.count_nonzero(targets.full_object & occlusion) / np.count_nonzero(targets.full_object))
                if np.any(targets.full_object) else None)})
        return output


@dataclass(frozen=True)
class ContourFragmentation:
    config: FragmentationConfig = FragmentationConfig()
    seed: int | None = None

    def __call__(self, sample: Sample) -> Sample:
        validate_sample(sample)
        if "occlusion_mask" in sample:
            raise ValueError("fragment the original target before applying any fence")
        result = generate_fragmentation(sample["image"], sample["segmentation"], self.config, self.seed)
        output = deepcopy(sample)
        output["image"] = result.pop("image")
        output["fragmentation"] = result
        output.setdefault("transforms", []).append({"name": "legacy_contour_fragmentation",
                                                    "parameters": result["parameters"],
                                                    "seed": self.seed,
                                                    "rng_protocol": result["rng_protocol"]})
        return output
