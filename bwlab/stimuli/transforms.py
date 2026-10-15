"""Compose-compatible dictionary transforms, at native image coordinates."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from bwlab.samples import Sample, validate_sample, binary_mask as _binary_mask, build_evaluation_targets
from bwlab.stimuli.fence import generate_fence
from bwlab.stimuli.fragmentation import FragmentationConfig, generate_fragmentation





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
