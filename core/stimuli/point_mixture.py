"""A balanced, seeded stimulus assignment: one condition per source per epoch."""

import random

from core.stimuli.point_transforms import PointTransform


class PointMixture:
    def __init__(self, conditions, image_ids, seed):
        if not conditions or len(image_ids) != len(set(image_ids)):
            raise ValueError("Mixture requires conditions and unique source IDs")
        names = [item["name"] for item in conditions]
        if len(names) != len(set(names)) or any(
            name not in ("identity", "fragmentation", "fence") for name in names
        ):
            raise ValueError("Mixture needs distinct supported conditions")
        self.transforms = [
            PointTransform(item["name"], seed, item.get("parameters", {})) for item in conditions
        ]
        self.names = names
        ids = sorted(image_ids)
        random.Random(f"mixture:{seed}").shuffle(ids)
        self.assignment = {image_id: index % len(conditions) for index, image_id in enumerate(ids)}
        self.counts = {
            name: sum(index == i for index in self.assignment.values())
            for i, name in enumerate(names)
        }

    def __call__(self, sample):
        index = self.assignment[sample["metadata"]["image_id"]]
        result = self.transforms[index](sample)
        result["metadata"]["training_condition"] = self.names[index]
        return result
