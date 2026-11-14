"""Seeded on-the-fly conditions for point-prompt experiments."""

from dataclasses import dataclass, field
import hashlib

from core.stimuli.fragmentation import FragmentationConfig
from core.stimuli.transforms import ContourFragmentation, FenceOcclusion


def point_transform_seed(seed, image_id):
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class PointTransform:
    name: str
    seed: int
    parameters: dict = field(default_factory=dict)

    def __call__(self, sample):
        if self.name == "identity":
            return sample
        image_id = sample["metadata"]["image_id"]
        seed = point_transform_seed(self.seed, image_id)
        if self.name == "fragmentation":
            transform = ContourFragmentation(FragmentationConfig(**self.parameters), seed=seed)
        elif self.name == "fence":
            transform = FenceOcclusion(seed=seed, **self.parameters)
        else:
            raise ValueError(f"Unknown point transform: {self.name}")
        return transform(sample)
