"""Seeded on-the-fly conditions for point-prompt experiments."""

from dataclasses import dataclass, field
import hashlib

from core.stimuli.fragmentation import FragmentationConfig
from core.stimuli.transforms import ContourFragmentation, FenceOcclusion


@dataclass(frozen=True)
class PointTransform:
    name: str
    seed: int
    parameters: dict = field(default_factory=dict)

    def __call__(self, sample):
        if self.name == "identity":
            return sample
        image_id = sample["metadata"]["image_id"]
        digest = hashlib.sha256(f"{self.seed}:{image_id}".encode()).digest()
        seed = int.from_bytes(digest[:4], "little")
        if self.name == "fragmentation":
            transform = ContourFragmentation(FragmentationConfig(**self.parameters), seed=seed)
        elif self.name == "fence":
            transform = FenceOcclusion(seed=seed, **self.parameters)
        else:
            raise ValueError(f"Unknown point transform: {self.name}")
        return transform(sample)
