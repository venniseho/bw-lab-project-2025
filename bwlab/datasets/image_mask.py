"""Native-resolution image/mask Dataset from explicit records."""
from __future__ import annotations
from copy import deepcopy
from typing import Callable
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from bwlab.stimuli.transforms import validate_sample


class ImageMaskDataset(Dataset):
    """Explicit records: __key__, image_path, mask_path, optional metadata."""

    def __init__(self, records: list[dict], transform: Callable | None = None):
        self.records = deepcopy(records)
        self.transform = transform
        keys = [r["__key__"] for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError("sample keys must be unique")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = np.array(source.convert("RGB"))
        with Image.open(record["mask_path"]) as source:
            mask = np.array(source)
        sample = {"__key__": record["__key__"], "image": image,
                  "segmentation": mask, "metadata": deepcopy(record.get("metadata", {}))}
        validate_sample(sample)
        return self.transform(sample) if self.transform is not None else sample


