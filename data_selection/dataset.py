"""Map-style native-resolution datasets; no downloads or automatic split choice."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_selection.coco import load_coco_image_instances
from stimuli_generation.transforms import validate_sample


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


class CocoInstanceDataset(Dataset):
    """Use an existing COCO API and explicit image IDs; record every decision.

    Index preparation decodes masks through the existing inclusive validator.
    No area, connectivity, shape, or centroid filtering is added. Images are
    loaded lazily; accepted instance masks and selection records are retained.
    """

    def __init__(self, coco, image_root: str | Path, image_ids: list[int],
                 transform: Callable | None = None, *, crowd_policy: str = "exclude"):
        self.transform = transform
        self.instances = []
        self.selection_records = []
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("image_ids must be unique")
        for image_id in sorted(image_ids):
            info = coco.loadImgs([image_id])[0]
            path = Path(image_root) / info["file_name"]
            if not path.is_file():
                path = Path(image_root) / Path(info["file_name"]).name
            with Image.open(path) as source:
                width, height = source.size
            selections = load_coco_image_instances(coco, image_id, (height, width),
                                                  crowd_policy=crowd_policy)
            self.selection_records.extend(selections)
            for selection in selections:
                if selection.accepted:
                    self.instances.append((path, selection))

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict:
        path, selection = self.instances[index]
        with Image.open(path) as source:
            image = np.array(source.convert("RGB"))
        sample = {"__key__": f"coco/{selection.metadata['image_id']}/{selection.metadata['annotation_id']}",
                  "image": image, "segmentation": selection.mask.copy(),
                  "metadata": deepcopy(selection.metadata)}
        validate_sample(sample)
        return self.transform(sample) if self.transform is not None else sample


def collate_samples(samples: list[dict]) -> list[dict]:
    """Keep variable native resolutions and optional fields; no tensor stacking."""
    return samples


def build_loader(dataset: Dataset, *, split: str, batch_size: int = 1,
                 num_workers: int = 0, prefetch_factor: int = 2,
                 seed: int = 0) -> DataLoader:
    """Shuffle training only. Callers explicitly provide disjoint split datasets."""
    import torch
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    options = {"prefetch_factor": prefetch_factor} if num_workers > 0 else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      num_workers=num_workers, collate_fn=collate_samples,
                      generator=torch.Generator().manual_seed(seed), **options)
