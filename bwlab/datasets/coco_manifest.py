"""Lazy COCO masks for explicit, reproducible validation manifests."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .selection import validate_instance
from bwlab.stimuli.transforms import validate_sample


class CocoManifestDataset(Dataset):
    """Decode only the requested instance when a loader worker reads a sample."""

    def __init__(self, records, transform=None):
        from pycocotools.coco import COCO
        self.records = deepcopy(records)
        self.transform = transform
        self.coco = {}
        for record in records:
            path = record["annotations_path"]
            if path not in self.coco:
                digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                if digest != record["annotations_sha256"]:
                    raise ValueError(f"COCO annotations changed: {path}")
                self.coco[path] = COCO(path)
        keys = [record["__key__"] for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError("sample keys must be unique")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        coco = self.coco[record["annotations_path"]]
        annotation = coco.loadAnns([record["annotation_id"]])[0]
        with Image.open(record["image_path"]) as source:
            image = np.array(source.convert("RGB"))
        selection = validate_instance(
            annotation, coco.annToMask(annotation), image.shape[:2],
            expected_image_id=record["image_id"])
        if not selection.accepted:
            raise ValueError(f"{record['__key__']}: {selection.metadata['exclusion_reason']}")
        sample = {"__key__": record["__key__"], "image": image,
                  "segmentation": selection.mask,
                  "metadata": {**deepcopy(record.get("metadata", {})), **selection.metadata}}
        validate_sample(sample)
        return self.transform(sample) if self.transform is not None else sample
