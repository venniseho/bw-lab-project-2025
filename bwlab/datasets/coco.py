"""COCO instance Dataset reusing the inclusive selection policy."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from bwlab.datasets.selection import load_coco_image_instances
from bwlab.stimuli.transforms import validate_sample


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
