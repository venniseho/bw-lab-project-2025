"""Lazy one-instance samples, with a hook for later in-memory transforms."""
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from bwlab.datasets.coco_manifest import CocoManifestDataset


def foreground_point(mask):
    """Use the foreground pixel nearest its centroid, always inside the mask."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("No visible foreground pixel available for a positive prompt")
    index = ((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).argmin()
    return [float(xs[index]), float(ys[index])]


def dataset(annotation_file, transform=None):
    path = Path(annotation_file).resolve()
    data = json.loads(path.read_text())
    images = {image["id"]: image for image in data["images"]}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records = [{"__key__": f"train2017_{ann['image_id']:012d}_ann{ann['id']}",
                "image_id": ann["image_id"], "annotation_id": ann["id"],
                "annotations_path": str(path), "annotations_sha256": digest,
                "image_path": images[ann["image_id"]]["file_name"]}
               for ann in sorted(data["annotations"], key=lambda item: item["id"])
               if ann.get("iscrowd") == 0]
    return CocoManifestDataset(records, transform=transform)


def single_sample(batch):
    return batch[0]


def loader(samples, workers):
    return DataLoader(samples, batch_size=1, shuffle=False, num_workers=workers,
                      collate_fn=single_sample, persistent_workers=workers > 0)
