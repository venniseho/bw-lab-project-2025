"""Lazy one-instance samples, with a hook for later in-memory transforms."""

import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.data_loading.coco_manifest import CocoManifestDataset


def dataset(annotation_file, transform=None, *, one_per_image=False, seed=42, image_count=None):
    path = Path(annotation_file).resolve()
    data = json.loads(path.read_text())
    images = {image["id"]: image for image in data["images"]}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records = [
        {
            "__key__": f"{'val2017' if Path(images[ann['image_id']]['file_name']).parent.name == 'val2017' else 'train2017'}_{ann['image_id']:012d}_ann{ann['id']}",
            "image_id": ann["image_id"],
            "annotation_id": ann["id"],
            "annotations_path": str(path),
            "annotations_sha256": digest,
            "image_path": images[ann["image_id"]]["file_name"],
        }
        for ann in sorted(data["annotations"], key=lambda item: item["id"])
        if ann.get("iscrowd") == 0
    ]
    if image_count is not None:
        ids = sorted({record["image_id"] for record in records})
        random.Random(seed).shuffle(ids)
        if not 0 < image_count <= len(ids):
            raise ValueError("Validation image count exceeds available eligible images")
        chosen = set(ids[:image_count])
        records = [record for record in records if record["image_id"] in chosen]
    if one_per_image:
        groups = {}
        for record in records:
            groups.setdefault(record["image_id"], []).append(record)
        records = [
            random.Random(f"{seed}:{image_id}").choice(group)
            for image_id, group in sorted(groups.items())
        ]
    for record in records:
        record["metadata"] = {"image_path": record["image_path"]}
    return CocoManifestDataset(records, transform=transform)


def single_sample(batch):
    return batch[0]


def loader(samples, workers, *, shuffle=False, seed=42):
    return DataLoader(
        samples,
        batch_size=1,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        collate_fn=single_sample,
        persistent_workers=workers > 0,
    )
