"""Local image/mask manifests, with explicit split and source provenance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ._base import DataModule
from .image_mask import ImageMaskDataset


def read_records(manifest: Path) -> list[dict]:
    records = json.loads(manifest.read_text(encoding="utf-8"))
    for record in records:
        for field in ("image_path", "mask_path", "annotations_path"):
            if field in record:
                record[field] = str((manifest.resolve().parent / record[field]).resolve())
    return records


class ManifestDataModule(DataModule):
    """Build only user-supplied splits and reject source-image leakage."""

    def __init__(self, manifests: dict[str, Path], transform: Callable | None = None,
                 *, seed: int = 0):
        datasets = {}
        source_splits = {}
        for subset, manifest in manifests.items():
            records = read_records(Path(manifest))
            for record in records:
                source = record["image_path"]
                if source in source_splits and source_splits[source] != subset:
                    raise ValueError(f"source image appears in multiple splits: {source}")
                source_splits[source] = subset
            coco_records = ["annotations_path" in record for record in records]
            if any(coco_records) and not all(coco_records):
                raise ValueError("do not mix COCO and image/mask records in one manifest")
            if any(coco_records):
                from .coco_manifest import CocoManifestDataset
                datasets[subset] = CocoManifestDataset(records, transform=transform)
            else:
                datasets[subset] = ImageMaskDataset(records, transform=transform)
        super().__init__(datasets, seed=seed)
