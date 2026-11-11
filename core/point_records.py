"""Reconstructable per-object provenance and compact selected predictions."""

import json

import numpy as np


def sample_record(sample):
    metadata = sample.get("metadata", {})
    return {
        "key": sample["__key__"],
        "image_id": metadata.get("image_id"),
        "annotation_id": metadata.get("annotation_id"),
        "image_path": metadata.get("image_path"),
        "cache_wait_seconds": metadata.get("cache_wait_seconds", 0.0),
        "image_height": sample["image"].shape[0],
        "image_width": sample["image"].shape[1],
        "target_pixels": int(np.count_nonzero(sample["segmentation"])),
        "visible_pixels": int(
            np.count_nonzero(sample.get("visible_segmentation", sample["segmentation"]))
        ),
        "transform_json": json.dumps(sample.get("transforms", []), sort_keys=True),
    }


def mask_rle(mask):
    from pycocotools import mask as masks

    encoded = masks.encode(np.asfortranarray(mask, dtype=np.uint8))
    return json.dumps({"size": encoded["size"], "counts": encoded["counts"].decode("ascii")})
