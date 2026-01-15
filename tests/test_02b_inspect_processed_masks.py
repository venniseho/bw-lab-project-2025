"""
Command to run:
python -m pytest -q tests/test_02b_inspect_processed_masks.py -s

NOTE:
  Use -s so pytest does NOT capture stdout.
  This test is intentionally verbose.
"""

from pathlib import Path
import cv2
import numpy as np
from pycocotools.coco import COCO

from coco_pipeline_revised import (
    get_image_annotation_info,
    get_instance_masks,
    _resolve_image_path,
)

COCO_ANN = Path("COCO/annotations/instances_val2014.json")
COCO_IMGDIR = Path("COCO/val2014")


def test_inspect_instance_masks_for_one_image():
    coco = COCO(str(COCO_ANN))

    # ---- filtering thresholds (match pipeline defaults) ----
    MIN_MASK_AREA = 5000
    MIN_MASK_FRAC = 0.01

    # Pick a deterministic image (first one that loads)
    for img_id in coco.getImgIds():
        img_info, anns = get_image_annotation_info(coco, img_id)
        file_name = img_info.get("file_name", "")
        img_path = _resolve_image_path(COCO_IMGDIR, file_name)

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        H, W = img.shape[:2]
        masks = get_instance_masks(coco, anns, H, W)

        if masks:
            break
    else:
        raise AssertionError("Could not find a COCO image with valid instance masks.")

    print("\n====================================================")
    print(f"Image ID: {img_id}")
    print(f"File: {file_name}")
    print(f"Image size: H={H}, W={W}")
    print(f"Total annotations: {len(anns)}")
    print(f"Valid instance masks returned: {len(masks)}")
    print("----------------------------------------------------")

    kept = []
    rejected = []

    for a, m in masks:
        ann_id = a.get("id", "N/A")
        cat_id = a.get("category_id", "N/A")

        area = int(np.count_nonzero(m > 0))
        frac = area / float(H * W)

        passes = (area >= MIN_MASK_AREA) and (frac >= MIN_MASK_FRAC)

        entry = {
            "ann_id": ann_id,
            "cat_id": cat_id,
            "area_px": area,
            "area_frac": round(frac, 5),
        }

        if passes:
            kept.append(entry)
        else:
            rejected.append(entry)

    # ---- print summary ----
    print("KEPT INSTANCES:")
    for e in kept:
        print(
            f"  ann_id={e['ann_id']:>6} | "
            f"cat={e['cat_id']:>3} | "
            f"area_px={e['area_px']:>8} | "
            f"frac={e['area_frac']}"
        )

    print("\nREJECTED INSTANCES:")
    for e in rejected:
        print(
            f"  ann_id={e['ann_id']:>6} | "
            f"cat={e['cat_id']:>3} | "
            f"area_px={e['area_px']:>8} | "
            f"frac={e['area_frac']}"
        )

    print("\nSUMMARY:")
    print(f"  kept     = {len(kept)}")
    print(f"  rejected = {len(rejected)}")
    print("====================================================\n")

    # ---- minimal assertions (do not over-constrain) ----
    assert len(masks) > 0, "Expected at least one instance mask."
    assert len(kept) + len(rejected) == len(masks)
