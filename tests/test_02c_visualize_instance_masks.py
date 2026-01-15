"""
Command to run (IMPORTANT: use -s to see prints):
python -m pytest -q tests/test_02c_visualize_instance_masks.py -s

Outputs:
  outputs/tests/test02c_mask_visualization/
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

# ---------------------------
# Config
# ---------------------------
COCO_ANN = Path("COCO/annotations/instances_val2014.json")
COCO_IMGDIR = Path("COCO/val2014")

OUT_DIR = Path("outputs/tests/test02c_mask_visualization")

# Match pipeline defaults
MIN_MASK_AREA = 5000
MIN_MASK_FRAC = 0.01


def overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color=(0, 255, 0),
    alpha=0.5,
):
    """
    Overlay a binary mask on an image.
    mask: 0/1 or 0/255
    """
    overlay = image_bgr.copy()
    mask_bool = mask > 0

    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] +
        alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)

    return overlay


def test_visualize_instance_masks_for_one_image():
    if OUT_DIR.exists():
        for p in OUT_DIR.glob("*"):
            p.unlink()
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(COCO_ANN))

    # --------------------------------------------------
    # Pick first image that loads and has instance masks
    # --------------------------------------------------
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

    print("\n===============================================")
    print(f"Image ID: {img_id}")
    print(f"File: {file_name}")
    print(f"Image size: {H} x {W}")
    print(f"Total instance masks: {len(masks)}")
    print("Output dir:", OUT_DIR)
    print("===============================================\n")

    # Save original image for reference
    cv2.imwrite(str(OUT_DIR / "original.png"), img)

    kept_count = 0
    rejected_count = 0

    # --------------------------------------------------
    # Process each instance mask
    # --------------------------------------------------
    for a, m in masks:
        ann_id = a.get("id", "N/A")
        cat_id = a.get("category_id", "N/A")

        area = int(np.count_nonzero(m > 0))
        frac = area / float(H * W)

        passes = (area >= MIN_MASK_AREA) and (frac >= MIN_MASK_FRAC)

        status = "KEPT" if passes else "REJECTED"
        color = (0, 200, 0) if passes else (0, 0, 255)

        overlay = overlay_mask(img, m, color=color, alpha=0.5)

        out_name = (
            f"{status.lower()}_ann{ann_id}"
            f"_cat{cat_id}"
            f"_area{area}"
            f"_frac{frac:.4f}.png"
        )
        cv2.imwrite(str(OUT_DIR / out_name), overlay)

        print(
            f"{status:8} | ann_id={ann_id:>6} | "
            f"cat={cat_id:>3} | "
            f"area_px={area:>8} | "
            f"frac={frac:.4f}"
        )

        if passes:
            kept_count += 1
        else:
            rejected_count += 1

    print("\nSUMMARY")
    print(f"  kept     : {kept_count}")
    print(f"  rejected : {rejected_count}")
    print("===============================================\n")

    # Minimal sanity assertions
    assert len(masks) > 0
    assert kept_count + rejected_count == len(masks)
