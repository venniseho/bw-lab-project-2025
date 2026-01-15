"""
Command to run:
python -m pytest -q tests/test_02_mask_filtering.py
"""

from pathlib import Path
import cv2
import numpy as np
from pycocotools.coco import COCO

from coco_pipeline_revised import get_image_annotation_info, get_instance_masks

COCO_ANN = Path("COCO/annotations/instances_val2014.json")
COCO_IMGDIR = Path("COCO/val2014")


def _find_image_with_instances(coco: COCO, max_tries: int = 2000):
    """Find an image id that has at least 1 non-empty instance mask."""
    for img_id in coco.getImgIds()[:max_tries]:
        img_info, anns = get_image_annotation_info(coco, img_id)
        file_name = img_info.get("file_name", "")
        img_path = COCO_IMGDIR / Path(file_name).name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        masks = get_instance_masks(coco, anns, H, W)
        if masks:
            return img_id, img_info, img_path, img, masks

    raise AssertionError("Could not find any COCO image with at least 1 valid instance mask.")


def test_mask_filtering_thresholds_behave():
    coco = COCO(str(COCO_ANN))
    img_id, img_info, img_path, img, masks = _find_image_with_instances(coco)

    H, W = img.shape[:2]

    # Compute areas for this image's instance masks
    areas = []
    for a, m in masks:
        areas.append(int(np.count_nonzero(m > 0)))
    assert len(areas) > 0
    assert max(areas) > 0

    # Low thresholds should keep at least one mask
    min_area_low = 1
    min_frac_low = 1e-6
    kept_low = []
    for a, m in masks:
        area = int(np.count_nonzero(m > 0))
        frac = area / float(H * W)
        if area >= min_area_low and frac >= min_frac_low:
            kept_low.append((a, m))
    assert len(kept_low) >= 1, "Expected at least one instance to pass low thresholds."

    # Very high thresholds should keep none
    min_area_high = H * W  # impossible for a single instance in COCO usually
    min_frac_high = 0.99
    kept_high = []
    for a, m in masks:
        area = int(np.count_nonzero(m > 0))
        frac = area / float(H * W)
        if area >= min_area_high and frac >= min_frac_high:
            kept_high.append((a, m))
    assert len(kept_high) == 0, "Expected no instances to pass extreme high thresholds."
