"""
Command to run:
python -m pytest -q tests/test_03_fragmenter_outputs.py
"""

from pathlib import Path
import shutil
import cv2
import numpy as np
from pycocotools.coco import COCO
import mask_fragmenter_clean as frag

COCO_ANN = Path("COCO/annotations/instances_val2014.json")
COCO_IMGDIR = Path("COCO/val2014")

OUT = Path("outputs/tests/test03_fragmenter")
TMP = Path("tests/tmp/test03")


def test_fragmenter_writes_expected_files():
    if OUT.exists():
        shutil.rmtree(OUT)
    if TMP.exists():
        shutil.rmtree(TMP)

    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(COCO_ANN))
    img_id = coco.getImgIds()[0]
    img_info = coco.loadImgs([img_id])[0]
    img_path = COCO_IMGDIR / Path(img_info["file_name"]).name
    assert img_path.exists()

    img = cv2.imread(str(img_path))
    assert img is not None
    H, W = img.shape[:2]

    # Build a union mask as fragmenter input (any valid binary mask is fine)
    union = np.zeros((H, W), dtype=np.uint8)
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        m = coco.annToMask(ann).astype(np.uint8)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        union = np.maximum(union, m)

    assert union.sum() > 0

    mask_path = TMP / "mask.png"
    cv2.imwrite(str(mask_path), (union * 255).astype(np.uint8))

    frag.fragment_one(
        image_path=str(img_path),
        mask_path=str(mask_path),
        out_dir=str(OUT),
        outline_mode="scan",
        noise_mode="uniform",
        noise_count=200,
    )

    fragments_dir = OUT / "fragments"
    outlines_dir = OUT / "outlines"
    panels_dir = OUT / "panels"
    metrics_dir = OUT / "metrics"

    assert fragments_dir.exists()
    assert outlines_dir.exists()
    assert panels_dir.exists()
    assert metrics_dir.exists()

    assert len(list(fragments_dir.glob("*_fragmented.png"))) >= 1
    assert len(list(outlines_dir.glob("*_outline.png"))) >= 1
    assert len(list(panels_dir.glob("*_panel.png"))) >= 1
    assert len(list(metrics_dir.glob("*_metrics.json"))) >= 1
