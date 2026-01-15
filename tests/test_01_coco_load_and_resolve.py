"""
Command to run:
python -m pytest -q tests/test_01_coco_load_and_resolve.py
"""

from pathlib import Path
from pycocotools.coco import COCO

COCO_ANN = Path("COCO/annotations/instances_val2014.json")
COCO_IMGDIR = Path("COCO/val2014")

def test_coco_annotation_exists():
    assert COCO_ANN.exists(), f"Missing COCO ann file: {COCO_ANN}"

def test_coco_imgdir_exists():
    assert COCO_IMGDIR.exists(), f"Missing COCO image dir: {COCO_IMGDIR}"

def test_load_coco_and_resolve_one_image():
    coco = COCO(str(COCO_ANN))
    img_ids = coco.getImgIds()
    assert len(img_ids) > 0, "No image ids found in COCO annotations"

    img_info = coco.loadImgs(img_ids[0])[0]
    file_name = img_info["file_name"]

    img_path = COCO_IMGDIR / file_name
    assert img_path.exists(), f"COCO image not found: {img_path}"
