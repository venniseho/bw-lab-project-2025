"""
coco_utils.py
--------------------------------------------------------------
Basic COCO API helpers for full-image processing.
--------------------------------------------------------------
"""

from pycocotools.coco import COCO
from typing import List, Dict

def get_ann_ids_by_category(coco: COCO, category_names: List[str]) -> List[int]:
    """Retrieves all annotation IDs for specified categories."""
    return coco.getAnnIds(catIds=coco.getCatIds(catNms=category_names))

def get_category_name(coco: COCO, cat_id: int) -> str:
    """Returns the string name of a category ID."""
    return coco.loadCats([cat_id])[0]['name']