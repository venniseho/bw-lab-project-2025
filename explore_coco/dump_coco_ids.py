"""
dump_coco_full.py
--------------------------------------------------------------
Save all images and annotations from a COCO JSON file
into a single text file for inspection.
--------------------------------------------------------------
"""

from pycocotools.coco import COCO
from pprint import pprint

ann_path = "COCO/annotations/instances_val2014.json"
out_path = "coco_full_dump.txt"

print("Loading COCO annotations...")
coco = COCO(ann_path)

with open(out_path, "w", encoding="utf-8") as f:
    # -----------------------------
    # Write image metadata section
    # -----------------------------
    f.write("========== IMAGES ==========\n")
    f.write(f"Total images: {len(coco.imgs)}\n\n")

    for img_id, img_data in coco.imgs.items():
        f.write(
            f"image_id: {img_id}, "
            f"file_name: {img_data.get('file_name', 'N/A')}, "
            f"width: {img_data.get('width', 'N/A')}, "
            f"height: {img_data.get('height', 'N/A')}, "
            f"license: {img_data.get('license', 'N/A')}, "
            f"date_captured: {img_data.get('date_captured', 'N/A')}\n"
        )

    f.write("\n========== ANNOTATIONS ==========\n")
    f.write(f"Total annotations: {len(coco.anns)}\n\n")

    # -----------------------------
    # Write annotation data section
    # -----------------------------
    for ann_id, ann_data in coco.anns.items():
        f.write(
            f"ann_id: {ann_id}, "
            f"image_id: {ann_data.get('image_id', 'N/A')}, "
            f"category_id: {ann_data.get('category_id', 'N/A')}, "
            f"iscrowd: {ann_data.get('iscrowd', 'N/A')}, "
            f"area: {ann_data.get('area', 'N/A')}, "
            f"bbox: {ann_data.get('bbox', 'N/A')}\n"
        )

print(f"Saved {len(coco.imgs)} images and {len(coco.anns)} annotations to {out_path}")
