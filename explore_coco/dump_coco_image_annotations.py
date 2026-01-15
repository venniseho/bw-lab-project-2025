"""
dump_coco_image_annotations.py
--------------------------------------------------------------
Iterate over every image in a COCO dataset and list:
  - image_id, file_name, width, height
  - all annotations for that image (category, bbox, area, etc.)
Saves everything to a text file for easy inspection.
--------------------------------------------------------------
"""

from pycocotools.coco import COCO
from pprint import pprint

# --- Configuration ---
ann_path = "COCO/annotations/instances_val2014.json"   # path to JSON
out_path = "coco_image_annotations.txt"                # output text file

# --- Load dataset ---
print("Loading COCO annotations...")
coco = COCO(ann_path)
print(f"Loaded {len(coco.imgs)} images and {len(coco.anns)} annotations.\n")

# --- Open output file ---
with open(out_path, "w", encoding="utf-8") as f:
    for idx, (img_id, img_data) in enumerate(coco.imgs.items()):
        f.write(f"=== Image {idx+1}/{len(coco.imgs)} ===\n")
        f.write(
            f"image_id: {img_id}\n"
            f"file_name: {img_data.get('file_name', 'N/A')}\n"
            f"size: {img_data.get('width', 'N/A')}x{img_data.get('height', 'N/A')}\n"
        )

        # get all annotation IDs linked to this image
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        if not anns:
            f.write("  (no annotations)\n\n")
            continue

        for a in anns:
            cat_id = a.get("category_id")
            cat_name = coco.loadCats([cat_id])[0]["name"] if cat_id else "unknown"
            f.write(
                f"  ann_id: {a.get('id', 'N/A')}, "
                f"category: {cat_name} ({cat_id}), "
                f"bbox: {a.get('bbox', 'N/A')}, "
                f"area: {a.get('area', 'N/A')}, "
                f"iscrowd: {a.get('iscrowd', 'N/A')}\n"
            )
        f.write("\n")

print(f"Finished. Saved results to: {out_path}")

