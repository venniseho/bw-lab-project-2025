"""
coco_load.py
--------------------------------------------------------------
1) Load COCO JSON + images directory
--------------------------------------------------------------
"""
from pycocotools.coco import COCO
import numpy as np

# information needed
# - image_info
# - ann_info
# - binary mask
def get_image_annotation_info(coco, img_id):
    img_info = coco.loadImgs([img_id])[0]
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    return img_info, anns

def get_binary_mask(coco, img_info, anns, cat_ids=None):
    masks = []
    for a in anns:
        m = coco.annToMask(a)           # 0/1 mask
        if m.sum() == 0:
            print("all zero mask for ann_id:", a.get("id", "N/A"))
            continue
        masks.append(m)
        
    return masks

def main():
    # --- Configuration ---
    ann_path = "COCO/annotations/instances_val2014.json"   # path to JSON
    
    # --- Load dataset ---
    coco = COCO(ann_path)
    print(f"Loaded {len(coco.imgs)} images and {len(coco.anns)} annotations.\n")
    
    for img_id, _ in coco.imgs.items():
        img_info, anns = get_image_annotation_info(coco, img_id)
        
        print(f"image_id: {img_id}, file_name: {img_info.get('file_name', 'N/A')}")
        for a in anns:
            cat_id = a.get("category_id")
            cat_name = coco.loadCats([cat_id])[0]["name"] if cat_id else "unknown"
            print(
                f"  ann_id: {a.get('id', 'N/A')}, "
                f"category: {cat_name} ({cat_id}), "
                f"bbox: {a.get('bbox', 'N/A')}, "
                f"area: {a.get('area', 'N/A')}, "
                f"iscrowd: {a.get('iscrowd', 'N/A')}"
            )
            
        # get binary mask
        masks = get_binary_mask(coco, img_info, anns)
        print(masks)
        print(f"    Number of binary masks: {len(masks)}")
        break

if __name__ == "__main__":
    main()
    