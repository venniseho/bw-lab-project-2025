"""
coco_load.py
--------------------------------------------------------------
1) Load COCO JSON + images directory
2) For each image (up to --limit):
   - fetch image+annotations
   - build binary masks (list)
   - save original image (PNG) and union mask PNG
   - (optional) save per-instance masks
--------------------------------------------------------------
"""
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

import mask_fragmenter_clean as frag

# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Helpers (administrative?)
# ------------------------------------------------------------
def _resolve_image_path(img_dir: Path, file_name: str) -> Path:
    """
    Robustly resolve an on-disk path for COCO file_name which may or may not
    contain subfolders like 'COCO/val2014/...'.
    """
    p = img_dir / file_name
    if p.exists():
        return p
    # Try basename-only fallback
    q = img_dir / Path(file_name).name
    return q if q.exists() else p  # return p (non-existent) if both fail

def main():
    ap = argparse.ArgumentParser()
    
    # --- Arguments (administrative)---
    ap.add_argument("--coco_ann", required=True, help="path to instances_*.json")
    ap.add_argument("--coco_imgdir", required=True, help="folder with COCO images (e.g., COCO/val2014)")
    ap.add_argument("--out_root", default=".", help="where to write images/ and masks/")
    ap.add_argument("--limit", type=int, default=10, help="max images to process")
    ap.add_argument("--save_instances", action="store_true",
                    help="Also save per-instance masks and outline-only fragments.")
    
    # --- Fragmentation parameters ---
    ap.add_argument("--target_frag_per_100px", type=float, default=6.0)
    ap.add_argument("--gap_factor", type=float, default=0.35)
    ap.add_argument("--jitter_deg", type=int, default=15)
    ap.add_argument("--thickness", type=int, default=1)
    ap.add_argument("--noise_mode", choices=["uniform", "grid"], default="uniform")
    ap.add_argument("--noise_count", type=int, default=400)
    ap.add_argument("--sep_pad", type=int, default=1)
    ap.add_argument("--grid", type=int, default=40)
    
    args = ap.parse_args()

    # --- Output directories ---
    out_root          = Path(args.out_root)
    out_images        = out_root / "images"
    out_masks         = out_root / "masks"
    out_clean         = out_root / "output_clean"
    out_inst_outlines = out_root / "output" / "instance_outlines"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_clean.mkdir(parents=True, exist_ok=True)
    
    if args.save_instances:
        out_inst_outlines.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    coco = COCO(args.coco_ann)
    print(f"Loaded {len(coco.imgs)} images and {len(coco.anns)} annotations.\n")

    # --- Process images ---
    processed = 0
    for img_id in coco.getImgIds():
        if processed >= args.limit:
            break

        img_info, anns = get_image_annotation_info(coco, img_id)
        file_name = img_info.get("file_name", "")
        img_path = _resolve_image_path(Path(args.coco_imgdir), file_name)
        if not img_path.exists():
            print(f"Missing image on disk, skip: {img_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Could not read: {img_path}")
            continue
        H, W = img_bgr.shape[:2]

        # gather instance masks and build union mask
        masks = get_binary_mask(coco, img_info, anns)
        if masks:
            union = np.zeros((H, W), dtype=np.uint8)
            for m in masks:
                # annToMask returns HxW {0,1}
                if m.shape[0] != H or m.shape[1] != W:
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                union |= (m.astype(np.uint8) & 1)
            union_u8 = (union * 255).astype(np.uint8)
        else:
            print(f"no annotations for {file_name}")
            union_u8 = np.zeros((H, W), dtype=np.uint8)

        # save original image as PNG +
        stem = Path(file_name).stem
        dst_img = out_images / f"{stem}.png"
        cv2.imwrite(str(dst_img), img_bgr)

        # save union mask that fragmenter expects: masks/<stem>_mask.png
        dst_mask = out_masks / f"{stem}_mask.png"
        cv2.imwrite(str(dst_mask), union_u8)

        # call fragmenter (writes outline/fragmented/panel in output_clean/)
        frag.fragment_one(
            image_path=str(dst_img),
            mask_path=str(dst_mask),
            out_dir=str(out_clean),
            edge_len=-1,                              # auto from perimeter
            target_frag_per_100px=args.target_frag_per_100px,
            grid=args.grid,
            gap_factor=args.gap_factor,
            jitter_deg=args.jitter_deg,
            thickness=args.thickness,
            noise_mode=args.noise_mode,               # "uniform" or "grid"
            noise_count=args.noise_count,             # used if uniform
            sep_pad=args.sep_pad,
            noise_per_cell=1                          # ignored in uniform mode
        )

        # ---- (optional) per-instance OUTLINES (no background noise) ----
        if args.save_instances and masks and anns:
            for a, m in zip(anns, masks):
                ann_id = a.get("id", "N")
                m_u8 = (m.astype(np.uint8) * 255)
                inst_mask_path = out_masks / f"{stem}_ann{ann_id}.png"
                cv2.imwrite(str(inst_mask_path), m_u8)

                # outline-only: call fragmenter with noise_count=0
                frag.fragment_one(
                    image_path=str(dst_img),
                    mask_path=str(inst_mask_path),
                    out_dir=str(out_inst_outlines),
                    edge_len=-1,                          # auto from perimeter
                    target_frag_per_100px=args.target_frag_per_100px,
                    grid=args.grid,
                    gap_factor=args.gap_factor,
                    jitter_deg=args.jitter_deg,
                    thickness=args.thickness,
                    noise_mode="uniform",
                    noise_count=0,                        # <- no background lines
                    sep_pad=args.sep_pad,
                    noise_per_cell=0                      # ignored
                )

        processed += 1

    print(f"\nDone. Processed {processed} images.")
    print(f"Images        -> {out_images}")
    print(f"Masks         -> {out_masks}")
    print(f"Output (frag) -> {out_clean}")
    if args.save_instances:
        print(f"Instance outlines     -> {out_inst_outlines}")

if __name__ == "__main__":
    main()
