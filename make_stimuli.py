# make_stimuli.py
"""
make_stimuli.py
--------------------------------------------------------------
STAGE 1: COCO -> per-instance masks -> fragmented stimuli (+ metrics)

This script is the "stimuli generation" half of the pipeline.

  1) loads COCO images + instance annotations
  2) writes:
       out_root/assets/images/<img_stem>.png
       out_root/assets/masks/<instance_id>_mask.png
  3) calls mask_fragmenter_clean.fragment_one(...) to produce:
       out_root/stimuli/fragments/<instance_id>_fragmented.png         (SAFE stimulus)
       out_root/debug/outlines/<instance_id>_outline.png               (debug)
       out_root/debug/panels/<instance_id>_panel.png                   (debug)
       out_root/debug/metrics/<instance_id>_metrics.json (+ hists)     (debug/analysis)
  4) writes an index CSV:
       out_root/indexes/stimuli_index.csv

SAFETY:
  - Only out_root/stimuli/fragments is intended for model/human input.
  - Everything under out_root/debug is debug/analysis only.

Requires:
  pip install pycocotools opencv-python numpy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import time
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from skimage.measure import label 

import mask_fragmenter_clean as frag


def _resolve_image_path(img_dir: Path, file_name: str) -> Path:
    """
    COCO file_name is sometimes nested (e.g. 'COCO_val2014_....jpg' or 'val2014/....jpg').
    We try:
      1) img_dir / file_name
      2) img_dir / basename(file_name)
    """
    p = img_dir / file_name
    if p.exists():
        return p

    q = img_dir / Path(file_name).name
    if q.exists():
        return q

    return p  # let caller .exists() fail


def get_image_annotation_info(coco: COCO, img_id: int):
    img_info = coco.loadImgs([img_id])[0]
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    return img_info, anns


def get_instance_masks(coco: COCO, anns: list, H: int, W: int):
    """
    Returns list of tuples: (ann_dict, mask01_uint8)
    """
    masks = []
    for a in anns:
        m = coco.annToMask(a)
        if m is None or m.sum() == 0:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append((a, (m.astype(np.uint8) & 1)))
    return masks


def write_index_row(writer: csv.DictWriter, row: dict):
    # Always stringify paths for CSV portability
    row2 = row.copy()
    for k, v in row2.items():
        if isinstance(v, Path):
            row2[k] = str(v)
    writer.writerow(row2)
    
def is_mask_valid(mask_u8: np.ndarray) -> tuple[bool, str]:
    """
    Validates mask based on experiment constraints:
      1. Must be a single connected component (no disconnected fragments).
      2. Centre point must be inside the mask (convexity check).
    """
    # 1. Check connectivity
    # label() returns a labeled array and the number of labels (background=0 is ignored)
    _, num_components = label(mask_u8 > 0, return_num=True, connectivity=1)
    if num_components != 1:
        return False, f"disconnected components ({num_components})"

    # 2. Check centroid containment
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return False, "empty"
    
    # Simple geometric centroid
    cy, cx = int(ys.mean()), int(xs.mean())
    
    # Ensure coordinates are within bounds
    H, W = mask_u8.shape
    cy = np.clip(cy, 0, H - 1)
    cx = np.clip(cx, 0, W - 1)

    if mask_u8[cy, cx] == 0:
        return False, "centroid outside mask"

    return True, "ok"

def main():
    ap = argparse.ArgumentParser(description="STAGE 1: COCO -> fragmented-contour stimuli + metrics + indexes")

    ap.add_argument("--coco_ann", required=True, help="path to instances_*.json")
    ap.add_argument("--coco_imgdir", required=True, help="folder with COCO images (e.g., COCO/val2014 or COCO)")
    ap.add_argument("--out_root", default="outputs/make_stimuli", help="where to write outputs")
    ap.add_argument("--limit", type=int, default=10, help="max COCO images to process")

    # Instance filtering (helps avoid tiny outlines / trivial cases)
    ap.add_argument("--min_mask_area", type=int, default=5000,
                    help="Skip instance masks whose area < this many pixels")
    ap.add_argument("--min_mask_frac", type=float, default=0.01,
                    help="Skip instance masks whose area fraction < this value")

    # Fragmentation params (passed to fragment_one)
    ap.add_argument("--target_frag_per_100px", type=float, default=6.0)
    ap.add_argument("--gap_factor", type=float, default=0.35)
    ap.add_argument("--jitter_deg", type=int, default=0)   # 0 = hug boundary (your default)
    ap.add_argument("--thickness", type=int, default=1)
    ap.add_argument("--noise_mode", choices=["uniform", "grid"], default="uniform")
    ap.add_argument("--noise_count", type=int, default=400)
    ap.add_argument("--sep_pad", type=int, default=1)
    ap.add_argument("--grid", type=int, default=40)

    args = ap.parse_args()

    out_root = Path(args.out_root)

    # Folder layout (safe + modular)
    assets_images = out_root / "assets" / "images"
    assets_masks = out_root / "assets" / "masks"
    indexes_dir = out_root / "indexes"
    stimuli_subdir = "stimuli/fragments"  # relative to out_root
    debug_subdir = "debug"                # relative to out_root

    assets_images.mkdir(parents=True, exist_ok=True)
    assets_masks.mkdir(parents=True, exist_ok=True)
    indexes_dir.mkdir(parents=True, exist_ok=True)

    index_path = indexes_dir / "stimuli_index.csv"
    manifest_path = indexes_dir / "manifest.jsonl"

    coco = COCO(args.coco_ann)
    print(f"Loaded {len(coco.imgs)} images and {len(coco.anns)} annotations.\n")

    processed = 0
    t_total0 = time.perf_counter()

    fieldnames = [
    "instance_id",
    "coco_image_id",
    "coco_ann_id",
    "orig_img_path",
    "frag_img_path",
    "gt_mask_path",
    "outline_img_path",
    "panel_img_path",
    "metrics_json_path",
]

    with open(index_path, "w", newline="") as f_csv, open(manifest_path, "w") as f_manifest:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for img_id in coco.getImgIds():
            if processed >= args.limit:
                break

            t_img0 = time.perf_counter()

            img_info, anns = get_image_annotation_info(coco, img_id)
            file_name = img_info.get("file_name", "")
            img_path = _resolve_image_path(Path(args.coco_imgdir), file_name)
            if not img_path.exists():
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            H, W = img_bgr.shape[:2]
            masks = get_instance_masks(coco, anns, H, W)
            if not masks:
                continue

            img_stem = Path(file_name).stem
            dst_img = assets_images / f"{img_stem}.png"
            cv2.imwrite(str(dst_img), img_bgr)

            kept_instances = 0

            for a, m01 in masks:
                ann_id = a.get("id", "N")

                inst_area = int(m01.sum())
                inst_frac = float(inst_area) / float(H * W)

                if inst_area < args.min_mask_area or inst_frac < args.min_mask_frac:
                    continue

                instance_id = f"{img_stem}_ann{ann_id}"

                # mask filtering based on connectivity + centroid check
                m_u8 = (m01 * 255).astype(np.uint8)
                is_valid, reason = is_mask_valid(m_u8)
                
                if not is_valid:
                    # Optional: print skipped reason for debugging
                    print(f"Skipping {instance_id}: {reason}")
                    continue
                
                # Save GT mask per instance
                inst_mask_path = assets_masks / f"{instance_id}_mask.png"
                cv2.imwrite(str(inst_mask_path), m_u8)

                # Generate stimulus + debug + metrics
                frag.fragment_one(
                    image_path=str(dst_img),
                    mask_path=str(inst_mask_path),
                    out_dir=str(out_root),
                    output_stem=instance_id,
                    edge_len=-1,  # auto from perimeter
                    target_frag_per_100px=args.target_frag_per_100px,
                    grid=args.grid,
                    gap_factor=args.gap_factor,
                    jitter_deg=args.jitter_deg,
                    thickness=args.thickness,
                    noise_mode=args.noise_mode,
                    noise_count=args.noise_count,
                    sep_pad=args.sep_pad,
                    noise_per_cell=1,
                    outline_mode="scan",
                    max_outline_segments=None,

                    stimuli_subdir=stimuli_subdir,
                    debug_subdir=debug_subdir,
                )

                # Paths that fragment_one produces (based on its directory conventions)
                panel_img_path = out_root / debug_subdir / "panels" / f"{instance_id}_panel.png"
                frag_img_path = out_root / stimuli_subdir / f"{instance_id}_fragmented.png"
                outline_img_path = out_root / debug_subdir / "outlines" / f"{instance_id}_outline.png"
                metrics_json_path = out_root / debug_subdir / "metrics" / f"{instance_id}_metrics.json"

                # Write index row
                write_index_row(writer, {
                    "instance_id": instance_id,
                    "coco_image_id": img_id,
                    "coco_ann_id": ann_id,
                    "orig_img_path": dst_img,
                    "frag_img_path": frag_img_path,
                    "gt_mask_path": inst_mask_path,
                    "outline_img_path": outline_img_path,
                    "panel_img_path": panel_img_path,
                    "metrics_json_path": metrics_json_path,
                })
                
                rec = {
                    "stem": instance_id,
                    "orig_img_path": str(dst_img),
                    "frag_img_path": str(frag_img_path),
                    "gt_mask_path": str(inst_mask_path),
                }
                f_manifest.write(json.dumps(rec) + "\n")

                kept_instances += 1

            if kept_instances == 0:
                continue

            processed += 1
            t_img1 = time.perf_counter()
            print(
                f"[{processed}] {file_name} processed in {t_img1 - t_img0:.3f}s "
                f"(instances={len(masks)} | kept={kept_instances})"
            )

    t_total1 = time.perf_counter()

    print(f"\nDone. Processed {processed} COCO images in {t_total1 - t_total0:.3f}s.")
    print(f"Assets images     -> {assets_images}")
    print(f"Assets masks      -> {assets_masks}")
    print(f"Stimuli (SAFE)    -> {out_root / stimuli_subdir}")
    print(f"Debug (UNSAFE)    -> {out_root / debug_subdir}")
    print(f"Index CSV      -> {index_path}")
    print(f"Manifest JSONL -> {manifest_path}")



if __name__ == "__main__":
    main()
