"""
make_stimuli.py
----------------
DESCRIPTION:
STAGE 1: This script converts COCO dataset instances into fragmented stimuli.
It filters object masks for quality (size, connectivity) and uses the 
'mask_fragmenter' engine to create dashed-contour objects hidden in 
statistically matched noise.

FUNCTIONS:
- StimuliGenerator: Class to manage COCO loading, mask filtering, and directory setup.
- is_mask_valid: Ensures the mask is a single connected blob with a central centroid.
- process_image: Extracts all valid instances from a single COCO image and fragments them.
- main: CLI entry point for processing the batch.
"""

from __future__ import annotations

import argparse
import csv
import time
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from pycocotools.coco import COCO
from skimage.measure import label 

# Custom fragmenter module
import mask_fragmenter as frag

class StimuliGenerator:
    """
    Manages the end-to-end generation of fragmented stimuli from a COCO dataset.
    """

    def __init__(self, coco_ann: str, coco_imgdir: str, out_root: str, from_manifest: Optional[str] = None):
        self.coco = COCO(coco_ann)
        self.img_dir = Path(coco_imgdir)
        self.out_root = Path(out_root)
        
        # If we are using a manifest, assets are in the seed (n0) directory
        if from_manifest:
            seed_root = Path(from_manifest).parents[1] 
            self.assets_images = seed_root / "assets" / "images"
            self.assets_masks = seed_root / "assets" / "masks"
        else:
            self.assets_images = self.out_root / "assets" / "images"
            self.assets_masks = self.out_root / "assets" / "masks"

        self.indexes_dir = self.out_root / "indexes"
        self.stimuli_subdir = "stimuli"
        self.debug_subdir = "debug"

        # Create local output directories for this noise level
        for d in [self.indexes_dir, self.out_root / self.stimuli_subdir]:
            d.mkdir(parents=True, exist_ok=True)

        # Only create assets if this is the original seed run (no manifest)
        if not from_manifest:
            self.assets_images.mkdir(parents=True, exist_ok=True)
            self.assets_masks.mkdir(parents=True, exist_ok=True)

    def _resolve_image_path(self, file_name: str) -> Path:
        p = self.img_dir / file_name
        if p.exists(): return p
        return self.img_dir / Path(file_name).name

    def is_mask_valid(self, mask_u8: np.ndarray) -> Tuple[bool, str]:
        _, num_components = label(mask_u8 > 0, return_num=True, connectivity=1)
        if num_components != 1:
            return False, f"disconnected components ({num_components})"

        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0: return False, "empty"
        
        cy, cx = int(ys.mean()), int(xs.mean())
        H, W = mask_u8.shape
        if mask_u8[np.clip(cy, 0, H-1), np.clip(cx, 0, W-1)] == 0:
            return False, "centroid outside mask"

        return True, "ok"

    def process_image(self, img_id: int, writer: csv.DictWriter, f_manifest, args: argparse.Namespace) -> int:
        img_info = self.coco.loadImgs([img_id])[0]
        
        # Pull specific annotations if re-running from a manifest
        if args.from_manifest:
            with open(args.from_manifest, "r") as f:
                valid_ann_ids = [
                    json.loads(line)["coco_ann_id"] 
                    for line in f if line.strip() and json.loads(line)["coco_image_id"] == img_id
                ]
            ann_ids = valid_ann_ids
        else:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            
        anns = self.coco.loadAnns(ann_ids)
        file_name = img_info.get("file_name", "")
        img_path = self._resolve_image_path(file_name)
        if not img_path.exists(): return 0

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: return 0

        H, W = img_bgr.shape[:2]
        img_stem = Path(file_name).stem
        dst_img = self.assets_images / f"{img_stem}.png"
        
        # Save image only if it doesn't already exist (Prevents Quota bloat)
        if not dst_img.exists():
            cv2.imwrite(str(dst_img), img_bgr)

        kept_count = 0
        for ann in anns:
            mask_01 = self.coco.annToMask(ann)
            area = int(mask_01.sum())
            if area < args.min_mask_area or (area / (H * W)) < args.min_mask_frac:
                continue

            m_u8 = (mask_01 * 255).astype(np.uint8)
            is_valid, reason = self.is_mask_valid(m_u8)
            if not is_valid: continue

            instance_id = f"{img_stem}_ann{ann['id']}"
            inst_mask_path = self.assets_masks / f"{instance_id}_mask.png"
            
            if not inst_mask_path.exists():
                cv2.imwrite(str(inst_mask_path), m_u8)

            # Fragmentation Engine
            frag.fragment_one(
                image_path=str(dst_img),
                mask_path=str(inst_mask_path),
                out_dir=str(self.out_root),
                output_stem=instance_id,
                target_frag_per_100px=args.target_frag_per_100px,
                grid=args.grid,
                gap_factor=args.gap_factor,
                jitter_deg=args.jitter_deg,
                thickness=args.thickness,
                noise_mode=args.noise_mode,
                noise_count=args.noise_count,
                sep_pad=args.sep_pad,
                stimuli_subdir=self.stimuli_subdir,
                debug_subdir=self.debug_subdir
            )

            frag_img_p = self.out_root / self.stimuli_subdir / f"{instance_id}_fragmented.png"
            
            # CSV Indexing
            paths = {
                "instance_id": instance_id,
                "coco_image_id": img_id,
                "coco_ann_id": ann['id'],
                "orig_img_path": dst_img,
                "frag_img_path": frag_img_p,
                "gt_mask_path": inst_mask_path,
                "outline_img_path": self.out_root / self.debug_subdir / "outlines" / f"{instance_id}_outline.png",
                "panel_img_path": self.out_root / self.debug_subdir / "panels" / f"{instance_id}_panel.png",
                "metrics_json_path": self.out_root / self.debug_subdir / "metrics" / f"{instance_id}_metrics.json"
            }
            writer.writerow({k: str(v) for k, v in paths.items()})
            
            # --- FIXED: ADDED MISSING KEYS TO MANIFEST ---
            manifest_data = {
                "stem": instance_id,
                "coco_image_id": img_id, # Required for re-runs
                "coco_ann_id": ann['id'],   # Required for re-runs
                "orig_img_path": str(dst_img),
                "frag_img_path": str(frag_img_p),
                "gt_mask_path": str(inst_mask_path), 
                "gt_area": area
            }
            f_manifest.write(json.dumps(manifest_data) + "\n")
            kept_count += 1
        
        return kept_count

def main():
    ap = argparse.ArgumentParser(description="STAGE 1: Stimuli Generation")
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--coco_imgdir", required=True)
    ap.add_argument("--out_root", default="outputs/make_stimuli")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--min_mask_area", type=int, default=5000)
    ap.add_argument("--min_mask_frac", type=float, default=0.01)
    ap.add_argument("--from_manifest", help="Path to manifest.jsonl to reuse image/ann IDs.")
    
    ap.add_argument("--target_frag_per_100px", type=float, default=6.0)
    ap.add_argument("--gap_factor", type=float, default=0.35)
    ap.add_argument("--jitter_deg", type=int, default=0)
    ap.add_argument("--thickness", type=int, default=1)
    ap.add_argument("--noise_mode", default="uniform")
    ap.add_argument("--noise_count", type=int, default=400)
    ap.add_argument("--sep_pad", type=int, default=1)
    ap.add_argument("--grid", type=int, default=40)
    args = ap.parse_args()

    # Pass manifest to init to enable shared assets
    gen = StimuliGenerator(args.coco_ann, args.coco_imgdir, args.out_root, from_manifest=args.from_manifest)
    
    manifest_p = gen.indexes_dir / "manifest.jsonl"
    csv_p = gen.indexes_dir / "stimuli_index.csv"

    if manifest_p.exists():
        manifest_p.unlink()

    fieldnames = ["instance_id", "coco_image_id", "coco_ann_id", "orig_img_path", 
                  "frag_img_path", "gt_mask_path", "outline_img_path", 
                  "panel_img_path", "metrics_json_path"]

    processed = 0
    t0 = time.perf_counter()

    with open(csv_p, "w", newline="", encoding="utf-8") as f_csv, \
         open(manifest_p, "w", encoding="utf-8") as f_manifest:
        
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        # FIXED: Indentation ensures loop happens while files are open
        if args.from_manifest:
            with open(args.from_manifest, "r", encoding="utf-8") as f:
                target_img_ids = list(dict.fromkeys([
                    json.loads(line)["coco_image_id"] 
                    for line in f if line.strip()
                ]))
            print(f"Re-using {len(target_img_ids)} image IDs from manifest.")
        else:
            target_img_ids = gen.coco.getImgIds()

        for img_id in target_img_ids:
            if not args.from_manifest and processed >= args.limit: 
                break
            
            try:
                count = gen.process_image(img_id, writer, f_manifest, args)
                if count > 0:
                    processed += 1
                    f_manifest.flush()
                    os.fsync(f_manifest.fileno())
                    print(f"[{processed}] Processed image {img_id} ({count} instances kept)")
            except Exception as e:
                print(f"!! Error processing image {img_id}: {e}")
                continue

    print(f"\nDone. Processed {processed} images in {time.perf_counter() - t0:.2f}s.")

if __name__ == "__main__":
    main()