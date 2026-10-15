"""
coco_pipeline_revised.py
--------------------------------------------------------------
COCO → union mask → fragmented-contour stimuli

For each image (up to --limit):
  - load RGB image + annotations
  - build union binary mask (0/255)
  - save image as PNG to images/
  - save union mask to masks/<stem>_mask.png
  - run mask_fragmenter_clean.fragment_one(...) to create
    fragmented stimuli in output/fragments/
  - (optional) also save per-instance masks + outline-only
    fragments in output/instance_outlines/
--------------------------------------------------------------
"""

import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

import mask_fragmenter_clean as frag
from sam_runner import run_sam_on_pair

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _resolve_image_path(img_dir: Path, file_name: str) -> Path:
    """
    Robustly resolve an on-disk path for COCO file_name which may or may not
    contain subfolders like 'COCO/val2014/...'.

    Works with:
      --coco_imgdir COCO
      --coco_imgdir COCO/val2014
    """
    p = img_dir / file_name
    if p.exists():
        return p

    # Try basename-only in the given directory
    q = img_dir / Path(file_name).name
    if q.exists():
        return q

    # Try stripping any leading 'COCO/val2014/' from file_name
    parts = Path(file_name).parts
    if len(parts) >= 3 and parts[0] == "COCO" and parts[1].startswith("val"):
        r = img_dir / parts[-1]
        if r.exists():
            return r

    return p  # fallback (likely non-existent, caller checks .exists())


def get_image_annotation_info(coco, img_id):
    """Return (img_info, anns) for a given image id."""
    img_info = coco.loadImgs([img_id])[0]
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    return img_info, anns


def get_binary_masks(coco, anns, H, W):
    """
    Return a list of instance masks (H x W uint8, values 0 or 1).
    """
    masks = []
    for a in anns:
        m = coco.annToMask(a)  # 0/1 mask at original size
        if m.sum() == 0:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append((m.astype(np.uint8) & 1))
    return masks


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="COCO → fragmented-contour stimuli (revised pipeline)"
    )

    # --- IO arguments ---
    ap.add_argument("--coco_ann", required=True, help="path to instances_*.json")
    ap.add_argument(
        "--coco_imgdir",
        required=True,
        help="folder with COCO images (e.g., COCO/val2014 or COCO)",
    )
    ap.add_argument("--out_root", default=".", help="where to write outputs")
    ap.add_argument("--limit", type=int, default=10, help="max images to process")
    ap.add_argument(
        "--save_instances",
        action="store_true",
        help="Also save per-instance masks and outline-only fragments",
    )

    # --- Fragmentation parameters (forwarded to fragment_one) ---
    ap.add_argument("--target_frag_per_100px", type=float, default=6.0)
    ap.add_argument("--gap_factor", type=float, default=0.35)
    ap.add_argument("--jitter_deg", type=int, default=15)
    ap.add_argument("--thickness", type=int, default=1)
    ap.add_argument("--noise_mode", choices=["uniform", "grid"], default="uniform")
    ap.add_argument("--noise_count", type=int, default=400)
    ap.add_argument("--sep_pad", type=int, default=1)
    ap.add_argument("--grid", type=int, default=40)
    
    # --- SAM parameters ---
    ap.add_argument("--sam_ckpt", type=str, default=None, help="Path to SAM checkpoint (.pth). If not provided, SAM is skipped.")
    ap.add_argument("--sam_model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM backbone type matching the checkpoint (vit_h, vit_l, vit_b)",
    )


    args = ap.parse_args()

    # --- Output directories ---
    out_root = Path(args.out_root)
    out_images = out_root / "images"
    out_masks = out_root / "masks"
    out_fragments = out_root / "output" / "fragments"
    out_inst_outlines = out_root / "output" / "instance_outlines"

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_fragments.mkdir(parents=True, exist_ok=True)
    if args.save_instances:
        out_inst_outlines.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    coco = COCO(args.coco_ann)
    print(f"Loaded {len(coco.imgs)} images and {len(coco.anns)} annotations.\n")

    processed = 0
    t_total0 = time.perf_counter()

    for img_id in coco.getImgIds():
        if processed >= args.limit:
            break

        t_img0 = time.perf_counter()

        img_info, anns = get_image_annotation_info(coco, img_id)
        file_name = img_info.get("file_name", "")
        img_path = _resolve_image_path(Path(args.coco_imgdir), file_name)

        if not img_path.exists():
            # silently skip if we really can't find the file
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        # --- instance masks + union mask ---
        masks = get_binary_masks(coco, anns, H, W)
        if not masks:
            # no usable instances
            continue

        union = np.zeros((H, W), dtype=np.uint8)
        for m in masks:
            union |= m
        union_u8 = (union * 255).astype(np.uint8)

        # --- save original image & union mask ---
        stem = Path(file_name).stem
        dst_img = out_images / f"{stem}.png"
        dst_mask = out_masks / f"{stem}_mask.png"
        cv2.imwrite(str(dst_img), img_bgr)
        cv2.imwrite(str(dst_mask), union_u8)

        # --- run fragmenter on union mask (writes outline/fragmented/panel) ---
        frag.fragment_one(
            image_path=str(dst_img),
            mask_path=str(dst_mask),
            out_dir=str(out_fragments),
            edge_len=-1,  # auto from perimeter
            target_frag_per_100px=args.target_frag_per_100px,
            grid=args.grid,
            gap_factor=args.gap_factor,
            jitter_deg=0,
            thickness=args.thickness,
            noise_mode=args.noise_mode,
            noise_count=args.noise_count,
            sep_pad=args.sep_pad,
            noise_per_cell=1,
            outline_mode="random",      # use rejection-sampling outline
            max_outline_segments=None,
        )

        # --- optional: per-instance outline-only fragments ---
        if args.save_instances:
            for a, m in zip(anns, masks):
                ann_id = a.get("id", "N")
                m_u8 = (m * 255).astype(np.uint8)
                inst_mask_path = out_masks / f"{stem}_ann{ann_id}.png"
                cv2.imwrite(str(inst_mask_path), m_u8)

                frag.fragment_one(
                    image_path=str(dst_img),
                    mask_path=str(inst_mask_path),
                    out_dir=str(out_inst_outlines),
                    edge_len=-1,
                    target_frag_per_100px=args.target_frag_per_100px,
                    grid=args.grid,
                    gap_factor=args.gap_factor,
                    jitter_deg=0,
                    thickness=args.thickness,
                    noise_mode="uniform",
                    noise_count=0,        # no background noise
                    sep_pad=args.sep_pad,
                    noise_per_cell=0,
                    outline_mode="random",
                    max_outline_segments=None,
                )

        processed += 1
        t_img1 = time.perf_counter()
        print(
            f"[{processed}] {file_name} processed in {t_img1 - t_img0:.3f}s "
            f"(instances={len(masks)})"
        )
        
        # --- run SAM on original + fragmented image (optional) ---
        if args.sam_ckpt:
            # fragmented image path 
            frag_img_path = out_fragments / f"{stem}_fragmented.png"
            if frag_img_path.exists():
                t_sam0 = time.perf_counter()
                run_sam_on_pair(
                    orig_img_path=str(dst_img),
                    frag_img_path=str(frag_img_path),
                    gt_mask_path=str(dst_mask),
                    out_dir=str(out_fragments),
                    sam_checkpoint=args.sam_ckpt,
                    model_type=args.sam_model_type,   # change if you use a different SAM
                    device=None,          # auto-select cuda/cpu
                )
                t_sam1 = time.perf_counter()
                print(f"   SAM on {file_name} took {t_sam1 - t_sam0:.3f}s")


    t_total1 = time.perf_counter()
    print(f"\nDone. Processed {processed} images in {t_total1 - t_total0:.3f}s.")
    print(f"Images           -> {out_images}")
    print(f"Masks            -> {out_masks}")
    print(f"Fragment outputs -> {out_fragments}")
    if args.save_instances:
        print(f"Instance outlines -> {out_inst_outlines}")


if __name__ == "__main__":
    main()
