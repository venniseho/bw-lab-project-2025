"""
coco_pipeline_revised.py
--------------------------------------------------------------
COCO → per-instance mask → fragmented-contour stimuli

Also (optional):
  - run SAM on original + fragmented with a centroid point prompt
  - write sam_iou.csv
  - write sam_iou_scatter.png
--------------------------------------------------------------
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

import mask_fragmenter_clean as frag
from sam_runner import run_sam_on_pair


def infer_sam_model_type_from_ckpt(ckpt_path: str) -> str:
    """
    Heuristic: infer model type from checkpoint filename.
    Common SAM ckpts:
      sam_vit_b_*.pth -> vit_b
      sam_vit_l_*.pth -> vit_l
      sam_vit_h_*.pth -> vit_h
    """
    name = Path(ckpt_path).name.lower()
    if "vit_b" in name:
        return "vit_b"
    if "vit_l" in name:
        return "vit_l"
    if "vit_h" in name:
        return "vit_h"
    # fallback
    return "vit_h"


def _resolve_image_path(img_dir: Path, file_name: str) -> Path:
    p = img_dir / file_name
    if p.exists():
        return p

    q = img_dir / Path(file_name).name
    if q.exists():
        return q

    parts = Path(file_name).parts
    if len(parts) >= 3 and parts[0] == "COCO" and parts[1].startswith("val"):
        r = img_dir / parts[-1]
        if r.exists():
            return r

    return p


def get_image_annotation_info(coco, img_id):
    img_info = coco.loadImgs([img_id])[0]
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    return img_info, anns


def get_instance_masks(coco, anns, H, W):
    masks = []
    for a in anns:
        m = coco.annToMask(a)
        if m.sum() == 0:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append((a, (m.astype(np.uint8) & 1)))
    return masks


def main():
    ap = argparse.ArgumentParser(description="COCO → fragmented-contour stimuli (revised pipeline)")

    ap.add_argument("--coco_ann", required=True, help="path to instances_*.json")
    ap.add_argument("--coco_imgdir", required=True, help="folder with COCO images (e.g., COCO/val2014 or COCO)")
    ap.add_argument("--out_root", default="outputs/coco_pipeline_revised", help="where to write outputs")
    ap.add_argument("--limit", type=int, default=10, help="max images to process")
    # Instance mask filtering (helps avoid tiny outlines)
    ap.add_argument("--min_mask_area", type=int, default=5000,
                    help="Skip instance masks whose area < this many pixels")
    ap.add_argument("--min_mask_frac", type=float, default=0.01,
                    help="Skip instance masks whose area fraction < this value")

    # Fragmentation params
    ap.add_argument("--target_frag_per_100px", type=float, default=6.0)
    ap.add_argument("--gap_factor", type=float, default=0.35)
    ap.add_argument("--jitter_deg", type=int, default=0)  # default 0: follow mask boundary
    ap.add_argument("--thickness", type=int, default=1)
    ap.add_argument("--noise_mode", choices=["uniform", "grid"], default="uniform")
    ap.add_argument("--noise_count", type=int, default=400)
    ap.add_argument("--sep_pad", type=int, default=1)
    ap.add_argument("--grid", type=int, default=40)

    # SAM params
    ap.add_argument("--sam_ckpt", type=str, default=None,
                    help="Path to SAM checkpoint (.pth). If not provided, SAM is skipped.")
    ap.add_argument("--sam_model_type", default=None,
                    choices=[None, "vit_h", "vit_l", "vit_b"],
                    help="SAM backbone type matching the checkpoint. If omitted, inferred from ckpt filename.")

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_images = out_root / "images"
    out_masks = out_root / "masks"
    out_fragments = out_root
    out_fragment_images = out_root / "fragments"
    out_sam_overlays = out_root / "sam_overlays"
    out_metrics = out_root / "metrics"

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_fragments.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_sam_overlays.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.coco_ann)
    print(f"Loaded {len(coco.imgs)} images and {len(coco.anns)} annotations.\n")

    processed = 0
    t_total0 = time.perf_counter()
    sam_rows = []  # each row: {"stem":..., "iou_orig":..., "iou_frag":...}

    # Decide SAM model_type now (if using SAM)
    sam_model_type = None
    if args.sam_ckpt:
        sam_model_type = args.sam_model_type or infer_sam_model_type_from_ckpt(args.sam_ckpt)
        print(f"SAM enabled | ckpt={args.sam_ckpt} | model_type={sam_model_type}\n")

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

        stem = Path(file_name).stem
        dst_img = out_images / f"{stem}.png"
        cv2.imwrite(str(dst_img), img_bgr)

        kept_instances = 0
        for a, m in masks:
            ann_id = a.get("id", "N")
            inst_area = int(m.sum())
            inst_frac = float(inst_area) / float(H * W)
            if inst_area < args.min_mask_area or inst_frac < args.min_mask_frac:
                continue

            m_u8 = (m * 255).astype(np.uint8)
            inst_mask_path = out_masks / f"{stem}_ann{ann_id}.png"
            cv2.imwrite(str(inst_mask_path), m_u8)

            instance_stem = f"{stem}_ann{ann_id}"
            frag.fragment_one(
                image_path=str(dst_img),
                mask_path=str(inst_mask_path),
                out_dir=str(out_fragments),
                output_stem=instance_stem,
                edge_len=-1,
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
            )

            kept_instances += 1

            if args.sam_ckpt:
                frag_img_path = out_fragment_images / f"{instance_stem}_fragmented.png"
                if frag_img_path.exists():
                    t_sam0 = time.perf_counter()
                    result = run_sam_on_pair(
                        orig_img_path=str(dst_img),
                        frag_img_path=str(frag_img_path),
                        gt_mask_path=str(inst_mask_path),
                        out_dir=str(out_sam_overlays),
                        sam_checkpoint=args.sam_ckpt,
                        model_type=sam_model_type,
                        device=None,
                    )
                    t_sam1 = time.perf_counter()

                    sam_rows.append({
                        "stem": result["stem"],
                        "iou_orig": float(result["iou_orig"]),
                        "iou_frag": float(result["iou_frag"]),
                    })

                    print(
                        f"   SAM took {t_sam1 - t_sam0:.3f}s"
                        f" | IoU(orig)={result['iou_orig']:.3f} IoU(frag)={result['iou_frag']:.3f}"
                    )

        if kept_instances == 0:
            continue

        processed += 1
        t_img1 = time.perf_counter()
        print(
            f"[{processed}] {file_name} processed in {t_img1 - t_img0:.3f}s "
            f"(instances={len(masks)} | kept={kept_instances})"
        )

    t_total1 = time.perf_counter()
    print(f"\nDone. Processed {processed} images in {t_total1 - t_total0:.3f}s.")
    print(f"Images           -> {out_images}")
    print(f"Masks            -> {out_masks}")
    print(f"Fragment outputs -> {out_fragment_images}")

    # Write SAM IoU CSV
    if len(sam_rows) > 0:
        csv_path = out_metrics / "sam_iou.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "iou_orig", "iou_frag"])
            w.writeheader()
            w.writerows(sam_rows)

        print(f"\nSaved SAM IoU CSV -> {csv_path}")
    else:
        print("\nNo SAM results collected (sam_ckpt not set or no fragments found).")


if __name__ == "__main__":
    main()