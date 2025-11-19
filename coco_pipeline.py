"""
coco_pipeline.py
--------------------------------------------------------------
1) Load COCO JSON + images directory
2) Build binary GT mask(s) with pycocotools
3) (Optional) filter by area / object count / categories
4) Save:
     images/<file_name>.png
     masks/<stem>_mask.png
5) Call mask_fragmenter_clean.fragment_one(...) with:
     - contour density knobs (edge_len, gap_factor, jitter)
     - fully RANDOM background noise (noise_count)
     - collision-avoidance enabled (from your current code)
--------------------------------------------------------------
"""

import os, argparse, shutil
from pathlib import Path
import numpy as np
import cv2

from pycocotools.coco import COCO

# local fragmenter utilities
import mask_fragmenter_clean as frag


def save_mask_png(mask_bool, out_path):
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(str(out_path), mask_u8)


def build_gt_mask(coco: COCO, img_info, cat_ids=None, min_obj_frac=0.0, fallback_bbox=True):
    """
    Return a single binary mask by OR-ing selected instances.
    - If cat_ids is None -> use all categories present.
    - min_obj_frac filters tiny instances (fraction of image area).
    - fallback_bbox=True: if annToMask() yields empty (e.g., missing polygons),
      approximate each instance by filling its bbox.
    """
    H, W = img_info["height"], img_info["width"]
    ann_ids = coco.getAnnIds(imgIds=[img_info["id"]], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    if not anns:
        return None

    mask = np.zeros((H, W), dtype=np.uint8)
    img_area = H * W
    kept = 0

    for a in anns:
        m = coco.annToMask(a).astype(np.uint8)
        if m.sum() == 0 and fallback_bbox and "bbox" in a:
            # polygon missing -> approximate by bbox
            x, y, w, h = map(int, a["bbox"])
            x2, y2 = min(x + w, W), min(y + h, H)
            if x < x2 and y < y2:
                m[y:y2, x:x2] = 1

        if m.sum() < (min_obj_frac * img_area):
            continue

        mask |= m
        kept += 1

    return mask if (kept > 0 and mask.any()) else None


def choose_frag_params(perimeter_px, target_frag_per_100px=6, min_edge=10, max_edge=24):
    """
    Heuristic to 'check density around outline':
    Given the rough perimeter, choose edge_len so we place about
    target_frag_per_100px fragments along the boundary.
      fragments ≈ perimeter / edge_len  ->  edge_len ≈ 100 / target_frag_per_100px
    We clamp to [min_edge, max_edge].
    """
    desired_edge = int(round(100.0 / max(1, target_frag_per_100px)))
    edge_len = int(np.clip(desired_edge, min_edge, max_edge))
    # modest gaps so contour is visibly fragmented
    gap_factor = 0.35
    return edge_len, gap_factor


def approx_perimeter(mask_u8):
    cnt = frag.largest_external_contour(mask_u8)
    if cnt is None:
        return 0.0
    arc = cv2.arcLength(cnt, closed=True)
    return float(arc)


def main():
    ap = argparse.ArgumentParser(description="COCO → GT masks → fragmented stimuli")
    ap.add_argument("--coco_ann", required=True, help="path to COCO annotations JSON (instances_*.json)")
    ap.add_argument("--coco_imgdir", required=True, help="directory containing the COCO images")
    ap.add_argument("--out_root", default=".", help="root where images/, masks/, output_clean/ live")
    ap.add_argument("--limit", type=int, default=30, help="max images to process")
    ap.add_argument("--categories", nargs="*", default=None, help="category names to keep (default: all)")

    ap.add_argument("--skip_filters", action="store_true",
                    help="Disable ALL mask size/object-count filtering.")
    ap.add_argument("--min_frac", type=float, default=0.03, help="min total mask area fraction")
    ap.add_argument("--max_frac", type=float, default=0.5,  help="max total mask area fraction")
    ap.add_argument("--min_obj_frac", type=float, default=0.01, help="min single instance area fraction")
    ap.add_argument("--min_objs", type=int, default=1, help="min #objects after filtering")
    ap.add_argument("--max_objs", type=int, default=3, help="max #objects after filtering")

    # contour density + appearance
    ap.add_argument("--target_frag_per_100px", type=float, default=6.0,
                    help="~fragments placed per 100px of contour (higher → denser)")
    ap.add_argument("--jitter", type=int, default=18, help="angular jitter (deg)")
    ap.add_argument("--thickness", type=int, default=1, help="line thickness (px)")

    # fully-random background noise
    ap.add_argument("--noise_count", type=int, default=350, help="N random background segments")
    ap.add_argument("--sep_pad", type=int, default=1, help="min separation (px) between segments")

    args = ap.parse_args()

    # resolve & create output folders
    out_root = Path(args.out_root).expanduser().resolve()
    images_dir = out_root / "images"
    masks_dir  = out_root / "masks"
    out_dir    = out_root / "output_clean"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_path = Path(args.coco_ann).expanduser().resolve()
    imgdir   = Path(args.coco_imgdir).expanduser().resolve()
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations JSON not found: {ann_path}")
    if not imgdir.exists():
        raise FileNotFoundError(f"Image directory not found: {imgdir}")

    coco = COCO(ann_path)

    # map category names to ids if provided
    cat_ids = None
    if args.categories:
        cats = coco.loadCats(coco.getCatIds(catNms=args.categories))
        if not cats:
            print(f"no categories matched: {args.categories}")
            cat_ids = None
        else:
            cat_ids = [c["id"] for c in cats]
            print("using categories:", [c["name"] for c in cats])

    # iterate only images that have at least one annotation (for selected cats or all)
    if cat_ids is None:
        ann_ids = coco.getAnnIds()
    else:
        ann_ids = coco.getAnnIds(catIds=cat_ids)
    if not ann_ids:
        print("No annotations found (for selected categories).")
        print("Tip: run without --categories to include all.")
        print(f"Done. Processed 0 images. Outputs in: {out_dir}")
        return

    anns = coco.loadAnns(ann_ids)
    img_ids = sorted({a["image_id"] for a in anns})

    processed = 0
    for img_id in img_ids:
        if processed >= args.limit:
            break

        info = coco.loadImgs([img_id])[0]
        src_path = imgdir / info["file_name"]
        if not src_path.exists():
            # silently skip missing files
            continue

        # --- try with requested categories (if any), else all ---
        mask = build_gt_mask(
            coco, info,
            cat_ids=cat_ids,
            min_obj_frac=(0.0 if args.skip_filters else args.min_obj_frac),
            fallback_bbox=True
        )

        # If still None and you had category filters, retry with ALL categories
        if mask is None and cat_ids is not None:
            mask = build_gt_mask(
                coco, info,
                cat_ids=None,
                min_obj_frac=(0.0 if args.skip_filters else args.min_obj_frac),
                fallback_bbox=True
            )

        if mask is None:
            # uncomment to debug:
            # print(f"skip {info['file_name']}: no instances even after bbox fallback")
            continue

        mask_bin = (mask > 0).astype(np.uint8) * 255

        # optional global filters
        if not args.skip_filters:
            if not frag.mask_passes_filters(
                mask_bin,
                min_frac=args.min_frac,
                max_frac=args.max_frac,
                min_objs=args.min_objs,
                max_objs=args.max_objs,
                min_obj_frac=args.min_obj_frac
            ):
                continue

        # copy/convert image to PNG in images/
        stem = Path(info["file_name"]).stem
        dst_img = images_dir / f"{stem}.png"
        if str(src_path).lower().endswith(".png"):
            shutil.copy2(src_path, dst_img)
        else:
            img_bgr = cv2.imread(str(src_path))
            if img_bgr is None:
                continue
            cv2.imwrite(str(dst_img), img_bgr)

        # save mask to masks/<stem>_mask.png
        dst_msk = masks_dir / f"{stem}_mask.png"
        save_mask_png(mask > 0, dst_msk)

        # ---------- choose contour density from perimeter ----------
        perim = approx_perimeter(mask_bin)
        edge_len, gap_factor = choose_frag_params(
            perimeter_px=perim,
            target_frag_per_100px=args.target_frag_per_100px
        )

        # ---------- call fragmenter ----------
        frag.fragment_one(
            str(dst_img), str(dst_msk), str(out_dir),
            edge_len=edge_len,
            grid=40,                    # used for gap calc only
            gap_factor=gap_factor,
            jitter_deg=args.jitter,
            noise_per_cell=1,           # ignored in uniform mode
            thickness=args.thickness,
            noise_mode="uniform",       # fully random
            noise_count=int(args.noise_count),
            sep_pad=int(args.sep_pad),
            target_frag_per_100px=args.target_frag_per_100px
        )

        processed += 1

    print(f"Done. Processed {processed} images. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
