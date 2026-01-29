"""
run_sam_batch.py
--------------------------------------------------------------
STAGE 2: Run SAM on a batch of already-generated stimuli.

Designed to pair with STAGE 1 (make_stimuli.py) which writes:
  outputs/<exp>/
    assets/images/<img_stem>.png
    assets/masks/<instance_id>_mask.png
    stimuli/fragments/<instance_id>_fragmented.png
    indexes/manifest.jsonl   (recommended input to this script)
    debug/...                (stage 1 debug + where stage 2 can write overlays)

This script:
  - Reads a JSONL manifest (preferred) OR tries to derive triples by globbing.
  - Runs SAM on (original, fragmented) using centroid point prompt from GT mask.
  - Writes overlays + panels into:
      out_root/debug/sam_overlays/
  - Writes CSV metrics into:
      out_root/metrics/sam_iou.csv

EVAL VALIDITY:
  - GT is used ONLY to compute the centroid point prompt.
  - We do NOT use GT to select among SAM's multimask outputs.

Usage:
  python3 run_sam_batch.py \
    --out_root outputs/tests/manual_coco_check \
    --manifest outputs/tests/manual_coco_check/indexes/manifest.jsonl \
    --sam_ckpt checkpoints/sam_vit_h_4b8939.pth

If you omit --manifest, it will try to glob:
  stimuli/fragments/*_fragmented.png
and infer matching assets/images + assets/masks.

--------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sam_runner import run_sam_on_pair


def infer_sam_model_type_from_ckpt(ckpt_path: str) -> str:
    name = Path(ckpt_path).name.lower()
    if "vit_b" in name:
        return "vit_b"
    if "vit_l" in name:
        return "vit_l"
    if "vit_h" in name:
        return "vit_h"
    return "vit_h"


# ---------------------------
# Manifest reading
# ---------------------------

def _read_manifest_jsonl(manifest_path: Path) -> List[Dict[str, str]]:
    """
    Manifest format (JSONL, one dict per line), recommended from make_stimuli.py:

      {"stem": "<instance_id>",
       "orig_img_path": "...",
       "frag_img_path": "...",
       "gt_mask_path": "..."}

    Extra keys are allowed and ignored by this runner.
    """
    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _get_col(row: Dict[str, str], *names: str) -> Optional[str]:
    """
    Case-insensitive key lookup with fallbacks.
    """
    lower = {k.lower(): k for k in row.keys()}
    for n in names:
        k = lower.get(n.lower())
        if k is not None:
            v = row.get(k)
            if v:
                return str(v).strip()
    return None


def _resolve_path(p: str) -> Path:
    """
    The manifest produced by make_stimuli.py typically writes project-relative paths
    like "outputs/.../assets/images/...". We interpret these as-is.

    If you later change stage 1 to write absolute paths, this still works.
    """
    return Path(p).expanduser()


# ---------------------------
# Globbing fallback (no manifest)
# ---------------------------

def _glob_mode(out_root: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Derive triples (orig_img, frag_img, mask) without a manifest.

    Assumes STAGE 1 layout:
      - fragments: out_root/stimuli/fragments/<instance_id>_fragmented.png
      - masks:     out_root/assets/masks/<instance_id>_mask.png
      - images:    out_root/assets/images/<img_stem>.png
                  where img_stem is instance_id up to first "_ann"

    We do:
      stem := fragment filename with "_fragmented" removed
      mask := assets/masks/{stem}_mask.png
      orig := assets/images/{img_stem}.png   (img_stem = stem.split("_ann")[0])
    """
    frags_dir = out_root / "stimuli" / "fragments"
    masks_dir = out_root / "assets" / "masks"
    images_dir = out_root / "assets" / "images"

    triples: List[Tuple[Path, Path, Path]] = []

    if not frags_dir.exists():
        return triples

    for frag_path in sorted(frags_dir.glob("*_fragmented.png")):
        stem = frag_path.stem.replace("_fragmented", "")

        mask_path = masks_dir / f"{stem}_mask.png"
        if not mask_path.exists():
            # fallback: any mask that starts with stem
            cand = sorted(masks_dir.glob(f"{stem}*_mask.png"))
            if cand:
                mask_path = cand[0]
        if not mask_path.exists():
            continue

        img_stem = stem.split("_ann")[0]
        orig_path = images_dir / f"{img_stem}.png"
        if not orig_path.exists():
            # fallback: any extension
            cand2 = sorted(images_dir.glob(f"{img_stem}.*"))
            if cand2:
                orig_path = cand2[0]
        if not orig_path.exists():
            continue

        triples.append((orig_path, frag_path, mask_path))

    return triples


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="STAGE 2: Run SAM on stimuli batch")

    ap.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Stage1 output root (contains assets/, stimuli/, indexes/...)",
    )
    ap.add_argument("--sam_ckpt", type=str, required=True, help="Path to SAM checkpoint (.pth)")
    ap.add_argument("--sam_model_type", type=str, default=None, choices=[None, "vit_h", "vit_l", "vit_b"])
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    ap.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional JSONL manifest from stage1 (recommended): out_root/indexes/manifest.jsonl",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional limit #instances")
    args = ap.parse_args()

    out_root = Path(args.out_root)

    # Write stage 2 outputs into debug/ (keeps stimuli folder clean)
    debug_dir = out_root / "debug"
    overlays_dir = debug_dir / "sam_overlays"
    metrics_dir = out_root / "metrics"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_type = args.sam_model_type or infer_sam_model_type_from_ckpt(args.sam_ckpt)
    print(f"SAM ckpt={args.sam_ckpt} | model_type={model_type} | device={args.device or 'auto'}")

    # Build worklist
    work: List[Tuple[Path, Path, Path]] = []

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise SystemExit(f"Manifest not found: {manifest_path}")

        rows = _read_manifest_jsonl(manifest_path)

        for r in rows:
            orig_s = _get_col(r, "orig_img_path", "orig")
            frag_s = _get_col(r, "frag_img_path", "frag")
            mask_s = _get_col(r, "gt_mask_path", "mask")

            if not (orig_s and frag_s and mask_s):
                continue

            orig_p = _resolve_path(orig_s)
            frag_p = _resolve_path(frag_s)
            mask_p = _resolve_path(mask_s)

            if orig_p.exists() and frag_p.exists() and mask_p.exists():
                work.append((orig_p, frag_p, mask_p))
    else:
        work = _glob_mode(out_root)

    if args.limit is not None:
        work = work[: max(0, int(args.limit))]

    if not work:
        raise SystemExit(
            "No (orig, frag, mask) triples found. Provide --manifest or check out_root structure."
        )

    print(f"Found {len(work)} instances to run.")

    # Run SAM
    rows_out: List[Dict[str, object]] = []

    for i, (orig_p, frag_p, mask_p) in enumerate(work, start=1):
        stem = frag_p.stem.replace("_fragmented", "")
        outline_p = out_root / "debug" / "outlines" / f"{stem}_outline.png"

        res = run_sam_on_pair(
            orig_img_path=str(orig_p),
            frag_img_path=str(frag_p),
            gt_mask_path=str(mask_p),
            out_dir=str(overlays_dir),
            sam_checkpoint=args.sam_ckpt,
            model_type=model_type,
            device=args.device,
            outline_img_path=str(outline_p) if outline_p.exists() else None,
        )

        rows_out.append(
            {
                "stem": res["stem"],
                "orig_img": str(orig_p),
                "frag_img": str(frag_p),
                "gt_mask": str(mask_p),
                "iou_orig": res["iou_orig"],
                "iou_frag": res["iou_frag"],
                "chance_iou": res["chance_iou"],
                "niou_orig": res.get("niou_orig", None),
                "niou_frag": res.get("niou_frag", None),
                "score_orig": res["score_orig"],
                "score_frag": res["score_frag"],
            }
        )

        if i % 25 == 0 or i == len(work):
            print(f"[{i}/{len(work)}] done")

    # Save CSV
    csv_path = metrics_dir / "sam_iou.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "orig_img",
                "frag_img",
                "gt_mask",
                "iou_orig",
                "iou_frag",
                "chance_iou",
                "niou_orig",
                "niou_frag",
                "score_orig",
                "score_frag",
            ],
        )
        w.writeheader()
        w.writerows(rows_out)

    print("\nStage2 complete.")
    print(f"Overlays/panels -> {overlays_dir}")
    print(f"SAM IoU CSV     -> {csv_path}")


if __name__ == "__main__":
    main()
