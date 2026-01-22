"""
run_sam_batch.py
--------------------------------------------------------------
Stage 2: Run SAM on a batch of already-generated stimuli.

Inputs (typical out_root structure):
  outputs/<exp>/
    images/      (original images saved by stage 1)
    masks/       (instance masks: *_mask.png)
    fragments/   (stimuli: *_fragmented.png)
    analysis/    (this script writes here)

Writes:
  outputs/<exp>/analysis/sam_overlays/
    *_overlay_orig.png
    *_overlay_frag.png
    *_panel_3x2.png

  outputs/<exp>/analysis/metrics/sam_iou.csv

Notes:
  - GT is used ONLY to place the centroid prompt point.
  - We do NOT use GT to choose among SAM multimask outputs.
"""

from __future__ import annotations

import argparse
import csv
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


def _read_manifest_csv(manifest_path: Path) -> List[Dict[str, str]]:
    """
    Expected columns (flexible, case-insensitive):
      - orig_img_path  (or orig)
      - frag_img_path  (or frag)
      - gt_mask_path   (or mask)
    Any extra columns ignored.
    """
    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


def _get_col(row: Dict[str, str], *names: str) -> Optional[str]:
    lower = {k.lower(): k for k in row.keys()}
    for n in names:
        k = lower.get(n.lower())
        if k is not None:
            v = row.get(k)
            if v:
                return v
    return None


def _resolve_under_root(out_root: Path, p: str) -> Path:
    """
    If path is absolute, keep it. If relative, interpret relative to out_root.
    """
    q = Path(p)
    if q.is_absolute():
        return q
    return out_root / q


def _glob_mode(out_root: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Derive triples (orig_img, frag_img, mask) without a manifest.

    Assumes stage1 naming:
      - fragments: <stem>_fragmented.png
      - masks:     <stem>_mask.png OR <stem>_ann<ID>_mask.png (same stem as fragment without suffix)
      - images:    <image_stem>.png (stage1 saved originals as PNG)

    We use:
      stem := fragment filename with "_fragmented" removed
      mask := masks/{stem}_mask.png  (or best match)
      orig := images/{image_stem}.png where image_stem is stem up to first "_ann"
    """
    frags_dir = out_root / "fragments"
    masks_dir = out_root / "masks"
    images_dir = out_root / "images"

    triples: List[Tuple[Path, Path, Path]] = []

    for frag_path in sorted(frags_dir.glob("*_fragmented.png")):
        stem = frag_path.stem.replace("_fragmented", "")

        # mask path: direct hit first
        mask_path = masks_dir / f"{stem}_mask.png"
        if not mask_path.exists():
            # fallback: any mask that starts with stem
            cand = sorted(masks_dir.glob(f"{stem}*_mask.png"))
            if cand:
                mask_path = cand[0]

        if not mask_path.exists():
            continue

        # original image stem: strip instance suffix like "_ann123"
        img_stem = stem.split("_ann")[0]
        orig_path = images_dir / f"{img_stem}.png"
        if not orig_path.exists():
            # fallback: try jpg
            cand2 = sorted(images_dir.glob(f"{img_stem}.*"))
            if cand2:
                orig_path = cand2[0]

        if not orig_path.exists():
            continue

        triples.append((orig_path, frag_path, mask_path))

    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True, help="Stage1 output root (contains images/, masks/, fragments/)")
    ap.add_argument("--sam_ckpt", type=str, required=True, help="Path to SAM checkpoint (.pth)")
    ap.add_argument("--sam_model_type", type=str, default=None, choices=[None, "vit_h", "vit_l", "vit_b"])
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--manifest", type=str, default=None, help="Optional CSV manifest from stage1")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit #instances")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    analysis_dir = out_root / "analysis"
    overlays_dir = analysis_dir / "sam_overlays"
    metrics_dir = analysis_dir / "metrics"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_type = args.sam_model_type or infer_sam_model_type_from_ckpt(args.sam_ckpt)

    # Build worklist
    work: List[Tuple[Path, Path, Path]] = []

    if args.manifest:
        manifest_path = Path(args.manifest)
        rows = _read_manifest_csv(manifest_path)

        for r in rows:
            orig_s = _get_col(r, "orig_img_path", "orig")
            frag_s = _get_col(r, "frag_img_path", "frag")
            mask_s = _get_col(r, "gt_mask_path", "mask")

            if not (orig_s and frag_s and mask_s):
                continue

            orig_p = _resolve_under_root(out_root, orig_s)
            frag_p = _resolve_under_root(out_root, frag_s)
            mask_p = _resolve_under_root(out_root, mask_s)

            if orig_p.exists() and frag_p.exists() and mask_p.exists():
                work.append((orig_p, frag_p, mask_p))
    else:
        work = _glob_mode(out_root)

    if args.limit is not None:
        work = work[: max(0, int(args.limit))]

    if not work:
        raise SystemExit("No (orig, frag, mask) triples found. Provide --manifest or check out_root structure.")

    # Run SAM
    rows_out: List[Dict[str, object]] = []
    for i, (orig_p, frag_p, mask_p) in enumerate(work, start=1):
        res = run_sam_on_pair(
            orig_img_path=str(orig_p),
            frag_img_path=str(frag_p),
            gt_mask_path=str(mask_p),
            out_dir=str(overlays_dir),
            sam_checkpoint=args.sam_ckpt,
            model_type=model_type,
            device=args.device,
        )

        rows_out.append({
            "stem": res["stem"],
            "orig_img": str(orig_p),
            "frag_img": str(frag_p),
            "gt_mask": str(mask_p),
            "iou_orig": res["iou_orig"],
            "iou_frag": res["iou_frag"],
            "chance_iou": res["chance_iou"],
            "score_orig": res["score_orig"],
            "score_frag": res["score_frag"],
        })

        if i % 25 == 0:
            print(f"[{i}/{len(work)}] done")

    # Save CSV
    csv_path = metrics_dir / "sam_iou.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stem", "orig_img", "frag_img", "gt_mask",
                "iou_orig", "iou_frag", "chance_iou", "score_orig", "score_frag"
            ],
        )
        w.writeheader()
        w.writerows(rows_out)

    print("\nStage2 complete.")
    print(f"Overlays/panels -> {overlays_dir}")
    print(f"SAM IoU CSV     -> {csv_path}")


if __name__ == "__main__":
    main()
