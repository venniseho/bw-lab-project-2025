"""
Noise density sweep experiment (Stage1 -> Stage2 -> aggregation)

Runs:
  - make_stimuli.py for each noise_count into outputs/noise_sweep/n<COUNT>/
  - run_sam_batch.py on each folder
  - aggregates sam_iou.csv into one summary CSV
  - plots mean IoU (orig/frag) + mean normalised IoU (orig/frag) vs noise_count

Assumes:
  - make_stimuli.py exists at repo root
  - run_sam_batch.py exists at repo root
  - Stage1 writes manifest to: out_root/indexes/manifest.jsonl
  - Stage2 writes CSV to: out_root/metrics/sam_iou.csv

Usage example:
  python3 analysis/noise_density_experiment.py \
    --coco_ann COCO/annotations/instances_val2014.json \
    --coco_imgdir COCO/val2014 \
    --sam_ckpt checkpoints/sam_vit_h_4b8939.pth \
    --limit 5 \
    --noise_counts 50 100 200 400 800
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def normalise_iou(iou: pd.Series, chance: pd.Series) -> pd.Series:
    # (IoU - chance) / (1 - chance)
    denom = (1.0 - chance).replace(0, pd.NA)
    return (iou - chance) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--coco_imgdir", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--noise_counts", type=int, nargs="+", required=True)
    ap.add_argument("--out_root", default="outputs/noise_sweep", help="base output folder")
    ap.add_argument("--device", default=None, help="cuda or cpu (optional)")
    args = ap.parse_args()

    base_out = Path(args.out_root)
    base_out.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for nc in args.noise_counts:
        exp_dir = base_out / f"n{nc}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        manifest = exp_dir / "indexes" / "manifest.jsonl"
        sam_csv = exp_dir / "metrics" / "sam_iou.csv"

        # -------------------
        # Stage 1: stimuli
        # -------------------
        run_cmd([
            "python3", "make_stimuli.py",
            "--coco_ann", args.coco_ann,
            "--coco_imgdir", args.coco_imgdir,
            "--out_root", str(exp_dir),
            "--limit", str(args.limit),
            "--noise_count", str(nc),
        ])

        if not manifest.exists():
            raise SystemExit(f"Missing manifest after Stage1: {manifest}")

        # -------------------
        # Stage 2: SAM batch
        # -------------------
        cmd2 = [
            "python3", "run_sam_batch.py",
            "--out_root", str(exp_dir),
            "--manifest", str(manifest),
            "--sam_ckpt", args.sam_ckpt,
        ]
        if args.device:
            cmd2 += ["--device", args.device]
        run_cmd(cmd2)

        if not sam_csv.exists():
            raise SystemExit(f"Missing SAM CSV after Stage2: {sam_csv}")

        # -------------------
        # Aggregate this condition
        # -------------------
        df = pd.read_csv(sam_csv)

        # Add normalised IoUs
        df["norm_iou_orig"] = normalise_iou(df["iou_orig"], df["chance_iou"])
        df["norm_iou_frag"] = normalise_iou(df["iou_frag"], df["chance_iou"])

        row = {
            "noise_count": nc,
            "n_instances": len(df),

            "mean_iou_orig": df["iou_orig"].mean(),
            "mean_iou_frag": df["iou_frag"].mean(),

            "mean_norm_iou_orig": df["norm_iou_orig"].mean(),
            "mean_norm_iou_frag": df["norm_iou_frag"].mean(),
        }
        summary_rows.append(row)

        print(f"\n[n={nc}] mean_iou_orig={row['mean_iou_orig']:.4f} mean_iou_frag={row['mean_iou_frag']:.4f} "
              f"mean_norm_iou_orig={row['mean_norm_iou_orig']:.4f} mean_norm_iou_frag={row['mean_norm_iou_frag']:.4f}")

    # -------------------
    # Save + plot
    # -------------------
    summary = pd.DataFrame(summary_rows).sort_values("noise_count")
    out_csv = base_out / "noise_sweep_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary -> {out_csv}")

    # Plot: unnormalised + normalised (two panels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: unnormalised
    axes[0].plot(summary["noise_count"], summary["mean_iou_orig"], marker="o", label="orig")
    axes[0].plot(summary["noise_count"], summary["mean_iou_frag"], marker="o", label="frag")
    axes[0].set_title("Mean IoU vs noise_count (unnormalised)")
    axes[0].set_xlabel("noise_count")
    axes[0].set_ylabel("mean IoU")
    axes[0].legend()

    # Right: normalised
    axes[1].plot(summary["noise_count"], summary["mean_norm_iou_orig"], marker="o", label="orig")
    axes[1].plot(summary["noise_count"], summary["mean_norm_iou_frag"], marker="o", label="frag")
    axes[1].set_title("Mean normalised IoU vs noise_count (0 = chance)")
    axes[1].set_xlabel("noise_count")
    axes[1].set_ylabel("mean normalised IoU")
    axes[1].axhline(0.0, linestyle="--")
    axes[1].legend()

    fig.tight_layout()
    out_png = base_out / "noise_sweep_plot.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved plot -> {out_png}")


if __name__ == "__main__":
    main()
