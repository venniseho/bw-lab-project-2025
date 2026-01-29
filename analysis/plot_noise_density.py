"""
plot_noise_density.py
--------------------------------------------------------------
Plots SAM performance vs noise density from a sweep of outputs.

Expected structure (recommended):
  outputs/noise_density_exp/
    noise_050/
      metrics/sam_iou.csv
    noise_100/
      metrics/sam_iou.csv
    ...

Each sam_iou.csv should have columns at least:
  iou_orig, iou_frag, chance_iou

This script:
  - reads all matching CSVs under --root
  - extracts noise density from folder name (noise_XXX or noiseXXX)
  - computes normalised IoU:
        norm = (iou - chance) / (1 - chance)
    (this can be negative; that's okay)
  - plots mean ± SEM vs noise density for orig & frag
  - plots Δnorm = norm_orig - norm_frag vs noise density

Run:
  python3 analysis/noise_density/plot_noise_density.py \
    --root outputs/noise_density_exp \
    --out_dir analysis/noise_density
--------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NOISE_RE = re.compile(r"(?:n?|n)(\d+)", re.IGNORECASE) 


def parse_noise_from_path(p: Path) -> int | None:
    """
    Extract integer noise density from any parent folder name like:
      noise_050, noise-200, noise300
    """
    for part in p.parts[::-1]:
        m = NOISE_RE.search(part)
        if m:
            return int(m.group(1))
    return None


def normalise_iou(iou: np.ndarray, chance: np.ndarray) -> np.ndarray:
    """
    Normalised IoU where 0 = chance, 1 = perfect.
      norm = (iou - chance) / (1 - chance)

    Notes:
      - If chance is 1 (mask covers whole image), denominator is 0; handle safely.
      - Negative values are valid: worse than chance.
    """
    denom = (1.0 - chance)
    denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
    return (iou - chance) / denom


def mean_sem(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(x))
    sem = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return m, sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Root folder containing noise sweep subfolders (noise_XXX/...)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Where to save plots + merged CSV")
    ap.add_argument("--pattern", type=str, default="**/metrics/sam_iou.csv",
                    help="Glob pattern under root to find CSVs")
    ap.add_argument("--save_merged", action="store_true",
                    help="Also write merged per-instance table used for plotting")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(root.glob(args.pattern))
    if not csv_paths:
        raise SystemExit(f"No sam_iou.csv found under {root} with pattern {args.pattern}")

    all_rows = []

    for csv_path in csv_paths:
        noise = parse_noise_from_path(csv_path)
        if noise is None:
            print(f"⚠ Skipping (could not parse noise from path): {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        required = {"iou_orig", "iou_frag", "chance_iou"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{csv_path} missing columns: {sorted(missing)}")

        df = df.copy()
        df["noise_density"] = noise
        df["source_csv"] = str(csv_path)

        # normalised IoU (0 = chance)
        df["norm_iou_orig"] = normalise_iou(df["iou_orig"].to_numpy(), df["chance_iou"].to_numpy())
        df["norm_iou_frag"] = normalise_iou(df["iou_frag"].to_numpy(), df["chance_iou"].to_numpy())
        df["delta_norm"] = df["norm_iou_orig"] - df["norm_iou_frag"]

        all_rows.append(df)

    if not all_rows:
        raise SystemExit("No usable CSVs found (noise parsing failed on all).")

    merged = pd.concat(all_rows, ignore_index=True)

    # Aggregate by noise density
    dens = np.array(sorted(merged["noise_density"].unique()), dtype=int)

    mean_o, sem_o = [], []
    mean_f, sem_f = [], []
    mean_d, sem_d = [], []

    for d in dens:
        sub = merged[merged["noise_density"] == d]
        m, s = mean_sem(sub["norm_iou_orig"].to_numpy())
        mean_o.append(m); sem_o.append(s)
        m, s = mean_sem(sub["norm_iou_frag"].to_numpy())
        mean_f.append(m); sem_f.append(s)
        m, s = mean_sem(sub["delta_norm"].to_numpy())
        mean_d.append(m); sem_d.append(s)

    mean_o = np.array(mean_o); sem_o = np.array(sem_o)
    mean_f = np.array(mean_f); sem_f = np.array(sem_f)
    mean_d = np.array(mean_d); sem_d = np.array(sem_d)

    # ---------------------------
    # Plot 1: norm IoU vs noise
    # ---------------------------
    plt.figure()
    plt.errorbar(dens, mean_o, yerr=sem_o, marker="o", linestyle="-", label="Original (normalised IoU)")
    plt.errorbar(dens, mean_f, yerr=sem_f, marker="o", linestyle="-", label="Fragmented (normalised IoU)")
    plt.axhline(0.0, linestyle="--", linewidth=1)  # chance line
    plt.xlabel("Noise density (e.g., number of noise segments)")
    plt.ylabel("Normalised IoU (0 = chance)")
    plt.title("SAM performance vs noise density")
    plt.legend()
    plt.tight_layout()
    out1 = out_dir / "noise_vs_norm_iou.png"
    plt.savefig(out1)
    plt.close()

    # ---------------------------
    # Plot 2: delta vs noise
    # ---------------------------
    plt.figure()
    plt.errorbar(dens, mean_d, yerr=sem_d, marker="o", linestyle="-")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Noise density (e.g., number of noise segments)")
    plt.ylabel("Δ normalised IoU (orig − frag)")
    plt.title("Fragmentation cost vs noise density")
    plt.tight_layout()
    out2 = out_dir / "noise_vs_delta_norm_iou.png"
    plt.savefig(out2)
    plt.close()

    # Optional merged output
    if args.save_merged:
        merged_path = out_dir / "noise_sweep_merged.csv"
        merged.to_csv(merged_path, index=False)
        print(f"Saved merged table -> {merged_path}")

    print(f"Saved plot -> {out1}")
    print(f"Saved plot -> {out2}")


if __name__ == "__main__":
    main()
