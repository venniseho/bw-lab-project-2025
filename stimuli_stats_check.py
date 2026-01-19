"""
stimuli_stats_check.py
Validate "no local cues" by comparing OUTLINE vs NOISE dash stats.

Uses standard p-value tests:
  - Chi-square test on binned histograms

Inputs:
  metrics JSONs produced by mask_fragmenter_clean.py, which must include:
    metrics["hists"]["length" / "orientation_deg" / "nn_distance"]

Outputs:
  - prints per-metric p-values (pooled across all images)
  - saves a JSON report with per-image + pooled results
  - saves overlay plots of pooled histograms

Run:
  python stimuli_stats_check.py \
    --metrics_dir outputs/tests/manual_coco_check/metrics \
    --out_dir outputs/tests/manual_coco_check/metrics/stats_check


"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, ttest_ind


# ---------------------------
# Helpers
# ---------------------------

def safe_np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)

def chi_square_pvalue(out_counts: np.ndarray, noise_counts: np.ndarray) -> float:
    """
    Chi-square test of homogeneity on two binned distributions.

    We do a 2 x K contingency table:
      [outline_counts]
      [noise_counts]

    Notes:
      - If many bins are 0 for both, chi2_contingency can be unstable.
      - We drop bins where (outline + noise) == 0.
      - We also add a tiny epsilon to avoid division issues in degenerate cases.
    """
    out_counts = safe_np(out_counts)
    noise_counts = safe_np(noise_counts)

    total = out_counts + noise_counts
    keep = total > 0
    out_counts = out_counts[keep]
    noise_counts = noise_counts[keep]

    # If too few bins remain, return NaN (not enough info)
    if out_counts.size < 2:
        return float("nan")

    table = np.vstack([out_counts, noise_counts])
    # chi2_contingency returns (chi2, p, dof, expected)
    _, p, _, _ = chi2_contingency(table, correction=False)
    return float(p)

def pooled_counts(metrics_list: List[dict], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool histogram counts across all images for a given hist key.
    Returns: (edges, pooled_outline_counts, pooled_noise_counts)
    """
    edges_ref = None
    out_sum = None
    noise_sum = None

    for m in metrics_list:
        h = m.get("hists", {}).get(key, None)
        if h is None:
            continue

        edges = np.asarray(h["edges"], dtype=np.float64)
        out_c = np.asarray(h["outline_counts"], dtype=np.float64)
        no_c  = np.asarray(h["noise_counts"], dtype=np.float64)

        if edges_ref is None:
            edges_ref = edges
            out_sum = out_c.copy()
            noise_sum = no_c.copy()
        else:
            # Require identical bin edges; otherwise results are not comparable.
            if len(edges) != len(edges_ref) or not np.allclose(edges, edges_ref):
                raise ValueError(
                    f"Bin edges mismatch for hist '{key}'. "
                    f"Make sure all metrics were generated with consistent bin settings."
                )
            out_sum += out_c
            noise_sum += no_c

    if edges_ref is None:
        raise ValueError(f"No histograms found for key '{key}'. Did you patch metrics to include 'hists'?")

    return edges_ref, out_sum, noise_sum

def plot_overlay_hist(edges: np.ndarray, out_counts: np.ndarray, noise_counts: np.ndarray, title: str, out_path: Path):
    """
    Overlay plot (normalized) for outline vs noise.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to probabilities
    out_p = out_counts / max(out_counts.sum(), 1.0)
    no_p  = noise_counts / max(noise_counts.sum(), 1.0)

    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1:] - edges[:-1])

    plt.figure()
    plt.bar(centers, out_p, width=width, alpha=0.6, label="outline")
    plt.bar(centers, no_p,  width=width, alpha=0.6, label="noise")
    plt.title(title)
    plt.xlabel("bin")
    plt.ylabel("probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def approx_expand_from_hist(edges: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Expand a histogram into sample-like data by repeating bin centers.
    This is only used for a familiar Welch t-test sanity check.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    reps = counts.astype(int)
    # cap to avoid massive arrays
    cap = 200000
    if reps.sum() > cap:
        scale = cap / reps.sum()
        reps = np.maximum(1, (reps * scale).astype(int))
    return np.repeat(centers, reps)


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", type=str, required=True, help="Folder containing *_metrics.json files")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write report + plots")
    ap.add_argument("--glob", type=str, default="*_metrics.json", help="Glob pattern for metrics JSONs")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for interpretation")
    ap.add_argument("--also_ttest", action="store_true", help="Also run Welch t-tests (sanity check)")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(metrics_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No metrics JSONs found in {metrics_dir} matching {args.glob}")

    metrics_list = []
    per_image = []

    # Load all metrics
    for fp in files:
        with open(fp, "r") as f:
            m = json.load(f)
        metrics_list.append(m)

    # Per-image p-values (useful to see outliers)
    for fp, m in zip(files, metrics_list):
        h = m.get("hists", {})
        if not h:
            continue

        rec = {"file": fp.name}

        for key in ["length", "orientation_deg", "nn_distance"]:
            hk = h.get(key, None)
            if hk is None:
                rec[f"{key}_p"] = None
                continue
            p = chi_square_pvalue(hk["outline_counts"], hk["noise_counts"])
            rec[f"{key}_p"] = p

        per_image.append(rec)

    # Pooled p-values (strongest single statement)
    pooled_results: Dict[str, dict] = {}

    for key, title in [
        ("length", "Dash length distribution (pooled)"),
        ("orientation_deg", "Orientation distribution (pooled)"),
        ("nn_distance", "Nearest-neighbor spacing distribution (pooled)"),
    ]:
        edges, out_counts, noise_counts = pooled_counts(metrics_list, key=key)
        p = chi_square_pvalue(out_counts, noise_counts)

        pooled_results[key] = {
            "chi2_pvalue": p,
            "outline_total": float(out_counts.sum()),
            "noise_total": float(noise_counts.sum()),
        }

        plot_overlay_hist(
            edges=edges,
            out_counts=out_counts,
            noise_counts=noise_counts,
            title=f"{title}\nchi-square p={p:.4g}",
            out_path=out_dir / f"pooled_{key}_overlay.png",
        )

        # Optional Welch t-test sanity check on approximate expanded samples
        if args.also_ttest:
            x = approx_expand_from_hist(edges, out_counts)
            y = approx_expand_from_hist(edges, noise_counts)
            if len(x) > 2 and len(y) > 2:
                t_p = float(ttest_ind(x, y, equal_var=False).pvalue)
            else:
                t_p = float("nan")
            pooled_results[key]["welch_t_pvalue"] = t_p

    # Summary interpretation
    summary = {}
    for key in pooled_results:
        p = pooled_results[key]["chi2_pvalue"]
        summary[key] = {
            "passes_alpha": (p >= args.alpha) if np.isfinite(p) else False,
            "alpha": args.alpha,
        }

    report = {
        "metrics_dir": str(metrics_dir),
        "n_files": len(files),
        "alpha": args.alpha,
        "pooled": pooled_results,
        "per_image": per_image,
        "summary": summary,
        "notes": {
            "test": "Chi-square test of homogeneity on binned histograms (outline vs noise).",
            "goal": "Fail to reject difference (p >= alpha) => no detectable local cue in that statistic.",
        },
    }

    out_json = out_dir / "stimuli_stats_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    # Print quick console summary
    print("\nPOOLED CHI-SQUARE RESULTS (outline vs noise)")
    for key in ["length", "orientation_deg", "nn_distance"]:
        p = pooled_results[key]["chi2_pvalue"]
        ok = summary[key]["passes_alpha"]
        print(f"{key:16s} p={p:.4g}  -> {'OK (no diff detected)' if ok else 'FLAG (diff detected)'}")

    print(f"\nSaved report: {out_json}")
    print(f"Saved plots:  {out_dir}")

if __name__ == "__main__":
    main()
