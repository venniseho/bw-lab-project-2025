"""Pair clean and transformed full-target IoU by source ID; no inference needed."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CONDITIONS = ("light", "medium", "heavy", "fragmentation")
COLORS = ("#009E73", "#E69F00", "#D55E00", "#CC79A7")


def read_scores(path):
    groups = {}
    seen = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["condition"], row["source_key"])
            if key in seen:
                raise ValueError(f"Duplicate source-condition row: {key}")
            seen.add(key)
            if row["status"] != "complete" or not row["full_iou"]:
                continue
            score = float(row["full_iou"])
            if math.isfinite(score):
                if not 0 <= score <= 1:
                    raise ValueError("IoU outside [0, 1]")
                groups.setdefault(row["condition"], {})[row["source_key"]] = score
    return groups


def pair_scores(clean, transformed):
    sources = sorted(set(clean) & set(transformed))
    if not sources:
        raise ValueError("No matched sources with finite full-target IoU")
    x = np.array([clean[source] for source in sources])
    y = np.array([transformed[source] for source in sources])
    return sources, x, y


def plot(fence_run, fragmentation_run, output):
    fence = read_scores(fence_run / "predictions/image_results.csv")
    frag = read_scores(fragmentation_run / "predictions/image_results.csv")
    if "clean" not in fence or set(fence) - {"clean", "light", "medium", "heavy"}:
        raise ValueError("Fence run must contain clean/light/medium/heavy conditions")
    if set(frag) != {"fragmentation"}:
        raise ValueError("Fragmentation run must contain the fragmentation condition")
    clean = fence["clean"]
    groups = {**fence, **frag}
    pairs = {condition: pair_scores(clean, groups[condition]) for condition in CONDITIONS}
    output.mkdir(parents=True, exist_ok=False)
    # Retain the exact pairings used for the figures, not just a visual summary.
    with (output / "paired_iou.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_key", "condition", "clean_full_iou", "transformed_full_iou", "delta_iou"])
        writer.writeheader()
        for condition, (sources, x, y) in pairs.items():
            for source, before, after in zip(sources, x, y):
                writer.writerow({"source_key": source, "condition": condition,
                                 "clean_full_iou": before, "transformed_full_iou": after,
                                 "delta_iou": after-before})
    audit = {}
    fig, axis = plt.subplots(figsize=(7.5, 7.5), layout="constrained")
    panels, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained", sharex=True, sharey=True)
    handles = []
    for condition, color, panel in zip(CONDITIONS, COLORS, axes.flat):
        sources, x, y = pairs[condition]
        audit[condition] = {"matched_sources": len(sources),
                            "clean_only_sources": len(set(clean)-set(groups[condition])),
                            "transformed_only_sources": len(set(groups[condition])-set(clean)),
                            "mean_delta_iou": float(np.mean(y-x)),
                            "below_diagonal": int(np.sum(y < x)),
                            "equal": int(np.sum(y == x)),
                            "above_diagonal": int(np.sum(y > x))}
        for ax in (axis, panel):
            ax.scatter(x, y, color=color, s=8, alpha=0.17, linewidths=0, rasterized=True)
        handles.append(Line2D([], [], marker="o", color=color, linestyle="",
                              label=f"{condition.capitalize()} (n={len(sources):,})"))
        panel.set_title(f"{condition.capitalize()} (n={len(sources):,})")
        print(f"{condition}: n={len(sources)}, mean change={np.mean(y-x):+.4f}; "
              f"below/equal/above={np.sum(y<x)}/{np.sum(y==x)}/{np.sum(y>x)}")
    for ax in (axis, *axes.flat):
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, zorder=0)
        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(-0.025, 1.025)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    axis.set_xlabel("Clean-image full-target IoU")
    axis.set_ylabel("Transformed-image full-target IoU")
    axis.legend(handles=handles, loc="upper left", fontsize=9)
    fig.suptitle("SAM3: clean versus transformed target IoU")
    panels.suptitle("SAM3: paired IoU by transformation")
    panels.supxlabel("Clean-image full-target IoU")
    panels.supylabel("Transformed-image full-target IoU")
    note = ("One point per matched source and condition; no jitter. Below diagonal = lower target IoU.\n"
            "Highest-score grid candidate selected independently in each condition; selected object may change.")
    fig.supxlabel("Clean-image full-target IoU\n\n"+note, fontsize=9)
    axis.set_xlabel("")
    panels.text(0.5, -0.015, note, ha="center", fontsize=9)
    for figure, stem in ((fig, "paired_iou"), (panels, "paired_iou_by_condition")):
        for extension in ("png", "svg"):
            path = output / f"{stem}.{extension}"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            print(path.resolve())
        plt.close(figure)
    (output / "pairing_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fence-run", type=Path, required=True)
    parser.add_argument("--fragmentation-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.fence_run, args.fragmentation_run, args.output_dir)
