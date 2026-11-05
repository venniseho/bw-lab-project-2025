"""Scatter measured target IoU by condition from saved SAM3 evaluation CSVs."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CONDITIONS = ("clean", "light", "medium", "heavy", "fragmentation")
COLORS = ("#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7")


def read_points(run, metric):
    groups = {}
    with (run / "predictions/image_results.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "complete" or not row.get(metric):
                continue
            value = float(row[metric])
            if math.isfinite(value):
                if not 0 <= value <= 1:
                    raise ValueError("IoU must be in [0, 1]")
                groups.setdefault(row["condition"], []).append(value)
    return groups


def plot(fence_run, fragmentation_run, output, metric="full_iou"):
    groups = read_points(fence_run, metric)
    fragmentation = read_points(fragmentation_run, metric)
    if set(groups) - set(CONDITIONS[:4]) or set(fragmentation) - {"fragmentation"}:
        raise ValueError("Unexpected conditions; check the two run directory arguments")
    groups.update(fragmentation)
    if not all(groups.get(condition) for condition in CONDITIONS):
        raise ValueError("Need finite complete IoU results for all five conditions")
    output.mkdir(parents=True, exist_ok=False)
    fig, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    rng = np.random.default_rng(42)
    handles = []
    for index, (condition, color) in enumerate(zip(CONDITIONS, COLORS)):
        values = groups[condition]
        x = index + rng.uniform(-0.23, 0.23, len(values))
        axis.scatter(
            x, values, color=color, s=7, alpha=0.14, linewidths=0, rasterized=True, zorder=2
        )
        average = float(np.mean(values))
        axis.scatter(index, average, color="black", marker="D", s=42, zorder=4)
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=color,
                label=f"{condition.capitalize()} (n={len(values):,})",
            )
        )
        print(f"{condition}: n={len(values)}, mean {metric}={average:.4f}")
    handles.append(Line2D([], [], marker="D", linestyle="", color="black", label="Condition mean"))
    axis.legend(handles=handles, loc="upper right", fontsize=9)
    axis.set_xticks(range(len(CONDITIONS)), [condition.capitalize() for condition in CONDITIONS])
    axis.set_ylim(-0.025, 1.025)
    axis.set_ylabel("Full-target mask IoU" if metric == "full_iou" else "Visible-target mask IoU")
    axis.set_xlabel("Stimulus condition")
    axis.set_title("SAM3 segmentation accuracy across stimulus conditions")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    fig.supxlabel(
        "Each dot: one source–condition result. Horizontal jitter only separates overlapping dots.\n"
        "Highest model-score grid candidate; one noncrowd target per source. Complete finite results only.",
        fontsize=9,
    )
    for extension in ("png", "svg"):
        path = output / f"{metric}_scatter.{extension}"
        fig.savefig(path, dpi=200)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fence-run", type=Path, required=True)
    parser.add_argument("--fragmentation-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=("full_iou", "visible_iou"), default="full_iou")
    args = parser.parse_args()
    plot(args.fence_run, args.fragmentation_run, args.output_dir, args.metric)
