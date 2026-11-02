"""Plot exploratory score-selected mask accuracy from existing evaluation CSVs."""
import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path):
    groups = defaultdict(lambda: {"full_iou": [], "visible_iou": []})
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "complete":
                continue
            for metric in ("full_iou", "visible_iou"):
                if row.get(metric) and math.isfinite(float(row[metric])):
                    groups[row["condition"]][metric].append(float(row[metric]))
    return groups


def plot(root, output):
    # Fix batch size across arms; this is a scientific diagnostic, not a sweep average.
    paths = {arm: root / arm / "batch-1" / "predictions" / "image_results.csv"
             for arm in ("fence", "fragmentation")}
    data = {arm: read_metrics(path) for arm, path in paths.items()}
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained", sharey=True)
    with (output / "accuracy_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "condition", "metric", "n", "mean_iou"])
        writer.writeheader()
        for axis, (arm, groups) in zip(axes, data.items()):
            order = [c for c in ("clean", "light", "medium", "heavy", "fragmentation") if c in groups]
            order += sorted(set(groups) - set(order))
            for offset, metric, label in ((-0.18, "full_iou", "Full target"),
                                          (0.18, "visible_iou", "Visible target")):
                for index, condition in enumerate(order):
                    values = groups[condition][metric]
                    if not values:
                        continue
                    value = mean(values)
                    axis.bar(index+offset, value, width=0.34,
                             color="tab:blue" if offset < 0 else "tab:orange",
                             label=label if index == 0 else None)
                    writer.writerow({"arm": arm, "condition": condition, "metric": metric,
                                     "n": len(values), "mean_iou": value})
            axis.set_xticks(range(len(order)),
                            [f"{c}\nn={len(groups[c]['full_iou'])}" for c in order])
            axis.set_title(arm.capitalize())
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.2)
            axis.legend()
    axes[0].set_ylabel("Mean measured mask IoU")
    fig.suptitle("Exploratory SAM3 accuracy across stimulus conditions")
    fig.supxlabel("Batch 1; model-score-selected candidate, not label-selected best mask.\n"
                  "Grid prompts; one noncrowd target per source. Single run; no uncertainty intervals.", fontsize=9)
    for extension in ("png", "svg"):
        path = output / f"pilot_accuracy.{extension}"
        fig.savefig(path, dpi=180)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.benchmark_dir, args.output_dir)
