"""Plot clean training and retained clean IoU from CPU-readable summary CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, root, name):
    for extension in ("png", "svg"):
        fig.savefig(root / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot(root):
    training = rows(root / "training_summary.csv")
    evaluations = rows(root / "evaluation_summary.csv")
    baseline = float(training[0]["mean_full_iou"])
    if int(training[0]["step"]) != 0 or len(evaluations) != 7:
        raise ValueError("Expected baseline, clean curve and seven saved-model evaluations")
    if any(int(row["instances"]) != 4952 for row in training + evaluations):
        raise ValueError("Validation cohorts must have 4,952 targets")
    if any(abs(float(row["pretrained_iou"]) - baseline) > 1e-7 for row in evaluations):
        raise ValueError("Pretrained baselines differ")
    steps = [int(row["step"]) for row in training]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for axis, metric, title in zip(
        axes, ("mean_full_iou", "mean_full_loss"), ("Clean validation IoU", "Clean validation loss")
    ):
        values = [float(row[metric]) for row in training]
        axis.plot(steps, values, color="#007c79", marker="o", markersize=4, lw=2)
        axis.axhline(values[0], color="#666666", linestyle="--", lw=1, label="Pretrained baseline")
        axis.set(xlabel="Training updates (one clean image per update)", ylabel=title)
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(alpha=0.2)
        axis.legend()
        if metric.endswith("iou"):
            axis.set_ylim(0.60, 0.76)
            axis.annotate(
                f"Final: {values[-1]:.4f}",
                (steps[-1], values[-1]),
                xytext=(-5, 12),
                textcoords="offset points",
                ha="right",
            )
        else:
            axis.set_ylim(0.35, 0.56)
    fig.suptitle(
        "SAM3: clean-image mask-decoder fine-tuning\n58,633 training images; 4,952 validation targets"
    )
    save(fig, root, "clean_training_learning_curves")

    labels = ["Original pretrained SAM3", "Clean-trained decoder"]
    values = [baseline, float(training[-1]["mean_full_iou"])]
    colors = ["#777777", "#007c79"]
    for row in evaluations:
        label = (
            row["condition"]
            .replace("-visible-loss", " (visible loss)")
            .replace("fence-", "Fence: ")
            .replace("fragmentation", "Fragmentation")
        )
        labels.append(label)
        values.append(float(row["finetuned_iou"]))
        colors.append(
            "#784899"
            if row["condition"] == "fragmentation"
            else "#316ca0"
            if "visible-loss" in row["condition"]
            else "#b65020"
        )
    fig, axis = plt.subplots(figsize=(12, 6.5), layout="constrained")
    bars = axis.barh(labels, values, color=colors)
    axis.bar_label(bars, fmt="%.4f", padding=5)
    axis.axvline(
        baseline, color="#555555", linestyle="--", lw=1, label=f"Pretrained IoU: {baseline:.4f}"
    )
    axis.invert_yaxis()
    axis.set(xlim=(0, 0.83), xlabel="Mean full-target IoU on original COCO validation images")
    axis.set_axisbelow(True)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right")
    axis.set_title(
        "Clean-image performance after condition-specific fine-tuning\nSame 4,952 targets and clean-image point prompts; final checkpoints"
    )
    save(fig, root, "clean_performance_after_finetuning")
    (root / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>SAM3 clean-image results</title>"
        "<style>body{font:16px Arial;max-width:1250px;margin:32px auto;padding:16px}img{max-width:100%}</style>"
        "<h1>SAM3: clean-image training and retention</h1>"
        "<p>Figures reconstructed from the user's pasted terminal CSV output. Raw run files and contact sheets have not been downloaded here. These are validation results, not independent test results. Full and visible IoU coincide on clean inputs.</p>"
        "<h2>Clean-only training</h2><p>Both learning-curve y-axes are zoomed; they do not start at zero.</p>"
        "<img src='clean_training_learning_curves.png'><a href='clean_training_learning_curves.svg'>SVG</a>"
        "<h2>Clean performance after fine-tuning</h2><p>Bar-chart x-axis starts at zero. Image and prompt encoders were frozen; only the mask decoder was trained. Training cohorts differ slightly across conditions due to fence visibility exclusions.</p>"
        "<img src='clean_performance_after_finetuning.png'><a href='clean_performance_after_finetuning.svg'>SVG</a>"
        "<p>Severe clean-performance degradation warrants inspecting masks and verifying own-condition checkpoint replay before attributing a mechanism.</p>",
        encoding="utf-8",
    )
    print(f"Report: {root / 'index.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.results_dir)
