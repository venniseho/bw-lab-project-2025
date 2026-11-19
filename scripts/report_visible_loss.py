"""Compare saved matched fence training runs with full versus visible-region loss."""

import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_point_training import read_curve, write_csv
from scripts.report_point_staged import MATCHED_KEYS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np


CONDITIONS = {"fence-low": "Low fence", "fence-medium": "Medium fence", "fence-high": "High fence"}
OBJECTIVES = {"full": ("Full loss", "#006d77"), "visible": ("Visible loss", "#ad4718")}
METRICS = {
    "mean_full_iou": "Full-target IoU",
    "mean_visible_iou": "Visible-region IoU",
    "mean_full_loss": "Full-region loss",
    "mean_visible_loss": "Visible-region loss",
}


def load_pairs(control, visible_root):
    sources = json.loads((control / "sources.json").read_text())
    with (control / "validation_curves.csv").open(newline="") as stream:
        control_rows = list(csv.DictReader(stream))
    runs = {}
    for condition in CONDITIONS:
        fixed = sources[f"{condition}/fixed"]["metadata"]
        root = visible_root / f"training-{condition}-visible-loss"
        changed = json.loads((root / "metadata.json").read_text())
        complete = json.loads((root / "complete.json").read_text())
        for metadata, region in ((fixed, "full"), (changed, "visible")):
            if (
                metadata["scope"] != "mask_decoder_only"
                or metadata["config"]["loss_region"] != region
            ):
                raise ValueError(f"Unexpected scope or loss region: {condition}")
        for key in (
            *[key for key in MATCHED_KEYS if key != "loss_region"],
            "lr_schedule",
            "checkpoint",
            "native_revision",
        ):
            if fixed["config"].get(key) != changed["config"].get(key):
                raise ValueError(f"Paired configs differ in {key}: {condition}")
        rows = [
            row
            for row in control_rows
            if row["condition"] == condition and row["schedule"] == "fixed"
        ]
        full = [
            {
                "step": int(row["step"]),
                "instances": int(row["instances"]),
                **{key: float(row[key]) for key in METRICS},
            }
            for row in rows
        ]
        full.sort(key=lambda row: row["step"])
        visible = read_curve(root)
        if [(row["step"], row["instances"]) for row in full] != [
            (row["step"], row["instances"]) for row in visible
        ]:
            raise ValueError(f"Validation steps or counts differ: {condition}")
        if visible[-1]["step"] != complete["steps"] or complete["steps"] != len(
            changed["config"]["training_order"]
        ):
            raise ValueError(f"Incomplete training budget: {condition}")
        for key in METRICS:
            if not np.isclose(full[0][key], visible[0][key], atol=1e-6, rtol=0):
                raise ValueError(f"Pretrained baseline mismatch for {key}: {condition}")
            if not np.isclose(
                visible[-1][key], complete["final_validation_summary"][key], atol=1e-7, rtol=0
            ):
                raise ValueError(f"Final summary mismatch: {condition}")
        runs[condition] = {
            "full": full,
            "visible": visible,
            "metadata": changed,
            "complete": complete,
            "path": root,
        }
    return runs


def save(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def learning_curves(runs, output):
    for metric, title in METRICS.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, layout="constrained")
        for axis, (condition, label) in zip(axes, CONDITIONS.items()):
            for objective, (legend, color) in OBJECTIVES.items():
                rows = runs[condition][objective]
                axis.plot(
                    [row["step"] for row in rows],
                    [row[metric] for row in rows],
                    label=legend,
                    color=color,
                    marker="o",
                    markersize=4,
                    lw=2,
                )
            axis.set_title(f"{label} (n={runs[condition]['full'][0]['instances']:,})")
            axis.set_xlabel("Training updates")
            axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            axis.grid(alpha=0.2)
            axis.legend()
        axes[0].set_ylabel(
            title + (" (higher is better)" if metric.endswith("iou") else " (lower is better)")
        )
        if metric.endswith("iou"):
            axes[0].set_ylim(0, 1)
        else:
            axes[0].set_ylim(bottom=0)
        fig.suptitle(f"SAM3 decoder-only: {title.lower()} under both training objectives")
        save(fig, output, metric.removeprefix("mean_") + "_learning_curves")


def final_scores(runs, output):
    x = np.arange(3)
    for gain in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
        for axis, metric in zip(axes, ("mean_full_iou", "mean_visible_iou")):
            if gain:
                values = [
                    runs[c]["visible"][-1][metric] - runs[c]["full"][-1][metric] for c in CONDITIONS
                ]
                bars = axis.bar(
                    x, values, color=["#006d77" if v >= 0 else "#ad4718" for v in values]
                )
                axis.axhline(0, color="black", lw=1)
                axis.bar_label(bars, fmt="%+.4f", padding=4)
                axis.margins(y=0.2)
                axis.set_ylabel("Final IoU change (visible-loss minus full-loss training)")
            else:
                for objective, offset in (("full", -0.19), ("visible", 0.19)):
                    legend, color = OBJECTIVES[objective]
                    bars = axis.bar(
                        x + offset,
                        [runs[c][objective][-1][metric] for c in CONDITIONS],
                        width=0.36,
                        color=color,
                        label=legend,
                    )
                    axis.bar_label(bars, fmt="%.3f", padding=3)
                axis.set_ylim(0, 1)
                axis.set_ylabel("Final validation IoU")
                axis.legend()
            axis.set_title(METRICS[metric])
            axis.set_xticks(x, CONDITIONS.values())
            axis.set_axisbelow(True)
            axis.grid(axis="y", alpha=0.2)
        fig.suptitle("Matched data, seed and training budget; final checkpoints")
        save(fig, output, "final_iou_changes" if gain else "final_iou_comparison")


def report(control, visible_root, output):
    runs = load_pairs(control, visible_root)
    output.mkdir(parents=True, exist_ok=False)
    curves, summaries = [], []
    for condition, run in runs.items():
        summary = {
            "condition": condition,
            "training_updates": run["complete"]["steps"],
            "validation_objects": run["visible"][0]["instances"],
        }
        for objective in OBJECTIVES:
            curves.extend(
                {
                    "condition": condition,
                    "objective": objective,
                    "step": row["step"],
                    "instances": row["instances"],
                    **{key: row[key] for key in METRICS},
                }
                for row in run[objective]
            )
            for key in METRICS:
                summary[f"{objective}_training_final_{key}"] = run[objective][-1][key]
        for key in ("mean_full_iou", "mean_visible_iou"):
            summary[f"change_{key}"] = run["visible"][-1][key] - run["full"][-1][key]
        summaries.append(summary)
    write_csv(output / "run_summary.csv", summaries)
    write_csv(output / "validation_curves.csv", curves)
    learning_curves(runs, output)
    final_scores(runs, output)
    page = [
        "<!doctype html><meta charset='utf-8'><title>SAM3 fence loss comparison</title>",
        "<style>body{font:16px Arial;max-width:1250px;margin:30px auto;padding:16px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}</style>",
        "<h1>SAM3: full-loss vs visible-loss fence training</h1>",
        "<p>Only the mask decoder is trainable. Matched targets, prompts, training order, fixed learning rate and update budget. Full-loss training supervises the whole target; visible-loss training ignores fence bars in focal, Dice and quality-score supervision. Both metrics and both losses are evaluated for both models.</p>",
        "<p>These are own-condition validation results, not cross-condition or independent test results. Comparisons use final checkpoints. Fully hidden targets were excluded.</p>",
    ]
    for name in (
        "full_iou_learning_curves",
        "visible_iou_learning_curves",
        "full_loss_learning_curves",
        "visible_loss_learning_curves",
        "final_iou_comparison",
        "final_iou_changes",
    ):
        page.append(
            f'<h2>{name.replace("_", " ").capitalize()}</h2><a href="{name}.svg">SVG</a><img src="{name}.png">'
        )
    page.append(
        "<h2>Final before/after examples</h2><p>First four validation targets per condition; not selected successes or representative random samples. Each sheet compares the pretrained prediction with that objective's fine-tuned prediction. Displayed sheet IoUs are full-target IoUs.</p>"
    )
    for condition, label in CONDITIONS.items():
        for objective, (legend, _) in OBJECTIVES.items():
            destination = output / "examples" / f"{condition}-{objective}"
            destination.mkdir(parents=True)
            if objective == "full":
                source = control / "examples" / f"{label.lower().replace(' ', '-')}-fixed"
                images = sorted(source.glob("final-*.png"))
            else:
                run = runs[condition]
                source = run["path"] / "previews"
                images = sorted(source.glob(f"epoch-1-step-{run['complete']['steps']}-*.png"))
            page.append(f"<h3>{html.escape(label)}: {legend}</h3>")
            for image in images:
                shutil.copy2(image, destination / image.name)
                page.append(
                    f'<img loading="lazy" src="examples/{condition}-{objective}/{image.name}">'
                )
    (output / "index.html").write_text("\n".join(page), encoding="utf-8")
    (output / "sources.json").write_text(
        json.dumps(
            {
                "control_report": str(control.resolve()),
                "visible_root": str(visible_root.resolve()),
                "visible_run_metadata": {key: run["metadata"] for key, run in runs.items()},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Report: {output / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--visible-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.control_report, args.visible_root, args.output_dir)


if __name__ == "__main__":
    main()
