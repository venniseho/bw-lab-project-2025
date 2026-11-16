"""Report matched fixed-rate versus warmup/cosine runs from saved CPU-readable files."""

import argparse
import csv
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_point_training import collect_previews, read_curve, write_csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


CONDITIONS = {
    "fragmentation": "Fragmentation",
    "fence-low": "Low fence",
    "fence-medium": "Medium fence",
    "fence-high": "High fence",
}
SCHEDULES = {
    "fixed": ("", "#006d77", "Fixed LR"),
    "warmup_cosine": ("-warmup-cosine", "#c05018", "Warmup + cosine"),
}
MATCHED_KEYS = (
    "annotation_sha256",
    "training_order",
    "seed",
    "epochs",
    "resolution",
    "learning_rate",
    "transform",
    "transform_parameters",
    "loss_region",
    "validation_interval_steps",
    "validation_image_count",
    "drop_fully_hidden",
)


def csv_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_runs(root):
    runs = {}
    for condition in CONDITIONS:
        for schedule, (suffix, _, _) in SCHEDULES.items():
            path = root / f"training-{condition}{suffix}"
            complete = json.loads((path / "complete.json").read_text())
            metadata = json.loads((path / "metadata.json").read_text())
            curve = read_curve(path)
            if metadata["scope"] != "mask_decoder_only":
                raise ValueError(f"Expected mask-decoder-only scope: {path}")
            config = metadata["config"]
            expected_transform = "fragmentation" if condition == "fragmentation" else "fence"
            if (
                config["transform"] != expected_transform
                or config.get("loss_region", "full") != "full"
            ):
                raise ValueError(f"Unexpected condition or objective for Stage 1: {path}")
            if condition != "fragmentation":
                width = {"fence-low": 4, "fence-medium": 12, "fence-high": 24}[condition]
                parameters = config["transform_parameters"]
                if parameters["bar_width"] != width or parameters["bar_spacing"] != 40:
                    raise ValueError(f"Fence severity does not match its folder label: {path}")
            if config["epochs"] != 1 or complete["steps"] != len(config["training_order"]):
                raise ValueError(f"Expected a completed one-epoch training budget: {path}")
            if curve[-1]["step"] != complete["steps"]:
                raise ValueError(f"Final summary and completion steps differ: {path}")
            if any("mean_visible_iou" not in row for row in curve):
                raise ValueError(f"Visible-region measurements missing: {path}")
            expected = "constant" if schedule == "fixed" else "warmup_cosine"
            if metadata["config"].get("lr_schedule", {}).get("name", "constant") != expected:
                raise ValueError(f"Unexpected learning-rate schedule: {path}")
            runs[condition, schedule] = {
                "path": path,
                "complete": complete,
                "metadata": metadata,
                "curve": curve,
            }
        fixed, changed = (runs[condition, name] for name in SCHEDULES)
        for key in MATCHED_KEYS:
            if fixed["metadata"]["config"].get(key) != changed["metadata"]["config"].get(key):
                raise ValueError(f"{condition}: paired runs differ in {key}")
        if [(row["step"], row["instances"]) for row in fixed["curve"]] != [
            (row["step"], row["instances"]) for row in changed["curve"]
        ]:
            raise ValueError(f"{condition}: paired validation steps or counts differ")
    return runs


def learning_rates(path):
    """Stream large training CSVs and retain only a small trace."""
    kept, last = [], None
    with (path / "train.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            last = {"step": int(row["step"]), "learning_rate": float(row["learning_rate"])}
            if last["step"] == 1 or last["step"] % 500 == 0:
                kept.append(last)
    if last is None:
        raise ValueError(f"No training rows: {path}")
    if kept[-1]["step"] != last["step"]:
        kept.append(last)
    return kept


def run_summary(condition, schedule, run):
    curve = run["curve"]
    best = max(curve, key=lambda row: row["mean_iou"])
    timings = csv_rows(run["path"] / "timing.csv")
    train = [row for row in timings if row["phase"] == "train"]
    train_seconds = sum(float(row["wall_seconds"]) for row in train)
    presentations = sum(int(row["image_presentations"]) for row in train)
    return {
        "condition": condition,
        "schedule": schedule,
        "training_updates": run["complete"]["steps"],
        "validation_objects": curve[0]["instances"],
        "validation_passes": len(curve),
        "baseline_full_iou": curve[0]["mean_iou"],
        "final_full_iou": curve[-1]["mean_iou"],
        "final_visible_iou": curve[-1]["mean_visible_iou"],
        "best_full_iou": best["mean_iou"],
        "best_full_step": best["step"],
        "fit_wall_seconds": run["complete"]["wall_seconds"],
        "train_wall_seconds": train_seconds or None,
        "training_images_per_second": presentations / train_seconds if train_seconds else None,
        "validation_wall_seconds": sum(
            float(row["wall_seconds"]) for row in timings if row["phase"] == "validation"
        ),
        "checkpoint_wall_seconds": sum(
            float(row["wall_seconds"]) for row in csv_rows(run["path"] / "checkpoints.csv")
        ),
        "peak_gpu_memory_bytes": run["complete"]["peak_gpu_memory_bytes"],
        "gpu": run["metadata"].get("gpu"),
        "run": str(run["path"].resolve()),
    }


def draw_comparison(runs, traces, output, metric, ylabel, filename):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for axis, (condition, label) in zip(axes.flat, CONDITIONS.items()):
        for schedule, (_, color, legend) in SCHEDULES.items():
            rows = (
                traces[condition, schedule]
                if metric == "learning_rate"
                else runs[condition, schedule]["curve"]
            )
            axis.plot(
                [row["step"] for row in rows],
                [row[metric] for row in rows],
                color=color,
                label=legend,
                linewidth=2,
                marker=None if metric == "learning_rate" else "o",
                markersize=4,
            )
        count = runs[condition, "fixed"]["curve"][0]["instances"]
        axis.set_title(f"{label} (validation n={count:,})")
        axis.set_xlabel("Training updates")
        axis.set_ylabel(ylabel)
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(alpha=0.2)
        axis.legend()
        axis.set_ylim(0, 1) if metric.endswith("iou") else axis.set_ylim(bottom=0)
    fig.suptitle("SAM3: mask-decoder-only training, fixed LR versus warmup/cosine")
    fig.supxlabel(
        "Matched targets and training budget within each pair; visibility exclusions differ across conditions.\n"
        "Step 0 is pretrained baseline. Single seed; validation results, without uncertainty estimates.",
        fontsize=9,
    )
    for extension in ("png", "svg"):
        fig.savefig(output / f"{filename}.{extension}", dpi=180)
    plt.close(fig)


def write_gallery(output, summaries, previews):
    parts = [
        "<!doctype html><meta charset='utf-8'><title>SAM3 learning-rate comparison</title>",
        "<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:15px}"
        "img{max-width:100%}td,th{padding:8px;text-align:left}tr{border-bottom:1px solid #ccc}</style>",
        "<h1>Mask-decoder-only training: eight-run comparison</h1>",
        "<p>Full IoU scores the original target. Visible IoU ignores pixels under fence bars, "
        "including predictions there. Both score the same highest-quality candidate. "
        "Learning-rate pairs share targets; cohorts can differ across conditions.</p>",
    ]
    for filename, title in (
        ("full_iou", "Full-target IoU"),
        ("visible_iou", "Visible-region IoU"),
        ("full_loss", "Full-target loss"),
        ("visible_loss", "Visible-region loss"),
        ("learning_rate", "Recorded learning rate"),
    ):
        parts.append(f"<h2>{title}</h2><a href='{filename}.svg'><img src='{filename}.png'></a>")
    parts.append(
        "<h2>Final validation results</h2><table><tr><th>Condition</th><th>Schedule</th>"
        "<th>Updates</th><th>Full IoU</th><th>Visible IoU</th><th>Fit hours</th><th>Train images/s</th></tr>"
    )
    for row in summaries:
        rate = row["training_images_per_second"]
        rate_label = f"{rate:.2f}" if rate is not None else "unavailable"
        parts.append(
            f"<tr><td>{html.escape(CONDITIONS[row['condition']])}</td><td>{html.escape(row['schedule'])}</td>"
            f"<td>{row['training_updates']:,}</td><td>{row['final_full_iou']:.4f}</td>"
            f"<td>{row['final_visible_iou']:.4f}</td><td>{row['fit_wall_seconds'] / 3600:.2f}</td>"
            f"<td>{rate_label}</td></tr>"
        )
    parts.append(
        "</table><p>Fit duration includes validation and checkpoint saves, and excludes model initialization. "
        "Train images/s excludes validation and checkpoint pauses. Best checkpoints use validation full IoU. "
        "These are validation-selected results, not independent test results.</p>"
        "<h2>Saved before/after examples</h2><p>First four validation targets in each run; "
        "these are neither a random sample nor selected success cases. Panels show input and positive "
        "point, COCO target, pretrained prediction, and trained prediction. The green dot marks the prompt. "
        "Only compare sample indices within a learning-rate pair; visibility exclusions can change indices "
        "across conditions. Object keys and individual full-target IoUs are printed on the sheets.</p>"
    )
    for label, entries in previews.items():
        parts.append(f"<h3>{html.escape(label)}</h3>")
        for entry in entries:
            parts.append(
                f"<p>{entry['snapshot'].title()}, update {entry['step']:,}, sample {entry['sample_index'] + 1}</p>"
                f"<a href='{entry['image']}'><img loading='lazy' src='{entry['image']}'></a>"
            )
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def report(root, output):
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    runs = load_runs(root)
    traces = {key: learning_rates(run["path"]) for key, run in runs.items()}
    for key, run in runs.items():
        if traces[key][-1]["step"] != run["complete"]["steps"]:
            raise ValueError(f"Training CSV does not reach the completion step: {run['path']}")
    output.mkdir(parents=True)
    summaries = [run_summary(*key, run) for key, run in runs.items()]
    write_csv(output / "run_summary.csv", summaries)
    write_csv(
        output / "validation_curves.csv",
        [
            {"condition": condition, "schedule": schedule, **row}
            for (condition, schedule), run in runs.items()
            for row in run["curve"]
        ],
    )
    write_csv(
        output / "learning_rates.csv",
        [
            {"condition": condition, "schedule": schedule, **row}
            for (condition, schedule), rows in traces.items()
            for row in rows
        ],
    )
    differences = []
    for condition in CONDITIONS:
        fixed, changed = (runs[condition, name]["curve"][-1] for name in SCHEDULES)
        differences.append(
            {
                "condition": condition,
                "full_iou_change_warmup_minus_fixed": changed["mean_iou"] - fixed["mean_iou"],
                "visible_iou_change_warmup_minus_fixed": changed["mean_visible_iou"]
                - fixed["mean_visible_iou"],
            }
        )
    write_csv(output / "paired_differences.csv", differences)
    for metric, ylabel, filename in (
        ("mean_full_iou", "Mean full-target IoU", "full_iou"),
        ("mean_visible_iou", "Mean visible-region IoU", "visible_iou"),
        ("mean_full_loss", "Mean full-target loss", "full_loss"),
        ("mean_visible_loss", "Mean visible-region loss", "visible_loss"),
        ("learning_rate", "Learning rate", "learning_rate"),
    ):
        draw_comparison(runs, traces, output, metric, ylabel, filename)
    previews = {}
    for (condition, schedule), run in runs.items():
        label = f"{CONDITIONS[condition]} {schedule}"
        previews[label] = collect_previews(label, run["path"], run["curve"], output)
    write_gallery(output, summaries, previews)
    (output / "sources.json").write_text(
        json.dumps(
            {
                f"{condition}/{schedule}": {
                    "run": str(run["path"].resolve()),
                    "metadata": run["metadata"],
                }
                for (condition, schedule), run in runs.items()
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Report: {output.resolve()}")
    for row in summaries:
        print(
            f"{row['condition']} {row['schedule']}: full={row['final_full_iou']:.4f}; "
            f"visible={row['final_visible_iou']:.4f}; train images/s={row['training_images_per_second']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.plan_root, args.output_dir)
