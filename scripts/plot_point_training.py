"""Compare saved point-training validation curves and collect existing previews."""

import argparse
import csv
import html
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


COLORS = {
    "Fragmentation": "#6a238e",
    "Low fence": "#007c79",
    "Medium fence": "#b17800",
    "High fence": "#ba3627",
}


def read_curve(run):
    with (run / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["validation_set"] == "interval"]
    result = [
        {
            "step": int(row["step"]),
            "instances": int(row["instances"]),
            "mean_iou": float(row["mean_iou"]),
            "mean_loss": float(row["mean_loss"]),
            "epoch": int(row["epoch"]),
        }
        for row in rows
    ]
    result.sort(key=lambda row: row["step"])
    if not result or result[0]["step"] != 0:
        raise ValueError(f"Missing before-training baseline: {run}")
    if len({row["step"] for row in result}) != len(result):
        raise ValueError(f"Duplicate validation steps: {run}")
    if len({row["instances"] for row in result}) != 1 or result[0]["instances"] < 1:
        raise ValueError(f"Validation sample count changes: {run}")
    if any(
        not 0 <= row["mean_iou"] <= 1 or not math.isfinite(row["mean_loss"]) or row["mean_loss"] < 0
        for row in result
    ):
        raise ValueError(f"Invalid validation metrics: {run}")
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_curves(curves, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for label, rows in curves.items():
        for axis, metric in zip(axes, ("mean_iou", "mean_loss")):
            axis.plot(
                [row["step"] for row in rows],
                [row[metric] for row in rows],
                color=COLORS[label],
                label=label,
                marker="o",
                linewidth=2,
                markersize=5,
            )
    for axis in axes:
        axis.set_xlabel("Training updates (one image per update)")
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Validation mean IoU (higher is better)")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("Validation mean loss (lower is better)")
    axes[1].set_ylim(bottom=0)
    fig.suptitle("SAM3 full point-path fine-tuning: one epoch")
    fig.supxlabel(
        "Separate model per condition; validation uses that model's training transformation.\n"
        "Step 0 = pretrained baseline. Fixed held-out targets; no independent test set or repeated seeds.",
        fontsize=9,
    )
    for extension in ("png", "svg"):
        fig.savefig(output / f"learning_curves.{extension}", dpi=180)
    plt.close(fig)


def collect_previews(label, run, rows, output):
    """Copy saved panels; never recompute masks or load a checkpoint."""
    best = max(rows, key=lambda row: row["mean_iou"])
    selected = {"best": best, "final": rows[-1]}
    slug = label.lower().replace(" ", "-")
    destination = output / "examples" / slug
    destination.mkdir(parents=True)
    entries = []
    for snapshot, row in selected.items():
        for index in range(4):
            source = run / "previews" / f"epoch-{row['epoch']}-step-{row['step']}-{index}.png"
            target = destination / f"{snapshot}-{index}.png"
            if source.is_file():
                shutil.copyfile(source, target)
                entries.append(
                    {
                        "snapshot": snapshot,
                        "step": row["step"],
                        "sample_index": index,
                        "image": target.relative_to(output).as_posix(),
                    }
                )
            else:
                print(f"Preview unavailable (no inference rerun): {source}")
    return entries


def gallery(output, summaries, previews):
    parts = [
        "<!doctype html><meta charset='utf-8'><title>SAM3 training results</title>",
        "<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:15px}"
        "img{max-width:100%;height:auto}td,th{padding:8px;text-align:left}table{border-collapse:collapse}"
        "tr{border-bottom:1px solid #ddd}</style>",
        "<h1>SAM3 fine-tuning results</h1><img src='learning_curves.png' alt='Validation learning curves'>",
        "<p>Each model was trained and validated on its own condition. Best checkpoints were selected "
        "on this validation set; these are not independent test results.</p>",
        "<table><tr><th>Condition</th><th>Baseline IoU</th><th>Best IoU</th><th>Best step</th><th>Final IoU</th></tr>",
    ]
    for row in summaries:
        parts.append(
            f"<tr><td>{html.escape(row['condition'])}</td><td>{row['baseline_iou']:.4f}</td>"
            f"<td>{row['best_iou']:.4f}</td><td>{row['best_step']:,}</td><td>{row['final_iou']:.4f}</td></tr>"
        )
    parts.append(
        "</table><h2>Saved before/after examples</h2><p>These are the first four validation samples, "
        "not selected successes or a representative random sample. Each sheet shows input + positive "
        "point, COCO target, pretrained prediction, and fine-tuned prediction. "
        "The green dot is the target point. IoUs on sheets refer to that individual object.</p>"
    )
    for label, entries in previews.items():
        parts.append(f"<h2>{html.escape(label)}</h2>")
        for entry in entries:
            parts.append(
                f"<p>{entry['snapshot'].title()} checkpoint, update {entry['step']:,}, "
                f"validation sample {entry['sample_index'] + 1}</p>"
                f"<a href='{entry['image']}'><img src='{entry['image']}' loading='lazy' "
                "alt='Input, target, before and after segmentation'></a>"
            )
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def plot(runs, output):
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    curves = {label: read_curve(run) for label, run in runs.items()}
    counts = {rows[0]["instances"] for rows in curves.values()}
    if len(counts) != 1:
        raise ValueError("Runs have different validation sample counts")
    metadata = {
        label: json.loads((run / "metadata.json").read_text()) for label, run in runs.items()
    }
    signatures = {
        json.dumps(item["config"]["annotation_sha256"], sort_keys=True)
        for item in metadata.values()
    }
    if len(signatures) != 1:
        raise ValueError("Prepared train/validation annotation hashes differ between runs")
    policies = {
        (item["scope"], item["config"]["seed"], item["config"].get("validation_image_count"))
        for item in metadata.values()
    }
    if len(policies) != 1:
        raise ValueError("Training scope or validation selection policies differ between runs")
    for label, item in metadata.items():
        config = item["config"]
        expected = "fragmentation" if label == "Fragmentation" else "fence"
        if config["transform"] != expected:
            raise ValueError(f"{label}: unexpected transform")
        if expected == "fence":
            parameters = config["transform_parameters"]
            width = {"Low fence": 4, "Medium fence": 12, "High fence": 24}[label]
            if parameters["bar_width"] != width or parameters["bar_spacing"] != 40:
                raise ValueError(f"{label}: expected width {width} and spacing 40")
    output.mkdir(parents=True)
    summaries, previews = [], {}
    for label, rows in curves.items():
        best, final = max(rows, key=lambda row: row["mean_iou"]), rows[-1]
        summaries.append(
            {
                "condition": label,
                "validation_objects": rows[0]["instances"],
                "baseline_iou": rows[0]["mean_iou"],
                "best_iou": best["mean_iou"],
                "best_step": best["step"],
                "final_iou": final["mean_iou"],
                "final_step": final["step"],
                "run": str(runs[label].resolve()),
            }
        )
        previews[label] = collect_previews(label, runs[label], rows, output)
        print(
            f"{label}: baseline={rows[0]['mean_iou']:.4f}; best={best['mean_iou']:.4f} "
            f"at {best['step']:,}; final={final['mean_iou']:.4f}"
        )
    write_csv(output / "run_summary.csv", summaries)
    write_csv(
        output / "validation_curves.csv",
        [{"condition": label, **row} for label, rows in curves.items() for row in rows],
    )
    draw_curves(curves, output)
    gallery(output, summaries, previews)
    (output / "sources.json").write_text(
        json.dumps(
            {
                "runs": {label: str(run.resolve()) for label, run in runs.items()},
                "annotation_sha256": next(iter(metadata.values()))["config"]["annotation_sha256"],
                "previews": previews,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Figures and gallery: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fragmentation", "low", "medium", "high"):
        parser.add_argument(f"--{name}-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(
        {
            "Fragmentation": args.fragmentation_run,
            "Low fence": args.low_run,
            "Medium fence": args.medium_run,
            "High fence": args.high_run,
        },
        args.output_dir,
    )
