"""Plot paired cross-condition IoUs from saved evaluation records; no model loading."""

import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_point_training import write_csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCES = (
    "fragmentation",
    "fence-low",
    "fence-medium",
    "fence-high",
    "fence-low-visible-loss",
    "fence-medium-visible-loss",
    "fence-high-visible-loss",
)
FENCES = ("fence-low", "fence-medium", "fence-high")
METRICS = ("full_iou", "visible_iou")
COLORS = {"full_iou": "#006d77", "visible_iou": "#ad4718"}


def csv_rows(path):
    csv.field_size_limit(2**31 - 1)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def paired_records(rows, final_step):
    """Pair by target identity, and reject changes in prompts or rendered stimuli."""
    phases = {"pretrained": {}, "finetuned": {}}
    for row in rows:
        step = int(row["step"])
        if step not in (0, final_step) or final_step <= 0:
            raise ValueError("Expected only baseline and final-checkpoint records")
        phase = "pretrained" if step == 0 else "finetuned"
        key = row["key"]
        if key in phases[phase]:
            raise ValueError(f"Duplicate {phase} target: {key}")
        phases[phase][key] = row
    before, after = phases.values()
    if not before or before.keys() != after.keys():
        raise ValueError("Baseline and fine-tuned target keys differ or are empty")
    pairs = []
    for key in sorted(before):
        left, right = before[key], after[key]
        for field in (
            "image_id",
            "annotation_id",
            "point_x",
            "point_y",
            "point_seed",
            "transform_json",
            "target_pixels",
            "visible_pixels",
        ):
            if left[field] != right[field]:
                raise ValueError(f"Paired target {key} differs in {field}")
        result = {"key": key, "image_id": left["image_id"], "annotation_id": left["annotation_id"]}
        for metric in METRICS:
            for phase, row in (("pretrained", left), ("finetuned", right)):
                value = float(row[metric])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {metric}: {key}")
                result[f"{phase}_{metric}"] = value
            result[f"change_{metric}"] = (
                result[f"finetuned_{metric}"] - result[f"pretrained_{metric}"]
            )
        pairs.append(result)
    return pairs


def load_comparisons(root):
    comparisons, timing, provenance, baselines = [], [], {}, {}
    for source in SOURCES:
        run = root / f"{source}-transfer"
        complete = json.loads((run / "complete.json").read_text())
        metadata = json.loads((run / "metadata.json").read_text())
        if metadata["scope"] != "cross-condition-evaluation" or complete["training_updates"] != 0:
            raise ValueError(f"Not an evaluation-only output: {run}")
        expected = FENCES if source == "fragmentation" else ("fragmentation",)
        if set(complete["results"]) != set(expected):
            raise ValueError(f"Unexpected target conditions: {run}")
        provenance[source] = metadata
        for target in expected:
            pairs = paired_records(csv_rows(run / target / "heldout.csv"), complete["steps"])
            summary = {
                "source": source,
                "target": target,
                "instances": len(pairs),
                "source_step": complete["steps"],
            }
            for phase in ("pretrained", "finetuned"):
                saved = complete["results"][target][phase]
                if saved["instances"] != len(pairs):
                    raise ValueError(f"Summary/record counts differ: {source} -> {target}")
                for metric in METRICS:
                    mean = float(np.mean([row[f"{phase}_{metric}"] for row in pairs]))
                    if not np.isclose(mean, saved[f"mean_{metric}"], atol=1e-7, rtol=0):
                        raise ValueError(f"Summary/record IoUs differ: {source} -> {target}")
                    summary[f"{phase}_{metric}"] = mean
            signature = metadata["targets"][target]
            baseline = [
                (row["key"], row["pretrained_full_iou"], row["pretrained_visible_iou"])
                for row in pairs
            ]
            if target in baselines:
                old_signature, old_baseline = baselines[target]
                if signature != old_signature or [row[0] for row in baseline] != [
                    row[0] for row in old_baseline
                ]:
                    raise ValueError(f"Target config or cohort changed across sources: {target}")
                difference = max(
                    abs(new - old)
                    for left, right in zip(baseline, old_baseline)
                    for new, old in zip(left[1:], right[1:])
                )
                if difference > 1e-7:
                    print(
                        f"Note: {target} pretrained IoUs vary across jobs (max delta={difference:.6g}); each comparison uses its own paired baseline."
                    )
            baselines[target] = signature, baseline
            for metric in METRICS:
                change = np.array([row[f"change_{metric}"] for row in pairs])
                summary[f"change_{metric}"] = float(change.mean())
                summary[f"improved_{metric}"] = int((change > 0).sum())
                summary[f"equal_{metric}"] = int((change == 0).sum())
                summary[f"worse_{metric}"] = int((change < 0).sum())
            timing.extend(
                {"source": source, "target": target, **row}
                for row in csv_rows(run / target / "timing.csv")
            )
            comparisons.append({"summary": summary, "pairs": pairs, "path": run / target})
    return comparisons, timing, provenance


def label(summary):
    source = summary["source"].replace("-visible-loss", " (visible loss)")
    return f"{source} → {summary['target']}"


def save_figure(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_means(comparisons, output):
    labels = [label(item["summary"]) for item in comparisons]
    y = np.arange(len(labels))
    for gain in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, layout="constrained")
        for axis, metric in zip(axes, METRICS):
            if gain:
                values = [item["summary"][f"change_{metric}"] for item in comparisons]
                axis.barh(y, values, color=COLORS[metric])
                axis.axvline(0, color="black", lw=1)
                axis.set_xlabel("Mean IoU change (fine-tuned − pretrained)")
            else:
                for phase, offset, color in (
                    ("pretrained", -0.18, "#777777"),
                    ("finetuned", 0.18, COLORS[metric]),
                ):
                    values = [item["summary"][f"{phase}_{metric}"] for item in comparisons]
                    axis.barh(
                        y + offset, values, height=0.34, color=color, label=phase.capitalize()
                    )
                axis.set_xlim(0, 1)
                axis.set_xlabel("Mean validation IoU")
                axis.legend()
            axis.set_title("Full target" if metric == "full_iou" else "Visible region")
            axis.grid(axis="x", alpha=0.2)
            axis.set_axisbelow(True)
        axes[0].set_yticks(y, labels)
        axes[0].invert_yaxis()
        fig.suptitle("SAM3 decoder-only cross-condition evaluation: final checkpoints")
        save_figure(fig, output, "iou_changes" if gain else "mean_iou")


def plot_pairs(comparisons, output):
    for metric in METRICS:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15), layout="constrained")
        for axis, item in zip(axes.flat, comparisons):
            x = np.array([row[f"pretrained_{metric}"] for row in item["pairs"]])
            y = np.array([row[f"finetuned_{metric}"] for row in item["pairs"]])
            axis.scatter(
                x, y, s=8, alpha=0.55, color=COLORS[metric], edgecolors="none", rasterized=True
            )
            axis.plot([0, 1], [0, 1], "k--", lw=1)
            axis.set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025), aspect="equal")
            axis.set_title(label(item["summary"]).replace(" → ", "\n→ "), fontsize=10)
            axis.text(
                0.03,
                0.97,
                f"n={len(x):,}\n(0,0): {int(((x == 0) & (y == 0)).sum()):,}",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
            )
            axis.grid(alpha=0.18)
        fig.supxlabel("Original pretrained model: target IoU")
        fig.supylabel("Fine-tuned model: target IoU")
        fig.suptitle(
            f"{'Full-target' if metric == 'full_iou' else 'Visible-region'} IoU: same target and prompt\nAbove diagonal = improved after fine-tuning; overlapping points have no jitter"
        )
        save_figure(fig, output, f"paired_{metric}")


def report(root, output):
    comparisons, timing, provenance = load_comparisons(root)
    output.mkdir(parents=True, exist_ok=False)
    summaries = [item["summary"] for item in comparisons]
    write_csv(output / "run_summary.csv", summaries)
    write_csv(
        output / "paired_image_results.csv",
        [
            {"source": item["summary"]["source"], "target": item["summary"]["target"], **row}
            for item in comparisons
            for row in item["pairs"]
        ],
    )
    write_csv(output / "timing.csv", timing)
    (output / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot_means(comparisons, output)
    plot_pairs(comparisons, output)
    page = [
        "<!doctype html><meta charset='utf-8'><title>SAM3 cross-condition results</title>",
        "<style>body{font-family:Arial;max-width:1200px;margin:32px auto;padding:16px}img{max-width:100%}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>",
        "<h1>SAM3: fence ↔ fragmentation transfer</h1>",
        "<p>Final decoder-only checkpoints versus the original pretrained model. No training occurs here. Each comparison uses identical targets, transforms and prompts. Validation results, not independent test results.</p>",
        "<p>Full IoU includes the original target behind artificial fence bars. Visible IoU ignores bars in prediction and target. Cohorts differ across target conditions. No confidence intervals or repeated training seeds are estimated.</p>",
        "<table><tr><th>Training → evaluation</th><th>n</th><th>Full IoU: before → after</th><th>Visible IoU: before → after</th></tr>",
    ]
    for row in summaries:
        page.append(
            f"<tr><td>{html.escape(label(row))}</td><td>{row['instances']:,}</td><td>{row['pretrained_full_iou']:.4f} → {row['finetuned_full_iou']:.4f}</td><td>{row['pretrained_visible_iou']:.4f} → {row['finetuned_visible_iou']:.4f}</td></tr>"
        )
        print(
            f"{label(row).replace('→', '->')}: n={row['instances']}, full change={row['change_full_iou']:+.4f}, visible change={row['change_visible_iou']:+.4f}"
        )
    page.append(
        "</table><p>In paired scatterplots, x = pretrained IoU and y = fine-tuned IoU. Above diagonal = improvement. (0,0) = no target overlap in either prediction; points can overlap.</p>"
    )
    for name in ("mean_iou", "iou_changes", "paired_full_iou", "paired_visible_iou"):
        page.append(
            f'<h2>{name.replace("_", " ").capitalize()}</h2><a href="{name}.svg">SVG</a><img src="{name}.png">'
        )
    page.append(
        "<h2>Before/after examples</h2><p>First four validation targets per comparison; these are not selected successes or representative samples.</p>"
    )
    for item in comparisons:
        name = f"{item['summary']['source']}-to-{item['summary']['target']}"
        destination = output / "examples" / name
        destination.mkdir(parents=True)
        page.append(f"<h3>{html.escape(label(item['summary']))}</h3>")
        for image in sorted((item["path"] / "previews").glob("*.png")):
            shutil.copy2(image, destination / image.name)
            page.append(f'<img loading="lazy" src="examples/{name}/{image.name}">')
    (output / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"Report: {output / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.cross_root, args.output_dir)


if __name__ == "__main__":
    main()
