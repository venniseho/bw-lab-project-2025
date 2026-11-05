"""Plot exploratory score-selected mask accuracy from existing evaluation CSVs."""

import argparse
import csv
from collections import Counter, defaultdict
import math
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path):
    groups = defaultdict(lambda: {"full_iou": [], "visible_iou": []})
    statuses = Counter()
    sources = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            statuses[row["status"]] += 1
            sources.add(row.get("source_key") or row.get("__key__"))
            if row["status"] != "complete":
                continue
            for metric in ("full_iou", "visible_iou"):
                if row.get(metric) and math.isfinite(float(row[metric])):
                    groups[row["condition"]][metric].append(float(row[metric]))
    return groups, {
        "csv": str(path.resolve()),
        "total_rows": sum(statuses.values()),
        "distinct_source_keys": len(sources),
        "status_counts": dict(statuses),
    }


def plot(root, output, fence_run=None, fragmentation_run=None):
    # Fix batch size across arms; this is a scientific diagnostic, not a sweep average.
    if root is not None:
        paths = {
            arm: root / arm / "batch-1" / "predictions" / "image_results.csv"
            for arm in ("fence", "fragmentation")
        }
    else:
        paths = {
            "fence": fence_run / "predictions/image_results.csv",
            "fragmentation": fragmentation_run / "predictions/image_results.csv",
        }
    loaded = {arm: read_metrics(path) for arm, path in paths.items()}
    data = {arm: groups for arm, (groups, _) in loaded.items()}
    audit = {arm: details for arm, (_, details) in loaded.items()}
    for arm, path in paths.items():
        config_path = path.parent.parent / "config.json"
        if config_path.exists():
            saved = json.loads(config_path.read_text())
            config = saved.get("config", saved)
            if config.get("condition") != arm:
                raise ValueError(
                    f"{arm}: run config has a different condition; check run directories"
                )
            audit[arm]["batch_size"] = config.get("batch_size")
            audit[arm]["max_inputs"] = config.get("max_inputs")
        timing_path = path.parent.parent / "timing.json"
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            audit[arm]["timing_input_count"] = timing["input_count"]
            if timing["input_count"] != audit[arm]["total_rows"]:
                raise ValueError(f"{arm}: CSV row count differs from timing summary")
        if not data[arm]:
            raise ValueError(f"{arm}: no complete finite IoU results")
        print(
            f"{arm}: {audit[arm]['total_rows']} rows; "
            f"{audit[arm]['distinct_source_keys']} source keys; statuses={audit[arm]['status_counts']}"
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained", sharey=True)
    with (output / "accuracy_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "condition", "metric", "n", "mean_iou"])
        writer.writeheader()
        for axis, (arm, groups) in zip(axes, data.items()):
            order = [
                c for c in ("clean", "light", "medium", "heavy", "fragmentation") if c in groups
            ]
            order += sorted(set(groups) - set(order))
            for offset, metric, label in (
                (-0.18, "full_iou", "Full target"),
                (0.18, "visible_iou", "Visible target"),
            ):
                for index, condition in enumerate(order):
                    values = groups[condition][metric]
                    if not values:
                        continue
                    value = mean(values)
                    axis.bar(
                        index + offset,
                        value,
                        width=0.34,
                        color="tab:blue" if offset < 0 else "tab:orange",
                        label=label if index == 0 else None,
                    )
                    print(f"  {arm}/{condition}: {metric} mean={value:.4f}, n={len(values)}")
                    writer.writerow(
                        {
                            "arm": arm,
                            "condition": condition,
                            "metric": metric,
                            "n": len(values),
                            "mean_iou": value,
                        }
                    )
            axis.set_xticks(
                range(len(order)), [f"{c}\nn={len(groups[c]['full_iou'])}" for c in order]
            )
            axis.set_title(arm.capitalize())
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.2)
            axis.legend()
    axes[0].set_ylabel("Mean measured mask IoU")
    fig.suptitle("Exploratory SAM3 accuracy across stimulus conditions")
    fig.supxlabel(
        "Model-score-selected grid candidate; mean over complete, finite results (exclusions in input_audit.json).\n"
        "One noncrowd target per source. Single run; no uncertainty intervals.",
        fontsize=8,
    )
    for extension in ("png", "svg"):
        stem = "pilot_accuracy" if root is not None else "evaluation_accuracy"
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=180)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--fence-run", type=Path)
    parser.add_argument("--fragmentation-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.benchmark_dir is not None:
        if args.fence_run or args.fragmentation_run:
            parser.error("use benchmark-dir OR the two explicit run directories")
    elif args.fence_run is None or args.fragmentation_run is None:
        parser.error("provide benchmark-dir or both fence-run and fragmentation-run")
    plot(args.benchmark_dir, args.output_dir, args.fence_run, args.fragmentation_run)
