"""Plot saved throughput sweeps without loading SAM3 or allocating a GPU."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_sweeps(root):
    sweeps = {}
    for path in sorted(root.rglob("batch_sweep.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["status"] == "complete"]
        rows.sort(key=lambda row: int(row["batch_size"]))
        if rows:
            sweeps[str(path.parent.relative_to(root))] = rows
    if not sweeps:
        raise ValueError(f"No completed batch sweeps under {root}")
    return sweeps


def plot(root, output):
    sweeps = read_sweeps(root)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for label, rows in sweeps.items():
        sizes = [int(row["batch_size"]) for row in rows]
        rates = [float(row["inputs_per_second"]) for row in rows]
        latency = [1000 * float(row["evaluation_seconds"]) / int(row["input_count"]) for row in rows]
        line, = axes[0].plot(sizes, rates, marker="o", label=label)
        axes[1].plot(sizes, latency, marker="o", label=label, color=line.get_color())
        best = max(range(len(rates)), key=rates.__getitem__)
        axes[0].annotate(f"{rates[best]:.2f}", (sizes[best], rates[best]),
                         xytext=(5, 8), textcoords="offset points", color=line.get_color())
    for axis in axes:
        axis.set_xscale("log", base=2)
        sizes = sorted({int(row["batch_size"]) for rows in sweeps.values() for row in rows})
        axis.set_xticks(sizes, labels=[str(size) for size in sizes])
        axis.set_xlabel("Image batch size")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Inputs per second (higher is faster)")
    axes[1].set_ylabel("Evaluation ms per input (lower is faster)")
    fig.suptitle("SAM3 evaluation throughput by image batch size")
    fig.supxlabel("Includes loading, transforms, inference and metrics; excludes model loading and warm-up.\n"
                  "Amortized time per input, not isolated prediction latency. Single sweeps; no error bars.", fontsize=9)
    for extension in ("png", "svg"):
        path = output / f"batch_throughput.{extension}"
        fig.savefig(path, dpi=180)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.benchmark_dir, args.output_dir)
