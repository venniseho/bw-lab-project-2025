"""Read existing benchmark CSVs on a login node; no model or GPU required."""
import argparse
import csv
from pathlib import Path


FIELDS = ("batch_wait_seconds", "image_embedding_seconds", "point_query_seconds",
          "inference_wall_seconds", "metrics_seconds", "export_seconds")


def summarize(root):
    """Print aggregate wall times; overlapping columns must not be added together."""
    print("Times are total seconds over recorded inputs. Encoder/query are parts of inference.")
    print("Encoder includes transfer/preprocessing/projections; query includes CPU mask collection.")
    for sweep in sorted(root.rglob("batch_sweep.csv")):
        print(f"\n{sweep.parent}")
        with sweep.open(newline="", encoding="utf-8") as stream:
            runs = list(csv.DictReader(stream))
        for run in runs:
            if run["status"] != "complete":
                print(f"batch={run['batch_size']}: {run['status']}")
                continue
            path = sweep.parent / f"batch-{run['batch_size']}" / "predictions" / "image_results.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            totals = {field: sum(float(row[field]) for row in rows if row.get(field)) for field in FIELDS}
            print(f"batch={run['batch_size']} inputs={len(rows)} "
                  f"inputs/s={float(run['inputs_per_second']):.3f} "
                  f"evaluation={float(run['evaluation_seconds']):.3f}s")
            print("  " + "; ".join(f"{key.removesuffix('_seconds')}={value:.3f}s" for key, value in totals.items()))
    if not list(root.rglob("batch_sweep.csv")):
        raise FileNotFoundError(f"No batch_sweep.csv under {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    summarize(parser.parse_args().benchmark_dir)
