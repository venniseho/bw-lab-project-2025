"""Wall-clock summaries for pilot runtime checks, not segmentation metrics."""
import csv
from pathlib import Path

import numpy as np


def summarize_timings(records: list[dict]) -> dict:
    """Report counts, totals, median and p95, grouped by stimulus condition."""
    groups = {"all": records}
    for row in records:
        groups.setdefault(row["condition"], []).append(row)
    summary = {}
    for name, rows in groups.items():
        fields = sorted({key for row in rows for key in row.get("timing", {})})
        summary[name] = {}
        for field in fields:
            values = [row["timing"][field] for row in rows if field in row.get("timing", {})]
            summary[name][field] = {"count": len(values), "total": float(sum(values)),
                                     "median": float(np.median(values)),
                                     "p95": float(np.percentile(values, 95))}
    return summary


def write_timing_csv(records: list[dict], path: Path) -> None:
    """Save one timing row per transformed input, including the cold first call."""
    fields = sorted({key for row in records for key in row.get("timing", {})})
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["__key__", "condition", "first_inference", *fields])
        writer.writeheader()
        for row in records:
            writer.writerow({"__key__": row["__key__"], "condition": row["condition"],
                             "first_inference": row["first_inference"], **row["timing"]})
