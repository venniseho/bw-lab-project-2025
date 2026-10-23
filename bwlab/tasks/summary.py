"""Streaming numerical diagnostics with bounded scene previews."""
import csv
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import time

import numpy as np

from bwlab.models.proposals import Proposal, packed_mask_iou, select_proposals
from bwlab.monitoring import summarize_timings
from bwlab.samples import binary_mask
from bwlab.tasks._scene_export import save_scene


TIMINGS = ("source_load_seconds", "input_copy_seconds", "transform_seconds",
           "batch_wait_seconds", "inference_wall_seconds", "image_embedding_seconds",
           "point_query_seconds", "collection_seconds", "metrics_seconds", "export_seconds")
FIELDS = ("input_index", "__key__", "source_key", "condition", "batch_index",
          "input_sha256", "target_sha256",
          "raw_proposals", "chosen_proposal_id", "predicted_iou_score", "full_iou",
          "visible_iou", "oracle_full_iou", "oracle_visible_iou", "object_occluded_fraction",
          "status", "source_metadata", "transforms", *TIMINGS)


def _overlaps(proposals, mask):
    mask = binary_mask(mask, "target")
    if not mask.any():
        return None
    target = Proposal(np.packbits(mask), mask.shape, {"area_pixels": int(mask.sum())})
    return [packed_mask_iou(proposal, target) for proposal in proposals]


@dataclass(frozen=True)
class SummaryTask:
    preview_batches: int = 1

    def get_evaluator(self, output_path):
        return SummaryEvaluator(Path(output_path), self.preview_batches)


class SummaryEvaluator:
    def __init__(self, path, preview_batches):
        self.path, self.preview_batches = path, preview_batches
        path.mkdir(parents=True, exist_ok=False)
        self.stream = (path / "image_results.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS)
        self.writer.writeheader()
        self.records = []

    def update(self, sample, prediction):
        proposals, timing = prediction
        start = time.perf_counter()
        chosen = min(range(len(proposals)), key=lambda i: (
            -proposals[i].metadata["predicted_iou_score"], proposals[i].metadata["proposal_id"])) if proposals else None
        full = _overlaps(proposals, sample["segmentation"])
        visible = _overlaps(proposals, sample.get("visible_segmentation", sample["segmentation"]))
        transforms = sample.get("transforms", [])
        coverage = next((t.get("cumulative_object_occluded_fraction") for t in reversed(transforms)
                         if t.get("name") == "fence_occlusion"), None)
        elapsed = {**sample.get("timing", {}), **{k: v for k, v in timing.items() if k in TIMINGS},
                   "metrics_seconds": time.perf_counter()-start}
        row = {"input_index": len(self.records), "__key__": sample["__key__"],
               "source_key": sample.get("source_key", sample["__key__"]),
               "condition": sample.get("condition"), "batch_index": sample["batch_index"],
               "input_sha256": hashlib.sha256(sample["image"].tobytes()).hexdigest(),
               "target_sha256": hashlib.sha256(binary_mask(sample["segmentation"], "target").tobytes()).hexdigest(),
               "raw_proposals": len(proposals), "chosen_proposal_id": chosen,
               "predicted_iou_score": proposals[chosen].metadata["predicted_iou_score"] if chosen is not None else None,
               "full_iou": full[chosen] if full is not None and chosen is not None else None,
               "visible_iou": visible[chosen] if visible is not None and chosen is not None else None,
               "oracle_full_iou": max(full) if full else None,
               "oracle_visible_iou": max(visible) if visible else None,
               "object_occluded_fraction": coverage,
               "status": "empty_target" if full is None else ("no_proposals" if chosen is None else "complete"),
               "source_metadata": json.dumps(sample.get("metadata", {}), sort_keys=True),
               "transforms": json.dumps(transforms, sort_keys=True)}
        export_start = time.perf_counter()
        if sample["batch_index"] < self.preview_batches:
            retained, ledger, rules = select_proposals(proposals, dedup_iou=None)
            save_scene(sample["image"], proposals, retained, ledger,
                       self.path / "previews" / f"{len(self.records):06d}",
                       {"__key__": sample["__key__"], **rules})
        elapsed["export_seconds"] = time.perf_counter()-export_start
        self.writer.writerow({**row, **{k: elapsed.get(k) for k in TIMINGS}})
        # Small records only. Do not rewrite all results/percentiles per input.
        self.records.append({"__key__": row["__key__"], "condition": row["condition"], "timing": elapsed})
        if len(self.records) % 100 == 0:
            self.stream.flush()

    def close(self):
        self.stream.close()

    def finalize(self):
        self.close()
        report = {"status": "complete", "input_count": len(self.records),
                  "records": self.records,
                  "scoring_protocol": "highest predicted score across independent grid proposals; oracle is diagnostic only",
                  "aggregation_unit": "one transformed dataset sample (not necessarily one unique source image)",
                  "timing_summary_seconds": summarize_timings(self.records)}
        (self.path / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return report
