"""Archive the existing automatic proposal protocol; no new quantitative metric."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np

from bwlab.models.proposals import select_proposals
from bwlab.monitoring import summarize_timings, write_timing_csv
from ._scene_export import save_scene


@dataclass(frozen=True)
class SceneProposalTask:
    """Keep candidate generation separate from recorded display-selection policy."""

    min_score: float | None = None
    dedup_iou: float | None = None
    proposal_limit: int = 100
    save_candidate_limit: int | None = None
    model_metadata: dict = field(default_factory=dict)

    def get_evaluator(self, output_path: Path) -> "SceneProposalEvaluator":
        return SceneProposalEvaluator(output_path, self)


class SceneProposalEvaluator:
    """Incremental scene export using the already verified save_scene implementation.

    Finalize reports counts and artifact locations, not accuracy. Targets are
    stored separately; they never affect proposal generation or selection.
    """

    def __init__(self, output_path: Path, task: SceneProposalTask):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=False)
        self.task = task
        self.records = []
        self._write_manifest("running")

    def _write_manifest(self, status: str) -> dict:
        report = {"status": status, "protocol": "automatic_independent_positive_point_grid",
                  "quantitative_evaluation": False,
                  "model": deepcopy(self.task.model_metadata), "records": self.records,
                  "timing_summary_seconds": summarize_timings(self.records),
                  "timing_summary_excluding_first_inference_seconds": summarize_timings(self.records[1:])}
        (self.output_path / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report

    def update(self, sample: dict, prediction: tuple) -> None:
        proposals, timing = prediction
        start = time.perf_counter()
        retained, rows, rules = select_proposals(
            proposals, min_score=self.task.min_score, dedup_iou=self.task.dedup_iou,
            proposal_limit=self.task.proposal_limit)
        selection_seconds = time.perf_counter()-start
        export_start = time.perf_counter()
        scene_path = self.output_path / f"{len(self.records):06d}"
        metadata = {"__key__": sample["__key__"], "source_key": sample.get("source_key"),
                    "condition": sample.get("condition"),
                    "source_metadata": deepcopy(sample.get("metadata", {})),
                    "transforms": deepcopy(sample.get("transforms", [])),
                    "model": self.task.model_metadata, **timing, **rules}
        metadata["sample_timing_seconds"] = deepcopy(sample.get("timing", {}))
        save_scene(sample["image"], proposals, retained, rows, scene_path, metadata,
                   self.task.save_candidate_limit)
        targets = {name: sample[name] for name in
                   ("segmentation", "visible_segmentation", "occlusion_mask") if name in sample}
        if "fragmentation" in sample:
            targets.update({name: sample["fragmentation"][name]
                            for name in ("outline_segments", "noise_segments")})
        np.savez_compressed(scene_path / "targets.npz", **targets)
        self.records.append({"__key__": sample["__key__"], "source_key": sample.get("source_key"),
                             "condition": sample.get("condition"),
                             "scene": scene_path.name, "raw_proposals": len(proposals),
                             "retained_proposals": len(retained),
                             "first_inference": len(self.records) == 0,
                             "timing": {**sample.get("timing", {}),
                                        **{k: v for k, v in timing.items() if k.endswith("_seconds")},
                                        "selection_seconds": selection_seconds,
                                        "export_seconds": time.perf_counter()-export_start}})
        self._write_manifest("running")

    def finalize(self) -> dict:
        write_timing_csv(self.records, self.output_path / "timings.csv")
        return self._write_manifest("complete")
