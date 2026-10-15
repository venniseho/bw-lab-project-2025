"""Automatic grid queries, explicit mask-IoU deduplication, and RGB previews."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
from PIL import Image


@dataclass
class Proposal:
    packed_mask: np.ndarray
    shape: tuple[int, int]
    metadata: dict

    def mask(self) -> np.ndarray:
        return np.unpackbits(self.packed_mask, count=self.shape[0] * self.shape[1]).reshape(self.shape).astype(bool)


def point_grid(height: int, width: int, points_per_side: int) -> np.ndarray:
    """Row-major grid at equal cell centers, in original zero-based (x, y).

    Coordinates span [0, width-1] / [0, height-1]; no out-of-bounds prompts
    occur even on a one-pixel image. Each point is an independent object query.
    """
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
           for v in (height, width, points_per_side)):
        raise ValueError("image dimensions and points_per_side must be positive integers")
    centers = (np.arange(points_per_side, dtype=np.float32) + 0.5) / points_per_side
    x, y = np.meshgrid(centers * (width - 1), centers * (height - 1))
    return np.stack((x.ravel(), y.ravel()), axis=1)


def collect_proposals(model, processor, rgb: np.ndarray, points_per_side: int,
                      synchronize=lambda: None) -> tuple[list[Proposal], dict]:
    """Use Sam3Processor.set_image then Sam3Image.predict_inst with cached state.

    Query one positive point at a time, never combine the grid as prompts for
    one object. Keep every returned candidate, including empty/low-score masks.
    Packed masks bound host memory; only one query's dense outputs are live.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an (H,W,3) uint8 RGB array")
    height, width = rgb.shape[:2]
    points = point_grid(height, width, points_per_side)
    synchronize()
    start = time.perf_counter()
    # PIL is required here: Meta's processor reads ndarray shape[-2:], which
    # would interpret an HWC RGB array's width/channel dimensions as H/W.
    inference_state = processor.set_image(Image.fromarray(rgb))
    if (not isinstance(inference_state, dict)
            or inference_state.get("original_height") != height
            or inference_state.get("original_width") != width
            or "sam2_backbone_out" not in inference_state.get("backbone_out", {})):
        raise RuntimeError("Meta processor did not return aligned interactive image features; "
                           "check installed SAM3 API and enable_inst_interactivity=True")
    synchronize()
    embedding_seconds = time.perf_counter() - start
    proposals = []
    query_seconds = 0.0
    for point_index, point in enumerate(points):
        synchronize()
        query_start = time.perf_counter()
        masks, scores, _low_resolution_logits = model.predict_inst(
            inference_state,
            point_coords=point[None, :], point_labels=np.array([1], dtype=np.int32),
            multimask_output=True, return_logits=False, normalize_coords=True,
        )
        synchronize()
        elapsed = time.perf_counter() - query_start
        query_seconds += elapsed
        if (not isinstance(masks, np.ndarray) or masks.ndim != 3
                or masks.shape[1:] != (height, width) or masks.shape[0] < 1):
            raise ValueError("Meta predictor must return nonempty (K,H,W) candidate arrays")
        if not np.all((masks == 0) | (masks == 1)):
            raise ValueError("Meta predictor must return binary masks, not logits")
        if not isinstance(scores, np.ndarray) or scores.shape != (len(masks),) or not np.isfinite(scores).all():
            raise ValueError("Meta predictor must return one finite quality score per candidate")
        for candidate_index, (mask, score) in enumerate(zip(masks, scores)):
            proposals.append(Proposal(np.packbits(mask.astype(bool)), (height, width), {
                "proposal_id": len(proposals), "grid_point_index": point_index,
                "candidate_index": candidate_index, "point_xy": point.astype(float).tolist(),
                "point_label": 1, "predicted_iou_score": float(score),
                "area_pixels": int(np.count_nonzero(mask)), "query_seconds": elapsed,
            }))
        # Do not retain dense low-resolution logits across queries.
        del masks, scores, _low_resolution_logits, mask
    return proposals, {
        "grid_point_count": len(points), "raw_proposal_count": len(proposals),
        "image_embedding_seconds": embedding_seconds, "point_query_seconds": query_seconds,
        "collection_seconds": time.perf_counter() - start,
    }


_BIT_COUNTS = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def packed_mask_iou(a: Proposal, b: Proposal) -> float:
    if a.shape != b.shape:
        raise ValueError("IoU candidates must have aligned masks")
    intersection = int(_BIT_COUNTS[np.bitwise_and(a.packed_mask, b.packed_mask)].sum())
    union = a.metadata["area_pixels"] + b.metadata["area_pixels"] - intersection
    return intersection / union if union else 0.0


def select_proposals(proposals: list[Proposal], *, min_score: float | None = None,
                     dedup_iou: float | None = 0.95, proposal_limit: int = 100):
    """Exact-mask greedy NMS by descending predicted IoU, then raw ID on ties.

    Empty masks are excluded only from retained visualization, not the archive.
    No area, bounding-box, stability, class, background or connectedness filter.
    Apply the retained limit AFTER deduplication and record every decision.
    """
    if min_score is not None and not np.isfinite(min_score):
        raise ValueError("min_score must be finite or None")
    if dedup_iou is not None and (not np.isfinite(dedup_iou) or not 0 < dedup_iou <= 1):
        raise ValueError("dedup_iou must be in (0,1] or None")
    if isinstance(proposal_limit, bool) or not isinstance(proposal_limit, int) or proposal_limit < 0:
        raise ValueError("proposal_limit must be a nonnegative integer")
    identifiers = [p.metadata["proposal_id"] for p in proposals]
    if identifiers != list(range(len(proposals))):
        raise ValueError("raw proposal IDs must be consecutive in collection order")
    rows = [deepcopy(p.metadata) for p in proposals]
    unique = []
    for proposal in sorted(proposals, key=lambda p: (-p.metadata["predicted_iou_score"], p.metadata["proposal_id"])):
        row = rows[proposal.metadata["proposal_id"]]
        row["duplicate_of"] = None
        row["duplicate_mask_iou"] = None
        if row["area_pixels"] == 0:
            row["retention_status"] = "empty_mask"
        elif min_score is not None and row["predicted_iou_score"] < min_score:
            row["retention_status"] = "score_filter"
        else:
            duplicate = None
            if dedup_iou is not None:
                for representative in unique:
                    overlap = packed_mask_iou(proposal, representative)
                    if overlap >= dedup_iou:
                        duplicate = (representative.metadata["proposal_id"], overlap)
                        break
            if duplicate is not None:
                row["retention_status"] = "duplicate"
                row["duplicate_of"], row["duplicate_mask_iou"] = duplicate
            else:
                unique.append(proposal)
                row["retention_status"] = "retained"
    retained = unique[:proposal_limit]
    for proposal in unique[proposal_limit:]:
        rows[proposal.metadata["proposal_id"]]["retention_status"] = "proposal_limit"
    counts = Counter(row["retention_status"] for row in rows)
    return retained, rows, {
        "raw_proposal_count": len(proposals), "retained_count": len(retained),
        "unique_before_limit_count": len(unique), "decision_counts": dict(sorted(counts.items())),
        "rules": {"min_score": min_score, "empty_masks_removed_from_visualization": True,
                  "deduplication": "disabled" if dedup_iou is None else "greedy_exact_binary_mask_IoU",
                  "dedup_iou": dedup_iou, "ranking": "predicted_iou_desc_then_raw_id_asc",
                  "proposal_limit": proposal_limit, "limit_applied_after_deduplication": True,
                  "area_filter": None, "stability_filter": None, "crop_layers": 0},
    }








