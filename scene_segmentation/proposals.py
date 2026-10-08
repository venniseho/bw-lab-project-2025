"""Automatic grid queries, explicit mask-IoU deduplication, and RGB previews."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import colorsys
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps


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


def collect_proposals(predictor, rgb: np.ndarray, points_per_side: int,
                      synchronize=lambda: None) -> tuple[list[Proposal], dict]:
    """Use Meta's set_image/predict API; cache one embedding per image.

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
    predictor.set_image(rgb)
    synchronize()
    embedding_seconds = time.perf_counter() - start
    proposals = []
    query_seconds = 0.0
    for point_index, point in enumerate(points):
        synchronize()
        query_start = time.perf_counter()
        masks, scores, _low_resolution_logits = predictor.predict(
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


def proposal_color(identifier: int) -> tuple[int, int, int]:
    return tuple(round(255 * channel) for channel in colorsys.hsv_to_rgb(
        (identifier * 0.618033988749895) % 1, 0.75, 0.95
    ))


def all_proposal_overlay(rgb: np.ndarray, retained: list[Proposal]) -> np.ndarray:
    """Average proposal colors at overlaps; preserve all overlapping masks.

    No exclusive instance assignment or later-mask-wins rendering is imposed.
    This overlay is an exploratory multi-label visualization, not a panoptic map.
    """
    colors = np.zeros(rgb.shape, dtype=np.float64)
    counts = np.zeros(rgb.shape[:2], dtype=np.uint32)
    for proposal in retained:
        mask = proposal.mask()
        colors[mask] += proposal_color(proposal.metadata["proposal_id"])
        counts[mask] += 1
    active = counts > 0
    overlay = rgb.copy()
    overlay[active] = np.rint(0.55 * rgb[active] + 0.45 * colors[active] / counts[active, None]).astype(np.uint8)
    return overlay


def contact_sheet(rgb: np.ndarray, overlay: np.ndarray, retained: list[Proposal]) -> Image.Image:
    columns, tile_width, tile_height = 4, 240, 180
    top_height = 320
    rows = max(1, (len(retained) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_width, top_height + rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (pixels, caption) in enumerate((
        (rgb, "Original RGB"), (overlay, f"Automatic visual proposals: {len(retained)} retained"),
    )):
        tile = ImageOps.contain(Image.fromarray(pixels), (2 * tile_width - 12, top_height - 40))
        x = column * 2 * tile_width
        draw.text((x + 6, 6), caption, fill="black")
        sheet.paste(tile, (x + (2 * tile_width - tile.width) // 2, 30))
    if not retained:
        draw.text((6, top_height + 6), "No retained proposals (see metadata for all decisions).", fill="black")
    for index, proposal in enumerate(retained):
        identifier = proposal.metadata["proposal_id"]
        x, y = (index % columns) * tile_width, top_height + (index // columns) * tile_height
        silhouette = np.zeros(rgb.shape, dtype=np.uint8)
        silhouette[proposal.mask()] = proposal_color(identifier)
        tile = ImageOps.contain(Image.fromarray(silhouette), (tile_width - 12, tile_height - 32),
                                method=Image.Resampling.NEAREST)
        draw.text((x + 6, y + 3), f"P{identifier:05d}  score={proposal.metadata['predicted_iou_score']:.3f}", fill="black")
        sheet.paste(tile, (x + (tile_width - tile.width) // 2, y + 25))
    return sheet


def save_scene(rgb: np.ndarray, proposals: list[Proposal], retained: list[Proposal],
               rows: list[dict], output_dir: Path, metadata: dict,
               save_candidate_limit: int | None = None) -> dict:
    """Archive all raw binary masks by default, even filtered/duplicate ones.

    Optional archive limit saves the first N raw masks; every candidate still
    has a ledger row. Retained masks are always saved, even outside that raw
    archive limit, so final proposals remain independently inspectable.
    """
    if save_candidate_limit is not None and (
        not isinstance(save_candidate_limit, int) or isinstance(save_candidate_limit, bool)
        or save_candidate_limit < 0
    ):
        raise ValueError("save_candidate_limit must be nonnegative or None")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "candidates").mkdir()
    rows = deepcopy(rows)
    retained_ids = {p.metadata["proposal_id"] for p in retained}
    raw_saved = retained_extra_saved = 0
    for index, proposal in enumerate(proposals):
        row = rows[index]
        save_raw = save_candidate_limit is None or index < save_candidate_limit
        row["raw_archive_status"] = "saved" if save_raw else "archive_limit"
        row["mask_file"] = None
        if save_raw or index in retained_ids:
            filename = f"candidates/proposal_{index:05d}.png"
            Image.fromarray(proposal.mask().astype(np.uint8) * 255).save(output_dir / filename)
            row["mask_file"] = filename
            raw_saved += int(save_raw)
            retained_extra_saved += int(not save_raw)
    overlay = all_proposal_overlay(rgb, retained)
    Image.fromarray(rgb).save(output_dir / "original.png")
    Image.fromarray(overlay).save(output_dir / "all_proposals_overlay.png")
    contact_sheet(rgb, overlay, retained).save(output_dir / "contact_sheet.png")
    report = deepcopy(metadata)
    report.update({
        "proposals": rows, "retained_proposal_ids": [p.metadata["proposal_id"] for p in retained],
        "raw_masks_saved_count": raw_saved, "retained_masks_saved_outside_archive_limit": retained_extra_saved,
        "raw_archive_limit": save_candidate_limit,
        "raw_archive_omitted_count": len(proposals) - raw_saved,
        "overlay_rule": "mean proposal color at overlaps; 45 percent RGB blend",
        "contact_sheet_panels": "original, retained overlay, every retained individual mask",
        "status": "complete",
    })
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
