"""Existing scene archive format, shared by the CLI and proposal evaluator."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from PIL import Image
from bwlab.models.proposals import Proposal
from bwlab.visualization import all_proposal_overlay, contact_sheet, prompt_overlay


def _save_candidate(rgb: np.ndarray, proposal: Proposal, row: dict, index: int,
                    output_dir: Path, retained: bool) -> None:
    """Write one archived mask/overlay and optionally its retained copy."""
    filename = f"candidates/proposal_{index:05d}.png"
    pixels = Image.fromarray(proposal.mask().astype(np.uint8)*255)
    pixels.save(output_dir / filename)
    row["mask_file"] = filename
    overlay_filename = f"candidates/proposal_{index:05d}_overlay.png"
    Image.fromarray(all_proposal_overlay(rgb, [proposal])).save(output_dir / overlay_filename)
    row["overlay_file"] = overlay_filename
    if retained:
        filename = f"retained/proposal_{index:05d}.png"
        pixels.save(output_dir / filename)
        row["retained_mask_file"] = filename


def _archive_masks(rgb: np.ndarray, proposals: list[Proposal], retained: list[Proposal],
                   rows: list[dict], output_dir: Path,
                   limit: int | None) -> tuple[list[dict], int, int]:
    """Apply the existing archive limit while always saving retained candidates."""
    rows = deepcopy(rows)
    retained_ids = {p.metadata["proposal_id"] for p in retained}
    raw_saved = extra_saved = 0
    for index, proposal in enumerate(proposals):
        raw = limit is None or index < limit
        keep = index in retained_ids
        row = rows[index]
        row.update(raw_archive_status="saved" if raw else "archive_limit",
                   mask_file=None, overlay_file=None, retained_mask_file=None)
        if raw or keep:
            _save_candidate(rgb, proposal, row, index, output_dir, keep)
            raw_saved += int(raw)
            extra_saved += int(not raw)
    return rows, raw_saved, extra_saved


def _save_previews(rgb: np.ndarray, proposals: list[Proposal], retained: list[Proposal],
                   output_dir: Path, metadata: dict) -> None:
    """Save unchanged data previews separately from annotated prompt displays."""
    overlay = all_proposal_overlay(rgb, retained)
    Image.fromarray(rgb).save(output_dir / "original.png")
    Image.fromarray(overlay).save(output_dir / "all_proposals_overlay.png")
    Image.fromarray(all_proposal_overlay(rgb, proposals)).save(output_dir / "raw_all_proposals_overlay.png")
    prompts = [p.metadata for p in proposals]
    prompt_overlay(Image.fromarray(rgb), prompts).save(output_dir / "prompt_locations.png")
    title = metadata.get("__key__") or metadata.get("condition")
    contact_sheet(rgb, overlay, retained, prompts=prompts, title=title).save(output_dir / "contact_sheet.png")


def _scene_report(metadata: dict, proposals: list[Proposal], retained: list[Proposal],
                  rows: list[dict], limit: int | None, raw_saved: int, extra_saved: int) -> dict:
    """Describe artifacts and selection decisions using the existing report schema."""
    report = deepcopy(metadata)
    report.update({
        "proposals": rows, "retained_proposal_ids": [p.metadata["proposal_id"] for p in retained],
        "raw_masks_saved_count": raw_saved, "retained_masks_saved_outside_archive_limit": extra_saved,
        "raw_archive_limit": limit, "raw_archive_omitted_count": len(proposals)-raw_saved,
        "overlay_rule": "mean proposal color at overlaps; 45 percent RGB blend",
        "all_proposals_overlay_set": "retained; raw candidates shown separately in raw_all_proposals_overlay.png",
        "contact_sheet_panels": "RGB and retained overlay with all prompt dots; individual RGB/mask overlays with their query dot",
        "prompt_locations_file": "prompt_locations.png",
        "prompt_marker_rule": "red dots; G# is zero-based grid_point_index; display-only",
        "status": "complete",
    })
    return report


def save_scene(rgb: np.ndarray, proposals: list[Proposal], retained: list[Proposal],
               rows: list[dict], output_dir: Path, metadata: dict,
               save_candidate_limit: int | None = None) -> dict:
    """Archive raw candidates and retain their original geometry and provenance.

    The optional archive limit omits raw files only; retained masks are always
    saved and every candidate keeps its ledger row. Input arrays are unchanged.
    """
    if save_candidate_limit is not None and (
        not isinstance(save_candidate_limit, int) or isinstance(save_candidate_limit, bool)
        or save_candidate_limit < 0
    ):
        raise ValueError("save_candidate_limit must be nonnegative or None")
    output_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("candidates", "retained"):
        (output_dir/folder).mkdir()
    rows, raw, extra = _archive_masks(rgb, proposals, retained, rows, output_dir, save_candidate_limit)
    _save_previews(rgb, proposals, retained, output_dir, metadata)
    report = _scene_report(metadata, proposals, retained, rows, save_candidate_limit, raw, extra)
    (output_dir/"metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report
