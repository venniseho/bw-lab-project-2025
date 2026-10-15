"""Existing scene archive format, shared by the CLI and proposal evaluator."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from PIL import Image
from bwlab.models.proposals import Proposal
from bwlab.visualization import all_proposal_overlay, contact_sheet


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
    (output_dir / "retained").mkdir()
    rows = deepcopy(rows)
    retained_ids = {p.metadata["proposal_id"] for p in retained}
    raw_saved = retained_extra_saved = 0
    for index, proposal in enumerate(proposals):
        row = rows[index]
        save_raw = save_candidate_limit is None or index < save_candidate_limit
        row["raw_archive_status"] = "saved" if save_raw else "archive_limit"
        row["mask_file"] = None
        row["overlay_file"] = None
        row["retained_mask_file"] = None
        if save_raw or index in retained_ids:
            filename = f"candidates/proposal_{index:05d}.png"
            Image.fromarray(proposal.mask().astype(np.uint8) * 255).save(output_dir / filename)
            row["mask_file"] = filename
            overlay_filename = f"candidates/proposal_{index:05d}_overlay.png"
            Image.fromarray(all_proposal_overlay(rgb, [proposal])).save(output_dir / overlay_filename)
            row["overlay_file"] = overlay_filename
            if index in retained_ids:
                retained_filename = f"retained/proposal_{index:05d}.png"
                Image.fromarray(proposal.mask().astype(np.uint8) * 255).save(output_dir / retained_filename)
                row["retained_mask_file"] = retained_filename
            raw_saved += int(save_raw)
            retained_extra_saved += int(not save_raw)
    overlay = all_proposal_overlay(rgb, retained)
    Image.fromarray(rgb).save(output_dir / "original.png")
    Image.fromarray(overlay).save(output_dir / "all_proposals_overlay.png")
    Image.fromarray(all_proposal_overlay(rgb, proposals)).save(output_dir / "raw_all_proposals_overlay.png")
    contact_sheet(rgb, overlay, retained).save(output_dir / "contact_sheet.png")
    report = deepcopy(metadata)
    report.update({
        "proposals": rows, "retained_proposal_ids": [p.metadata["proposal_id"] for p in retained],
        "raw_masks_saved_count": raw_saved, "retained_masks_saved_outside_archive_limit": retained_extra_saved,
        "raw_archive_limit": save_candidate_limit,
        "raw_archive_omitted_count": len(proposals) - raw_saved,
        "overlay_rule": "mean proposal color at overlaps; 45 percent RGB blend",
        "all_proposals_overlay_set": "retained; raw candidates shown separately in raw_all_proposals_overlay.png",
        "contact_sheet_panels": "original, retained overlay, every retained individual mask",
        "status": "complete",
    })
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
