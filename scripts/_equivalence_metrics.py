"""Candidate correspondence and numerical drift diagnostics; no acceptance relaxation."""
import numpy as np

from bwlab.samples import binary_mask


def mask_iou(a, b):
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b)/union) if union else 1.0


def comparison_rows(key, reference, actual, stage, score_atol, target):
    if len(reference) != len(actual):
        raise RuntimeError(f"candidate count changed in {stage}")
    target = binary_mask(target, "target")
    for old, new in zip(reference, actual):
        for field in ("proposal_id", "grid_point_index", "candidate_index", "point_xy"):
            if old.metadata[field] != new.metadata[field]:
                raise RuntimeError(f"prompt/candidate correspondence changed: {field} in {stage}")
        if old.shape != new.shape or old.shape != target.shape:
            raise RuntimeError(f"mask dimensions changed in {stage}")
        a, b = old.mask(), new.mask()
        different = int(np.count_nonzero(a != b))
        score_delta = abs(old.metadata["predicted_iou_score"]-new.metadata["predicted_iou_score"])
        old_iou = mask_iou(a, target) if target.any() else None
        new_iou = mask_iou(b, target) if target.any() else None
        yield {"stage": stage, "__key__": key, "proposal_id": old.metadata["proposal_id"],
               "different_pixels": different, "different_fraction": different/a.size,
               "reference_area": int(a.sum()), "actual_area": int(b.sum()),
               "mask_iou": mask_iou(a, b), "score_difference": score_delta,
               "reference_target_iou": old_iou, "actual_target_iou": new_iou,
               "target_iou_difference": abs(old_iou-new_iou) if old_iou is not None else None,
               "strict_mismatch": different > 0 or score_delta > score_atol}


def summarize(rows, reference, actual):
    def choice(proposals):
        return min(proposals, key=lambda p: (-p.metadata["predicted_iou_score"], p.metadata["proposal_id"])).metadata["proposal_id"]
    return {"candidate_count": len(rows),
            "strict_mismatch_count": sum(row["strict_mismatch"] for row in rows),
            "max_different_pixels": max(row["different_pixels"] for row in rows),
            "max_different_fraction": max(row["different_fraction"] for row in rows),
            "min_mask_iou": min(row["mask_iou"] for row in rows),
            "median_mask_iou": float(np.median([row["mask_iou"] for row in rows])),
            "max_score_difference": max(row["score_difference"] for row in rows),
            "max_target_iou_difference": max((row["target_iou_difference"] for row in rows
                                              if row["target_iou_difference"] is not None), default=None),
            "selected_candidate_changes": sum(choice(old) != choice(new)
                                               for old, new in zip(reference, actual))}
