"""SAM-inspired single-point mask supervision; not Meta's full training recipe."""

import hashlib

import numpy as np
import torch
from torch.nn import functional as F


def random_target_point(mask, key, seed=42):
    """Sample uniformly inside the target; keep each seeded instance reproducible."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("No visible foreground pixel available for a positive prompt")
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    index = rng.integers(len(xs))
    return [float(xs[index]), float(ys[index])]


def candidate_ious(logits, target, valid_pixels=None):
    masks, target = logits.detach() > 0, target.bool()
    if valid_pixels is not None:
        valid = torch.as_tensor(valid_pixels, device=logits.device, dtype=torch.bool)
        masks, target = masks & valid, target & valid
    intersection = (masks & target).sum((-2, -1)).float()
    union = (masks | target).sum((-2, -1)).float()
    return intersection / union.clamp_min(1)


def mask_loss(logits, scores, target, valid_pixels=None):
    """Train the lowest-error mask and all three predicted-quality scores."""
    target = target.to(device=logits.device, dtype=torch.float32)
    truth = target.expand_as(logits)
    probabilities = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    p_correct = probabilities * truth + (1 - probabilities) * (1 - truth)
    alpha = 0.25 * truth + 0.75 * (1 - truth)
    focal_pixels = alpha * (1 - p_correct).square() * bce
    valid = (
        torch.ones_like(target)
        if valid_pixels is None
        else torch.as_tensor(valid_pixels, device=logits.device, dtype=torch.float32)
    )
    if valid.shape != target.shape or not torch.any(valid):
        raise ValueError("Loss needs a nonempty valid region aligned with the target")
    focal = (focal_pixels * valid).sum((-2, -1)) / valid.sum()
    dice = 1 - (2 * (probabilities * truth * valid).sum((-2, -1)) + 1) / (
        (probabilities * valid).sum((-2, -1)) + (truth * valid).sum((-2, -1)) + 1
    )
    winner = (20 * focal + dice).argmin()
    quality = F.mse_loss(scores, candidate_ious(logits, target, valid_pixels))
    total = 20 * focal[winner] + dice[winner] + quality
    return total, {
        "loss": float(total.detach()),
        "focal": float(focal[winner].detach()),
        "dice": float(dice[winner].detach()),
        "quality_mse": float(quality.detach()),
        "training_candidate": int(winner),
        "selected_iou": float(
            candidate_ious(logits, target, valid_pixels)[scores.detach().argmax()]
        ),
    }


def forward_sample(segmenter, sample, seed=42, encoded=None, loss_region="full"):
    """Build the target prompt, then supervise one object's three candidates."""
    prompt_mask = sample.get("visible_segmentation", sample["segmentation"])
    if not np.any(prompt_mask):
        raise ValueError(
            f"Fully hidden target must be excluded before loading: {sample['__key__']}"
        )
    if loss_region not in ("full", "visible"):
        raise ValueError("loss_region must be full or visible")
    point = random_target_point(prompt_mask, sample["__key__"], seed)
    if encoded is None:
        logits, scores = segmenter.predict(sample["image"], point)
    else:
        logits, scores = segmenter.predict_encoded(encoded, point, sample["image"].shape[:2])
    target = torch.as_tensor(sample["segmentation"], device=logits.device, dtype=torch.float32)
    valid = ~torch.as_tensor(
        sample.get("occlusion_mask", np.zeros(target.shape, bool)),
        device=logits.device,
        dtype=torch.bool,
    )
    if loss_region == "visible":
        loss, visible_metrics = mask_loss(logits, scores, target, valid)
        metrics = visible_metrics
        with torch.no_grad():
            _, full_metrics = mask_loss(logits, scores, target)
    else:
        loss, full_metrics = mask_loss(logits, scores, target)
        metrics = full_metrics
        with torch.no_grad():
            _, visible_metrics = mask_loss(logits, scores, target, valid)
    metrics = dict(metrics)
    selected = int(scores.detach().argmax())
    full_ious = candidate_ious(logits, target)
    visible_ious = candidate_ious(logits, target, valid)
    metrics.update(
        selected_iou=float(full_ious[selected]),
        full_iou=float(full_ious[selected]),
        visible_iou=float(visible_ious[selected]),
        loss_region=loss_region,
        full_loss=full_metrics["loss"],
        visible_loss=visible_metrics["loss"],
    )
    for component in ("focal", "dice", "quality_mse"):
        metrics[f"full_{component}"] = full_metrics[component]
        metrics[f"visible_{component}"] = visible_metrics[component]
    metrics.update(selected_candidate=selected, point_x=point[0], point_y=point[1], point_seed=seed)
    for index, (score, iou) in enumerate(zip(scores.detach(), full_ious)):
        metrics[f"candidate_{index}_score"] = float(score)
        metrics[f"candidate_{index}_iou"] = float(iou)
        metrics[f"candidate_{index}_visible_iou"] = float(visible_ious[index])
    metrics["prompt_on_visible_target"] = bool(
        sample.get("visible_segmentation", sample["segmentation"])[int(point[1]), int(point[0])]
    )
    metrics["prompt_on_rendered_signal"] = bool(
        np.any(sample["image"][int(point[1]), int(point[0])])
    )
    return loss, metrics, logits, scores, point
