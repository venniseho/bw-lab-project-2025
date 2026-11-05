"""SAM-inspired single-point mask supervision; not Meta's full training recipe."""

import torch
from torch.nn import functional as F
from core.samples import foreground_point


def candidate_ious(logits, target):
    masks, target = logits.detach() > 0, target.bool()
    intersection = (masks & target).sum((-2, -1)).float()
    union = (masks | target).sum((-2, -1)).float()
    return intersection / union.clamp_min(1)


def mask_loss(logits, scores, target):
    """Train the lowest-error mask and all three predicted-quality scores."""
    target = target.to(device=logits.device, dtype=torch.float32)
    truth = target.expand_as(logits)
    probabilities = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    p_correct = probabilities * truth + (1 - probabilities) * (1 - truth)
    alpha = 0.25 * truth + 0.75 * (1 - truth)
    focal = (alpha * (1 - p_correct).square() * bce).mean((-2, -1))
    dice = 1 - (2 * (probabilities * truth).sum((-2, -1)) + 1) / (
        probabilities.sum((-2, -1)) + truth.sum((-2, -1)) + 1
    )
    winner = (20 * focal + dice).argmin()
    quality = F.mse_loss(scores, candidate_ious(logits, target))
    total = 20 * focal[winner] + dice[winner] + quality
    return total, {
        "loss": float(total.detach()),
        "focal": float(focal[winner].detach()),
        "dice": float(dice[winner].detach()),
        "quality_mse": float(quality.detach()),
        "training_candidate": int(winner),
        "selected_iou": float(candidate_ious(logits, target)[scores.detach().argmax()]),
    }


def forward_sample(segmenter, sample):
    """Build the target prompt, then supervise one object's three candidates."""
    prompt_mask = sample.get("visible_segmentation", sample["segmentation"])
    point = foreground_point(prompt_mask)
    logits, scores = segmenter.predict(sample["image"], point)
    target = torch.as_tensor(sample["segmentation"], device=logits.device, dtype=torch.float32)
    loss, metrics = mask_loss(logits, scores, target)
    return loss, metrics, logits, scores, point
