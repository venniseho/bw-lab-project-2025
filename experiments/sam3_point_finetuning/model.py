"""Differentiable version of Meta's pinned interactive image path."""
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from bwlab.models.batched_sam3 import _tensor_image


ENCODER = ("backbone.vision_backbone.trunk.", "backbone.vision_backbone.sam2_convs.")
DECODER = ("inst_interactive_predictor.model.sam_prompt_encoder.",
           "inst_interactive_predictor.model.sam_mask_decoder.")


def set_scope(model, scope):
    """The full scope is the active point path, excluding text/video heads."""
    if scope not in ("full", "frozen_encoder"):
        raise ValueError(scope)
    prefixes = DECODER + (ENCODER if scope == "full" else ())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    groups = {group: sum(p.numel() for n, p in model.named_parameters()
                         if p.requires_grad and n.startswith(prefix))
              for group, prefix in (("encoder", ENCODER), ("decoder", DECODER))}
    if not groups["decoder"] or (scope == "full" and not groups["encoder"]):
        raise RuntimeError(f"Installed SAM3 does not match the expected point modules: {groups}")
    return groups


def train_mode(model, scope, training):
    # Leave unused language, detector and video modules in evaluation mode.
    model.eval()
    tracker = model.inst_interactive_predictor.model
    tracker.sam_prompt_encoder.train(training)
    tracker.sam_mask_decoder.train(training)
    model.backbone.training = training and scope == "full"
    model.backbone.vision_backbone.train(training and scope == "full")


def point_forward(model, processor, image, point_xy, *, freeze_encoder):
    """One image, one positive point, three differentiable candidate masks.

    Do not use set_image/predict_inst here: those convenience methods disable
    autograd. Decoder feature projections must remain outside no_grad.
    """
    predictor = model.inst_interactive_predictor
    tracker = predictor.model
    pixels = _tensor_image(processor, image).unsqueeze(0)
    with torch.no_grad() if freeze_encoder else nullcontext():
        features = model.backbone.forward_image(pixels)["sam2_backbone_out"]
    if features is None:
        raise RuntimeError("Interactive image features are unavailable")
    pyramid = features["backbone_fpn"]
    decoder = tracker.sam_mask_decoder
    pyramid[0] = decoder.conv_s0(pyramid[0])
    pyramid[1] = decoder.conv_s1(pyramid[1])
    _, vision, _, sizes = tracker._prepare_backbone_features(features)
    vision[-1] = vision[-1] + tracker.no_mem_embed
    maps = [feat.permute(1, 2, 0).reshape(1, -1, *size)
            for feat, size in zip(vision, sizes)]
    point = torch.as_tensor(point_xy, device=pixels.device, dtype=torch.float32).reshape(1, 1, 2)
    point = predictor._transforms.transform_coords(point, normalize=True, orig_hw=image.shape[:2])
    labels = torch.ones((1, 1), dtype=torch.int64, device=pixels.device)
    sparse, dense = tracker.sam_prompt_encoder(points=(point, labels), boxes=None, masks=None)
    logits, scores, _, _ = decoder(
        image_embeddings=maps[-1], image_pe=tracker.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=True, repeat_image=False, high_res_features=maps[:-1])
    logits = F.interpolate(logits.float(), size=image.shape[:2], mode="bilinear", align_corners=False)
    return logits[0], scores[0].float()
