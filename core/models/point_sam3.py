"""Differentiable version of Meta's pinned interactive image path."""

from contextlib import nullcontext
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from core.models.batched_sam3 import _tensor_image
from core.samples import foreground_point


ENCODER = ("backbone.vision_backbone.trunk.", "backbone.vision_backbone.sam2_convs.")
PROMPT_ENCODER = ("inst_interactive_predictor.model.sam_prompt_encoder.",)
MASK_DECODER = ("inst_interactive_predictor.model.sam_mask_decoder.",)
DECODER = PROMPT_ENCODER + MASK_DECODER


def _autograd_mlp_forward(self, x):
    """Keep native inference, but use differentiable linear+activation in training."""
    if not torch.is_grad_enabled():
        return type(self).forward(self, x)
    x = self.act(self.fc1(x))
    x = self.norm(self.drop1(x))
    return self.drop2(self.fc2(x))


def enable_encoder_autograd(model):
    """Adapt the pinned ViT MLPs without changing weights or checkpoint keys."""
    for module in model.backbone.vision_backbone.trunk.modules():
        cls = type(module)
        if cls.__module__ == "sam3.model.vitdet" and cls.__name__ == "Mlp":
            module.forward = MethodType(_autograd_mlp_forward, module)


def set_scope(model, scope):
    """The full scope is the active point path, excluding text/video heads."""
    if scope not in ("full", "frozen_encoder", "mask_decoder_only"):
        raise ValueError(scope)
    prefixes = MASK_DECODER if scope == "mask_decoder_only" else DECODER
    prefixes += ENCODER if scope == "full" else ()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    groups = {
        group: sum(
            p.numel()
            for n, p in model.named_parameters()
            if p.requires_grad and n.startswith(prefix)
        )
        for group, prefix in (("encoder", ENCODER), ("decoder", DECODER))
    }
    if not groups["decoder"] or (scope == "full" and not groups["encoder"]):
        raise RuntimeError(f"Installed SAM3 does not match the expected point modules: {groups}")
    if scope == "mask_decoder_only":
        groups["prompt_encoder"] = 0
    return groups


def train_mode(model, scope, training):
    # Leave unused language, detector and video modules in evaluation mode.
    model.eval()
    tracker = model.inst_interactive_predictor.model
    tracker.sam_prompt_encoder.train(training and scope != "mask_decoder_only")
    tracker.sam_mask_decoder.train(training)
    model.backbone.training = training and scope == "full"
    model.backbone.vision_backbone.train(training and scope == "full")


def encode_point_image(model, processor, image, *, freeze_encoder):
    """Encode once, retaining decoder projection gradients when encoder is frozen."""
    tracker = model.inst_interactive_predictor.model
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
    maps = [feat.permute(1, 2, 0).reshape(1, -1, *size) for feat, size in zip(vision, sizes)]
    return maps


def decode_point(model, maps, point_xy, original_shape):
    """Decode an independent positive point using the current image's features."""
    predictor = model.inst_interactive_predictor
    tracker = predictor.model
    decoder = tracker.sam_mask_decoder
    point = torch.as_tensor(point_xy, device=maps[-1].device, dtype=torch.float32).reshape(1, 1, 2)
    point = predictor._transforms.transform_coords(point, normalize=True, orig_hw=original_shape)
    labels = torch.ones((1, 1), dtype=torch.int64, device=maps[-1].device)
    sparse, dense = tracker.sam_prompt_encoder(points=(point, labels), boxes=None, masks=None)
    logits, scores, _, _ = decoder(
        image_embeddings=maps[-1],
        image_pe=tracker.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
        repeat_image=False,
        high_res_features=maps[:-1],
    )
    logits = F.interpolate(
        logits.float(), size=original_shape, mode="bilinear", align_corners=False
    )
    return logits[0], scores[0].float()


def point_forward(model, processor, image, point_xy, *, freeze_encoder):
    """One image and point, with gradients through the requested model scope."""
    maps = encode_point_image(model, processor, image, freeze_encoder=freeze_encoder)
    return decode_point(model, maps, point_xy, image.shape[:2])


def gradient_probes(model, scope):
    """Check actual nonzero gradients in each required part before the first step."""
    result = {}
    if scope == "mask_decoder_only":
        for name, parameter in model.named_parameters():
            if not name.startswith(MASK_DECODER) and (
                parameter.requires_grad or parameter.grad is not None
            ):
                raise RuntimeError(f"Non-mask-decoder parameter participates in backward: {name}")
    for group, prefix in (("encoder", ENCODER), ("decoder", DECODER)):
        parameters = [(name, p) for name, p in model.named_parameters() if name.startswith(prefix)]
        active = [
            (name, p)
            for name, p in parameters
            if p.grad is not None and torch.count_nonzero(p.grad)
        ]
        if group == "encoder" and scope != "full":
            if active or any(p.requires_grad or p.grad is not None for _, p in parameters):
                raise RuntimeError("Frozen encoder unexpectedly participates in backward")
            continue
        if not active:
            raise RuntimeError(f"No nonzero {group} gradients")
        name, parameter = min(active, key=lambda item: item[1].numel())
        result[group] = (name, parameter, parameter.detach().clone())
    return result


def check_step(probes):
    result = {}
    for group, (name, parameter, before) in probes.items():
        delta = float((parameter.detach() - before).abs().max())
        if delta == 0:
            raise RuntimeError(f"Optimizer did not change the checked {group} weight: {name}")
        result[group] = {"parameter": name, "max_weight_change": delta, "nonzero_gradient": True}
    return result


@torch.no_grad()
def native_parity(model, processor, sample):
    """Compare the new forward with native prediction before any training."""
    from PIL import Image

    point = foreground_point(sample["segmentation"])
    logits, scores = point_forward(model, processor, sample["image"], point, freeze_encoder=True)
    state = processor.set_image(Image.fromarray(sample["image"]))
    reference, quality, _ = model.predict_inst(
        state,
        point_coords=np.array([point], dtype=np.float32),
        point_labels=np.array([1]),
        multimask_output=True,
        return_logits=True,
        normalize_coords=True,
    )
    reference = torch.as_tensor(reference, device=logits.device)
    if reference.shape != logits.shape:
        raise RuntimeError(f"Native/new mask shapes differ: {reference.shape}, {logits.shape}")
    left, right = logits > 0, reference > 0
    union = (left | right).sum((-2, -1))
    iou = torch.where(union > 0, (left & right).sum((-2, -1)) / union.clamp_min(1), 1.0)
    result = {
        "min_candidate_iou": float(iou.min()),
        "max_score_delta": float((scores.cpu() - torch.as_tensor(quality)).abs().max()),
    }
    if result["min_candidate_iou"] < 0.999 or result["max_score_delta"] > 0.001:
        raise RuntimeError(
            f"New forward does not closely match native single-point inference: {result}"
        )
    return result


class PointSegmenter:
    """Own native SAM3 point inference and its trainable/frozen module scope."""

    def __init__(self, model, processor, scope):
        self.model, self.processor, self.scope = model, processor, scope
        self.parameter_counts = set_scope(model, scope)
        if scope == "full":
            enable_encoder_autograd(model)
        model.backbone.act_ckpt_whole_vision_backbone = scope == "full"

    def predict(self, image, point):
        device_type = torch.device(self.processor.device).type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            return point_forward(
                self.model,
                self.processor,
                image,
                point,
                freeze_encoder=self.scope != "full",
            )

    def encode(self, image):
        device_type = torch.device(self.processor.device).type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            return encode_point_image(
                self.model, self.processor, image, freeze_encoder=self.scope != "full"
            )

    def predict_encoded(self, maps, point, original_shape):
        device_type = torch.device(self.processor.device).type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            return decode_point(self.model, maps, point, original_shape)

    def train(self, training=True):
        train_mode(self.model, self.scope, training)

    def gradient_probes(self):
        return gradient_probes(self.model, self.scope)

    def check_update(self, probes):
        return check_step(probes)
