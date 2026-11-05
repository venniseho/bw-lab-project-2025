"""Explicit trainable-parameter selection for native concept segmentation."""
import json
from pathlib import Path


def add_eval_matching(model, inputs, outputs):
    """Native image forward computes assignment only in train mode.

    Add the same native assignment in eval mode so a held-out mask loss can
    be computed, without changing prediction tensors or enabling gradients.
    """
    if not model.training:
        for stage, target in zip(outputs.output, inputs[0].find_targets):
            for prediction in stage:
                if "indices" not in prediction:
                    model._compute_matching(prediction, model.back_convert(target))


def set_trainable_scope(model, scope):
    if scope not in ("mask_head", "full", "encoder"):
        raise ValueError("Unknown training scope")
    names = []
    for name, parameter in model.named_parameters():
        trainable = (scope == "full" or
                     (scope == "mask_head" and name.startswith("segmentation_head.")) or
                     (scope == "encoder" and name.startswith("backbone.vision_backbone.")))
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    if not names:
        raise ValueError(f"No trainable parameters matched scope {scope}")
    return {"scope": scope, "trainable_names": names,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "interactive_predictor_enabled": False}


def build_training_model(checkpoint_path, bpe_path, scope, audit_output):
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path, bpe_path=bpe_path, load_from_HF=False,
        device="cpu", eval_mode=False, enable_segmentation=True,
        enable_inst_interactivity=False, compile=False)
    report = set_trainable_scope(model, scope)
    model.register_forward_hook(add_eval_matching)
    (Path(audit_output) / "trainable_parameters.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Training scope: {scope}; {report['trainable_parameters']:,} / "
          f"{report['total_parameters']:,} parameters trainable", flush=True)
    return model
