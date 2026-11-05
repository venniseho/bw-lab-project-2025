"""Run a small genuine point-prompt fine-tuning loop on one allocated GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from bwlab.models.sam3 import load_scene_model, VERIFIED_SAM3_API_REVISION
from .data import dataset, foreground_point, loader
from .loss import mask_loss
from .model import DECODER, ENCODER, point_forward, set_scope, train_mode
from .output import append_json, contact_sheet


def check_native(config):
    revision = subprocess.check_output(
        ["git", "-C", config["native_root"], "rev-parse", "HEAD"], text=True).strip()
    if revision != config["native_revision"] or revision != VERIFIED_SAM3_API_REVISION:
        raise RuntimeError(f"Unexpected native revision: {revision}")
    import sam3
    if Path(sam3.__file__).resolve().parent.parent != Path(config["native_root"]).resolve():
        raise RuntimeError(f"Python imported a different SAM3 checkout: {sam3.__file__}")


def forward_sample(model, processor, sample, scope):
    prompt_mask = sample.get("visible_segmentation", sample["segmentation"])
    point = foreground_point(prompt_mask)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, scores = point_forward(model, processor, sample["image"], point,
                                       freeze_encoder=scope == "frozen_encoder")
    target = torch.as_tensor(sample["segmentation"], device=logits.device, dtype=torch.float32)
    loss, metrics = mask_loss(logits, scores, target)
    return loss, metrics, logits, scores, point


def gradient_probes(model, scope):
    """Check actual nonzero gradients in each required part before the first step."""
    result = {}
    for group, prefix in (("encoder", ENCODER), ("decoder", DECODER)):
        parameters = [(name, p) for name, p in model.named_parameters() if name.startswith(prefix)]
        active = [(name, p) for name, p in parameters if p.grad is not None and torch.count_nonzero(p.grad)]
        if group == "encoder" and scope == "frozen_encoder":
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
def evaluate(model, processor, samples, scope, output, epoch, baseline=None):
    train_mode(model, scope, False)
    rows, examples = [], {}
    for index, sample in enumerate(samples):
        _, metrics, logits, scores, point = forward_sample(model, processor, sample, scope)
        row = {"phase": "heldout", "epoch": epoch, "key": sample["__key__"], **metrics}
        rows.append(row)
        append_json(output / "heldout.jsonl", row)
        if index < 4:
            prediction = {"mask": (logits[scores.argmax()] > 0).cpu().numpy(), "iou": metrics["selected_iou"]}
            examples[sample["__key__"]] = prediction
            if baseline is not None:
                contact_sheet(output / "previews" / f"epoch-{epoch}-{index}.png", sample, point,
                              baseline[sample["__key__"]], prediction)
    summary = {"phase": "heldout_summary", "epoch": epoch, "instances": len(rows),
               "mean_iou": float(np.mean([row["selected_iou"] for row in rows])),
               "mean_loss": float(np.mean([row["loss"] for row in rows]))}
    append_json(output / "summary.jsonl", summary)
    print(json.dumps(summary), flush=True)
    return examples, summary


@torch.no_grad()
def native_parity(model, processor, sample):
    """Compare the new forward with native prediction before any training."""
    from PIL import Image
    point = foreground_point(sample["segmentation"])
    logits, scores = point_forward(model, processor, sample["image"], point, freeze_encoder=True)
    state = processor.set_image(Image.fromarray(sample["image"]))
    reference, quality, _ = model.predict_inst(
        state, point_coords=np.array([point], dtype=np.float32), point_labels=np.array([1]),
        multimask_output=True, return_logits=True, normalize_coords=True)
    reference = torch.as_tensor(reference, device=logits.device)
    if reference.shape != logits.shape:
        raise RuntimeError(f"Native/new mask shapes differ: {reference.shape}, {logits.shape}")
    left, right = logits > 0, reference > 0
    union = (left | right).sum((-2, -1))
    iou = torch.where(union > 0, (left & right).sum((-2, -1)) / union.clamp_min(1), 1.)
    result = {"min_candidate_iou": float(iou.min()),
              "max_score_delta": float((scores.cpu() - torch.as_tensor(quality)).abs().max())}
    if result["min_candidate_iou"] < .999 or result["max_score_delta"] > .001:
        raise RuntimeError(f"New forward does not closely match native single-point inference: {result}")
    return result


def checkpoint(model, optimizer, output, epoch, config, scope):
    path = output / "checkpoint.pt"
    temporary = output / "checkpoint.part"
    torch.save({"format": "bwlab_point_finetune_v1", "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "epoch": epoch, "config": config,
                "scope": scope}, temporary)
    temporary.replace(path)
    # Read back metadata, rather than treating an existing filename as success.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != "bwlab_point_finetune_v1" or saved["epoch"] != epoch:
        raise RuntimeError("Saved checkpoint is invalid")
    return {"path": str(path), "bytes": path.stat().st_size, "epoch": epoch}


def run(config, scope, output, wandb_mode="offline", max_steps=None):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Run training inside a Slurm GPU allocation, not the login node")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    check_native(config)
    if config["transform"] != "identity":
        raise ValueError("Only the clean-image smoke test is configured")
    for split, key in (("train", "train_annotations"), ("test", "heldout_annotations")):
        digest = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
        if digest != config["annotation_sha256"][split]:
            raise RuntimeError(f"Prepared {split} annotations changed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "previews").mkdir()
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    train_set, heldout_set = dataset(config["train_annotations"]), dataset(config["heldout_annotations"])
    if not len(train_set) or not len(heldout_set):
        raise ValueError("Both instance splits must be nonempty")
    if {r["image_id"] for r in train_set.records} & {r["image_id"] for r in heldout_set.records}:
        raise ValueError("Training and held-out source images overlap")
    train_data = loader(train_set, config["workers"])
    heldout_data = loader(heldout_set, config["workers"])
    model, processor, provenance = load_scene_model(config["checkpoint"], None, torch)
    groups = set_scope(model, scope)
    # Native non-reentrant checkpointing saves activation memory in the full run.
    model.backbone.act_ckpt_whole_vision_backbone = scope == "full"
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=.01)
    metadata = {"config": config, "scope": scope, "trainable_parameters": groups,
                "provenance": provenance, "gpu": torch.cuda.get_device_name(),
                "precision": "bf16 forward, float32 loss/parameters", "max_steps": max_steps}
    metadata["native_parity"] = native_parity(model, processor, heldout_set[0])
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tracking = None
    if wandb_mode != "disabled":
        import wandb
        tracking = wandb.init(project="bwlab-point-finetuning", name=output.name,
                              dir=str(output), config=metadata, mode=wandb_mode)
    print(json.dumps({"scope": scope, "parameters": groups, "train_instances": len(train_set),
                      "heldout_instances": len(heldout_set), "parity": metadata["native_parity"]}), flush=True)
    start = time.perf_counter()
    baseline, summary = evaluate(model, processor, heldout_data, scope, output, 0)
    if tracking:
        tracking.log({"heldout/mean_iou": summary["mean_iou"], "epoch": 0})
    steps, audit = 0, {}
    try:
        for epoch in range(1, config["epochs"] + 1):
            train_mode(model, scope, True)
            for sample in train_data:
                optimizer.zero_grad(set_to_none=True)
                loss, metrics, logits, scores, _ = forward_sample(model, processor, sample, scope)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                loss.backward()
                probes = gradient_probes(model, scope) if steps == 0 else None
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                if probes is not None:
                    audit = check_step(probes)
                    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
                steps += 1
                record = {"phase": "train", "epoch": epoch, "step": steps,
                          "key": sample["__key__"], "gradient_norm": float(norm), **metrics}
                append_json(output / "train.jsonl", record)
                print(json.dumps(record), flush=True)
                if tracking:
                    tracking.log({f"train/{key}": value for key, value in metrics.items()} | {"step": steps})
                del loss, logits, scores
                if max_steps is not None and steps >= max_steps:
                    break
            _, summary = evaluate(model, processor, heldout_data, scope, output, epoch, baseline)
            saved = checkpoint(model, optimizer, output, epoch, config, scope)
            if tracking:
                tracking.log({"heldout/mean_iou": summary["mean_iou"], "epoch": epoch})
            if max_steps is not None and steps >= max_steps:
                break
        summary = {"steps": steps, "checkpoint": saved, "gradient_and_step_audit": audit,
                   "wall_seconds": time.perf_counter() - start,
                   "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
        (output / "complete.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        if tracking:
            tracking.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", choices=("full", "frozen_encoder"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.scope, args.output, args.wandb_mode, args.max_steps)


if __name__ == "__main__":
    main()
