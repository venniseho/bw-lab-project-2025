"""Recompute a few matched source examples across all five conditions on a GPU.

These are new previews, not archived predictions from the original run. Both
the original CSV IoU and newly measured IoU are saved for comparison.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from bwlab.infrastructure import require_slurm_step, file_fingerprint
from bwlab.models import Sam3GridAdapter
from bwlab.models.batched_sam3 import BatchedSam3GridAdapter
from bwlab.models.sam3 import load_scene_model
from bwlab.runner import _predict_batch
from bwlab.runtime import configure_model
from bwlab.samples import binary_mask
from bwlab.visualization import prompt_overlay
from experiments.sam3_fragmentation.experiment import build_data_module, load_config
from scripts._equivalence_metrics import mask_iou


def read_rows(run):
    with (run / "predictions/image_results.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def overlay(rgb, mask, color):
    pixels = rgb.copy()
    pixels[mask] = np.rint(0.55*pixels[mask] + 0.45*np.array(color)).astype(np.uint8)
    return Image.fromarray(pixels)


def tile(image, title, prompts=()):
    original_size = image.size
    image = ImageOps.contain(image, (320, 240))
    image = prompt_overlay(image, list(prompts), source_size=original_size)
    panel = Image.new("RGB", (340, 285), "white")
    panel.paste(image, ((340-image.width)//2, 40+(240-image.height)//2))
    ImageDraw.Draw(panel).text((8, 7), title, font=ImageFont.load_default(size=14), fill="black")
    return panel


def config_for(run, output, arm):
    saved = json.loads((run / "config.json").read_text())
    config = saved["config"]
    if config["condition"] != arm:
        raise ValueError(f"Wrong {arm} run directory")
    if file_fingerprint(Path(config["manifest"]))["sha256"] != saved["input_manifest"]["sha256"]:
        raise ValueError("Source manifest changed")
    path = output / f"{arm}_config.json"
    path.write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    return load_config(path)


def collect(run, rows, sources, output, model, processor, info, arm):
    config = config_for(run, output, arm)
    original = json.loads((run / "timing.json").read_text())["model"]
    for field in ("checkpoint", "builder_source", "predictor_source", "processor_source", "image_model_source"):
        if info[field]["sha256"] != original[field]["sha256"]:
            raise ValueError(f"Native model changed: {field}")
    if config.runtime:
        configure_model(model, config.runtime)
    adapter = (BatchedSam3GridAdapter(model, processor, config.points_per_side,
                                    config.runtime.prompt_batch_size, config.runtime.pin_memory,
                                    torch.cuda.synchronize)
               if config.runtime and config.runtime.batched else
               Sam3GridAdapter(model, processor, config.points_per_side, torch.cuda.synchronize))
    module = build_data_module(config)
    dataset = module.val_dataset() if config.subset == "val" else module.test_dataset()
    wanted = {int(row["input_index"]): row for row in rows if row["source_key"] in sources}
    panels, records = {}, []
    for group in sorted({index//config.batch_size for index in wanted}):
        start = group*config.batch_size
        batch = [dataset[index] for index in range(start, min(start+config.batch_size, len(dataset)))]
        predictions = _predict_batch(adapter, batch)
        for offset, (sample, (proposals, _)) in enumerate(zip(batch, predictions)):
            row = wanted.get(start+offset)
            if row is None:
                continue
            if sample["__key__"] != row["__key__"]:
                raise ValueError("Sample identity changed")
            target = binary_mask(sample["segmentation"], "target")
            for field, pixels in (("input_sha256", sample["image"]), ("target_sha256", target)):
                if hashlib.sha256(pixels.tobytes()).hexdigest() != row[field]:
                    raise ValueError(f"Sample pixels changed: {field}")
            if not proposals:
                raise ValueError("No proposals for selected example")
            chosen = min(proposals, key=lambda p: (-p.metadata["predicted_iou_score"], p.metadata["proposal_id"]))
            iou = mask_iou(chosen.mask(), target) if target.any() else None
            source, condition = row["source_key"], row["condition"]
            title = f"{condition} | IoU={iou:.3f}" if iou is not None else f"{condition} | empty target"
            rgb = sample["image"]
            panels[source, condition] = [
                tile(Image.fromarray(rgb), "Input + independent grid prompts", [p.metadata for p in proposals]),
                tile(overlay(rgb, chosen.mask(), (0, 220, 80)), "Chosen prediction (green)\n"+title, [chosen.metadata]),
                tile(overlay(rgb, target, (220, 0, 220)), "Labelled full target (magenta)")]
            records.append({"source": source, "condition": condition,
                            "saved_full_iou": row["full_iou"], "recomputed_full_iou": iou,
                            "saved_proposal_id": row["chosen_proposal_id"],
                            "recomputed_proposal_id": chosen.metadata["proposal_id"]})
            print(f"{source}/{condition}: saved IoU={row['full_iou']}; preview IoU={iou}", flush=True)
    return panels, records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fence-run", type=Path, required=True)
    parser.add_argument("--fragmentation-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    import os
    require_slurm_step(os.environ)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    fence, frag = read_rows(args.fence_run), read_rows(args.fragmentation_run)
    common = {row["source_key"] for row in frag}
    sources = list(dict.fromkeys(row["source_key"] for row in fence
                                if row["condition"] == "clean" and row["source_key"] in common))[:args.count]
    if args.count < 1 or len(sources) != args.count:
        raise ValueError("Need a positive count with enough matched sources")
    args.output.mkdir(parents=True, exist_ok=False)
    config = config_for(args.fence_run, args.output, "fence")
    torch.manual_seed(0)
    model, processor, info = load_scene_model(config.checkpoint, config.bpe_path, torch)
    panels, records = {}, []
    for arm, run, rows in (("fence", args.fence_run, fence), ("fragmentation", args.fragmentation_run, frag)):
        current, evidence = collect(run, rows, sources, args.output, model, processor, info, arm)
        panels.update(current)
        records.extend(evidence)
    conditions = ("clean", "light", "medium", "heavy", "fragmentation")
    for number, source in enumerate(sources):
        sheet = Image.new("RGB", (340*5, 285*3+55), "white")
        ImageDraw.Draw(sheet).text((10, 8), f"{source} | newly recomputed previews\nFirst matched sources; masks selected by model score, not target IoU.", fill="black")
        for column, condition in enumerate(conditions):
            for row, panel in enumerate(panels[source, condition]):
                sheet.paste(panel, (column*340, 55+row*285))
        sheet.save(args.output / f"source_{number:02d}_conditions.png")
    (args.output / "examples.json").write_text(json.dumps({
        "selection": "first matched sources in original CSV order, not ranked by accuracy",
        "note": "Recomputed using current adapter; saved and recomputed IoU recorded separately",
        "model": info, "records": records}, indent=2)+"\n", encoding="utf-8")
    print(f"Contact sheets: {args.output}", flush=True)


if __name__ == "__main__":
    main()
