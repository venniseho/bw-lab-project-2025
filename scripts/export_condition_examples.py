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

from core.infrastructure import require_slurm_step, file_fingerprint
from core.models import Sam3GridAdapter
from core.models.batched_sam3 import BatchedSam3GridAdapter
from core.models.sam3 import load_scene_model
from core.runner import _predict_batch
from core.runtime import configure_model
from core.samples import binary_mask
from core.visualization import prompt_overlay
from experiments.sam3_fragmentation.experiment import build_data_module, load_config
from core.tasks.equivalence import mask_iou


def read_rows(run):
    with (run / "predictions/image_results.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def overlay(rgb, mask, color):
    pixels = rgb.copy()
    pixels[mask] = np.rint(0.55 * pixels[mask] + 0.45 * np.array(color)).astype(np.uint8)
    return Image.fromarray(pixels)


def tile(image, title, prompts=()):
    original_size = image.size
    image = ImageOps.contain(image, (320, 240))
    image = prompt_overlay(image, list(prompts), source_size=original_size)
    panel = Image.new("RGB", (340, 285), "white")
    panel.paste(image, ((340 - image.width) // 2, 40 + (240 - image.height) // 2))
    ImageDraw.Draw(panel).text((8, 7), title, font=ImageFont.load_default(size=14), fill="black")
    return panel


def save_point_details(rgb, target, proposals, chosen, source, condition, output, number):
    """Nine prompt rows, each with its input dot and all three candidates."""
    groups = {}
    for proposal in proposals:
        groups.setdefault(proposal.metadata["grid_point_index"], []).append(proposal)
    for candidates in groups.values():
        if sorted(p.metadata["candidate_index"] for p in candidates) != [0, 1, 2]:
            raise ValueError("Expected all three candidates per prompt")
        if any(p.metadata["point_xy"] != candidates[0].metadata["point_xy"] for p in candidates):
            raise ValueError("Candidates from one query must have the same prompt")
    sheet = Image.new("RGB", (340 * 4, 65 + 285 * (len(groups) + 1)), "white")
    ImageDraw.Draw(sheet).text(
        (10, 8),
        f"{source} | {condition} | NEWLY RECOMPUTED\n"
        "Each row is one independent positive dot. Columns: input, candidate 0, candidate 1, candidate 2.\n"
        "Green = candidate mask; gold border = globally score-selected mask. IoU uses labelled full target.",
        fill="black",
    )
    header = [
        tile(
            Image.fromarray(rgb),
            "All independent prompt locations",
            [p.metadata for p in proposals],
        ),
        tile(overlay(rgb, target, (220, 0, 220)), "Labelled target (magenta)"),
        tile(
            overlay(rgb, chosen.mask(), (0, 220, 80)),
            "Globally score-selected prediction",
            [chosen.metadata],
        ),
    ]
    for column, panel in enumerate(header):
        sheet.paste(panel, (column * 340, 65))
    rows = []
    for position, (grid, candidates) in enumerate(sorted(groups.items())):
        metadata = candidates[0].metadata
        x, y = np.rint(metadata["point_xy"]).astype(int)
        inside = (
            bool(target[y, x]) if 0 <= y < target.shape[0] and 0 <= x < target.shape[1] else False
        )
        panel = tile(
            Image.fromarray(rgb),
            f"G{grid}: independent positive dot\nDot inside full target: {inside}",
            [metadata],
        )
        top = 65 + (position + 1) * 285
        sheet.paste(panel, (0, top))
        for candidate in sorted(candidates, key=lambda p: p.metadata["candidate_index"]):
            meta = candidate.metadata
            mask = candidate.mask()
            iou = mask_iou(mask, target) if target.any() else None
            selected = meta["proposal_id"] == chosen.metadata["proposal_id"]
            label = f"G{grid} C{meta['candidate_index']} | score={meta['predicted_iou_score']:.4f}"
            label += f"\nIoU={iou:.4f}" if iou is not None else "\nIoU undefined: empty target"
            if selected:
                label += " | SELECTED"
            panel = tile(overlay(rgb, mask, (0, 220, 80)), label, [meta])
            if selected:
                ImageDraw.Draw(panel).rectangle((1, 1, 338, 283), outline="#D79B00", width=5)
            sheet.paste(panel, ((meta["candidate_index"] + 1) * 340, top))
            rows.append(
                {
                    **meta,
                    "measured_full_iou": iou,
                    "globally_selected": selected,
                    "point_inside_full_target": inside,
                    "area_pixels": int(mask.sum()),
                }
            )
    folder = output / "all_points"
    folder.mkdir(exist_ok=True)
    stem = f"source_{number:02d}_{condition}"
    sheet.save(folder / f"{stem}.png")
    (folder / f"{stem}.json").write_text(
        json.dumps(
            {
                "source": source,
                "condition": condition,
                "note": "Newly recomputed candidates, no candidate filtering",
                "candidates": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"All-dot sheet: {folder / (stem + '.png')}", flush=True)


def config_for(run, output, arm):
    saved = json.loads((run / "config.json").read_text())
    config = saved["config"]
    if config["condition"] != arm:
        raise ValueError(f"Wrong {arm} run directory")
    if file_fingerprint(Path(config["manifest"]))["sha256"] != saved["input_manifest"]["sha256"]:
        raise ValueError("Source manifest changed")
    path = output / f"{arm}_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return load_config(path)


def collect(run, rows, sources, output, model, processor, info, arm, all_points=False):
    config = config_for(run, output, arm)
    original = json.loads((run / "timing.json").read_text())["model"]
    for field in (
        "checkpoint",
        "builder_source",
        "predictor_source",
        "processor_source",
        "image_model_source",
    ):
        if info[field]["sha256"] != original[field]["sha256"]:
            raise ValueError(f"Native model changed: {field}")
    if config.runtime:
        configure_model(model, config.runtime)
    adapter = (
        BatchedSam3GridAdapter(
            model,
            processor,
            config.points_per_side,
            config.runtime.prompt_batch_size,
            config.runtime.pin_memory,
            torch.cuda.synchronize,
        )
        if config.runtime and config.runtime.batched
        else Sam3GridAdapter(model, processor, config.points_per_side, torch.cuda.synchronize)
    )
    module = build_data_module(config)
    dataset = module.val_dataset() if config.subset == "val" else module.test_dataset()
    wanted = {int(row["input_index"]): row for row in rows if row["source_key"] in sources}
    panels, records = {}, []
    for group in sorted({index // config.batch_size for index in wanted}):
        start = group * config.batch_size
        batch = [
            dataset[index] for index in range(start, min(start + config.batch_size, len(dataset)))
        ]
        predictions = _predict_batch(adapter, batch)
        for offset, (sample, (proposals, _)) in enumerate(zip(batch, predictions)):
            row = wanted.get(start + offset)
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
            chosen = min(
                proposals,
                key=lambda p: (-p.metadata["predicted_iou_score"], p.metadata["proposal_id"]),
            )
            iou = mask_iou(chosen.mask(), target) if target.any() else None
            source, condition = row["source_key"], row["condition"]
            title = (
                f"{condition} | IoU={iou:.3f}" if iou is not None else f"{condition} | empty target"
            )
            rgb = sample["image"]
            if all_points:
                save_point_details(
                    rgb, target, proposals, chosen, source, condition, output, sources.index(source)
                )
            panels[source, condition] = [
                tile(
                    Image.fromarray(rgb),
                    "Input + independent grid prompts",
                    [p.metadata for p in proposals],
                ),
                tile(
                    overlay(rgb, chosen.mask(), (0, 220, 80)),
                    "Chosen prediction (green)\n" + title,
                    [chosen.metadata],
                ),
                tile(overlay(rgb, target, (220, 0, 220)), "Labelled full target (magenta)"),
            ]
            records.append(
                {
                    "source": source,
                    "condition": condition,
                    "saved_full_iou": row["full_iou"],
                    "recomputed_full_iou": iou,
                    "saved_proposal_id": row["chosen_proposal_id"],
                    "recomputed_proposal_id": chosen.metadata["proposal_id"],
                }
            )
            print(
                f"{source}/{condition}: saved IoU={row['full_iou']}; preview IoU={iou}", flush=True
            )
    return panels, records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fence-run", type=Path, required=True)
    parser.add_argument("--fragmentation-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--all-points", action="store_true", help="Export every mask candidate for every prompt"
    )
    args = parser.parse_args()
    import os

    require_slurm_step(os.environ)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    fence, frag = read_rows(args.fence_run), read_rows(args.fragmentation_run)
    common = {row["source_key"] for row in frag}
    sources = list(
        dict.fromkeys(
            row["source_key"]
            for row in fence
            if row["condition"] == "clean" and row["source_key"] in common
        )
    )[: args.count]
    if args.count < 1 or len(sources) != args.count:
        raise ValueError("Need a positive count with enough matched sources")
    args.output.mkdir(parents=True, exist_ok=False)
    config = config_for(args.fence_run, args.output, "fence")
    torch.manual_seed(0)
    model, processor, info = load_scene_model(config.checkpoint, config.bpe_path, torch)
    panels, records = {}, []
    for arm, run, rows in (
        ("fence", args.fence_run, fence),
        ("fragmentation", args.fragmentation_run, frag),
    ):
        current, evidence = collect(
            run, rows, sources, args.output, model, processor, info, arm, args.all_points
        )
        panels.update(current)
        records.extend(evidence)
    conditions = ("clean", "light", "medium", "heavy", "fragmentation")
    for number, source in enumerate(sources):
        sheet = Image.new("RGB", (340 * 5, 285 * 3 + 55), "white")
        ImageDraw.Draw(sheet).text(
            (10, 8),
            f"{source} | newly recomputed previews\nFirst matched sources; masks selected by model score, not target IoU.",
            fill="black",
        )
        for column, condition in enumerate(conditions):
            for row, panel in enumerate(panels[source, condition]):
                sheet.paste(panel, (column * 340, 55 + row * 285))
        sheet.save(args.output / f"source_{number:02d}_conditions.png")
    (args.output / "examples.json").write_text(
        json.dumps(
            {
                "selection": "first matched sources in original CSV order, not ranked by accuracy",
                "note": "Recomputed using current adapter; saved and recomputed IoU recorded separately",
                "model": info,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Contact sheets: {args.output}", flush=True)


if __name__ == "__main__":
    main()
