"""Meta SAM3 visual point-grid proposals. No text queries or downloads."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scene_segmentation.proposals import collect_proposals, select_proposals, save_scene
from scripts.sam3_smoke_test import load_rgb, file_fingerprint, require_slurm_step


def checked_state(model, state):
    """Same native checkpoint prefixes as Meta's loader, but reject partial loads."""
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("Expected a native SAM3 checkpoint state dictionary")
    mapped = {}
    for key, value in state.items():
        if key.startswith("detector."):
            mapped[key.removeprefix("detector.")] = value
        elif key.startswith("tracker."):
            mapped["inst_interactive_predictor.model." + key.removeprefix("tracker.")] = value
    if not mapped:
        raise ValueError("Not a native Meta SAM3 detector/tracker checkpoint")
    result = model.load_state_dict(mapped, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Incomplete checkpoint: {result.missing_keys}")
    return list(result.unexpected_keys)


def load_predictor(checkpoint, bpe_path, torch):
    import sam3
    import sam3.model_builder as builder
    import sam3.model.sam1_task_predictor as visual
    bpe = bpe_path or Path(sam3.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz"
    if not Path(bpe).is_file():
        raise FileNotFoundError(f"Local tokenizer asset missing: {bpe}")
    model = builder.build_sam3_image_model(
        device="cpu", bpe_path=str(bpe), checkpoint_path=None,
        load_from_HF=False, enable_inst_interactivity=True, compile=False,
    )
    unexpected = checked_state(model, torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device="cuda", dtype=torch.float32).eval()
    # Meta's model_builder enables TF32 during import. Override after importing it.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predictor = visual.SAM3InteractiveImagePredictor(
        model.inst_interactive_predictor.model, mask_threshold=0.0,
        max_hole_area=0.0, max_sprinkle_area=0.0,
    ).eval()
    return predictor, {
        "checkpoint": file_fingerprint(Path(checkpoint)),
        "builder_source": file_fingerprint(Path(builder.__file__)),
        "predictor_source": file_fingerprint(Path(visual.__file__)),
        "tokenizer": file_fingerprint(Path(bpe)),
        "unexpected_checkpoint_keys": unexpected,
        "api": "Meta SAM3InteractiveImagePredictor.set_image/predict",
        "model_family": "SAM3 (not SAM3.1)", "text_queries": False,
        "mask_threshold": 0.0, "max_hole_area": 0.0, "max_sprinkle_area": 0.0,
    }


def image_paths(args):
    if args.image:
        paths = [args.image]
    else:
        paths = sorted((p for p in args.image_dir.iterdir() if p.is_file()
                        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}),
                       key=lambda p: p.name.casefold())
    if not paths or args.limit_images < 1:
        raise ValueError("Need at least one image and a positive --limit-images")
    return paths[:args.limit_images], len(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--image-dir", type=Path)
    parser.add_argument("--limit-images", type=int, default=1)
    parser.add_argument("--list-images", action="store_true", help="CPU-only input check")
    parser.add_argument("--checkpoint", type=Path, help="Local native Meta sam3.pt, not HF safetensors")
    parser.add_argument("--bpe-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--points-per-side", type=int, default=4)
    parser.add_argument("--proposal-limit", type=int, default=100)
    parser.add_argument("--save-candidate-limit", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--dedup-iou", type=float, default=0.95)
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args(argv)
    paths, source_count = image_paths(args)
    if args.list_images:
        for path in paths:
            image = load_rgb(path)
            print(f"{path}: RGB {image.width}x{image.height}")
        print(f"Selected {len(paths)} of {source_count} source images")
        return
    # Always prohibit login-node execution, before importing any GPU packages.
    require_slurm_step(os.environ)
    if not args.checkpoint or not args.checkpoint.is_file() or not args.output_dir:
        parser.error("Inference needs a local --checkpoint and new --output-dir")
    if args.points_per_side < 1 or args.proposal_limit < 0 or (args.save_candidate_limit is not None and args.save_candidate_limit < 0):
        parser.error("Invalid grid/proposal/archive limit")
    select_proposals([], min_score=args.min_score, dedup_iou=None if args.no_dedup else args.dedup_iou,
                     proposal_limit=args.proposal_limit)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU/model substitution")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.synchronize()
    start = time.perf_counter()
    predictor, model_info = load_predictor(args.checkpoint, args.bpe_path, torch)
    torch.cuda.synchronize()
    model_info.update(load_seconds=time.perf_counter() - start, torch_version=torch.__version__,
                      gpu=torch.cuda.get_device_name(), seed=0, precision="float32, no autocast, TF32 disabled")
    args.output_dir.mkdir(parents=True)
    manifest = {"status": "running", "model": model_info,
                "workflow_source": file_fingerprint(Path(__file__)),
                "proposal_source": file_fingerprint(Path(__file__).resolve().parents[1] / "scene_segmentation/proposals.py"),
                "source_image_count": source_count,
                "selected_images": [str(p) for p in paths], "completed_images": [],
                "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_step_id": os.environ["SLURM_STEP_ID"]}
    manifest_path = args.output_dir / "run.json"
    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_manifest()
    try:
        for index, path in enumerate(paths):
            image_start = time.perf_counter()
            rgb = np.asarray(load_rgb(path))
            predictor.reset_predictor()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                proposals, timing = collect_proposals(predictor, rgb, args.points_per_side, torch.cuda.synchronize)
            post_start = time.perf_counter()
            retained, rows, rules = select_proposals(proposals, min_score=args.min_score,
                dedup_iou=None if args.no_dedup else args.dedup_iou, proposal_limit=args.proposal_limit)
            timing["selection_seconds"] = time.perf_counter() - post_start
            destination = args.output_dir / f"{index:02d}_{path.stem}"
            metadata = {"source": file_fingerprint(path), "model": model_info, **timing, **rules,
                        "points_per_side": args.points_per_side, "color_order": "RGB; no EXIF rotation",
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
            save_start = time.perf_counter()
            report = save_scene(rgb, proposals, retained, rows, destination, metadata, args.save_candidate_limit)
            report["save_seconds"] = time.perf_counter() - save_start
            report["image_total_seconds"] = time.perf_counter() - image_start
            (destination / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            manifest["completed_images"].append(destination.name)
            write_manifest()
            print(f"{path.name}: {len(proposals)} raw, {len(retained)} retained -> {destination}")
            del proposals, retained
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_manifest()


if __name__ == "__main__":
    main()
