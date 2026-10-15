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
from bwlab.models.proposals import collect_proposals, select_proposals
from bwlab.tasks._scene_export import save_scene
from bwlab.infrastructure import load_rgb, file_fingerprint, require_slurm_step

from bwlab.models.sam3 import (VERIFIED_SAM3_API_REVISION, checked_state, validate_scene_api, load_scene_model)


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
    parser.add_argument("--points-per-side", type=int, default=3)
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
    model, processor, model_info = load_scene_model(args.checkpoint, args.bpe_path, torch)
    torch.cuda.synchronize()
    model_info.update(load_seconds=time.perf_counter() - start, torch_version=torch.__version__,
                      gpu=torch.cuda.get_device_name(), seed=0,
                      precision="native model dtypes preserved; no autocast; TF32 disabled")
    args.output_dir.mkdir(parents=True)
    manifest = {"status": "running", "model": model_info,
                "workflow_source": file_fingerprint(Path(__file__)),
                "proposal_source": file_fingerprint(Path(__file__).resolve().parents[1] / "bwlab/models/proposals.py"),
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
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                proposals, timing = collect_proposals(model, processor, rgb, args.points_per_side,
                                                     torch.cuda.synchronize)
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
