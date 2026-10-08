"""One RGB image + one positive point through SAM3 Tracker PVS on CUDA.

PyTorch/Transformers are imported only when running inference, so CPU mock
tests and --help do not require either package. No checkpoint downloads occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import NamedTuple

import numpy as np
import PIL
from PIL import Image


class PointPrediction(NamedTuple):
    masks: np.ndarray
    scores: np.ndarray
    selected_index: int
    object_score_logits: list | None
    inference_seconds: float
    cuda_forward_seconds: float


def require_slurm_step(environment: dict) -> None:
    """An salloc shell alone can still be on the login node: require srun."""
    if not environment.get("SLURM_JOB_ID") or not environment.get("SLURM_STEP_ID"):
        raise RuntimeError("Launch inside an allocated GPU job with srun; "
                           "both SLURM_JOB_ID and SLURM_STEP_ID are required.")


def load_rgb(path: str | Path) -> Image.Image:
    # Preserve decoded pixel coordinates: do not silently apply EXIF rotation.
    with Image.open(path) as source:
        return source.convert("RGB")


def validate_point(image: Image.Image, point_xy: tuple[float, float]) -> None:
    if image.mode != "RGB":
        raise ValueError("image must be RGB")
    x, y = point_xy
    if not np.isfinite([x, y]).all() or not (0 <= x < image.width and 0 <= y < image.height):
        raise ValueError("point must be finite (x, y) in original image bounds")


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def predict_point(model, processor, image: Image.Image, point_xy, torch) -> PointPrediction:
    """Use documented [image, object, point, xy] input and keep all candidates.

    Model-selected means highest predicted IoU among multimask candidates;
    first index wins ties. This is explicitly our score-selection policy,
    not the decoder's separate multimask_output=False behavior. No GT enters.
    """
    validate_point(image, point_xy)
    x, y = map(float, point_xy)
    torch.cuda.synchronize()
    start = time.perf_counter()
    inputs = processor(images=image, input_points=[[[[x, y]]]],
                       input_labels=[[[1]]], return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    forward_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=True, return_dict=True)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - forward_start
    processed = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu(),
        mask_threshold=0.0, binarize=True, max_hole_area=0.0,
        max_sprinkle_area=0.0, apply_non_overlapping_constraints=False,
    )
    if len(processed) != 1:
        raise ValueError("expected exactly one postprocessed image")
    masks = _numpy(processed[0])
    scores = _numpy(outputs.iou_scores)
    # No squeeze: verify image/object/candidate axes even for one candidate.
    if masks.ndim != 4 or masks.shape[0] != 1:
        raise ValueError("expected postprocessed masks shaped (1, K, H, W)")
    masks = masks[0]
    if masks.shape[0] == 0 or masks.shape[1:] != (image.height, image.width):
        raise ValueError("candidate masks must match original image dimensions")
    if masks.dtype != np.dtype(bool):
        raise ValueError("postprocessing must return binary boolean masks")
    if scores.shape != (1, 1, len(masks)) or not np.isfinite(scores).all():
        raise ValueError("expected finite IoU scores shaped (1, 1, K)")
    scores = scores[0, 0]
    logits = getattr(outputs, "object_score_logits", None)
    logits = None if logits is None else _numpy(logits).tolist()
    if logits is not None and not np.isfinite(logits).all():
        raise ValueError("object score logits must be finite")
    return PointPrediction(masks, scores, int(np.argmax(scores)), logits,
                           time.perf_counter() - start, forward_seconds)


def save_prediction(image: Image.Image, prediction: PointPrediction,
                    output_dir: Path, metadata: dict) -> dict:
    """Save 0/255 PNGs, RGB cyan overlay, and scores/provenance as JSON.

    Refuse to overwrite an existing directory so old candidates cannot be
    mixed with this inference. Inputs are not modified.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = prediction.masks[prediction.selected_index]
    Image.fromarray(selected.astype(np.uint8) * 255).save(output_dir / "prediction.png")
    rgb = np.asarray(image)
    overlay = rgb.copy()
    overlay[selected] = np.rint(
        0.55 * rgb[selected] + 0.45 * np.array([0, 255, 255])
    ).astype(np.uint8)
    Image.fromarray(overlay).save(output_dir / "overlay.png")
    candidates = []
    for index, (mask, score) in enumerate(zip(prediction.masks, prediction.scores)):
        filename = f"candidate_{index:02d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / filename)
        candidates.append({"index": index, "predicted_iou_score": float(score),
                           "mask_file": filename,
                           "model_selected": index == prediction.selected_index})
    report = dict(metadata)
    report.update({
        "selected_candidate_index": prediction.selected_index,
        "selection_policy": "highest_predicted_iou_first_index_on_ties",
        "candidates": candidates,
        "object_score_logits": prediction.object_score_logits,
        "inference_seconds": prediction.inference_seconds,
        "cuda_forward_seconds": prediction.cuda_forward_seconds,
        "prediction_file": "prediction.png",
        "overlay_file": "overlay.png",
    })
    (output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def file_fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Local HF SAM3 snapshot with configs and safetensors weights")
    parser.add_argument("--checkpoint-revision", help="Source HF commit, if known (provenance only)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--seed", type=int, default=0, help="Inference RNG seed, separate from fence seed")
    parser.add_argument("--require-slurm", action="store_true",
                        help="Require an srun job step before importing CUDA packages")
    args = parser.parse_args(argv)
    if args.require_slurm:
        require_slurm_step(os.environ)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.output_dir.exists():
        raise FileExistsError("--output-dir must be a new directory")
    checkpoint = args.checkpoint_dir.resolve()
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise FileNotFoundError("--checkpoint-dir must contain the HF config.json")
    weight_files = sorted(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError("local safetensors weights are required; no downloads occur")
    image = load_rgb(args.image)
    point = tuple(args.point)
    validate_point(image, point)

    try:
        import torch
        import transformers
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    except ImportError as error:
        raise RuntimeError("CUDA PyTorch, torchvision and Transformers with Sam3Tracker "
                           "are required; see docs/sam3_smoke_test.md") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is disabled")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.synchronize()
    load_start = time.perf_counter()
    processor = Sam3TrackerProcessor.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False
    )
    model, loading_info = Sam3TrackerModel.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32, attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(loading_info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Incomplete/incompatible tracker checkpoint: {loading_info}")
    # Transformers 5.17 returns sets for key collections; older versions
    # returned lists. Preserve the diagnostics as deterministic JSON data.
    loading_info = {key: sorted(value) if isinstance(value, set) else value
                    for key, value in loading_info.items()}
    model = model.to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    torch.cuda.reset_peak_memory_stats()
    prediction = predict_point(model, processor, image, point, torch)
    peak_bytes = int(torch.cuda.max_memory_allocated())
    fingerprint_start = time.perf_counter()
    checkpoint_files = [file_fingerprint(path) for path in
                        sorted(set(weight_files + list(checkpoint.glob("*.json"))))]
    fingerprint_seconds = time.perf_counter() - fingerprint_start
    try:
        repo_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=False,
        ).stdout.strip() or None
    except OSError:
        repo_commit = None
    metadata = {
        "inference_path": "transformers.Sam3TrackerModel.point_prompted_PVS",
        "image_path": str(args.image.resolve()),
        "image_fingerprint": file_fingerprint(args.image),
        "image_size_hw": [image.height, image.width],
        "image_color_order": "RGB", "exif_orientation_applied": False,
        "prompt_xy": list(point), "prompt_label": 1,
        "prompt_coordinate_system": "xy_zero_based_original_image_pixels",
        "checkpoint_dir": str(checkpoint), "checkpoint_revision": args.checkpoint_revision,
        "checkpoint_files": checkpoint_files, "loading_info": loading_info,
        "model_class": type(model).__name__, "model_config": model.config.to_dict(),
        "processor_class": type(processor).__name__,
        "image_processor_config": processor.image_processor.to_dict(),
        "processor_target_size": processor.target_size,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "numpy": np.__version__,
                     "pillow": PIL.__version__, "cuda": torch.version.cuda},
        "repository_commit": repo_commit,
        "device": "cuda", "gpu_name": torch.cuda.get_device_name(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "dtype": "float32", "attention_implementation": "sdpa", "tf32": False,
        "seed": args.seed, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "multimask_output": True, "mask_logit_threshold": 0.0,
        "max_hole_area": 0.0, "max_sprinkle_area": 0.0,
        "apply_non_overlapping_constraints": False,
        "model_load_seconds": load_seconds,
        "checkpoint_fingerprint_seconds": fingerprint_seconds,
        "peak_cuda_allocated_bytes": peak_bytes,
    }
    report = save_prediction(image, prediction, args.output_dir, metadata)
    print(f"Saved {len(report['candidates'])} candidates to {args.output_dir}; "
          f"selected index {prediction.selected_index}, "
          f"predicted IoU {prediction.scores[prediction.selected_index]:.4f}, "
          f"inference {prediction.inference_seconds:.3f}s")


if __name__ == "__main__":
    main()
