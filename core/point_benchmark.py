"""Measure validation with and without shared features, without weight updates."""

import torch

from core.point_records import sample_record, mask_rle
from core.tasks.point_segmentation import forward_sample
from core.training_timing import PhaseTiming


@torch.no_grad()
def validation_benchmark(
    segmenter, samples, output, *, seed, points_per_image=1, shared=True, repeat=0
):
    segmenter.train(False)
    mode = f"{'shared' if shared else 'reference'}_{points_per_image}points"
    timing = PhaseTiming(segmenter.model, "validation", 0)
    forwards, total_iou, comparisons = 0, 0.0, {}
    for sample in samples:
        features = segmenter.encode(sample["image"]) if shared else None
        forwards += int(shared)
        for prompt_index in range(points_per_image):
            _, metrics, logits, scores, _ = forward_sample(
                segmenter, sample, seed=seed + prompt_index, encoded=features
            )
            forwards += int(not shared)
            timing.observe(sample)
            total_iou += metrics["selected_iou"]
            comparisons[(sample["__key__"], prompt_index)] = metrics["selected_iou"]
            output.record(
                "benchmark_samples",
                {
                    "mode": mode,
                    "repeat": repeat,
                    "prompt_index": prompt_index,
                    **sample_record(sample),
                    **metrics,
                    "selected_mask_rle": mask_rle((logits[scores.argmax()] > 0).cpu().numpy()),
                },
            )
    report = timing.finish()
    report.update(
        mode=mode,
        repeat=repeat,
        encoder_forwards=forwards,
        points_per_image=points_per_image,
        mean_iou=total_iou / report["instances"],
    )
    output.record("benchmark", report)
    return report, comparisons
