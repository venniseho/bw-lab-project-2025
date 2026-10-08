"""Create a clean/light/medium/heavy RGB fence comparison with Pillow.

From the repository root:
    python scripts/preview_fence.py --image path/to/image.jpg
Optional: --mask path/to/object_mask.png --angle 45 --seed 123
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stimuli_generation.fence import generate_fence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/fence_preview.png"))
    parser.add_argument("--spacing", type=float, default=40.0, help="Bar center distance in pixels")
    parser.add_argument("--widths", type=float, nargs=3, default=[4.0, 12.0, 24.0],
                        metavar=("LIGHT", "MEDIUM", "HEAVY"))
    parser.add_argument("--angle", type=float, default=90.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--panel-width", type=int, default=400)
    parser.add_argument("--benchmark-repeats", type=int, default=0,
                        help="Time generation at native resolution; excludes loading and export")
    args = parser.parse_args()
    if not 0 <= args.widths[0] < args.widths[1] < args.widths[2] <= args.spacing:
        parser.error("widths must increase, stay nonnegative, and not exceed spacing")
    if args.panel_width <= 0 or args.benchmark_repeats < 0:
        parser.error("panel-width must be positive and benchmark-repeats nonnegative")

    with Image.open(args.image) as source:
        image_rgb = np.array(source.convert("RGB"))
    gt_mask = None
    if args.mask:
        with Image.open(args.mask) as source:
            gt_mask = np.array(source.convert("L"))

    panels = [image_rgb]
    captions = ["Clean image\nImage occluded: 0.0%"]
    for label, bar_width in zip(("Light", "Medium", "Heavy"), args.widths):
        result = generate_fence(image_rgb, bar_width, args.spacing, args.angle, args.seed, gt_mask)
        panels.append(result.corrupted_rgb)
        caption = f"{label}: width {bar_width:g} px\nImage occluded: {result.image_occluded_fraction:.1%}"
        if gt_mask is not None:
            coverage = result.object_occluded_fraction
            caption += "\nObject occluded: " + ("undefined (empty GT)" if coverage is None else f"{coverage:.1%}")
        captions.append(caption)

    # Generate at native resolution; resizing below is for display only.
    height, width = image_rgb.shape[:2]
    display_height = max(1, round(height * args.panel_width / width))
    header_height = 76
    comparison = Image.new("RGB", (4 * args.panel_width, display_height + header_height), "white")
    draw = ImageDraw.Draw(comparison)
    for i, (panel, caption) in enumerate(zip(panels, captions)):
        preview = Image.fromarray(panel).resize((args.panel_width, display_height), Image.Resampling.LANCZOS)
        x = i * args.panel_width
        comparison.paste(preview, (x, header_height))
        draw.multiline_text((x + 8, 8), caption, fill="black", spacing=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(args.output)
    print(f"Saved {args.output} (native image: {width} x {height}; spacing {args.spacing:g} px, angle {args.angle:g} deg)")

    if args.benchmark_repeats:
        for _ in range(3):
            generate_fence(image_rgb, args.widths[1], args.spacing, args.angle, args.seed, gt_mask)
        timings = []
        for _ in range(args.benchmark_repeats):
            start = time.perf_counter()
            generate_fence(image_rgb, args.widths[1], args.spacing, args.angle, args.seed, gt_mask)
            timings.append((time.perf_counter() - start) * 1000)
        print(f"CPU generation ({len(timings)} repeats, includes image copy" +
              (" and GT coverage" if gt_mask is not None else "") +
              f"): median {np.median(timings):.3f} ms, p95 {np.percentile(timings, 95):.3f} ms")


if __name__ == "__main__":
    main()
