"""Small, labelled before/after figures and structured experiment logs."""
import json

import numpy as np
from PIL import Image, ImageDraw


def append_json(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def contact_sheet(path, sample, point, before, after):
    image = sample["image"]
    target = sample["segmentation"].astype(bool)
    width, height = 300, 260
    sheet = Image.new("RGB", (4 * width, height + 58), "white")
    panels = [("Input + positive point", None), ("COCO target", target),
              (f"Before: IoU={before['iou']:.3f}", before["mask"]),
              (f"After: IoU={after['iou']:.3f}", after["mask"])]
    for index, (label, mask) in enumerate(panels):
        pixels = image.copy()
        if mask is not None:
            pixels[mask] = (.5 * pixels[mask] + .5 * np.array([255, 30, 30])).astype(np.uint8)
        panel = Image.fromarray(pixels)
        draw = ImageDraw.Draw(panel)
        x, y = point
        radius = max(3, min(image.shape[:2]) // 65)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="lime", outline="black", width=2)
        panel.thumbnail((width, height))
        sheet.paste(panel, (index * width, 28))
        ImageDraw.Draw(sheet).text((index * width + 4, 6), label, fill="black")
    ImageDraw.Draw(sheet).text((4, height + 36), sample["__key__"], fill="black")
    sheet.save(path)
