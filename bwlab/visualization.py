"""Proposal rendering; independent of model inference and result storage."""
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from bwlab.models.proposals import Proposal


def proposal_color(identifier: int) -> tuple[int, int, int]:
    return tuple(round(255 * channel) for channel in colorsys.hsv_to_rgb(
        (identifier * 0.618033988749895) % 1, 0.75, 0.95
    ))

def all_proposal_overlay(rgb: np.ndarray, retained: list[Proposal]) -> np.ndarray:
    """Average proposal colors at overlaps; preserve all overlapping masks.

    No exclusive instance assignment or later-mask-wins rendering is imposed.
    This overlay is an exploratory multi-label visualization, not a panoptic map.
    """
    colors = np.zeros(rgb.shape, dtype=np.float64)
    counts = np.zeros(rgb.shape[:2], dtype=np.uint32)
    for proposal in retained:
        mask = proposal.mask()
        colors[mask] += proposal_color(proposal.metadata["proposal_id"])
        counts[mask] += 1
    active = counts > 0
    overlay = rgb.copy()
    overlay[active] = np.rint(0.55 * rgb[active] + 0.45 * colors[active] / counts[active, None]).astype(np.uint8)
    return overlay

def contact_sheet(rgb: np.ndarray, overlay: np.ndarray, retained: list[Proposal]) -> Image.Image:
    columns, tile_width, tile_height = 4, 320, 270
    top_height = 420
    rows = max(1, (len(retained) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_width, top_height + rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    draw.text((8, 8), "SAM3 automatic point-grid prompting | each point is an independent query", fill="black", font=font)
    draw.text((8, 32), "Colors identify proposals, not object classes. Overlaps blend colors; this is not a final scene partition.", fill="black", font=font)
    draw.text((8, 56), "Cross = positive prompt (x,y in source pixels). Predicted IoU is model quality, not measured GT IoU.", fill="black", font=font)
    for column, (pixels, caption) in enumerate((
        (rgb, "A. Input RGB"), (overlay, f"B. Retained proposal overlay ({len(retained)} masks)"),
    )):
        tile = ImageOps.contain(Image.fromarray(pixels), (2 * tile_width - 12, top_height - 126))
        x = column * 2 * tile_width
        draw.text((x + 6, 88), caption, fill="black", font=font)
        sheet.paste(tile, (x + (2 * tile_width - tile.width) // 2, 116))
    if not retained:
        draw.text((6, top_height + 6), "No retained proposals (see metadata for all decisions).", fill="black")
    for index, proposal in enumerate(retained):
        identifier = proposal.metadata["proposal_id"]
        x, y = (index % columns) * tile_width, top_height + (index // columns) * tile_height
        silhouette = np.zeros(rgb.shape, dtype=np.uint8)
        silhouette[proposal.mask()] = proposal_color(identifier)
        tile = ImageOps.contain(Image.fromarray(silhouette), (tile_width - 12, tile_height - 94),
                                method=Image.Resampling.NEAREST)
        meta = proposal.metadata
        point_x, point_y = meta["point_xy"]
        caption = (f"P{identifier:05d} | grid point {meta['grid_point_index']} | candidate {meta['candidate_index']}\n"
                   f"Predicted IoU: {meta['predicted_iou_score']:.3f}\n"
                   f"Prompt (x,y): ({point_x:.1f}, {point_y:.1f}) | positive\n"
                   f"Mask area: {meta['area_pixels']} px | black = outside")
        draw.multiline_text((x + 6, y + 3), caption, fill="black", font=font, spacing=3)
        tile_x, tile_y = x + (tile_width - tile.width) // 2, y + 94
        sheet.paste(tile, (tile_x, tile_y))
        marker_x = tile_x + (point_x + 0.5) * tile.width / rgb.shape[1] - 0.5
        marker_y = tile_y + (point_y + 0.5) * tile.height / rgb.shape[0] - 0.5
        draw.line((marker_x - 5, marker_y, marker_x + 5, marker_y), fill="white", width=3)
        draw.line((marker_x, marker_y - 5, marker_x, marker_y + 5), fill="white", width=3)
        draw.line((marker_x - 4, marker_y, marker_x + 4, marker_y), fill="red", width=1)
        draw.line((marker_x, marker_y - 4, marker_x, marker_y + 4), fill="red", width=1)
    return sheet
