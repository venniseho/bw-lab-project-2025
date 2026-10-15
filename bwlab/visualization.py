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

def _draw_prompt(image: Image.Image, point_xy: tuple[float, float], label: str) -> None:
    """Draw one display-only dot and label, keeping the label inside the tile."""
    draw = ImageDraw.Draw(image)
    x, y = point_xy
    radius = 4
    font = ImageFont.load_default(size=14)
    box = draw.textbbox((0, 0), label, font=font)
    width, height = box[2]-box[0]+6, box[3]-box[1]+6
    left = max(0, min(x+7, image.width-width))
    top = max(0, min(y-12, image.height-height))
    draw.rectangle((left, top, left+width, top+height), fill="black")
    draw.text((left+3, top+3-box[1]), label, fill="white", font=font)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="red", outline="white", width=2)


def prompt_overlay(image: Image.Image, prompts: list[dict],
                   source_size: tuple[int, int] | None = None) -> Image.Image:
    """Mark unique query locations after display resizing; source_size is (W,H).

    Grid labels are zero-based, matching metadata. These marks never enter SAM3.
    Use all proposal metadata to retain points even if no masks were retained.
    """
    output = image.copy()
    source_width, source_height = source_size or image.size
    seen = set()
    for prompt in prompts:
        identifier = prompt["grid_point_index"]
        if identifier in seen:
            continue
        seen.add(identifier)
        x, y = prompt["point_xy"]
        point = ((x+0.5)*image.width/source_width-0.5,
                 (y+0.5)*image.height/source_height-0.5)
        _draw_prompt(output, point, f"G{identifier}")
    return output


def _header_panels(sheet: Image.Image, rgb: np.ndarray, overlay: np.ndarray,
                   retained_count: int, prompts: list[dict], title: str | None) -> None:
    """Place RGB and combined-mask previews with all queried grid points."""
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    heading = f"SAM3 | {title} | independent positive-point grid" if title else "SAM3 automatic point-grid prompting | each dot is an independent positive query"
    legends = [heading,
               "Colors identify proposals, not classes. Overlaps blend; this is not a final scene partition.",
               "Red dot = prompt; G# = zero-based grid ID. Predicted IoU is model quality, not measured GT IoU."]
    for y, text in zip((8, 32, 56), legends):
        draw.text((8, y), text, fill="black", font=font)
    panels = [(rgb, "A. Input RGB + all queried points"),
              (overlay, f"B. Retained overlay ({retained_count} masks) + all queried points")]
    for column, (pixels, caption) in enumerate(panels):
        tile = ImageOps.contain(Image.fromarray(pixels), (628, 294))
        tile = prompt_overlay(tile, prompts, (rgb.shape[1], rgb.shape[0]))
        draw.text((column*640+6, 88), caption, fill="black", font=font)
        sheet.paste(tile, (column*640+(640-tile.width)//2, 116))


def _candidate_caption(proposal: Proposal) -> str:
    meta = proposal.metadata
    x, y = meta["point_xy"]
    return (f"P{meta['proposal_id']:05d} | grid point {meta['grid_point_index']} | candidate {meta['candidate_index']}\n"
            f"Predicted IoU: {meta['predicted_iou_score']:.3f}\n"
            f"Prompt (x,y): ({x:.1f}, {y:.1f}) | positive\n"
            f"Mask area: {meta['area_pixels']} px | RGB + colored mask")


def _candidate_panel(sheet: Image.Image, rgb: np.ndarray, proposal: Proposal, index: int) -> None:
    """Show one mask over the source RGB and mark its own query only."""
    x, y = (index % 4)*320, 420+(index//4)*270
    draw = ImageDraw.Draw(sheet)
    draw.multiline_text((x+6, y+3), _candidate_caption(proposal), fill="black",
                        font=ImageFont.load_default(size=16), spacing=3)
    overlay = Image.fromarray(all_proposal_overlay(rgb, [proposal]))
    tile = ImageOps.contain(overlay, (308, 176))
    tile = prompt_overlay(tile, [proposal.metadata], (rgb.shape[1], rgb.shape[0]))
    sheet.paste(tile, (x+(320-tile.width)//2, y+94))


def contact_sheet(rgb: np.ndarray, overlay: np.ndarray, retained: list[Proposal],
                  *, prompts: list[dict] | None = None, title: str | None = None) -> Image.Image:
    """Render prompt context without changing source RGB or archived masks."""
    rows = max(1, (len(retained)+3)//4)
    sheet = Image.new("RGB", (1280, 420+rows*270), "white")
    prompts = [p.metadata for p in retained] if prompts is None else prompts
    _header_panels(sheet, rgb, overlay, len(retained), prompts, title)
    if not retained:
        ImageDraw.Draw(sheet).text((6, 426), "No retained proposals; all queried points shown above.", fill="black")
    for index, proposal in enumerate(retained):
        _candidate_panel(sheet, rgb, proposal, index)
    return sheet
