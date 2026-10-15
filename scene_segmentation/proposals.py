"""Compatibility imports; implementation lives in bwlab.models.proposals."""
from PIL import ImageDraw
from bwlab.models.proposals import (
    Proposal,
    point_grid,
    collect_proposals,
    packed_mask_iou,
    select_proposals,
)
from bwlab.visualization import proposal_color, all_proposal_overlay, contact_sheet
from bwlab.tasks._scene_export import save_scene
