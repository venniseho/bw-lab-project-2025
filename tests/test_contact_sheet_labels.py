"""Verify labels carry proposal provenance and a visible prompt marker."""
from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np
from PIL import ImageDraw

from scene_segmentation.proposals import Proposal, all_proposal_overlay, contact_sheet, save_scene, select_proposals
from scripts.relabel_scene import relabel_scene


class ContactSheetTests(TestCase):
    def test_rebuild_from_saved_archive_without_overwriting(self):
        rgb = np.full((10, 15, 3), 100, np.uint8)
        mask = np.ones((10, 15), bool)
        proposal = Proposal(np.packbits(mask), mask.shape, {
            "proposal_id": 0, "grid_point_index": 0, "candidate_index": 0,
            "point_xy": [7.0, 4.0], "predicted_iou_score": 0.8,
            "area_pixels": 150})
        retained, rows, rules = select_proposals([proposal])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_scene(rgb, [proposal], retained, rows, root / "scene", rules)
            original = (root / "scene/contact_sheet.png").read_bytes()
            relabel_scene(root / "scene", root / "labeled.png")
            self.assertTrue((root / "labeled.png").is_file())
            self.assertEqual((root / "scene/contact_sheet.png").read_bytes(), original)
            with self.assertRaises(FileExistsError):
                relabel_scene(root / "scene", root / "labeled.png")

    def test_labels_and_marker_are_aligned_with_proposal_metadata(self):
        rgb = np.full((100, 150, 3), 100, np.uint8)
        mask = np.zeros((100, 150), bool)
        mask[30:90, 40:130] = True
        proposal = Proposal(np.packbits(mask), mask.shape, {
            "proposal_id": 17, "grid_point_index": 5, "candidate_index": 2,
            "point_xy": [100.0, 60.0], "predicted_iou_score": 0.81234,
            "area_pixels": int(mask.sum())})
        original_draw = ImageDraw.Draw
        captions = []
        def capture(image):
            draw = original_draw(image)
            real_multiline_text = draw.multiline_text
            def record(position, text, **kwargs):
                captions.append(text)
                return real_multiline_text(position, text, **kwargs)
            draw.multiline_text = record
            return draw
        with patch("scene_segmentation.proposals.ImageDraw.Draw", side_effect=capture):
            sheet = contact_sheet(rgb, all_proposal_overlay(rgb, [proposal]), [proposal])
        caption = "\n".join(captions)
        for text in ("P00017", "grid point 5", "candidate 2", "Predicted IoU: 0.812",
                     "(100.0, 60.0)", "5400 px"):
            self.assertIn(text, caption)
        # Marker lies inside the first candidate tile and uses red for contrast.
        pixels = np.asarray(sheet)
        marker = np.all(pixels[420:, :320] == [255, 0, 0], axis=2)
        self.assertGreater(marker.sum(), 0)
        self.assertEqual(sheet.width, 1280)
        self.assertTrue(np.all(rgb == 100))
