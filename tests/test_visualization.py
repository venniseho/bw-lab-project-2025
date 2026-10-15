"""Verify labels carry proposal provenance and a visible prompt marker."""
from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np
from PIL import ImageDraw
from PIL import Image
from bwlab.visualization import prompt_overlay

from scene_segmentation.proposals import Proposal, all_proposal_overlay, contact_sheet, save_scene, select_proposals
from scripts.relabel_scene import relabel_scene


class ContactSheetTests(TestCase):
    def test_prompt_dots_scale_from_source_coordinates_and_deduplicate(self):
        from bwlab.visualization import _draw_prompt
        image = Image.new("RGB", (300, 150), (100, 100, 100))
        prompts = [{"grid_point_index": 2, "point_xy": [100, 60]},
                   {"grid_point_index": 2, "point_xy": [100, 60]},
                   {"grid_point_index": 8, "point_xy": [500, 240]}]
        with patch("bwlab.visualization._draw_prompt", wraps=_draw_prompt) as draw:
            result = prompt_overlay(image, prompts, source_size=(600, 300))
        self.assertEqual(draw.call_count, 2)
        self.assertEqual(draw.call_args_list[0].args[1], (49.75, 29.75))
        self.assertEqual(draw.call_args_list[0].args[2], "G2")
        self.assertEqual(result.getpixel((50, 30)), (255, 0, 0))
        self.assertEqual(image.getpixel((50, 30)), (100, 100, 100))

    def test_header_shows_queried_points_even_without_retained_masks(self):
        from bwlab.visualization import _draw_prompt
        rgb = np.full((100, 150, 3), 100, np.uint8)
        prompts = [{"grid_point_index": 7, "point_xy": [75, 50]}]
        with patch("bwlab.visualization._draw_prompt", wraps=_draw_prompt) as draw:
            contact_sheet(rgb, rgb.copy(), [], prompts=prompts)
        self.assertEqual(draw.call_count, 2)  # Same query on both header panels.
        self.assertTrue(all(call.args[2] == "G7" for call in draw.call_args_list))

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
            self.assertTrue((root / "scene/prompt_locations.png").is_file())
            np.testing.assert_array_equal(np.array(Image.open(root / "scene/original.png")), rgb)
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
