import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from core.models.proposals import Proposal, point_grid
from scripts.export_condition_examples import save_point_details


class AllPointSheetTests(unittest.TestCase):
    def proposals(self):
        proposals = []
        for grid, point in enumerate(point_grid(12, 20, 3)):
            for candidate in range(3):
                mask = np.zeros((12, 20), bool)
                if candidate:
                    mask[2:5, 3:7] = True
                proposals.append(
                    Proposal(
                        np.packbits(mask),
                        mask.shape,
                        {
                            "proposal_id": len(proposals),
                            "grid_point_index": grid,
                            "candidate_index": candidate,
                            "point_xy": point.tolist(),
                            "predicted_iou_score": 0.9 if len(proposals) == 14 else 0.5,
                        },
                    )
                )
        return proposals

    def test_all_27_candidates_including_empty_masks_are_exported(self):
        rgb = np.full((12, 20, 3), 120, np.uint8)
        target = np.zeros((12, 20), bool)
        target[2:5, 3:7] = True
        proposals = self.proposals()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_point_details(rgb, target, proposals, proposals[14], "source", "clean", output, 0)
            data = json.loads((output / "all_points/source_00_clean.json").read_text())
            self.assertEqual(len(data["candidates"]), 27)
            self.assertEqual([p["proposal_id"] for p in data["candidates"]], list(range(27)))
            self.assertEqual(sum(p["globally_selected"] for p in data["candidates"]), 1)
            self.assertTrue(data["candidates"][14]["globally_selected"])
            self.assertEqual(data["candidates"][0]["area_pixels"], 0)
            self.assertEqual(data["candidates"][0]["measured_full_iou"], 0)
            self.assertEqual(data["candidates"][1]["measured_full_iou"], 1)
            with Image.open(output / "all_points/source_00_clean.png") as image:
                self.assertEqual(image.size, (1360, 2915))

    def test_missing_candidate_is_rejected(self):
        proposals = self.proposals()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "all three candidates"):
                save_point_details(
                    np.zeros((12, 20, 3), np.uint8),
                    np.ones((12, 20), bool),
                    proposals[:-1],
                    proposals[0],
                    "a",
                    "clean",
                    Path(directory),
                    0,
                )
