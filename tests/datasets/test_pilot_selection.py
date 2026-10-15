import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
from PIL import Image

from scripts.prepare_pipeline_pilot import select_records


class PilotSelectionTests(TestCase):
    def test_one_instance_per_distinct_image_without_union_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"images").mkdir()
            (root/"masks").mkdir()
            for stem in ("image_b", "image_a"):
                Image.fromarray(np.full((8, 10, 3), 100, np.uint8)).save(root/f"images/{stem}.png")
                for annotation in (2, 1):
                    Image.fromarray(np.ones((8, 10), np.uint8)).save(root/f"masks/{stem}_ann{annotation}.png")
            records = select_records(root, 2)
            self.assertEqual([r["__key__"] for r in records], ["image_a", "image_b"])
            self.assertTrue(all(r["mask_path"].endswith("ann1.png") for r in records))
            with self.assertRaisesRegex(ValueError, "Need 3"):
                select_records(root, 3)

    def test_relocated_archive_pairs_are_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "legacy" / "main_2025"
            (archive / "images").mkdir(parents=True)
            (archive / "masks").mkdir()
            Image.fromarray(np.full((8, 10, 3), 100, np.uint8)).save(archive / "images/source.png")
            Image.fromarray(np.ones((8, 10), np.uint8)).save(archive / "masks/source_ann1.png")
            records = select_records(root, 1)
            self.assertEqual(Path(records[0]["image_path"]), (archive / "images/source.png").resolve())
            self.assertEqual(Path(records[0]["mask_path"]), (archive / "masks/source_ann1.png").resolve())
