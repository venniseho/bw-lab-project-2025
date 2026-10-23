import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from bwlab.datasets.manifest import ManifestDataModule
from scripts.prepare_coco_validation import download, main, select_records


class CocoManifestTests(TestCase):
    def test_download_is_atomic_and_reuses_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val2017" / "image.jpg"
            with patch("scripts.prepare_coco_validation.urlopen", return_value=io.BytesIO(b"image")) as fetch:
                download("https://example.invalid/image.jpg", path)
                download("https://example.invalid/image.jpg", path)
                self.assertEqual(fetch.call_count, 1)
            self.assertEqual(path.read_bytes(), b"image")
            self.assertFalse(path.with_suffix(".jpg.part").exists())
            failed = path.with_name("failed.jpg")
            with patch("scripts.prepare_coco_validation.urlopen", side_effect=OSError("network")):
                with self.assertRaises(OSError):
                    download("https://example.invalid/failed.jpg", failed)
            self.assertFalse(failed.exists())
            self.assertFalse(failed.with_suffix(".jpg.part").exists())

    def fixture(self, root):
        (root / "annotations").mkdir()
        (root / "val2017").mkdir()
        data = {"images": [{"id": 1, "file_name": "000000000001.jpg", "height": 8, "width": 10},
                           {"id": 2, "file_name": "000000000002.jpg", "height": 8, "width": 10}],
                "categories": [{"id": 1, "name": "object"}],
                "annotations": [{"id": 10, "image_id": 1, "category_id": 1, "iscrowd": 0,
                                 "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                                 "area": 9, "bbox": [1, 1, 3, 3]}]}
        path = root / "annotations/instances_val2017.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        Image.fromarray(np.full((8, 10, 3), 100, np.uint8)).save(root / "val2017/000000000001.jpg")
        return data, path

    def test_preparation_and_lazy_loading_without_mask_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            output = root / "configs"
            main(["--coco-root", str(root), "--output-dir", str(output), "--count", "2"])
            with patch.object(COCO, "annToMask", autospec=True, wraps=None) as decode:
                module = ManifestDataModule({"val": output / "records.json"})
                decode.assert_not_called()
            dataset = module.val_dataset()
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["segmentation"].sum(), 9)
            self.assertEqual(dataset[0]["metadata"]["annotation_id"], 10)
            self.assertFalse(list(root.rglob("*.png")))
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(selection["sources_without_noncrowd_instance"], 1)
            for condition in ("fence", "fragmentation"):
                config = json.loads((output / f"{condition}.json").read_text())
                self.assertEqual(config["manifest"], str(output / "records.json"))

    def test_annotation_changes_and_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, annotations = self.fixture(root)
            output = root / "configs"
            main(["--coco-root", str(root), "--output-dir", str(output), "--count", "2"])
            manifest = output / "records.json"
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                ManifestDataModule({"train": manifest, "val": manifest})
            annotations.write_text(annotations.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "annotations changed"):
                ManifestDataModule({"val": manifest})

    def test_selection_is_reproducible_and_does_not_rank_area(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, path = self.fixture(root)
            data["annotations"].extend([
                {"id": 2, "image_id": 1, "iscrowd": 1, "area": 1},
                {"id": 3, "image_id": 1, "iscrowd": 0, "area": 1}])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            first = select_records(data, 2, 42, root, path, digest)
            self.assertEqual(first, select_records(data, 2, 42, root, path, digest))
            self.assertEqual(first[0][0]["annotation_id"], 3)
