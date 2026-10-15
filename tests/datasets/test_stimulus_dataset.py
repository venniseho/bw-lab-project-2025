"""Native-coordinate dataset, model boundary, and preparation integration tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
from torchvision.transforms import Compose

from bwlab.datasets import ImageMaskDataset, CocoInstanceDataset, build_loader
from bwlab.runner import run_pipeline
from scripts.prepare_stimuli import main
from bwlab.stimuli.transforms import FenceOcclusion


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rgb = np.full((7, 11, 3), 127, np.uint8)
        self.mask = np.zeros((7, 11), np.uint8)
        self.mask[2:6, 3:9] = 255
        Image.fromarray(self.rgb).save(self.root / "rgb.png")
        Image.fromarray(self.mask).save(self.root / "mask.png")
        self.records = [{"__key__": "a", "image_path": str(self.root / "rgb.png"),
                         "mask_path": str(self.root / "mask.png")}]

    def test_dataset_compose_and_eval_loader(self):
        dataset = ImageMaskDataset(self.records, Compose([FenceOcclusion(2, 5)]))
        self.assertEqual(len(dataset), 1)
        result = next(iter(build_loader(dataset, split="test")))[0]
        self.assertEqual(result["__key__"], "a")
        np.testing.assert_array_equal(result["segmentation"], self.mask)
        result["segmentation"][:] = 0
        np.testing.assert_array_equal(dataset[0]["segmentation"], self.mask)

    def test_variable_resolution_collation_and_worker(self):
        Image.fromarray(np.ones((3, 4, 3), np.uint8)).save(self.root / "small.png")
        Image.fromarray(np.ones((3, 4), np.uint8)).save(self.root / "small_mask.png")
        records = self.records + [{"__key__": "b", "image_path": str(self.root / "small.png"),
                                   "mask_path": str(self.root / "small_mask.png")}]
        dataset = ImageMaskDataset(records)
        for workers in (0, 1):
            batch = next(iter(build_loader(dataset, split="val", batch_size=2, num_workers=workers)))
            self.assertEqual([s["image"].shape for s in batch], [(7, 11, 3), (3, 4, 3)])

    def test_coco_decisions_retained_without_shape_filters(self):
        coco = Mock()
        coco.loadImgs.return_value = [{"file_name": "rgb.png"}]
        coco.getAnnIds.return_value = [10, 11]
        coco.loadAnns.return_value = [{"id": 10, "image_id": 1, "iscrowd": 0},
                                     {"id": 11, "image_id": 1, "iscrowd": 1}]
        disconnected = np.zeros((7, 11), np.uint8)
        disconnected[0, 0] = disconnected[6, 10] = 1
        coco.annToMask.return_value = disconnected
        dataset = CocoInstanceDataset(coco, self.root, [1])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(len(dataset.selection_records), 2)
        self.assertEqual(dataset[0]["__key__"], "coco/1/10")
        np.testing.assert_array_equal(dataset[0]["segmentation"], disconnected)
        self.assertEqual(len(CocoInstanceDataset(coco, self.root, [1], crowd_policy="include")), 2)

    def test_duplicate_key_or_image_rejected(self):
        with self.assertRaises(ValueError):
            ImageMaskDataset(self.records * 2)
        with self.assertRaises(ValueError):
            CocoInstanceDataset(Mock(), self.root, [1, 1])

    def test_model_receives_only_pixels_evaluator_receives_targets(self):
        dataset = ImageMaskDataset(self.records, FenceOcclusion(2, 5))
        model = Mock(return_value="prediction")
        evaluator = Mock()
        evaluator.finalize.return_value = {"status": "test-only"}
        self.assertEqual(run_pipeline(dataset, model, evaluator), {"status": "test-only"})
        self.assertEqual(model.call_count, 1)
        self.assertIsInstance(model.call_args.args[0], np.ndarray)
        sample, prediction = evaluator.update.call_args.args
        self.assertEqual(prediction, "prediction")
        np.testing.assert_array_equal(sample["segmentation"], self.mask)

    def test_preparation_saves_separate_targets_and_refuses_overwrite(self):
        (self.root / "records.json").write_text(json.dumps([
            {"__key__": "a", "image_path": "rgb.png", "mask_path": "mask.png"}]))
        argv = ["--manifest", str(self.root / "records.json"), "--output-dir", str(self.root / "prepared"),
                "--transform", "fence", "--seed", "42"]
        main(argv)
        report = json.loads((self.root / "prepared/manifest.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["records"]), 4)
        for record in report["records"]:
            with np.load(self.root / "prepared" / record["targets"]) as targets:
                np.testing.assert_array_equal(targets["segmentation"], self.mask)
                np.testing.assert_array_equal(targets["visible_segmentation"],
                                              (self.mask > 0) & ~targets["occlusion_mask"])
        with self.assertRaises(FileExistsError):
            main(argv)


if __name__ == "__main__":
    unittest.main()
