"""Image-level sampling, feature reuse, transformed supervision and interval saves."""

import csv
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from core.data_loading.point_preparation import select_official_split
from core.data_loading.point_training import dataset
from core.models.point_sam3 import PointSegmenter
from core.point_benchmark import validation_benchmark
from core.stimuli.point_transforms import PointTransform
from core.tasks.point_segmentation import forward_sample
from core.trainer import PointTrainer
from core.training_output import TrainingOutput
from core.training_timing import PhaseTiming
from tests.test_point_finetuning import tiny_model
from core.models.sam3 import VERIFIED_SAM3_API_REVISION


def sample(key="image_1", image_id=1):
    mask = np.zeros((8, 8), bool)
    mask[1:7, 1:7] = True
    return {
        "__key__": key,
        "image": np.full((8, 8, 3), 100, np.uint8),
        "segmentation": mask,
        "metadata": {"image_id": image_id, "annotation_id": image_id, "image_path": "source.jpg"},
    }


class PointExperimentTests(unittest.TestCase):
    def test_preparation_writes_official_disjoint_single_target_manifests_and_csv(self):
        from scripts.prepare_point_training import main

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "annotations").mkdir()
            for split, ids in (("train2017", (1, 2)), ("val2017", (3,))):
                (root / split).mkdir()
                images, annotations = [], []
                for image_id in ids:
                    filename = f"{image_id:012d}.jpg"
                    Image.new("RGB", (8, 8), "white").save(root / split / filename)
                    images.append({"id": image_id, "file_name": filename, "width": 8, "height": 8})
                    for offset in (0, 1):
                        annotations.append(
                            {
                                "id": image_id * 10 + offset,
                                "image_id": image_id,
                                "category_id": 1,
                                "iscrowd": 0,
                                "area": 16,
                                "bbox": [1, 1, 4, 4],
                                "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]],
                            }
                        )
                (root / "annotations" / f"instances_{split}.json").write_text(
                    json.dumps(
                        {
                            "images": images,
                            "annotations": annotations,
                            "categories": [{"id": 1, "name": "object"}],
                        }
                    )
                )
            checkpoint = root / "original.pt"
            checkpoint.write_bytes(b"original checkpoint")
            output = root / "prepared"
            args = [
                "prepare_point_training.py",
                "--native-root",
                str(root),
                "--coco-root",
                str(root),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output),
                "--official-splits",
                "--train-images",
                "0",
                "--heldout-images",
                "0",
                "--transform",
                "fragmentation",
            ]
            with (
                patch.object(sys, "argv", args),
                patch(
                    "scripts.prepare_point_training.subprocess.check_output",
                    return_value=VERIFIED_SAM3_API_REVISION,
                ),
            ):
                main()
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["resolution"], 1008)
            self.assertEqual(config["epochs"], 1)
            self.assertTrue(config["one_object_per_image"])
            self.assertEqual(config["splits"]["train"]["instances"], 2)
            self.assertEqual(config["splits"]["test"]["instances"], 1)
            self.assertTrue((output / "validation.json").is_file())
            with (output / "samples.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["split"] for row in rows}, {"train", "validation"})

    def test_training_phase_timer_excludes_interval_validation_pause(self):
        model, _ = tiny_model()
        with patch("core.training_timing.time.perf_counter", side_effect=[100.0, 114.0]):
            timing = PhaseTiming(model, "train", 1)
            timing.observe(sample())
            timing.excluded_seconds = 10.0
            self.assertEqual(timing.finish()["wall_seconds"], 4.0)

    def test_full_validation_has_its_own_baseline_and_final_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "previews").mkdir()
            model, processor = tiny_model()
            segmenter = PointSegmenter(model, processor, "frozen_encoder")
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-5
            )
            trainer = PointTrainer(segmenter, optimizer, forward_sample, TrainingOutput(path))
            with patch("torch.cuda.max_memory_allocated", return_value=0):
                trainer.fit(
                    [sample("train", 1)],
                    [sample("interval", 2)],
                    {"epochs": 1},
                    final_validation_data=[sample("interval", 2), sample("full_only", 3)],
                )
            rows = [json.loads(line) for line in (path / "summary.jsonl").read_text().splitlines()]
            self.assertEqual(
                [(row["validation_set"], row["step"], row["instances"]) for row in rows],
                [("interval", 0, 1), ("full", 0, 2), ("interval", 1, 1), ("full", 1, 2)],
            )

    def test_full_split_selects_one_noncrowd_mask_per_eligible_source_reproducibly(self):
        data = {
            "images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(4)],
            "annotations": [
                {"id": i * 10 + j, "image_id": i, "iscrowd": int(i == 3)}
                for i in range(4)
                for j in range(3)
            ],
            "categories": [],
        }
        split = select_official_split(data, 0, 42)
        self.assertEqual(len(split["images"]), 3)
        self.assertEqual(len(split["annotations"]), 3)
        self.assertEqual({a["image_id"] for a in split["annotations"]}, {0, 1, 2})
        self.assertEqual(split, select_official_split(data, 0, 42))
        self.assertEqual(len(data["annotations"]), 12)

    def test_loader_uses_one_mask_per_image_even_with_old_all_instance_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            images = []
            for i in (1, 2):
                path = folder / f"{i}.jpg"
                Image.new("RGB", (8, 8), "white").save(path)
                images.append({"id": i, "file_name": str(path), "width": 8, "height": 8})
            annotations = [
                {
                    "id": i * 10 + j,
                    "image_id": i,
                    "category_id": 1,
                    "iscrowd": 0,
                    "area": 16,
                    "bbox": [1, 1, 4, 4],
                    "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]],
                }
                for i in (1, 2)
                for j in (0, 1)
            ]
            path = folder / "annotations.json"
            path.write_text(
                json.dumps(
                    {
                        "images": images,
                        "annotations": annotations,
                        "categories": [{"id": 1, "name": "object"}],
                    }
                )
            )
            chosen = dataset(path, one_per_image=True, seed=42)
            self.assertEqual(len(chosen), 2)
            self.assertEqual(chosen.records, dataset(path, one_per_image=True, seed=42).records)
            self.assertEqual(len(dataset(path)), 4)

    def test_shared_validation_encodes_once_for_three_independent_prompts(self):
        model, processor = tiny_model()
        segmenter = PointSegmenter(model, processor, "frozen_encoder")
        model.backbone.forward_image = Mock(wraps=model.backbone.forward_image)
        weights = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            output = TrainingOutput(Path(folder))
            _, reference = validation_benchmark(
                segmenter, [sample()], output, seed=42, points_per_image=3, shared=False
            )
            self.assertEqual(model.backbone.forward_image.call_count, 3)
            model.backbone.forward_image.reset_mock()
            report, shared = validation_benchmark(
                segmenter, [sample()], output, seed=42, points_per_image=3, shared=True
            )
            self.assertEqual(model.backbone.forward_image.call_count, 1)
            self.assertEqual(report["instances"], 3)
            self.assertEqual(report["unique_source_images"], 1)
            self.assertEqual(reference, shared)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(weights[key], tensor)

    def test_fence_and_fragmentation_are_seeded_and_preserve_full_target(self):
        original = sample()
        for name, parameters in (
            ("fence", {"bar_width": 3, "bar_spacing": 10}),
            ("fragmentation", {"noise_count": 0}),
        ):
            transform = PointTransform(name, 42, parameters)
            a, b = transform(original), transform(original)
            np.testing.assert_array_equal(a["image"], b["image"])
            np.testing.assert_array_equal(a["segmentation"], original["segmentation"])
            self.assertEqual(a["transforms"], b["transforms"])

    def test_interval_validation_saves_distinct_previews_csv_and_restoreable_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "previews").mkdir()
            model, processor = tiny_model()
            segmenter = PointSegmenter(model, processor, "full")
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-5
            )
            trainer = PointTrainer(segmenter, optimizer, forward_sample, TrainingOutput(path))
            with patch("torch.cuda.max_memory_allocated", return_value=0):
                trainer.fit(
                    [sample("train_1", 1), sample("train_2", 2)],
                    [sample("validation", 3)],
                    {"epochs": 1, "validation_interval_steps": 1},
                )
            rows = [json.loads(line) for line in (path / "summary.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [0, 1, 2])
            self.assertEqual(len(list((path / "previews").glob("*.png"))), 2)
            for filename in (
                "train.csv",
                "heldout.csv",
                "timing.csv",
                "summary.csv",
                "checkpoints.csv",
            ):
                with (path / filename).open(newline="") as stream:
                    records = list(csv.DictReader(stream))
                self.assertTrue(records)
                self.assertTrue(all(None not in row for row in records), filename)
            saved = torch.load(path / "checkpoint.pt", weights_only=True)
            self.assertEqual(saved["step"], 2)
            self.assertTrue((path / "checkpoint-best.pt").is_file())
            restored, _ = tiny_model()
            restored.load_state_dict(saved["model"], strict=True)
            for key, tensor in restored.state_dict().items():
                torch.testing.assert_close(tensor, model.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
