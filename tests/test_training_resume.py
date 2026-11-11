"""Interrupted training preserves completed samples, optimizer and random state."""

import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from core.data_loading.download_ahead import cached_first_order, DownloadReadyDataset
from core.data_loading.point_training import loader
from core.models.point_sam3 import PointSegmenter
from core.tasks.point_segmentation import forward_sample
from core.trainer import PointTrainer
from core.training_output import TrainingOutput
from core.training_resume import TrainingInterrupted, shutdown_signals
from tests.test_point_finetuning import tiny_model
from tests.test_point_experiments import sample


class Samples:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return sample(str(index), index)


def stochastic_task(segmenter, item, **kwargs):
    loss, *rest = forward_sample(segmenter, item, **kwargs)
    if torch.is_grad_enabled():
        loss = loss * (
            0.9 + 0.1 * torch.rand(()) + 0.01 * random.random() + 0.01 * np.random.rand()
        )
    return loss, *rest


def make_trainer(path):
    path.mkdir()
    (path / "previews").mkdir()
    model, processor = tiny_model()
    segmenter = PointSegmenter(model, processor, "full")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    return PointTrainer(segmenter, optimizer, stochastic_task, TrainingOutput(path))


class ResumeTests(unittest.TestCase):
    def test_preparation_can_defer_training_files_but_requires_validation_cache(self):
        from scripts.prepare_point_training import _prepare_splits

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            annotation = root / "annotations.json"
            annotation.write_text("{}")
            (root / "val2017").mkdir()
            (root / "val2017/2.jpg").touch()
            args = SimpleNamespace(
                download=True,
                download_ahead=True,
                official_splits=True,
                train_images=0,
                heldout_images=0,
                seed=42,
            )

            def splits(*unused):
                return {"images": [{"id": 1, "file_name": "1.jpg"}]}

            with (
                patch("scripts.prepare_point_training.ensure_annotations", return_value=annotation),
                patch(
                    "scripts.prepare_point_training.select_official_split",
                    side_effect=[splits(), {"images": [{"id": 2, "file_name": "2.jpg"}]}],
                ),
                patch("scripts.prepare_point_training.download") as download,
            ):
                result = _prepare_splits(args, root)
                self.assertEqual(
                    result["train"]["images"][0]["file_name"], str(root / "train2017/1.jpg")
                )
                self.assertEqual(download.call_count, 1)
                self.assertIn("val2017", download.call_args.args[0])

    def test_cached_prefix_and_pending_order_cover_every_image_once(self):
        with tempfile.TemporaryDirectory() as folder:
            images = [{"id": i, "file_name": str(Path(folder) / f"{i}.jpg")} for i in range(6)]
            for i in (1, 3, 5):
                Path(images[i]["file_name"]).touch()
            order, count = cached_first_order(images, 42)
            self.assertEqual(count, 3)
            self.assertEqual(set(order[:count]), {1, 3, 5})
            self.assertEqual(order[count:], [0, 2, 4])
            self.assertEqual(cached_first_order(images, 42)[0], order)
            self.assertEqual(sorted(order), list(range(6)))

    def test_partial_download_is_never_read_and_wait_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.jpg"
            path.with_suffix(".jpg.part").touch()
            samples = Samples([{"image_path": str(path)}])
            ready = DownloadReadyDataset(samples, timeout=0.01)
            with self.assertRaises(TimeoutError):
                ready[0]
            path.touch()
            self.assertIn("cache_wait_seconds", ready[0]["metadata"])

    def test_resume_matches_uninterrupted_weights_and_sample_sequence(self):
        config = {"epochs": 1, "training_order": [2, 0, 3, 1], "checkpoint_interval_steps": 1}
        samples = Samples([{}] * 4)

        def factory(epoch, cursor):
            return loader(samples, 0, order=config["training_order"], start=cursor)

        with (
            tempfile.TemporaryDirectory() as folder,
            patch("torch.cuda.max_memory_allocated", return_value=0),
        ):
            root = Path(folder)
            random.seed(4)
            np.random.seed(4)
            full = make_trainer(root / "full")
            full.fit([], [sample("validation", 99)], config, train_loader_factory=factory)
            random.seed(4)
            np.random.seed(4)
            partial = make_trainer(root / "partial")
            original = partial.output.record

            def stop(channel, row):
                original(channel, row)
                if channel == "train" and row["step"] == 2:
                    partial.stop_reason = "test shutdown"

            partial.output.record = stop
            with self.assertRaises(TrainingInterrupted):
                partial.fit([], [sample("validation", 99)], config, train_loader_factory=factory)
            self.assertTrue((root / "partial/interrupted.json").is_file())
            resumed = make_trainer(root / "resumed")
            resumed.resume(root / "partial/checkpoint.pt", config)
            self.assertEqual(resumed.epoch_step, 2)
            resumed.fit([], [sample("validation", 99)], config, train_loader_factory=factory)
            for name, value in full.segmenter.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, resumed.segmenter.model.state_dict()[name]), name
                )
            rows = []
            for name in ("partial", "resumed"):
                rows.extend(
                    json.loads(line)
                    for line in (root / name / "train.jsonl").read_text().splitlines()
                )
            self.assertEqual([row["image_id"] for row in rows], [2, 0, 3, 1])
            with (root / "partial/checkpoints.csv").open(newline="") as stream:
                saves = list(csv.DictReader(stream))
            self.assertEqual(saves[-1]["validation_mean_iou"], "")
            bad = {**config, "training_order": [0, 1, 2, 3]}
            with self.assertRaises(ValueError):
                resumed.resume(root / "partial/checkpoint.pt", bad)

    def test_signal_only_sets_stop_flag_and_restores_handlers(self):
        import signal

        trainer = type("Trainer", (), {"stop_reason": None})()
        original = signal.getsignal(signal.SIGTERM)
        with shutdown_signals(trainer):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            self.assertEqual(trainer.stop_reason, "SIGTERM")
        self.assertEqual(signal.getsignal(signal.SIGTERM), original)


if __name__ == "__main__":
    unittest.main()
