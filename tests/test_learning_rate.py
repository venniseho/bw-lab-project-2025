"""Learning-rate endpoints and interrupted-run behavior must match the saved update count."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.data_loading.point_training import loader
from core.learning_rate import learning_rate_for_update
from core.training_resume import TrainingInterrupted, check_resume_config
from scripts.prepare_point_followup import add_learning_rate_comparisons, CONDITIONS, prepare
from tests.test_visible_point_training import annotations
import hashlib
from tests.test_point_experiments import sample
from tests.test_training_resume import make_trainer, Samples


class LearningRateTests(unittest.TestCase):
    def test_staged_preparation_has_eight_initial_and_three_later_configs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {
                "seed": 42,
                "splits": {"train": {}, "test": {}},
                "annotation_sha256": {},
                "learning_rate": 1e-5,
            }
            for split, ids, tiny, key in (
                ("train", [1, 2, 3, 4], 3, "train_annotations"),
                ("test", [101, 102, 103, 104], 101, "heldout_annotations"),
            ):
                path = root / f"{split}.json"
                path.write_text(json.dumps(annotations(ids, tiny)))
                base[key] = str(path)
                base["annotation_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
            source = root / "base.json"
            source.write_text(json.dumps(base))
            output = root / "prepared"
            counts = prepare(source, output, compare_fence_losses=True, compare_learning_rates=True)
            self.assertEqual(len(list(output.glob("*/config.json"))), 11)
            self.assertEqual(len(counts), 22)
            for label in CONDITIONS:
                fixed = json.loads((output / label / "config.json").read_text())
                scheduled = json.loads((output / f"{label}-warmup-cosine/config.json").read_text())
                self.assertEqual(fixed["training_order"], scheduled["training_order"])
                self.assertEqual(fixed["annotation_sha256"], scheduled["annotation_sha256"])
                self.assertEqual(fixed["lr_schedule"]["name"], "constant")
                self.assertEqual(scheduled["lr_schedule"]["name"], "warmup_cosine")
                self.assertEqual(scheduled["loss_region"], "full")
                self.assertEqual(scheduled["validation_interval_steps"], 5000)

    def test_warmup_peak_cosine_and_final_rate(self):
        schedule = {
            "name": "warmup_cosine",
            "warmup_fraction": 0.2,
            "start_factor": 0.1,
            "end_factor": 0.1,
        }
        rates = [learning_rate_for_update(1e-5, index, 20, schedule) for index in range(20)]
        self.assertAlmostEqual(rates[0], 1e-6)
        self.assertAlmostEqual(rates[3], 1e-5)
        self.assertAlmostEqual(rates[-1], 1e-6)
        self.assertEqual(rates[:4], sorted(rates[:4]))
        self.assertEqual(rates[3:], sorted(rates[3:], reverse=True))
        self.assertEqual(learning_rate_for_update(1e-5, 10, 20), 1e-5)
        for invalid in ({"name": "unknown"}, {"name": "warmup_cosine", "warmup_fraction": 0}):
            with self.assertRaises(ValueError):
                learning_rate_for_update(1e-5, 0, 20, invalid)

    def test_all_four_schedule_comparisons_keep_their_control_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counts = []
            for label in CONDITIONS:
                control = root / label
                control.mkdir()
                config = {
                    "lr_schedule": {"name": "constant"},
                    "learning_rate": 1e-5,
                    "loss_region": "full",
                    "train_annotations": "same.json",
                    "training_order": [3, 1, 2],
                    "annotation_sha256": {"train": "hash"},
                }
                (control / "config.json").write_text(json.dumps(config))
                (control / "excluded_targets.csv").write_text("same exclusions\n")
                counts.append({"condition": label, "selected_images": 3})
            extra = add_learning_rate_comparisons(root, counts)
            self.assertEqual(len(extra), 4)
            for label in CONDITIONS:
                control = json.loads((root / label / "config.json").read_text())
                compared = json.loads((root / f"{label}-warmup-cosine/config.json").read_text())
                compared.pop("comparison_control")
                self.assertEqual(compared.pop("lr_schedule")["name"], "warmup_cosine")
                control.pop("lr_schedule")
                self.assertEqual(control, compared)

    def test_resume_uses_the_same_next_rate_and_weights_as_uninterrupted_training(self):
        config = {
            "epochs": 1,
            "training_order": [2, 0, 3, 1],
            "learning_rate": 0.01,
            "splits": {"train": {"instances": 4}},
            "lr_schedule": {
                "name": "warmup_cosine",
                "warmup_fraction": 0.5,
                "start_factor": 0.1,
                "end_factor": 0.1,
            },
        }
        samples = Samples([{}] * 4)

        def factory(epoch, cursor):
            return loader(samples, 0, order=config["training_order"], start=cursor)

        def seeded_trainer(path):
            random.seed(4)
            np.random.seed(4)
            return make_trainer(path)

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("torch.cuda.max_memory_allocated", return_value=0),
        ):
            root = Path(temporary)
            full = seeded_trainer(root / "full")
            full.fit([], [sample("val", 99)], config, train_loader_factory=factory)
            partial = seeded_trainer(root / "partial")
            original = partial.output.record

            def stop(channel, row):
                original(channel, row)
                if channel == "train" and row["step"] == 2:
                    partial.stop_reason = "test"

            partial.output.record = stop
            with self.assertRaises(TrainingInterrupted):
                partial.fit([], [sample("val", 99)], config, train_loader_factory=factory)
            resumed = seeded_trainer(root / "resumed")
            resumed.resume(root / "partial/checkpoint.pt", config)
            resumed.fit([], [sample("val", 99)], config, train_loader_factory=factory)
            for name, tensor in full.segmenter.model.state_dict().items():
                self.assertTrue(
                    torch.equal(tensor, resumed.segmenter.model.state_dict()[name]), name
                )
            rates = []
            for name in ("partial", "resumed"):
                rates.extend(
                    json.loads(line)["learning_rate"]
                    for line in (root / name / "train.jsonl").read_text().splitlines()
                )
            self.assertEqual(
                rates,
                [
                    learning_rate_for_update(0.01, index, 4, config["lr_schedule"])
                    for index in range(4)
                ],
            )
        with self.assertRaises(ValueError):
            check_resume_config({"lr_schedule": {"name": "constant"}}, config)


if __name__ == "__main__":
    unittest.main()
