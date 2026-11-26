"""Accumulated gradients must average correctly and survive a mid-window stop."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.data_loading.point_training import loader
from core.stimuli.point_mixture import PointMixture
from core.tasks.point_segmentation import forward_sample
from core.training_resume import TrainingInterrupted, check_resume_config
from scripts.prepare_point_week import prepare, FENCES
from tests.test_point_experiments import sample
from tests.test_training_resume import make_trainer, Samples
from tests.test_visible_point_training import annotations


class AccumulationTests(unittest.TestCase):
    def test_matches_mean_batch_gradients_and_flushes_short_final_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, reference = (
                make_trainer(root / "accumulated"),
                make_trainer(root / "reference"),
            )
            trainer.task = reference.task = forward_sample
            trainer.accumulation_steps = 2
            items = [sample(str(i), i) for i in range(3)]
            trainer.train_epoch(items, 1)
            reference.segmenter.train(True)
            for chunk in (items[:2], items[2:]):
                reference.optimizer.zero_grad(set_to_none=True)
                torch.stack(
                    [reference.task(reference.segmenter, item)[0] for item in chunk]
                ).mean().backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters, 1.0)
                reference.optimizer.step()
            self.assertEqual(trainer.steps, 3)
            self.assertEqual(trainer.optimizer_updates, 2)
            self.assertEqual(trainer.accumulated_images, 0)
            for name, value in trainer.segmenter.model.state_dict().items():
                self.assertTrue(
                    torch.allclose(
                        value, reference.segmenter.model.state_dict()[name], atol=1e-7, rtol=0
                    ),
                    name,
                )

    def test_resume_preserves_unfinished_gradient_window_and_exact_weights(self):
        config = {
            "epochs": 1,
            "training_order": [4, 0, 3, 1, 2],
            "checkpoint_interval_steps": 1,
            "accumulation_steps": 3,
        }
        items = Samples([{}] * 5)

        def factory(epoch, cursor):
            return loader(items, 0, order=config["training_order"], start=cursor)

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("torch.cuda.max_memory_allocated", return_value=0),
        ):
            root = Path(temporary)
            random.seed(7)
            np.random.seed(7)
            full = make_trainer(root / "full")
            full.fit([], [sample("val", 99)], config, train_loader_factory=factory)
            random.seed(7)
            np.random.seed(7)
            partial = make_trainer(root / "partial")
            original = partial.output.record

            def stop(channel, row):
                original(channel, row)
                if channel == "train" and row["step"] == 2:
                    partial.stop_reason = "test shutdown"

            partial.output.record = stop
            with self.assertRaises(TrainingInterrupted):
                partial.fit([], [sample("val", 99)], config, train_loader_factory=factory)
            resumed = make_trainer(root / "resumed")
            resumed.resume(root / "partial/checkpoint.pt", config)
            self.assertEqual(resumed.accumulated_images, 2)
            self.assertEqual(resumed.optimizer_updates, 0)
            resumed.fit([], [sample("val", 99)], config, train_loader_factory=factory)
            self.assertEqual(resumed.optimizer_updates, 2)
            self.assertEqual(resumed.accumulated_images, 0)
            for name, value in full.segmenter.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, resumed.segmenter.model.state_dict()[name]), name
                )
            with self.assertRaisesRegex(ValueError, "accumulation"):
                check_resume_config(config, {**config, "accumulation_steps": 16})

    def test_balanced_assignment_is_stable_per_source_and_does_not_expand_dataset(self):
        conditions = [{"name": "identity"}, {"name": "fence", "parameters": FENCES["fence-medium"]}]
        first = PointMixture(conditions, list(range(17)), 42)
        second = PointMixture(conditions, list(reversed(range(17))), 42)
        self.assertEqual(first.assignment, second.assignment)
        self.assertEqual(sum(first.counts.values()), 17)
        self.assertEqual(sorted(first.counts.values()), [8, 9])
        for image_id in range(17):
            left, right = (
                first(deepcopy(sample(str(image_id), image_id))),
                second(deepcopy(sample(str(image_id), image_id))),
            )
            self.assertEqual(
                left["metadata"]["training_condition"], right["metadata"]["training_condition"]
            )
            self.assertTrue(np.array_equal(left["image"], right["image"]))

    def test_batch_preparation_matches_cohorts_and_pilot_budgets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {
                "transform": "fence",
                "transform_parameters": FENCES["fence-medium"],
                "seed": 42,
                "training_order": list(range(24)),
                "annotation_sha256": {},
                "splits": {"train": {}, "test": {}},
                "requested_training_fraction": 0.5,
            }
            for split, ids, key in (
                ("train", range(24), "train_annotations"),
                ("test", range(100, 108), "heldout_annotations"),
            ):
                path = root / f"{split}.json"
                path.write_text(json.dumps(annotations(ids, -1)))
                base[key] = str(path)
                base["annotation_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
                base["splits"][split] = {"instances": len(ids)}
            path = root / "base.json"
            path.write_text(json.dumps(base))
            plan = prepare(path, root / "prepared", pilot_images=20)
            self.assertEqual(len(plan["jobs"]), 11)
            self.assertEqual(len(plan["validation_configs"]), 5)
            full_configs = []
            pilot_hashes = set()
            for job in plan["jobs"]:
                config = json.loads(Path(job["config"]).read_text())
                if job["evaluate_all"]:
                    full_configs.append(config)
                else:
                    pilot_hashes.add(config["annotation_sha256"]["train"])
                    self.assertEqual(job["images"], 20)
                    self.assertEqual(
                        job["optimizer_updates"], 20 if config["accumulation_steps"] == 1 else 2
                    )
            self.assertEqual(len(pilot_hashes), 1)
            self.assertTrue(
                all(c["training_order"] == base["training_order"] for c in full_configs)
            )
            self.assertTrue(
                all(
                    c["annotation_sha256"]["train"] == base["annotation_sha256"]["train"]
                    for c in full_configs
                )
            )


if __name__ == "__main__":
    unittest.main()
