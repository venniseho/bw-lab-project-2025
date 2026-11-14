"""Visible-region supervision must ignore bars without ignoring visible background."""

import csv
from functools import partial
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from pycocotools import mask as masks

from core.data_loading.point_followup import filter_targets, training_subset
from core.models.point_sam3 import PointSegmenter
from core.stimuli.fence import fence_mask
from core.stimuli.point_transforms import PointTransform, point_transform_seed
from core.tasks.point_segmentation import candidate_ious, forward_sample, mask_loss
from core.trainer import PointTrainer
from core.training_output import TrainingOutput
from core.training_resume import check_resume_config
from scripts.prepare_point_followup import prepare
from tests.test_point_finetuning import tiny_model
from tests.test_point_experiments import sample


def annotations(ids, tiny_id):
    images, objects = [], []
    for image_id in ids:
        image = {"id": image_id, "width": 40, "height": 20, "file_name": f"/{image_id}.jpg"}
        target = np.ones((20, 40), np.uint8)
        if image_id == tiny_id:
            occlusion = fence_mask(20, 40, 24, 40, seed=point_transform_seed(42, image_id))
            y, x = np.argwhere(occlusion)[0]
            target[:] = 0
            target[y, x] = 1
        encoded = masks.encode(np.asfortranarray(target))
        encoded["counts"] = encoded["counts"].decode("ascii")
        images.append(image)
        objects.append(
            {
                "id": image_id * 10,
                "image_id": image_id,
                "category_id": 1,
                "iscrowd": 0,
                "area": int(target.sum()),
                "bbox": [0, 0, 40, 20],
                "segmentation": encoded,
            }
        )
    return {"images": images, "annotations": objects, "categories": [{"id": 1, "name": "object"}]}


class VisiblePointTests(unittest.TestCase):
    def test_best_checkpoint_uses_the_explicit_visible_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            model, processor = tiny_model()
            segmenter = PointSegmenter(model, processor, "mask_decoder_only")
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
            trainer = PointTrainer(segmenter, optimizer, forward_sample, TrainingOutput(path))
            config = {"checkpoint_metric": "mean_visible_iou"}
            trainer.last_summary = {
                "validation_set": "interval",
                "mean_iou": 0.9,
                "mean_visible_iou": 0.2,
            }
            trainer.last_validation_step = 0
            trainer.save_checkpoint(config, 0)
            trainer.steps = trainer.last_validation_step = 1
            trainer.last_summary.update(mean_iou=0.7, mean_visible_iou=0.8)
            trainer.save_checkpoint(config, 1)
            saved = torch.load(path / "checkpoint-best.pt", weights_only=True)
            self.assertEqual(saved["step"], 1)
            self.assertEqual(trainer.best_iou, 0.8)

    def test_occluded_logits_do_not_affect_visible_loss_quality_targets_or_gradients(self):
        target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        valid = torch.tensor([[True, False, True, False]])
        logits = torch.tensor(
            [[[2.0, -2.0, -2.0, 2.0]], [[-2.0, 2.0, 2.0, -2.0]], [[0.5, 0.5, -0.5, -0.5]]],
            requires_grad=True,
        )
        scores = torch.tensor([0.3, 0.2, 0.4], requires_grad=True)
        loss, metrics = mask_loss(logits, scores, target, valid)
        changed = logits.detach().clone()
        changed[:, :, ~valid[0]] *= -100
        changed_loss, changed_metrics = mask_loss(changed, scores, target, valid)
        torch.testing.assert_close(loss, changed_loss)
        self.assertEqual(metrics, changed_metrics)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(logits.grad[:, :, ~valid[0]])), 0)
        self.assertGreater(int(torch.count_nonzero(logits.grad[:, :, valid[0]])), 0)
        self.assertGreater(int(torch.count_nonzero(scores.grad)), 0)
        expected = ((scores.detach() - candidate_ious(logits, target, valid)) ** 2).mean()
        self.assertAlmostEqual(metrics["quality_mse"], float(expected))

    def test_visible_iou_ignores_bars_but_counts_visible_background_false_positives(self):
        target = torch.tensor([[1, 1, 0, 0]])
        valid = torch.tensor([[True, False, True, False]])
        logits = torch.tensor([[[5.0, -5.0, -5.0, 5.0]]])
        self.assertAlmostEqual(float(candidate_ious(logits, target)[0]), 1 / 3)
        self.assertEqual(float(candidate_ious(logits, target, valid)[0]), 1)
        logits[0, 0, 2] = 5
        self.assertEqual(float(candidate_ious(logits, target, valid)[0]), 0.5)

    def test_no_occlusion_gives_identical_full_and_visible_metrics(self):
        model, processor = tiny_model()
        segmenter = PointSegmenter(model, processor, "mask_decoder_only")
        _, metrics, *_ = forward_sample(segmenter, sample())
        self.assertEqual(metrics["full_iou"], metrics["visible_iou"])
        self.assertEqual(metrics["full_loss"], metrics["visible_loss"])

    def test_fully_hidden_target_never_gets_a_fallback_point_or_model_forward(self):
        item = sample()
        item["visible_segmentation"] = np.zeros_like(item["segmentation"])
        segmenter = Mock()
        with self.assertRaisesRegex(ValueError, "Fully hidden"):
            forward_sample(segmenter, item)
        segmenter.predict.assert_not_called()

    def test_eligibility_filter_matches_runtime_fence_and_records_exclusions(self):
        data = annotations([1, 2, 3], tiny_id=1)
        parameters = {"bar_width": 24, "bar_spacing": 40, "bar_angle": 90.0}
        selected, order = filter_targets(data, [1, 2, 3], "fence", parameters, 42)
        self.assertEqual(order, [2, 3])
        self.assertEqual(selected["selection_counts"]["excluded_fully_hidden"], 1)
        for image, annotation in zip(data["images"], data["annotations"]):
            target = masks.decode(annotation["segmentation"]).astype(bool)
            item = {
                "__key__": str(image["id"]),
                "image": np.ones((20, 40, 3), np.uint8),
                "segmentation": target,
                "metadata": {"image_id": image["id"]},
            }
            transformed = PointTransform("fence", 42, parameters)(item)
            self.assertEqual(bool(transformed["visible_segmentation"].any()), image["id"] in order)
        self.assertEqual(len(data["images"]), 3)

    def test_half_preparation_saves_fixed_subset_counts_hashes_and_5000_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = annotations([1, 2, 3, 4], tiny_id=3)
            val = annotations([101, 102, 103, 104], tiny_id=101)
            base = {"seed": 42, "splits": {"train": {}, "test": {}}, "annotation_sha256": {}}
            for split, data, key in (
                ("train", train, "train_annotations"),
                ("test", val, "heldout_annotations"),
            ):
                path = root / f"{split}.json"
                path.write_text(json.dumps(data))
                base[key] = str(path)
                base["annotation_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
            base_path = root / "base.json"
            base_path.write_text(json.dumps(base))
            output = root / "prepared"
            counts = prepare(base_path, output, fence_loss_region="visible")
            ids = json.loads((output / "requested_training_ids.json").read_text())
            self.assertEqual(ids, training_subset(train, 0.5, 42))
            self.assertEqual(len(ids), 2)
            high = json.loads((output / "fence-high/config.json").read_text())
            self.assertEqual(high["validation_interval_steps"], 5000)
            self.assertEqual(high["loss_region"], "visible")
            self.assertEqual(high["training_order"], [2])
            self.assertEqual(high["splits"]["test"]["instances"], 3)
            self.assertTrue(high["drop_fully_hidden"])
            for row in counts:
                self.assertEqual(
                    row["candidate_images"], row["selected_images"] + row["excluded_fully_hidden"]
                )
            with (output / "fence-high/excluded_targets.csv").open() as stream:
                dropped = list(csv.DictReader(stream))
            self.assertEqual({row["split"] for row in dropped}, {"train", "validation"})
            self.assertEqual(base_path.read_text(), json.dumps(base))

    def test_both_metrics_reach_validation_csv_wandb_and_resume_guards(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("torch.cuda.max_memory_allocated", return_value=0),
        ):
            path = Path(temporary)
            model, processor = tiny_model()
            segmenter = PointSegmenter(model, processor, "mask_decoder_only")
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-5
            )
            tracking = Mock()
            output = TrainingOutput(path, tracking)
            item = PointTransform("fence", 42, {"bar_width": 2, "bar_spacing": 4})(sample())
            trainer = PointTrainer(
                segmenter, optimizer, partial(forward_sample, loss_region="visible"), output
            )
            trainer.evaluate([item], 0)
            with (path / "summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["mean_iou"], row["mean_full_iou"])
            self.assertIn("mean_visible_loss", row)
            summary_log = tracking.log.call_args.args[0]
            self.assertIn("heldout/mean_visible_iou", summary_log)
            self.assertIn("heldout/mean_full_loss", summary_log)
        with self.assertRaises(ValueError):
            check_resume_config({"loss_region": "full"}, {"loss_region": "visible"})


if __name__ == "__main__":
    unittest.main()
