"""Transfer evaluation must pair baseline/updated weights without optimization."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.evaluate_point_transfer import restore_final, run
from tests.test_point_experiments import sample


class FakeSegmenter:
    def __init__(self, model, processor, scope):
        self.model = model

    def train(self, training=True):
        self.model.train(training)

    def encode(self, image):
        return image.shape[:2]

    def predict_encoded(self, encoded, point, shape):
        logits = self.model.weight.reshape(1, 1, 1).expand(3, *shape)
        return logits, torch.tensor([0.9, 0.5, 0.1])


class PointTransferTests(unittest.TestCase):
    def test_end_to_end_pairs_predictions_saves_records_and_never_trains(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            config = {
                "seed": 42,
                "native_root": "native",
                "native_revision": "verified",
                "checkpoint": "native.pt",
                "workers": 0,
                "splits": {"test": {"instances": 1}},
            }
            metadata = {"config": config, "scope": "mask_decoder_only"}
            complete = {"steps": 10, "checkpoint": {"step": 10, "epoch": 1}}
            (source / "metadata.json").write_text(json.dumps(metadata))
            (source / "complete.json").write_text(json.dumps(complete))
            model = torch.nn.Linear(1, 1, bias=False)
            model.weight.data.fill_(1)
            torch.save(
                {
                    "format": "bwlab_point_finetune_v1",
                    "step": 10,
                    **metadata,
                    "model": model.state_dict(),
                },
                source / "checkpoint.pt",
            )
            model.weight.data.fill_(-1)
            target = root / "fragmentation" / "config.json"
            target.parent.mkdir()
            target.write_text(json.dumps(config))
            destination = root / "results"
            with (
                patch("scripts.evaluate_point_transfer.validate_config"),
                patch("scripts.evaluate_point_transfer.seed_everything"),
                patch(
                    "scripts.evaluate_point_transfer.load_scene_model",
                    return_value=(model, None, {}),
                ),
                patch("scripts.evaluate_point_transfer.PointSegmenter", FakeSegmenter),
                patch("scripts.evaluate_point_transfer.build_data", return_value=([], [sample()])),
                patch(
                    "scripts.evaluate_point_transfer.loader", side_effect=lambda data, workers: data
                ),
                patch("torch.cuda.reset_peak_memory_stats"),
                patch("torch.cuda.get_device_name", return_value="test GPU"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
            ):
                run(source, [target], destination, "disabled")
            result = json.loads((destination / "complete.json").read_text())
            summaries = result["results"]["fragmentation"]
            self.assertEqual(summaries["pretrained"]["mean_full_iou"], 0)
            self.assertGreater(summaries["finetuned"]["mean_full_iou"], 0)
            self.assertEqual(result["training_updates"], 0)
            self.assertIsNone(model.weight.grad)
            self.assertEqual(float(model.weight.detach().squeeze()), 1)
            with (destination / "fragmentation" / "heldout.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["step"] for row in rows], ["0", "10"])
            self.assertEqual(rows[0]["key"], rows[1]["key"])
            self.assertEqual(rows[0]["point_x"], rows[1]["point_x"])
            self.assertEqual(rows[0]["point_y"], rows[1]["point_y"])
            self.assertIn("selected_mask_rle", rows[1])
            self.assertEqual(
                len(list((destination / "fragmentation" / "previews").glob("*.png"))), 1
            )

    def test_checkpoint_rejects_wrong_step_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save({"format": "bwlab_point_finetune_v1", "step": 9}, path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                restore_final(torch.nn.Linear(1, 1), path, {}, {"steps": 10})


if __name__ == "__main__":
    unittest.main()
