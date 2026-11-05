"""Check cloud logging contracts without credentials, network calls or GPUs."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.training_output import TrainingOutput
from experiments.sam3_point_finetuning.experiment import build_output


class TrainingOutputTests(unittest.TestCase):
    def test_timing_is_saved_locally_and_logged_to_separate_cloud_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Mock()
            output = TrainingOutput(Path(folder), run)
            report = {
                "phase": "validation",
                "epoch": 0,
                "instances": 504,
                "images_per_second": 10.0,
            }
            output.record("timing", report)
            self.assertEqual(json.loads((Path(folder) / "timing.jsonl").read_text()), report)
            run.log.assert_called_with(
                {
                    "timing/validation/epoch": 0,
                    "timing/validation/instances": 504,
                    "timing/validation/images_per_second": 10.0,
                }
            )

    def test_online_setup_uses_workspace_and_distinct_metric_axes(self):
        run = Mock(url="https://wandb.ai/example/project/runs/test")
        sdk = SimpleNamespace(init=Mock(return_value=run), Image=Mock())
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(sys.modules, {"wandb": sdk}),
            patch.dict(
                os.environ,
                {
                    "WANDB_PROJECT": "research",
                    "WANDB_ENTITY": "lab",
                    "WANDB_RUN_GROUP": "comparison",
                },
            ),
        ):
            output = build_output(Path(folder), {"scope": "full"}, "online")
            options = sdk.init.call_args.kwargs
            self.assertEqual(
                (options["project"], options["entity"], options["group"]),
                ("research", "lab", "comparison"),
            )
            self.assertEqual(options["mode"], "online")
            run.define_metric.assert_any_call("train/*", step_metric="train/step")
            run.define_metric.assert_any_call("heldout/*", step_metric="heldout/epoch")
            output.close()
            run.finish.assert_called_once()

    def test_baseline_and_training_use_correct_axes_and_keep_local_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Mock()
            output = TrainingOutput(Path(folder), run)
            baseline = {
                "phase": "heldout_summary",
                "epoch": 0,
                "mean_iou": 0.6,
                "mean_loss": 0.4,
                "instances": 31,
            }
            output.record("summary", baseline)
            run.log.assert_called_with(
                {
                    "heldout/mean_iou": 0.6,
                    "heldout/mean_loss": 0.4,
                    "heldout/instances": 31,
                    "heldout/epoch": 0,
                }
            )
            self.assertEqual(json.loads((Path(folder) / "summary.jsonl").read_text()), baseline)
            output.record("train", {"step": 1, "epoch": 1, "loss": 0.2, "key": "object"})
            run.log.assert_called_with({"train/step": 1, "train/epoch": 1, "train/loss": 0.2})
            output.close(exit_code=1)
            run.finish.assert_called_once_with(exit_code=1)

    def test_preview_upload_uses_saved_image_and_caption(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("core.training_output.training_contact_sheet") as render,
        ):
            run, image = Mock(), Mock(return_value="media")
            output = TrainingOutput(Path(folder), run, image)
            output.preview({"__key__": "object"}, [2, 3], {"iou": 0.6}, {"iou": 0.7}, 1, 0)
            path = Path(folder) / "previews/epoch-1-0.png"
            self.assertEqual(render.call_args.args[0], path)
            self.assertEqual(image.call_args.args[0], str(path))
            self.assertIn("before=0.600, after=0.700", image.call_args.kwargs["caption"])
            run.log.assert_called_with({"examples/epoch": 1, "examples/instance_0": "media"})

    def test_disabled_tracking_needs_no_wandb_import(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"wandb": None}):
            output = build_output(Path(folder), {"scope": "frozen_encoder"}, "disabled")
            self.assertIsNone(output.tracking)
            output.close()
