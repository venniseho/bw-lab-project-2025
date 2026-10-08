"""Mock the documented Tracker API; no torch, transformers, CUDA or weights."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from scripts.sam3_smoke_test import (
    load_rgb, main, predict_point, require_slurm_step, save_prediction,
)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class Sam3SmokeTests(unittest.TestCase):
    def setUp(self):
        rgb = np.full((3, 5, 3), (210, 30, 10), dtype=np.uint8)
        self.image = Image.fromarray(rgb)
        self.masks = np.zeros((1, 3, 3, 5), dtype=bool)
        self.masks[0, 0, 0, 0] = True
        self.masks[0, 1, 1, 3] = True
        self.masks[0, 2, 2, 4] = True
        self.outputs = SimpleNamespace(
            pred_masks=FakeTensor(np.zeros((1, 1, 3, 2, 2), dtype=np.float32)),
            iou_scores=FakeTensor([[[0.2, 0.9, 0.7]]]),
            object_score_logits=FakeTensor([[[2.0]]]),
        )
        self.model = MagicMock()
        self.model.return_value = self.outputs
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.model.config.to_dict.return_value = {"model_type": "sam3_tracker"}
        self.batch = FakeBatch(original_sizes=FakeTensor([[3, 5]]),
                               pixel_values=FakeTensor(np.zeros((1, 3, 4, 4))))
        self.processor = MagicMock()
        self.processor.return_value = self.batch
        self.processor.post_process_masks.return_value = [FakeTensor(self.masks)]
        self.processor.image_processor.to_dict.return_value = {"size": {"height": 1008, "width": 1008}}
        self.processor.target_size = 1008
        self.torch = MagicMock()
        self.torch.__version__ = "mock-torch"
        self.torch.version.cuda = "mock-cuda"
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.get_device_name.return_value = "mock GPU"
        self.torch.cuda.max_memory_allocated.return_value = 1234
        self.torch.are_deterministic_algorithms_enabled.return_value = False
        self.torch.inference_mode.return_value = nullcontext()
        self.transformers = MagicMock()
        self.transformers.__version__ = "mock-transformers"
        self.transformers.Sam3TrackerModel.from_pretrained.return_value = (
            self.model, {"missing_keys": set(), "unexpected_keys": {"unused.pcs.weight"},
                         "mismatched_keys": set(), "error_msgs": []}
        )
        self.transformers.Sam3TrackerProcessor.from_pretrained.return_value = self.processor

    def predict(self):
        return predict_point(self.model, self.processor, self.image, (3, 1), self.torch)

    def fixture(self, root):
        checkpoint = root / "snapshot"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"model_type":"sam3"}', encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"mock weights only")
        image = root / "input.png"
        self.image.save(image)
        return ["--image", str(image), "--point", "3", "1",
                "--checkpoint-dir", str(checkpoint), "--output-dir", str(root / "output"),
                "--checkpoint-revision", "mock-revision"]

    def test_visual_point_api_shapes_rgb_and_postprocessing(self):
        result = self.predict()
        kwargs = self.processor.call_args.kwargs
        self.assertIs(kwargs["images"], self.image)
        self.assertEqual(kwargs["images"].getpixel((0, 0)), (210, 30, 10))
        self.assertEqual(kwargs["input_points"], [[[[3.0, 1.0]]]])
        self.assertEqual(kwargs["input_labels"], [[[1]]])
        self.assertNotIn("text", kwargs)
        self.assertEqual(self.batch.device, "cuda")
        self.assertTrue(self.model.call_args.kwargs["multimask_output"])
        self.assertTrue(self.model.call_args.kwargs["return_dict"])
        post = self.processor.post_process_masks.call_args.kwargs
        self.assertEqual(post, {"mask_threshold": 0.0, "binarize": True,
                                "max_hole_area": 0.0, "max_sprinkle_area": 0.0,
                                "apply_non_overlapping_constraints": False})
        np.testing.assert_array_equal(result.masks, self.masks[0])
        self.assertEqual(result.selected_index, 1)
        self.assertEqual(result.object_score_logits, [[[2.0]]])
        self.assertGreaterEqual(result.inference_seconds, result.cuda_forward_seconds)
        self.assertEqual(self.torch.cuda.synchronize.call_count, 3)

    def test_tied_scores_select_first_candidate_and_single_mask_is_safe(self):
        self.outputs.iou_scores = FakeTensor([[[0.9, 0.9, 0.7]]])
        self.assertEqual(self.predict().selected_index, 0)
        self.processor.post_process_masks.return_value = [FakeTensor(self.masks[:, :1])]
        self.outputs.iou_scores = FakeTensor([[[0.4]]])
        self.outputs.object_score_logits = None
        result = self.predict()
        self.assertEqual(result.masks.shape, (1, 3, 5))
        self.assertEqual(result.selected_index, 0)
        self.assertIsNone(result.object_score_logits)

    def test_bad_output_shapes_nonbinary_masks_and_scores_fail(self):
        bad_masks = [self.masks[0], self.masks[:, :, :2],
                     self.masks[:, :0], self.masks.astype(float)]
        for masks in bad_masks:
            with self.subTest(shape=masks.shape, dtype=masks.dtype):
                self.processor.post_process_masks.return_value = [FakeTensor(masks)]
                with self.assertRaises(ValueError):
                    self.predict()
        self.processor.post_process_masks.return_value = [FakeTensor(self.masks)]
        for scores in ([[0.2, 0.9, 0.7]], [[[0.2, 0.9]]], [[[0.2, np.nan, 0.7]]]):
            self.outputs.iou_scores = FakeTensor(scores)
            with self.assertRaises(ValueError):
                self.predict()

    def test_invalid_points_fail_before_processor_call(self):
        for point in ((-1, 1), (5, 1), (3, 3), (np.nan, 1), (1, np.inf)):
            with self.assertRaises(ValueError):
                predict_point(self.model, self.processor, self.image, point, self.torch)
        self.processor.assert_not_called()

    def test_png_loading_preserves_rgb_channel_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            self.image.save(path)
            np.testing.assert_array_equal(np.asarray(load_rgb(path)), np.asarray(self.image))
            Image.fromarray(np.full((3, 5), 21, dtype=np.uint8)).save(path)
            self.assertEqual(load_rgb(path).getpixel((0, 0)), (21, 21, 21))

    def test_saved_masks_overlay_scores_and_inputs_unchanged(self):
        image_before = np.asarray(self.image).copy()
        masks_before = self.masks.copy()
        result = self.predict()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            report = save_prediction(self.image, result, output, {"prompt_xy": [3, 1]})
            with Image.open(output / "prediction.png") as mask:
                expected = self.masks[0, 1].astype(np.uint8) * 255
                np.testing.assert_array_equal(np.asarray(mask), expected)
                self.assertEqual(mask.mode, "L")
            with Image.open(output / "overlay.png") as overlay:
                self.assertEqual(overlay.mode, "RGB")
                self.assertEqual(overlay.getpixel((0, 0)), (210, 30, 10))
                expected_color = tuple(np.rint(0.55 * np.array([210, 30, 10])
                                              + 0.45 * np.array([0, 255, 255])).astype(int))
                self.assertEqual(overlay.getpixel((3, 1)), expected_color)
            for index in range(3):
                with Image.open(output / f"candidate_{index:02d}.png") as mask:
                    np.testing.assert_array_equal(np.asarray(mask), self.masks[0, index] * 255)
            self.assertEqual([c["model_selected"] for c in report["candidates"]], [False, True, False])
            saved = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            with self.assertRaises(FileExistsError):
                save_prediction(self.image, result, output, {})
        np.testing.assert_array_equal(np.asarray(self.image), image_before)
        np.testing.assert_array_equal(self.masks, masks_before)

    def test_slurm_guard_requires_compute_step_not_just_salloc_job(self):
        for env in ({}, {"SLURM_JOB_ID": "7"}, {"SLURM_STEP_ID": "0"}):
            with self.assertRaises(RuntimeError):
                require_slurm_step(env)
        require_slurm_step({"SLURM_JOB_ID": "7", "SLURM_STEP_ID": "0"})

    def test_cli_end_to_end_mock_local_only_loading_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            with patch.dict("sys.modules", {"torch": self.torch, "transformers": self.transformers}):
                main(args)
            model_kwargs = self.transformers.Sam3TrackerModel.from_pretrained.call_args.kwargs
            self.assertTrue(model_kwargs["local_files_only"])
            self.assertFalse(model_kwargs["trust_remote_code"])
            self.assertTrue(model_kwargs["use_safetensors"])
            self.assertTrue(model_kwargs["output_loading_info"])
            self.assertEqual(model_kwargs["dtype"], self.torch.float32)
            self.assertEqual(model_kwargs["attn_implementation"], "sdpa")
            self.model.to.assert_called_once_with("cuda")
            self.model.eval.assert_called_once_with()
            processor_kwargs = self.transformers.Sam3TrackerProcessor.from_pretrained.call_args.kwargs
            self.assertTrue(processor_kwargs["local_files_only"])
            report = json.loads((root / "output" / "metadata.json").read_text(encoding="utf-8"))
            weights = next(f for f in report["checkpoint_files"] if f["file"] == "model.safetensors")
            self.assertEqual(weights["sha256"], hashlib.sha256(b"mock weights only").hexdigest())
            self.assertEqual(report["checkpoint_revision"], "mock-revision")
            self.assertEqual(report["selected_candidate_index"], 1)
            self.assertEqual(report["image_size_hw"], [3, 5])
            self.assertEqual(report["prompt_xy"], [3.0, 1.0])
            self.assertEqual(report["versions"]["torch"], "mock-torch")
            self.assertEqual(report["peak_cuda_allocated_bytes"], 1234)
            self.assertEqual(report["loading_info"]["unexpected_keys"], ["unused.pcs.weight"])

    def test_cuda_unavailable_fails_without_cpu_fallback_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            self.torch.cuda.is_available.return_value = False
            with patch.dict("sys.modules", {"torch": self.torch, "transformers": self.transformers}):
                with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                    main(args)
            self.transformers.Sam3TrackerModel.from_pretrained.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_incomplete_weights_fail_instead_of_using_random_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            self.transformers.Sam3TrackerModel.from_pretrained.return_value = (
                self.model, {"missing_keys": ["vision_encoder.some_weight"]}
            )
            with patch.dict("sys.modules", {"torch": self.torch, "transformers": self.transformers}):
                with self.assertRaisesRegex(RuntimeError, "Incomplete/incompatible"):
                    main(args)
            self.model.to.assert_not_called()
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
