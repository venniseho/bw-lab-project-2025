from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image

from bwlab.models.batched_sam3 import BatchedSam3GridAdapter, _tensor_state
from bwlab.runtime import RuntimeConfig, configure_model


class BatchModel:
    def __init__(self):
        self.batches = []
        self.queries = []
        self.backbone = SimpleNamespace(forward_image=self.embed)
        self.inst_interactive_predictor = SimpleNamespace(model=SimpleNamespace(
            sam_mask_decoder=SimpleNamespace(conv_s0=lambda x: x*2, conv_s1=lambda x: x*3)))

    def embed(self, batch):
        self.batches.append(batch.clone())
        return {"sam2_backbone_out": {"backbone_fpn": [batch.clone(), batch.clone()]}}

    def predict_inst_batch(self, state, **kwargs):
        if not torch.is_inference_mode_enabled():
            raise AssertionError("missing inference mode")
        self.queries.append(kwargs)
        masks, scores = [], []
        for h, w, points, labels in zip(state["original_heights"], state["original_widths"],
                                       kwargs["point_coords_batch"], kwargs["point_labels_batch"]):
            assert points.shape[1:] == (1, 2)
            np.testing.assert_array_equal(labels, np.ones((len(points), 1)))
            result = np.zeros((len(points), 3, h, w), dtype=bool)
            for i, point in enumerate(points[:, 0]):
                x, y = np.rint(point).astype(int)
                result[i, 0, y, x] = True
                result[i, 1] = True
            quality = np.tile([0.9, 0.8, 0.1], (len(points), 1))
            masks.append(result[0] if len(points) == 1 else result)
            scores.append(quality[0] if len(points) == 1 else quality)
        return masks, scores, None


def processor():
    return SimpleNamespace(device="cpu", transform=v2.Compose([
        v2.ToDtype(torch.uint8, scale=True), v2.Resize((8, 8)),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5]*3, [0.5]*3)]))


class BatchTests(unittest.TestCase):
    def test_tensor_preprocessing_matches_native_pil_and_projects_once(self):
        rng = np.random.default_rng(12)
        images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in [(7, 12), (10, 5)]]
        model, proc = BatchModel(), processor()
        state = _tensor_state(model, proc, images)
        expected = torch.stack([proc.transform(v2.functional.to_image(Image.fromarray(im))) for im in images])
        torch.testing.assert_close(model.batches[0], expected, rtol=0, atol=0)
        self.assertEqual(state["original_heights"], [7, 10])
        torch.testing.assert_close(state["backbone_out"]["sam2_backbone_out"]["backbone_fpn"][0], expected*2)

    def test_independent_queries_variable_sizes_order_and_chunk_tail(self):
        model = BatchModel()
        images = [np.full((h, w, 3), 127, np.uint8) for h, w in [(9, 15), (11, 7)]]
        results = BatchedSam3GridAdapter(model, processor(), prompt_batch_size=4).predict_batch(images)
        self.assertEqual(len(model.batches), 1)
        self.assertEqual([len(q["point_coords_batch"][0]) for q in model.queries], [4, 4, 1])
        for (proposals, timing), rgb in zip(results, images):
            self.assertEqual(len(proposals), 27)
            self.assertEqual([p.metadata["proposal_id"] for p in proposals], list(range(27)))
            self.assertEqual([p.metadata["grid_point_index"] for p in proposals], [i//3 for i in range(27)])
            self.assertEqual(proposals[0].mask().shape, rgb.shape[:2])
            self.assertEqual(timing["image_batch_size"], 2)

    def test_native_state_is_cleared_on_prediction_failure(self):
        model = BatchModel()
        def fail(*args, **kwargs):
            model.inst_interactive_predictor._features = "stale features"
            raise RuntimeError("test failure")
        model.predict_inst_batch = fail
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            BatchedSam3GridAdapter(model, processor()).predict_batch([np.zeros((8, 8, 3), np.uint8)])
        self.assertIsNone(model.inst_interactive_predictor._features)
        self.assertFalse(model.inst_interactive_predictor._is_batch)

    def test_runtime_compiles_only_encoder_and_preserves_complex_buffers(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(3, 2)
        model.backbone.forward_image = model.backbone.forward
        model.register_buffer("rope", torch.tensor([1+2j]))
        original = model.rope.clone()
        compiled = lambda x: x
        previous = torch.backends.cudnn.benchmark
        try:
            with patch("torch.compile", return_value=compiled) as compile_call:
                info = configure_model(model, RuntimeConfig(compile_encoder=True))
                self.assertTrue(torch.backends.cudnn.benchmark)
        finally:
            torch.backends.cudnn.benchmark = previous
        self.assertEqual(compile_call.call_count, 1)
        self.assertIs(model.backbone.forward_image, compiled)
        self.assertEqual(info["parameter_count"], 8)
        self.assertEqual(info["buffer_bytes"], 8)
        torch.testing.assert_close(model.rope, original)
