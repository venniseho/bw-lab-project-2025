"""Exercise real autograd through a small stand-in for the interactive API."""

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from PIL import Image

from core.data_loading.point_training import dataset, loader
from core.samples import foreground_point
from core.trainer import checkpoint, PointTrainer
from core.models.point_sam3 import check_step, gradient_probes, enable_encoder_autograd
from core.tasks.point_segmentation import mask_loss, forward_sample, random_target_point
from core.models.point_sam3 import point_forward, set_scope, PointSegmenter
from core.training_output import TrainingOutput
from core.training_timing import PhaseTiming


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = nn.Module()
        self.vision_backbone.trunk = nn.Conv2d(3, 4, 1)
        self.vision_backbone.sam2_convs = nn.Conv2d(4, 4, 1)
        self.vision_backbone.convs = nn.Conv2d(4, 4, 1)

    def forward_image(self, pixels):
        features = self.vision_backbone.sam2_convs(self.vision_backbone.trunk(pixels))
        return {"sam2_backbone_out": {"backbone_fpn": [features, features, features]}}


class Prompts(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Parameter(torch.ones(1, 4, 1, 1) * 0.1)

    def forward(self, points, boxes, masks):
        return self.embedding.flatten(2).transpose(1, 2), self.embedding

    def get_dense_pe(self):
        return torch.zeros_like(self.embedding)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_s0 = nn.Conv2d(4, 4, 1)
        self.conv_s1 = nn.Conv2d(4, 4, 1)
        self.masks = nn.Conv2d(4, 3, 1)
        self.quality = nn.Linear(4, 3)

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        repeat_image,
        high_res_features,
    ):
        features = image_embeddings + dense_prompt_embeddings + sum(high_res_features)
        return self.masks(features), self.quality(features.mean((-2, -1))), None, None


class Tracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam_prompt_encoder = Prompts()
        self.sam_mask_decoder = Decoder()
        self.register_buffer("no_mem_embed", torch.zeros(1, 1, 4))

    def _prepare_backbone_features(self, features):
        maps = features["backbone_fpn"]
        return (
            None,
            [x.flatten(2).permute(2, 0, 1) for x in maps],
            None,
            [x.shape[-2:] for x in maps],
        )


def tiny_model():
    torch.manual_seed(4)
    model = nn.Module()
    model.backbone = Backbone()
    model.inst_interactive_predictor = nn.Module()
    model.inst_interactive_predictor.model = Tracker()
    model.inst_interactive_predictor._transforms = SimpleNamespace(
        transform_coords=lambda coords, **kwargs: coords
    )
    processor = SimpleNamespace(device="cpu", transform=lambda pixels: pixels.float() / 255)
    return model, processor


class PointTrainingTests(unittest.TestCase):
    def test_phase_throughput_distinguishes_repeated_images_from_sources(self):
        model, _ = tiny_model()
        with patch("core.training_timing.time.perf_counter", side_effect=[100.0, 104.0]):
            timer = PhaseTiming(model, "train", 1)
            for key, image_id in (("a", 10), ("b", 10), ("c", 20)):
                timer.observe({"__key__": key, "metadata": {"image_id": image_id}})
            report = timer.finish()
        self.assertEqual(report["instances"], 3)
        self.assertEqual(report["unique_source_images"], 2)
        self.assertEqual(report["wall_seconds"], 4.0)
        self.assertEqual(report["images_per_second"], 0.75)
        self.assertEqual(report["unique_source_images_per_second"], 0.5)

    def test_inference_only_encoder_mlp_supports_backward_without_changing_state_keys(self):
        def inference_only_forward(module, x):
            if torch.is_grad_enabled():
                raise ValueError("Expected grad to be disabled.")
            return module.drop2(module.fc2(module.norm(module.drop1(module.act(module.fc1(x))))))

        native_mlp = type(
            "Mlp",
            (nn.Module,),
            {"__module__": "sam3.model.vitdet", "forward": inference_only_forward},
        )
        mlp = native_mlp()
        mlp.fc1, mlp.fc2 = nn.Linear(4, 8), nn.Linear(8, 4)
        mlp.act, mlp.norm = nn.GELU(), nn.LayerNorm(8)
        mlp.drop1, mlp.drop2 = nn.Identity(), nn.Identity()
        model, _ = tiny_model()
        model.backbone.vision_backbone.trunk = nn.Sequential(mlp)
        x = torch.randn(2, 4, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "Expected grad"):
            mlp(x)
        keys = list(model.state_dict())
        with torch.no_grad():
            reference = mlp(x)
        enable_encoder_autograd(model)
        actual = mlp(x)
        torch.testing.assert_close(actual, reference)
        actual.square().mean().backward()
        for parameter in (mlp.fc1.weight, mlp.fc2.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(keys, list(model.state_dict()))
        with torch.no_grad():
            torch.testing.assert_close(mlp(x), reference)

    def test_random_prompt_is_reproducible_inside_target_and_varies_across_seeds(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:3, 1:3] = mask[5:7, 5:7] = True
        points = [random_target_point(mask, "object", seed) for seed in range(20)]
        self.assertGreater(len({tuple(point) for point in points}), 1)
        for seed, (x, y) in enumerate(points):
            self.assertTrue(mask[int(y), int(x)])
            self.assertEqual([x, y], random_target_point(mask, "object", seed))
        with self.assertRaises(ValueError):
            random_target_point(np.zeros((2, 2)), "empty")

    def test_injected_trainer_runs_baseline_updates_preview_and_reloadable_checkpoint(self):
        for scope in ("full", "frozen_encoder"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                (output / "previews").mkdir()
                model, processor = tiny_model()
                segmenter = PointSegmenter(model, processor, scope)
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=0.01
                )
                trainer = PointTrainer(segmenter, optimizer, forward_sample, TrainingOutput(output))
                train = {
                    "__key__": "train",
                    "image": np.full((8, 8, 3), 100, np.uint8),
                    "segmentation": np.ones((8, 8), bool),
                }
                heldout = {**train, "__key__": "heldout"}
                with patch("torch.cuda.max_memory_allocated", return_value=0):
                    report = trainer.fit([train], [heldout], {"epochs": 2}, max_steps=1)
                self.assertEqual(report["steps"], 1)
                summaries = [
                    json.loads(line) for line in (output / "summary.jsonl").read_text().splitlines()
                ]
                self.assertEqual([row["epoch"] for row in summaries], [0, 1])
                timings = [
                    json.loads(line) for line in (output / "timing.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    [row["phase"] for row in timings], ["validation", "train", "validation"]
                )
                self.assertEqual([row["instances"] for row in timings], [1, 1, 1])
                self.assertTrue((output / "previews/epoch-1-0.png").is_file())
                self.assertTrue((output / "complete.json").is_file())
                state = torch.load(output / "checkpoint.pt", weights_only=True)
                restored, _ = tiny_model()
                restored.load_state_dict(state["model"], strict=True)
                self.assertEqual("encoder" in report["gradient_and_step_audit"], scope == "full")

    def test_coco_instance_is_loaded_lazily_with_an_in_memory_transform(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            image = folder / "source.jpg"
            Image.new("RGB", (8, 8), "white").save(image)
            annotations = {
                "images": [{"id": 1, "file_name": str(image), "width": 8, "height": 8}],
                "categories": [{"id": 1, "name": "object"}],
                "annotations": [
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "iscrowd": 0,
                        "area": 16,
                        "bbox": [1, 1, 4, 4],
                        "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]],
                    }
                ],
            }
            path = folder / "annotations.json"
            path.write_text(json.dumps(annotations))

            def transform(sample):
                sample["image"][:] = 0
                return sample

            samples = dataset(path, transform=transform)
            sample = next(iter(loader(samples, workers=0)))
            self.assertEqual(len(samples), 1)
            self.assertEqual(int(sample["segmentation"].sum()), 16)
            self.assertEqual(int(sample["image"].sum()), 0)
            self.assertEqual(np.asarray(Image.open(image)).min(), 255)

    def test_positive_point_is_inside_disconnected_nonconvex_mask(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1, 1] = mask[6, 6] = True
        x, y = foreground_point(mask)
        self.assertTrue(mask[int(y), int(x)])
        with self.assertRaises(ValueError):
            foreground_point(np.zeros((2, 2)))

    def test_encoder_and_decoder_updates_follow_scope_and_projections_keep_gradients(self):
        for scope in ("full", "frozen_encoder"):
            with self.subTest(scope=scope):
                model, processor = tiny_model()
                set_scope(model, scope)
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=0.01
                )
                image = np.full((8, 8, 3), 100, dtype=np.uint8)
                logits, scores = point_forward(
                    model, processor, image, [2.0, 2.0], freeze_encoder=scope == "frozen_encoder"
                )
                loss, _ = mask_loss(logits, scores, torch.ones(8, 8))
                loss.backward()
                probes = gradient_probes(model, scope)
                projection = model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0.weight
                self.assertGreater(int(torch.count_nonzero(projection.grad)), 0)
                optimizer.step()
                audit = check_step(probes)
                self.assertIn("decoder", audit)
                self.assertEqual("encoder" in audit, scope == "full")
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        self.assertTrue(torch.equal(p, before[name]), name)
                        self.assertIsNone(p.grad)

    def test_mask_loss_uses_one_candidate_but_trains_all_quality_scores(self):
        logits = torch.tensor([[[4.0, -4.0]], [[-4.0, 4.0]], [[0.0, 0.0]]], requires_grad=True)
        scores = torch.tensor([0.2, 0.4, 0.3], requires_grad=True)
        loss, metrics = mask_loss(logits, scores, torch.tensor([[1.0, 0.0]]))
        loss.backward()
        self.assertEqual(metrics["training_candidate"], 0)
        self.assertGreater(int(torch.count_nonzero(logits.grad[0])), 0)
        self.assertEqual(int(torch.count_nonzero(logits.grad[1:])), 0)
        self.assertEqual(int(torch.count_nonzero(scores.grad)), 3)
        # Evaluation selects score=.4, whose mask is wrong; no oracle score here.
        self.assertEqual(metrics["selected_iou"], 0.0)

    def test_small_model_can_reduce_loss_and_checkpoint_restores_weights(self):
        model, processor = tiny_model()
        set_scope(model, "frozen_encoder")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        losses = []
        for _ in range(12):
            optimizer.zero_grad()
            logits, scores = point_forward(model, processor, image, [2.0, 2.0], freeze_encoder=True)
            loss, _ = mask_loss(logits, scores, torch.ones(8, 8))
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint(model, optimizer, Path(folder), 1, {}, "frozen_encoder")
            saved = torch.load(Path(folder) / "checkpoint.pt", weights_only=True)
            restored, _ = tiny_model()
            restored.load_state_dict(saved["model"], strict=True)
            for name, p in restored.named_parameters():
                self.assertTrue(torch.equal(p, model.get_parameter(name)), name)


if __name__ == "__main__":
    unittest.main()
