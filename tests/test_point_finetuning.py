"""Exercise real autograd through a small stand-in for the interactive API."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from PIL import Image

from experiments.sam3_point_finetuning.data import dataset, foreground_point, loader
from experiments.sam3_point_finetuning.experiment import checkpoint, check_step, gradient_probes
from experiments.sam3_point_finetuning.loss import mask_loss
from experiments.sam3_point_finetuning.model import point_forward, set_scope


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
        self.embedding = nn.Parameter(torch.ones(1, 4, 1, 1) * .1)

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

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output, repeat_image, high_res_features):
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
        return None, [x.flatten(2).permute(2, 0, 1) for x in maps], None, [x.shape[-2:] for x in maps]


def tiny_model():
    torch.manual_seed(4)
    model = nn.Module()
    model.backbone = Backbone()
    model.inst_interactive_predictor = nn.Module()
    model.inst_interactive_predictor.model = Tracker()
    model.inst_interactive_predictor._transforms = SimpleNamespace(
        transform_coords=lambda coords, **kwargs: coords)
    processor = SimpleNamespace(device="cpu", transform=lambda pixels: pixels.float() / 255)
    return model, processor


class PointTrainingTests(unittest.TestCase):
    def test_coco_instance_is_loaded_lazily_with_an_in_memory_transform(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            image = folder / "source.jpg"
            Image.new("RGB", (8, 8), "white").save(image)
            annotations = {"images": [{"id": 1, "file_name": str(image), "width": 8, "height": 8}],
                           "categories": [{"id": 1, "name": "object"}],
                           "annotations": [{"id": 2, "image_id": 1, "category_id": 1,
                                            "iscrowd": 0, "area": 16, "bbox": [1, 1, 4, 4],
                                            "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]]}]}
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
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
                image = np.full((8, 8, 3), 100, dtype=np.uint8)
                logits, scores = point_forward(model, processor, image, [2., 2.],
                                               freeze_encoder=scope == "frozen_encoder")
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
        logits = torch.tensor([[[4., -4.]], [[-4., 4.]], [[0., 0.]]], requires_grad=True)
        scores = torch.tensor([.2, .4, .3], requires_grad=True)
        loss, metrics = mask_loss(logits, scores, torch.tensor([[1., 0.]]))
        loss.backward()
        self.assertEqual(metrics["training_candidate"], 0)
        self.assertGreater(int(torch.count_nonzero(logits.grad[0])), 0)
        self.assertEqual(int(torch.count_nonzero(logits.grad[1:])), 0)
        self.assertEqual(int(torch.count_nonzero(scores.grad)), 3)
        # Evaluation selects score=.4, whose mask is wrong; no oracle score here.
        self.assertEqual(metrics["selected_iou"], 0.)

    def test_small_model_can_reduce_loss_and_checkpoint_restores_weights(self):
        model, processor = tiny_model()
        set_scope(model, "frozen_encoder")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        losses = []
        for _ in range(12):
            optimizer.zero_grad()
            logits, scores = point_forward(model, processor, image, [2., 2.], freeze_encoder=True)
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
