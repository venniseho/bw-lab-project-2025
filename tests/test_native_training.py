from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.sam3_finetuning.config import training_config
from experiments.sam3_finetuning.model import add_eval_matching, set_trainable_scope
from scripts.prepare_native_training import select_splits


class NativeTrainingTests(unittest.TestCase):
    def test_splits_are_disjoint_reproducible_and_keep_all_noncrowd_instances(self):
        data = {"images": [{"id": index} for index in range(6)],
                "annotations": [{"id": index*2+offset, "image_id": index, "iscrowd": offset}
                                for index in range(6) for offset in (0, 1)], "categories": []}
        before = deepcopy(data)
        splits = select_splits(data, 3, 2, 42)
        self.assertEqual(splits, select_splits(data, 3, 2, 42))
        self.assertEqual(data, before)
        train_ids = {im["id"] for im in splits["train"]["images"]}
        heldout_ids = {im["id"] for im in splits["test"]["images"]}
        self.assertFalse(train_ids & heldout_ids)
        self.assertEqual(len(train_ids), 3)
        self.assertEqual(len(heldout_ids), 2)
        self.assertTrue(all(ann["iscrowd"] == 0 for split in splits.values() for ann in split["annotations"]))

    def test_mask_head_scope_freezes_backbone_and_other_decoders(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.vision_backbone = torch.nn.Linear(2, 2)
        model.segmentation_head = torch.nn.Linear(2, 1)
        model.transformer = torch.nn.Linear(2, 2)
        report = set_trainable_scope(model, "mask_head")
        self.assertEqual(report["trainable_parameters"], 3)
        self.assertTrue(all(name.startswith("segmentation_head.") for name in report["trainable_names"]))
        self.assertFalse(model.backbone.vision_backbone.weight.requires_grad)
        self.assertFalse(model.transformer.weight.requires_grad)
        set_trainable_scope(model, "encoder")
        self.assertTrue(model.backbone.vision_backbone.weight.requires_grad)
        self.assertFalse(model.segmentation_head.weight.requires_grad)
        set_trainable_scope(model, "full")
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_segmentation_config_enables_masks_in_loss_loaders_and_model(self):
        base = {"roboflow_train": {"loss": {"loss_fns_find": []},
                                   "val_transforms": [{"transforms": []}]},
                "scratch": {}, "trainer": {"data": {"train": {"dataset": {}}, "val": {"dataset": {}}},
                                              "logging": {}, "checkpoint": {}},
                "launcher": {}, "submitit": {"job_array": {"num_tasks": 100}}}
        original = deepcopy(base)
        config = training_config(base, native_root=Path("native"), dataset_root=Path("data"),
                                 output=Path("output"), checkpoint=Path("checkpoint"))
        self.assertEqual(base, original)
        self.assertTrue(config["scratch"]["enable_segmentation"])
        self.assertEqual(config["scratch"]["num_train_workers"], 8)
        self.assertIn(".Masks", config["roboflow_train"]["loss"]["loss_fns_find"][-1]["_target_"])
        self.assertTrue(config["trainer"]["data"]["val"]["collate_fn"]["with_seg_masks"])
        self.assertIn("DecodeRle", config["roboflow_train"]["val_transforms"][0]["transforms"][0]["_target_"])
        self.assertFalse(config["trainer"]["skip_saving_ckpts"])
        self.assertNotIn("job_array", config["submitit"])

    def test_eval_matching_adds_only_assignments_and_skips_training(self):
        prediction = {"pred_masks": object()}
        original_mask = prediction["pred_masks"]
        model = SimpleNamespace(training=False, back_convert=lambda value: value,
                                _compute_matching=lambda out, target: out.update(indices=target))
        add_eval_matching(model, [SimpleNamespace(find_targets=["target"])], SimpleNamespace(output=[[prediction]]))
        self.assertEqual(prediction["indices"], "target")
        self.assertIs(prediction["pred_masks"], original_mask)
        model.training = True
        untouched = {}
        add_eval_matching(model, [SimpleNamespace(find_targets=["target"])], SimpleNamespace(output=[[untouched]]))
        self.assertFalse(untouched)


def audit_class():
    """Test audit mechanics with real Torch autograd, without CUDA/native imports."""
    module = ModuleType("sam3.train.trainer")
    class NativeStub:
        def _run_step(self):
            prediction = self.model.segmentation_head(self.model.backbone(torch.ones(2, 2)))
            loss = (prediction-1).square().mean()
            self.pending_losses = {"loss": float(loss.detach())}
            loss.backward()
    module.Trainer = NativeStub
    module.unwrap_ddp_if_wrapped = lambda model: model
    path = Path(__file__).resolve().parents[1] / "experiments/sam3_finetuning/trainer.py"
    spec = importlib.util.spec_from_file_location("bwlab_audit_test", path)
    loaded = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"sam3.train.trainer": module}):
        spec.loader.exec_module(loaded)
    return loaded.AuditedTrainer


class NativeAuditTests(unittest.TestCase):
    def test_real_optimizer_update_is_recorded_and_frozen_weights_remain_fixed(self):
        cls = audit_class()
        audit = cls.__new__(cls)
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2, 2)
        model.segmentation_head = torch.nn.Linear(2, 1)
        set_trainable_scope(model, "mask_head")
        before = model.backbone.weight.detach().clone()
        audit.model, audit.wandb_run = model, None
        audit.completed_updates, audit.update_records = 0, []
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.register_step_post_hook(audit._after_update)
        with tempfile.TemporaryDirectory() as directory:
            audit.audit_output = Path(directory)
            audit._run_step()
            with patch("torch.cuda.synchronize"), patch("torch.cuda.memory_allocated", return_value=0), \
                    patch("torch.cuda.max_memory_allocated", return_value=0):
                optimizer.step()
            self.assertEqual(audit.completed_updates, 1)
            self.assertTrue(torch.equal(model.backbone.weight, before))
            record = json.loads((Path(directory) / "optimizer_steps.jsonl").read_text())
            self.assertTrue(record["changed_parameter_probes"])
