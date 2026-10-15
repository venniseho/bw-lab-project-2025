"""Integration tests for the mlvbench-inspired library/experiment boundaries."""
from dataclasses import replace
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from torch.utils.data import SequentialSampler

from bwlab.datasets import DataModule, ImageMaskDataset, ManifestDataModule
from bwlab.datasets.conditions import ConditionDataset
from bwlab.models import Sam3GridAdapter
from bwlab.runner import Runner
from bwlab.stimuli import FenceOcclusion
from bwlab.tasks.scene_proposals import SceneProposalTask
from experiments.sam3_fragmentation.experiment import Config, build_data_module, load_config, prepare, run
from tests.helpers import FakeProcessor, FakeSceneModel


class ExperimentStructureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rgb = np.full((60, 84, 3), 127, np.uint8)
        self.mask = np.zeros((60, 84), np.uint8)
        self.mask[12:48, 22:62] = 255
        Image.fromarray(self.rgb).save(self.root / "rgb.png")
        Image.fromarray(self.mask).save(self.root / "mask.png")
        self.manifest = self.root / "records.json"
        self.manifest.write_text(json.dumps([
            {"__key__": "object/1", "image_path": "rgb.png", "mask_path": "mask.png"}]))
        self.config = Config(self.manifest, "fence")

    def test_explicit_train_val_test_loaders_and_missing_splits(self):
        source = ManifestDataModule({"test": self.manifest})
        dataset = source.test_dataset()
        module = DataModule({"train": dataset, "val": dataset, "test": dataset})
        self.assertEqual(set(module.subsets()), {"train", "val", "test"})
        self.assertIsInstance(module.test_dataloader().sampler, SequentialSampler)
        self.assertIsInstance(module.val_dataloader().sampler, SequentialSampler)
        self.assertEqual(next(iter(module.train_dataloader()))[0]["__key__"], "object/1")
        with self.assertRaisesRegex(ValueError, "not supplied"):
            source.train_dataset()

    def test_manifest_module_rejects_source_image_leakage_between_splits(self):
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            ManifestDataModule({"train": self.manifest, "test": self.manifest})

    def test_conditions_start_from_original_and_have_stable_keys(self):
        dataset = build_data_module(self.config).test_dataset()
        self.assertEqual(len(dataset), 4)
        self.assertEqual([dataset[i]["condition"] for i in range(4)], ["clean", "light", "medium", "heavy"])
        masks = []
        for sample in dataset:
            np.testing.assert_array_equal(sample["segmentation"], self.mask)
            self.assertEqual(sample["source_key"], "object/1")
            self.assertEqual(sample["__key__"], f"object/1/{sample['condition']}")
            self.assertEqual(len(sample["transforms"]), 1)
            masks.append(sample["occlusion_mask"])
        for previous, current in zip(masks, masks[1:]):
            self.assertTrue(np.all(~previous | current))

    def test_cpu_prepare_both_arms_and_frozen_source_lookup(self):
        for condition in ("fence", "fragmentation"):
            with self.subTest(condition=condition):
                output = self.root / condition
                report = prepare(replace(self.config, condition=condition), output)
                self.assertEqual(report["status"], "complete")
                self.assertEqual(len(report["records"]), 4 if condition == "fence" else 1)
                self.assertTrue((output / "config.json").is_file())
                for record in report["records"]:
                    with np.load(output / record["targets"]) as targets:
                        np.testing.assert_array_equal(targets["segmentation"], self.mask)

    def test_runner_uses_real_adapter_with_mocked_native_model_and_exports(self):
        module = build_data_module(self.config)
        model = FakeSceneModel()
        processor = FakeProcessor(model)
        result = Runner(module, SceneProposalTask(), batch_size=2).evaluate(
            Sam3GridAdapter(model, processor, points_per_side=3), self.root / "results")
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["quantitative_evaluation"])
        self.assertEqual(len(result["records"]), 4)
        self.assertEqual(processor.embeddings, 4)
        self.assertEqual(len(model.points), 36)
        for record in result["records"]:
            scene = self.root / "results" / record["scene"]
            self.assertTrue((scene / "contact_sheet.png").is_file())
            with np.load(scene / "targets.npz") as targets:
                np.testing.assert_array_equal(targets["segmentation"], self.mask)

    def test_experiment_run_wires_existing_native_loader_without_real_gpu(self):
        checkpoint = self.root / "sam3.pt"
        checkpoint.write_bytes(b"mock checkpoint; never loaded")
        native = FakeSceneModel()
        processor = FakeProcessor(native)
        with patch.dict(os.environ, {"SLURM_JOB_ID": "test", "SLURM_STEP_ID": "0"}), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("experiments.sam3_fragmentation.experiment.load_scene_model",
                      return_value=(native, processor, {"mocked": True})) as loader:
            report = run(replace(self.config, checkpoint=checkpoint), self.root / "run")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(loader.call_count, 1)
        self.assertTrue((self.root / "run/config.json").is_file())

    def test_login_node_rejected_before_any_model_loading_or_output(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("experiments.sam3_fragmentation.experiment.load_scene_model") as loader:
            with self.assertRaisesRegex(RuntimeError, "SLURM"):
                run(self.config, self.root / "denied")
        loader.assert_not_called()
        self.assertFalse((self.root / "denied").exists())

    def test_runner_refuses_training_split_for_evaluation(self):
        model = FakeSceneModel()
        runner = Runner(build_data_module(self.config), SceneProposalTask())
        with self.assertRaisesRegex(ValueError, "val or test"):
            runner.evaluate(Sam3GridAdapter(model, FakeProcessor(model)), self.root / "bad", subset="train")

    def test_config_resolves_manifest_and_checkpoint_relative_to_config(self):
        path = self.root / "config.json"
        path.write_text(json.dumps({"manifest": "records.json", "condition": "fragmentation",
                                    "checkpoint": "sam3.pt", "fragmentation": {"noise_count": 5}}))
        config = load_config(path)
        self.assertEqual(config.manifest, self.manifest)
        self.assertEqual(config.checkpoint, self.root / "sam3.pt")
        self.assertEqual(config.fragmentation.noise_count, 5)

    def test_library_never_imports_scripts_experiments_or_compatibility_packages(self):
        root = Path(__file__).resolve().parents[2] / "bwlab"
        forbidden = {"scripts", "experiments", "data_selection", "stimuli_generation", "evaluation", "scene_segmentation"}
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [node.module] if isinstance(node, ast.ImportFrom) else (
                    [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    if name:
                        self.assertNotIn(name.split(".")[0], forbidden, str(path))


if __name__ == "__main__":
    unittest.main()
