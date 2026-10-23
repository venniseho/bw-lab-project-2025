from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from bwlab.datasets import DataModule
from bwlab.runtime import RuntimeConfig
from bwlab.runner import Runner
from bwlab.tasks.summary import SummaryTask
from experiments.sam3_fragmentation.experiment import Config, run
from tests.helpers import FakeProcessor, FakeSceneModel
from bwlab.models.proposals import collect_proposals
from scripts import check_batch_equivalence


class BatchAdapter:
    def __init__(self):
        self.sizes = []
        self.model = FakeSceneModel()
        self.processor = FakeProcessor(self.model)

    def predict_batch(self, images):
        assert torch.is_inference_mode_enabled()
        self.sizes.append(len(images))
        return [collect_proposals(self.model, self.processor, image, 1) for image in images]


class OptimizedRunnerTests(unittest.TestCase):
    def test_equivalence_without_gpu_stops_before_loading_data_or_model(self):
        with patch.dict("os.environ", {"SLURM_JOB_ID": "test", "SLURM_STEP_ID": "0"}), \
                patch("sys.argv", ["check", "--config", "unused.json", "--output", "unused.csv"]), \
                patch("torch.cuda.is_available", return_value=False), \
                patch.object(check_batch_equivalence, "build_data_module") as data, \
                patch.object(check_batch_equivalence, "load_scene_model") as model:
            with self.assertRaisesRegex(RuntimeError, "No CUDA GPU allocated"):
                check_batch_equivalence.main()
        data.assert_not_called()
        model.assert_not_called()

    def test_batches_warmup_and_first_batch_previews(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [{"__key__": str(i), "condition": "clean", "image": np.full((8, 9, 3), 127, np.uint8),
                        "segmentation": np.ones((8, 9), bool)} for i in range(5)]
            adapter = BatchAdapter()
            runner = Runner(DataModule({"test": samples}), SummaryTask(1), batch_size=2, warmup_batches=1)
            report = runner.evaluate(adapter, root/"run")
            self.assertEqual(adapter.sizes, [2, 2, 2, 1])
            self.assertEqual(len(report["records"]), 5)
            self.assertEqual(len(list((root/"run/previews").glob("*/contact_sheet.png"))), 2)
            self.assertGreater(runner.warmup_seconds, 0)
            with (root/"run/image_results.csv").open(newline="") as stream:
                self.assertEqual([r["input_index"] for r in csv.DictReader(stream)], [str(i) for i in range(5)])

    def test_optimized_entry_point_writes_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.full((8, 9, 3), 127, np.uint8)).save(root/"image.png")
            Image.fromarray(np.ones((8, 9), np.uint8)).save(root/"mask.png")
            (root/"records.json").write_text(json.dumps([
                {"__key__": "one", "image_path": "image.png", "mask_path": "mask.png"}]))
            checkpoint = root/"sam3.pt"
            checkpoint.write_bytes(b"fake")
            model = FakeSceneModel()
            config = Config(root/"records.json", "fence", checkpoint=checkpoint, batch_size=2,
                            runtime=RuntimeConfig(batched=False, allow_local=True, monitor_gpu=False, preview_batches=0))
            with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.synchronize"), \
                    patch("torch.cuda.reset_peak_memory_stats"), \
                    patch("torch.cuda.get_device_name", return_value="mock GPU"), \
                    patch("bwlab.runtime.configure_model", return_value={"parameter_count": 123}), \
                    patch("bwlab.runtime.gpu_memory", return_value={"peak_allocated_bytes": 456}), \
                    patch("experiments.sam3_fragmentation.experiment.load_scene_model",
                          return_value=(model, FakeProcessor(model), {"mock": True})):
                report = run(config, root/"output")
            timing = json.loads((root/"output/timing.json").read_text())
            self.assertEqual(timing["input_count"], 4)
            self.assertEqual(timing["parameter_count"], 123)
            self.assertGreater(timing["inputs_per_second"], 0)
            self.assertTrue((root/"output/predictions/image_results.csv").is_file())
            self.assertFalse((root/"output/predictions/previews").exists())
