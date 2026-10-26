"""Oversized batch candidates must not discard a valid small-data sweep."""
import csv
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import benchmark_pipeline
from scripts.summarize_batch_timings import summarize


class BenchmarkTests(unittest.TestCase):
    def test_summary_reads_nested_prediction_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "fence" / "batch-1" / "predictions"
            predictions.mkdir(parents=True)
            (root / "fence" / "batch_sweep.csv").write_text(
                "batch_size,status,inputs_per_second,evaluation_seconds\n1,complete,10,0.2\n64,skipped_insufficient_inputs,,\n")
            (predictions / "image_results.csv").write_text(
                "image_embedding_seconds,point_query_seconds\n0.01,0.08\n0.02,0.09\n")
            report = io.StringIO()
            with redirect_stdout(report):
                summarize(root)
            self.assertIn("inputs=2", report.getvalue())
            self.assertIn("image_embedding=0.030s", report.getvalue())
            self.assertIn("point_query=0.170s", report.getvalue())
            self.assertIn("skipped_insufficient_inputs", report.getvalue())

    def test_oversized_candidate_is_logged_without_launching_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep"
            module = SimpleNamespace(val_dataset=lambda: range(63))
            with patch("sys.argv", ["benchmark", "--config", "unused", "--output", str(output),
                                    "--batch-sizes", "64"]), \
                    patch.object(benchmark_pipeline, "require_slurm_step"), \
                    patch.object(benchmark_pipeline, "load_config", return_value=SimpleNamespace(subset="val")), \
                    patch.object(benchmark_pipeline, "build_data_module", return_value=module), \
                    patch.object(benchmark_pipeline.subprocess, "run") as launch:
                benchmark_pipeline.main()
            launch.assert_not_called()
            with (output / "batch_sweep.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["status"], "skipped_insufficient_inputs")
            self.assertEqual(rows[0]["input_count"], "63")
