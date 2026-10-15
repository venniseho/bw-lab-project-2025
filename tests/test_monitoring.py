import csv
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock
import numpy as np

from bwlab.monitoring import summarize_timings, write_timing_csv
from bwlab.models import Sam3GridAdapter
from tests.helpers import FakeProcessor, FakeSceneModel


class TimingTests(TestCase):
    def test_native_adapter_synchronizes_embedding_and_each_query(self):
        model = FakeSceneModel()
        synchronize = Mock()
        adapter = Sam3GridAdapter(model, FakeProcessor(model), 2, synchronize=synchronize)
        proposals, _ = adapter.predict(np.ones((8, 12, 3), np.uint8))
        self.assertEqual(len(proposals), 12)
        self.assertEqual(synchronize.call_count, 10)  # Embedding boundaries + 4 query pairs.

    def test_totals_percentiles_and_condition_groups(self):
        records = [{"__key__": str(i), "condition": name, "first_inference": i == 0,
                    "timing": {"inference_wall_seconds": value}}
                   for i, (name, value) in enumerate([( "clean", 1), ("clean", 3), ("fragmentation", 10)])]
        summary = summarize_timings(records)
        self.assertEqual(summary["all"]["inference_wall_seconds"]["total"], 14)
        self.assertEqual(summary["clean"]["inference_wall_seconds"]["median"], 2)
        self.assertAlmostEqual(summary["all"]["inference_wall_seconds"]["p95"], 9.3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"timings.csv"
            write_timing_csv(records, path)
            with path.open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["first_inference"], "True")
            self.assertEqual(rows[2]["inference_wall_seconds"], "10")

    def test_empty_run_has_no_fake_measurements(self):
        self.assertEqual(summarize_timings([]), {"all": {}})
