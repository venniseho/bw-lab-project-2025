"""Network retries must preserve complete caches and remove partial downloads."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch
from urllib.error import HTTPError, URLError

from core.data_loading.download import download


class BrokenStream(io.BytesIO):
    def read(self, size=-1):
        chunk = super().read(3)
        if not chunk:
            raise ConnectionResetError("connection dropped during transfer")
        return chunk


class DownloadRetryTests(unittest.TestCase):
    def test_transient_failures_retry_then_cache_complete_file(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("core.data_loading.download.sleep") as sleep,
        ):
            destination = Path(folder) / "image.jpg"
            with patch(
                "core.data_loading.download.urlopen",
                side_effect=[
                    URLError(ConnectionResetError()),
                    TimeoutError(),
                    io.BytesIO(b"complete"),
                ],
            ) as request:
                download("https://example.invalid/image.jpg", destination)
                download("https://example.invalid/image.jpg", destination)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(sleep.call_args_list, [call(1), call(2)])
            self.assertEqual(destination.read_bytes(), b"complete")
            self.assertFalse(destination.with_suffix(".jpg.part").exists())

    def test_failed_stream_is_replaced_not_appended_on_retry(self):
        with tempfile.TemporaryDirectory() as folder, patch("core.data_loading.download.sleep"):
            destination = Path(folder) / "image.jpg"
            with patch(
                "core.data_loading.download.urlopen",
                side_effect=[BrokenStream(b"partial-bad-body"), io.BytesIO(b"good")],
            ):
                download("https://example.invalid/image.jpg", destination)
            self.assertEqual(destination.read_bytes(), b"good")

    def test_exhausted_retries_leave_no_completed_or_partial_file(self):
        with tempfile.TemporaryDirectory() as folder, patch("core.data_loading.download.sleep"):
            destination = Path(folder) / "image.jpg"
            with patch(
                "core.data_loading.download.urlopen", side_effect=URLError("reset")
            ) as request:
                with self.assertRaises(URLError):
                    download("https://example.invalid/image.jpg", destination, attempts=2)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_permanent_http_failure_is_not_retried(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("core.data_loading.download.sleep") as sleep,
        ):
            destination = Path(folder) / "image.jpg"
            error = HTTPError("https://example.invalid", 404, "not found", {}, None)
            with patch("core.data_loading.download.urlopen", side_effect=error) as request:
                with self.assertRaises(HTTPError):
                    download("https://example.invalid/image.jpg", destination)
            request.assert_called_once()
            sleep.assert_not_called()

    def test_short_body_is_retried_before_publishing_cache(self):
        with tempfile.TemporaryDirectory() as folder, patch("core.data_loading.download.sleep"):
            destination = Path(folder) / "image.jpg"
            short = io.BytesIO(b"bad")
            short.headers = {"Content-Length": "100"}
            with patch(
                "core.data_loading.download.urlopen", side_effect=[short, io.BytesIO(b"complete")]
            ):
                download("https://example.invalid/image.jpg", destination)
            self.assertEqual(destination.read_bytes(), b"complete")


if __name__ == "__main__":
    unittest.main()
