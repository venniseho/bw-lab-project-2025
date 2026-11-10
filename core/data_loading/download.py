"""Cache official COCO files atomically; never run model inference."""

import shutil
from http.client import IncompleteRead
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from zipfile import ZipFile

BASE = "https://s3.amazonaws.com/images.cocodataset.org"


def _download_once(url, temporary):
    with urlopen(url, timeout=120) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
        expected = getattr(source, "headers", {}).get("Content-Length")
        if expected is not None and target.tell() != int(expected):
            raise IncompleteRead(b"", int(expected) - target.tell())


def download(url, destination, *, attempts=6):
    """Cache atomically; retry transient network failures without keeping partials."""
    if attempts < 1:
        raise ValueError("Download attempts must be positive")
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        try:
            _download_once(url, temporary)
            temporary.replace(destination)
            return
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as error:
            if isinstance(error, HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt + 1 == attempts:
                raise
            delay = min(2**attempt, 30)
            print(
                f"Download retry {attempt + 2}/{attempts} in {delay}s: {url} ({error})", flush=True
            )
            sleep(delay)
        finally:
            temporary.unlink(missing_ok=True)


def ensure_annotations(root, allow_download, split="train2017"):
    if split not in ("train2017", "val2017"):
        raise ValueError(f"Unknown COCO split: {split}")
    path = root / f"annotations/instances_{split}.json"
    if path.exists():
        return path
    archive = root / "annotations_trainval2017.zip"
    if not archive.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing {archive}; add --download")
        download(f"{BASE}/annotations/{archive.name}", archive)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    try:
        with (
            ZipFile(archive) as zipped,
            zipped.open(f"annotations/instances_{split}.json") as source,
        ):
            with temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
