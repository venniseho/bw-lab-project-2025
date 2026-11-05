"""Cache official COCO files atomically; never run model inference."""

import shutil
from urllib.request import urlopen
from zipfile import ZipFile

BASE = "https://s3.amazonaws.com/images.cocodataset.org"


def download(url, destination):
    """Stream to a temporary file; never publish a partially downloaded file."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=120) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(destination)
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
