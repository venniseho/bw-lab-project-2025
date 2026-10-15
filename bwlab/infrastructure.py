"""Shared I/O, provenance and SLURM guards; no job submission."""
from pathlib import Path
import hashlib
from PIL import Image

def require_slurm_step(environment: dict) -> None:
    """An salloc shell alone can still be on the login node: require srun."""
    if not environment.get("SLURM_JOB_ID") or not environment.get("SLURM_STEP_ID"):
        raise RuntimeError("Launch inside an allocated GPU job with srun; "
                           "both SLURM_JOB_ID and SLURM_STEP_ID are required.")

def load_rgb(path: str | Path) -> Image.Image:
    # Preserve decoded pixel coordinates: do not silently apply EXIF rotation.
    with Image.open(path) as source:
        return source.convert("RGB")

def file_fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
