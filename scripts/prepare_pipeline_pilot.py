"""Select existing archived image/instance-mask pairs for a timed integration pilot."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bwlab.datasets import ImageMaskDataset


def select_records(repo: Path, count: int) -> list[dict]:
    """Choose first annotation filename per distinct image; no shape/area ranking."""
    if count < 1:
        raise ValueError("count must be positive")
    archive = repo / "legacy" / "main_2025"
    source = archive if archive.is_dir() else repo
    records = []
    for image in sorted((source / "images").glob("*.png")):
        masks = sorted((source / "masks").glob(f"{image.stem}_ann*.png"))
        if not masks:
            continue
        records.append({"__key__": image.stem, "image_path": str(image.resolve()),
                        "mask_path": str(masks[0].resolve()),
                        "metadata": {"source": "archived_repository_pair",
                                     "selection": "sorted image; first annotation filename",
                                     "mask_file": masks[0].name}})
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError(f"Need {count} distinct paired images; found {len(records)}. No downloads or duplicated sources used.")
    for sample in ImageMaskDataset(records):
        if not sample["segmentation"].any():
            raise ValueError(f"Empty source mask: {sample['__key__']}; inspect rather than silently exclude")
    return records


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/scratch/vho/sam3-whole-scene/checkpoints/sam3.pt"))
    args = parser.parse_args(argv)
    records = select_records(args.repo_root, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir.resolve() / "records.json"
    manifest.write_text(json.dumps(records, indent=2)+"\n", encoding="utf-8")
    for condition in ("fence", "fragmentation"):
        config = {"manifest": str(manifest), "condition": condition, "seed": 42,
                  "points_per_side": 3, "num_workers": 0, "batch_size": 1,
                  "checkpoint": str(args.checkpoint.resolve())}
        (args.output_dir / f"{condition}.json").write_text(
            json.dumps(config, indent=2)+"\n", encoding="utf-8")
    print(f"Selected {len(records)} distinct images; fence inputs={4*len(records)}, contour inputs={len(records)}")
    print(f"Configs: {args.output_dir.resolve()}; CPU only, no inference run")


if __name__ == "__main__":
    main()
