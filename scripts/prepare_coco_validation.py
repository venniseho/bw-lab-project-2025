"""Download a small official COCO val2017 subset and prepare throughput configs.

Run on Vulcan. Images stay on scratch; masks are decoded in memory by the loader.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
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


def annotations(root, allow_download):
    path = root / "annotations" / "instances_val2017.json"
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing {path}; use --download on Vulcan")
        archive = root / "annotations_trainval2017.zip"
        download(f"{BASE}/annotations/{archive.name}", archive)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        try:
            with ZipFile(archive) as zipped, zipped.open("annotations/instances_val2017.json") as source:
                with temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def select_records(data, count, seed, root, annotation_path, digest, *, eligible_only=False):
    """Seeded source subset; smallest noncrowd annotation ID per image, no size filter."""
    images = sorted(data["images"], key=lambda image: image["id"])
    if not 1 <= count <= len(images):
        raise ValueError(f"count must be between 1 and {len(images)}")
    instances = {}
    for annotation in data["annotations"]:
        if annotation.get("iscrowd") == 0:
            instances.setdefault(annotation["image_id"], []).append(annotation)
    if eligible_only:
        images = [image for image in images if image["id"] in instances]
        if count > len(images):
            raise ValueError(f"Need {count} eligible sources; only {len(images)} available")
    selected = sorted(random.Random(seed).sample(images, count), key=lambda image: image["id"])
    records = []
    for image in selected:
        candidates = instances.get(image["id"], [])
        if not candidates:
            continue
        annotation = min(candidates, key=lambda item: item["id"])
        filename = image["file_name"]
        if Path(filename).name != filename or not filename.endswith(".jpg"):
            raise ValueError(f"Unexpected COCO filename: {filename}")
        records.append({
            "__key__": f"coco_val2017_{image['id']:012d}_ann{annotation['id']}",
            "image_path": str(root / "val2017" / filename),
            "annotations_path": str(annotation_path), "annotations_sha256": digest,
            "image_id": image["id"], "annotation_id": annotation["id"],
            "metadata": {"source": "COCO val2017", "selection_seed": seed,
                         "selection": "random source subset; smallest noncrowd annotation ID"}})
    return records, [image["id"] for image in selected]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--eligible-only", action="store_true",
                        help="Select exactly count sources with a noncrowd annotation; throughput sampling only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/scratch/vho/sam3-whole-scene/checkpoints/sam3.pt"))
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    root = args.coco_root.resolve()
    path = annotations(root, args.download)
    raw = path.read_bytes()
    records, image_ids = select_records(json.loads(raw), args.count, args.seed, root, path,
                                        hashlib.sha256(raw).hexdigest(), eligible_only=args.eligible_only)
    if not records:
        raise ValueError("Selected sources have no noncrowd instances")
    for record in records:
        image = Path(record["image_path"])
        if args.download:
            download(f"{BASE}/val2017/{image.name}", image)
        if not image.exists():
            raise FileNotFoundError(f"Missing {image}; use --download on Vulcan")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir.resolve() / "records.json"
    write_json(manifest, records)
    write_json(args.output_dir / "selection.json", {
        "dataset": "COCO val2017", "source_image_ids": image_ids,
        "sampling_pool": "noncrowd_annotated_sources" if args.eligible_only else "all_validation_sources",
        "requested_sources": args.count, "object_samples": len(records),
        "sources_without_noncrowd_instance": args.count - len(records),
        "policy": "one smallest-ID noncrowd instance per source; throughput pilot only"})
    for condition in ("fence", "fragmentation"):
        write_json(args.output_dir / f"{condition}.json", {
            "manifest": str(manifest), "condition": condition, "seed": args.seed,
            "points_per_side": 3, "batch_size": 4, "num_workers": 2,
            "checkpoint": str(args.checkpoint.resolve()),
            "runtime": {"batched": True, "preview_batches": 1, "warmup_batches": 1,
                        "cudnn_benchmark": True, "compile_encoder": False}})
    print(f"COCO val2017: {args.count} selected sources, {len(records)} object samples; "
          f"{args.count-len(records)} without noncrowd instances (not replaced)")
    print(f"Fence inputs: {4*len(records)}; fragmentation inputs: {len(records)}")
    print(f"Configs: {args.output_dir.resolve()}; no inference run")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
