"""Prepare a tiny clean COCO train2017 subset for native segmentation training."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from experiments.sam3_finetuning.config import NATIVE_REVISION, training_config
from scripts.prepare_coco_validation import BASE, download


def select_splits(data, train_count, heldout_count, seed):
    if train_count < 1 or heldout_count < 1:
        raise ValueError("Both split sizes must be positive")
    annotations = [ann for ann in data["annotations"] if ann.get("iscrowd") == 0]
    eligible = {ann["image_id"] for ann in annotations}
    images = sorted((im for im in data["images"] if im["id"] in eligible), key=lambda im: im["id"])
    if len(images) < train_count+heldout_count:
        raise ValueError("Not enough noncrowd training sources")
    chosen = random.Random(seed).sample(images, train_count+heldout_count)
    result = {}
    for name, selected in (("train", chosen[:train_count]), ("test", chosen[train_count:])):
        ids = {image["id"] for image in selected}
        result[name] = {"images": sorted(deepcopy(selected), key=lambda image: image["id"]),
                        "annotations": deepcopy([ann for ann in annotations if ann["image_id"] in ids]),
                        "categories": deepcopy(data["categories"])}
    return result


def ensure_annotations(root, allow_download):
    path = root / "annotations/instances_train2017.json"
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
        with ZipFile(archive) as zipped, zipped.open("annotations/instances_train2017.json") as source:
            with temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def validate_masks(data):
    from pycocotools.coco import COCO
    coco = COCO()
    coco.dataset = data
    coco.createIndex()
    for annotation in data["annotations"]:
        mask = coco.annToMask(annotation)
        image = coco.imgs[annotation["image_id"]]
        if mask.shape != (image["height"], image["width"]) or not mask.any():
            raise ValueError(f"Invalid/empty annotation mask: {annotation['id']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/scratch/vho/sam3-whole-scene/checkpoints/sam3.pt"))
    parser.add_argument("--train-images", type=int, default=8)
    parser.add_argument("--heldout-images", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scope", choices=("mask_head", "full", "encoder"), default="mask_head")
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    native = args.native_root.resolve()
    revision = subprocess.check_output(["git", "-C", str(native), "rev-parse", "HEAD"], text=True).strip()
    if revision != NATIVE_REVISION:
        raise ValueError(f"Expected native SAM3 {NATIVE_REVISION}; found {revision}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output_dir.exists():
        raise FileExistsError("Use a new preparation output directory")
    root, output = args.coco_root.resolve(), args.output_dir.resolve()
    annotation_file = ensure_annotations(root, args.download)
    print(f"Reading {annotation_file} (large training annotation JSON)...", flush=True)
    with annotation_file.open(encoding="utf-8") as stream:
        splits = select_splits(json.load(stream), args.train_images, args.heldout_images, args.seed)
    for name, data in splits.items():
        validate_masks(data)
        for image in data["images"]:
            filename = image["file_name"]
            if Path(filename).name != filename or not filename.endswith(".jpg"):
                raise ValueError(f"Unexpected COCO filename: {filename}")
            path = root / "train2017" / filename
            if args.download:
                print(f"{name}: caching {filename}", flush=True)
                download(f"{BASE}/train2017/{filename}", path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; add --download")
            image["file_name"] = str(path)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in splits.items():
        folder = output / "data/coco_smoke" / name
        folder.mkdir(parents=True)
        (folder / "_annotations.coco.json").write_text(json.dumps(data)+"\n", encoding="utf-8")
    template = native / "sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml"
    config = training_config(yaml.safe_load(template.read_text()), native_root=native,
                             dataset_root=output / "data", output=output / "run",
                             checkpoint=args.checkpoint, scope=args.scope, workers=args.workers,
                             epochs=args.epochs, wandb_mode=args.wandb_mode)
    (output / "training.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    selection = {"source_split": "COCO train2017 only; COCO val2017 untouched",
                 "native_revision": revision, "seed": args.seed,
                 "native_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
                 "splits": {name: {"source_ids": [im["id"] for im in data["images"]],
                                    "instance_count": len(data["annotations"])} for name, data in splits.items()}}
    (output / "selection.json").write_text(json.dumps(selection, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)
    print(f"Config: {output / 'training.json'}; no inference or training launched", flush=True)


if __name__ == "__main__":
    main()
