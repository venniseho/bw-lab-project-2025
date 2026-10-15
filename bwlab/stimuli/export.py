"""CPU-only stimulus export, independent of command-line configuration."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def prepare_stimuli(dataset, output_path: Path) -> dict:
    """Export model-ready inputs and separate targets to a new directory."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    (output_path / "inputs").mkdir()
    (output_path / "targets").mkdir()
    report = {"status": "preparing", "records": []}
    report_path = output_path / "manifest.json"

    def save_report():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_report()
    for index, sample in enumerate(dataset):
        condition = sample.get("condition", "stimulus")
        # File names contain only numeric indices; keys stay in the manifest.
        image_path = f"inputs/{index:06d}.png"
        target_path = f"targets/{index:06d}.npz"
        Image.fromarray(sample["image"]).save(output_path / image_path)
        targets = {name: sample[name] for name in
                   ("segmentation", "visible_segmentation", "occlusion_mask") if name in sample}
        if "fragmentation" in sample:
            targets.update({name: sample["fragmentation"][name]
                            for name in ("outline_segments", "noise_segments")})
        np.savez_compressed(output_path / target_path, **targets)
        report["records"].append({"__key__": sample["__key__"],
                                  "source_key": sample.get("source_key", sample["__key__"]),
                                  "condition": condition, "image": image_path,
                                  "targets": target_path, "transforms": sample.get("transforms", []),
                                  "metadata": sample.get("metadata", {})})
        save_report()
    report["status"] = "complete"
    save_report()
    return report
