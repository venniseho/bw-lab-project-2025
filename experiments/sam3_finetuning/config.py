"""Adapt the pinned native detection example into a segmentation smoke run."""
from copy import deepcopy
from pathlib import Path


NATIVE_REVISION = "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"


def training_config(base, *, native_root, dataset_root, output, checkpoint,
                    scope="mask_head", workers=8, epochs=2, wandb_mode="offline"):
    if scope not in ("mask_head", "full", "encoder"):
        raise ValueError("scope must be mask_head, full or encoder")
    if workers < 0 or epochs < 1:
        raise ValueError("nonnegative workers and positive epochs required")
    config = deepcopy(base)
    config.pop("defaults", None)
    config["paths"] = {
        "roboflow_vl_100_root": str(Path(dataset_root).resolve()),
        "experiment_log_dir": str(Path(output).resolve()),
        "bpe_path": str(Path(native_root).resolve() / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")}
    train = config["roboflow_train"]
    train.update(supercategory="coco_smoke", num_images=None)
    # The detection example's val transforms do not decode mask RLEs.
    train["val_transforms"][0]["transforms"].insert(0, {
        "_target_": "sam3.train.transforms.segmentation.DecodeRle"})
    # Native instance-mask criterion; do not use the commented semantic loss's
    # obsolete sam3.losses import path from the reference YAML.
    train["loss"]["loss_fns_find"].append({
        "_target_": "sam3.train.loss.loss_fns.Masks", "compute_aux": False,
        "focal_alpha": 0.25, "focal_gamma": 2.0,
        "weight_dict": {"loss_mask": 200.0, "loss_dice": 10.0},
        "num_sample_points": 12544, "oversample_ratio": 3.0,
        "importance_sample_ratio": 0.75})
    scratch = config["scratch"]
    scratch.update(enable_segmentation=True, num_train_workers=workers,
                   num_val_workers=workers, train_batch_size=1, val_batch_size=1,
                   gradient_accumulation_steps=1, max_data_epochs=epochs)
    trainer = config["trainer"]
    trainer.update({"_target_": "experiments.sam3_finetuning.trainer.AuditedTrainer",
                    "max_epochs": epochs, "mode": "train", "val_epoch_freq": 1,
                    "skip_first_val": True, "skip_saving_ckpts": False,
                    "audit_output": str(Path(output).resolve()), "wandb_mode": wandb_mode})
    trainer["model"] = {
        "_target_": "experiments.sam3_finetuning.model.build_training_model",
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "bpe_path": "${paths.bpe_path}", "scope": scope,
        "audit_output": str(Path(output).resolve())}
    trainer["meters"] = {}  # Loss-only held-out check; no claims of benchmark AP.
    trainer["loss"] = {"all": "${roboflow_train.loss}", "heldout": "${roboflow_train.loss}"}
    # Fixed AdamW avoids the reference's 20-epoch schedules during a 2-epoch
    # smoke check. The explicit native mask criterion still uses native matching.
    trainer["optim"] = {
        "amp": {"enabled": True, "amp_dtype": "bfloat16"},
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-5, "weight_decay": 0.01},
        "options": None, "param_group_modifiers": None,
        "gradient_clip": {"_target_": "sam3.train.optim.optimizer.GradientClipper",
                          "max_norm": 1.0, "norm_type": 2}}
    trainer["cuda"] = {"cudnn_benchmark": True, "allow_tf32": False}
    for split in ("train", "val"):
        loader = trainer["data"][split]
        loader["dataset"]["coco_json_loader"] = {
            "_target_": "sam3.train.data.coco_json_loaders.COCO_FROM_JSON",
            "_partial_": True, "include_negatives": False}
        loader["collate_fn"] = {
            "_target_": "sam3.train.data.collator.collate_fn_api", "_partial_": True,
            "repeats": 1, "dict_key": "all" if split == "train" else "heldout", "with_seg_masks": True}
    trainer["logging"].update(log_freq=1, log_scalar_frequency=1,
                              log_dir="${launcher.experiment_log_dir}/logs")
    trainer["checkpoint"].update(save_freq=0)  # Save latest, without redundant full-model copies.
    config["launcher"].update(num_nodes=1, gpus_per_node=1)
    config["submitit"].update(use_cluster=False, cpus_per_task=12)
    config["submitit"].pop("job_array", None)
    config.pop("all_roboflow_supercategories", None)
    config["bwlab"] = {"native_revision": NATIVE_REVISION, "scope": scope,
                       "task": "concept-driven instance segmentation, not interactive point decoder",
                       "purpose": "training smoke check; not reproduction of a published metric"}
    return config
