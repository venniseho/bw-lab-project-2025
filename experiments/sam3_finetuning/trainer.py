"""Audit Meta's real training loop without replacing its forward/backward logic."""
import json
import math
from pathlib import Path
import time

import torch
from sam3.train.trainer import Trainer, unwrap_ddp_if_wrapped


class AuditedTrainer(Trainer):
    def __init__(self, *, audit_output, wandb_mode="offline", **kwargs):
        self.audit_output = Path(audit_output)
        self.completed_updates = 0
        self.peak_allocated_bytes = 0
        self.update_records = []
        self.wandb_run = None
        super().__init__(**kwargs)
        self.optim.optimizer.register_step_post_hook(self._after_update)
        if wandb_mode != "disabled":
            import wandb
            self.wandb_run = wandb.init(
                project="bwlab-sam3", name=self.audit_output.name,
                dir=str(self.audit_output), mode=wandb_mode,
                config={"task": "native concept instance segmentation smoke check",
                        "model": dict(self.model_conf), "epochs": self.max_epochs})

    def _step(self, batch, model, phase):
        result = super()._step(batch, model, phase)
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, torch.cuda.max_memory_allocated())
        losses, _, detailed = result
        values = {key: float(value.detach().item()) for key, value in detailed.items()}
        values.update({key: float(value.detach().item()) for key, value in losses.items()})
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite native loss")
        if not any("loss_mask" in key for key in values) or not any("loss_dice" in key for key in values):
            raise RuntimeError("Native mask and dice losses were not computed")
        with (self.audit_output / "loss_steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"phase": phase, "native_step": self.steps[phase], **values})+"\n")
        if phase == "train":
            if not next(iter(losses.values())).requires_grad:
                raise RuntimeError("Training loss has no gradient graph")
            self.pending_losses = values
        elif self.wandb_run:
            self.wandb_run.log({f"validation/{key}": value for key, value in values.items()})
        return result

    def _run_step(self, *args, **kwargs):
        self.update_start = time.perf_counter()
        super()._run_step(*args, **kwargs)  # Native forward, loss and backward.
        model = unwrap_ddp_if_wrapped(self.model)
        self.probes = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                grad = parameter.grad.detach()
                if not torch.isfinite(grad).all():
                    raise FloatingPointError(f"Nonfinite gradient: {name}")
                if grad.count_nonzero().item() and len(self.probes) < 3:
                    self.probes.append((name, parameter, parameter.detach().clone()))
        if not self.probes:
            raise RuntimeError("Backward produced no nonzero trainable gradients")
        self.frozen_probe = next(((name, parameter, parameter.detach().flatten()[:1024].clone())
                                  for name, parameter in model.named_parameters()
                                  if not parameter.requires_grad), None)

    def _after_update(self, optimizer, args, kwargs):
        torch.cuda.synchronize()
        changed = [name for name, parameter, before in self.probes
                   if not torch.equal(parameter.detach(), before)]
        if not changed:
            raise RuntimeError("Optimizer step did not change any probed trainable weights")
        if self.frozen_probe:
            name, parameter, before = self.frozen_probe
            if not torch.equal(parameter.detach().flatten()[:1024], before):
                raise RuntimeError(f"Frozen parameter probe changed: {name}")
        self.completed_updates += 1
        self.peak_allocated_bytes = max(getattr(self, "peak_allocated_bytes", 0), torch.cuda.max_memory_allocated())
        record = {"optimizer_step": self.completed_updates,
                  "step_wall_seconds": time.perf_counter()-self.update_start,
                  "changed_parameter_probes": changed,
                  "frozen_parameter_probe": self.frozen_probe[0] if self.frozen_probe else None,
                  "allocated_gpu_bytes": torch.cuda.memory_allocated(),
                  **self.pending_losses}
        self.update_records.append(record)
        with (self.audit_output / "optimizer_steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record)+"\n")
        if self.wandb_run:
            self.wandb_run.log({key: value for key, value in record.items()
                                if isinstance(value, (float, int))})
        print(f"AUDIT: optimizer update {self.completed_updates}; nonzero gradients; "
              f"changed weights={changed}", flush=True)
        self.probes = []

    def run(self):
        started = time.perf_counter()
        status = "failed"
        try:
            super().run()
            if not self.completed_updates:
                raise RuntimeError("Training completed without any verified optimizer updates")
            checkpoint = Path(self.checkpoint_conf.save_dir) / "checkpoint.pt"
            if not checkpoint.is_file():
                raise RuntimeError("Native trainer did not save checkpoint.pt")
            status = "complete"
        finally:
            (self.audit_output / "training_audit.json").write_text(json.dumps({
                "status": status, "verified_optimizer_updates": self.completed_updates,
                "training_and_validation_wall_seconds": time.perf_counter()-started,
                "peak_allocated_gpu_bytes": max(self.peak_allocated_bytes, torch.cuda.max_memory_allocated()),
                "note": "Mask-head concept training; not an interactive point-decoder benchmark"}, indent=2)+"\n")
            if self.wandb_run:
                self.wandb_run.finish(exit_code=0 if status == "complete" else 1)
