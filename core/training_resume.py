"""Portable random state and cooperative shutdown for resumable training."""

from contextlib import contextmanager
import random
import signal

import numpy as np
import torch


class TrainingInterrupted(RuntimeError):
    pass


def random_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state):
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


@contextmanager
def shutdown_signals(trainer):
    """Never interrupt an optimizer write; request a stop at the next safe boundary."""
    previous = {}

    def request_stop(number, frame):
        trainer.stop_reason = signal.Signals(number).name

    for name in ("SIGUSR1", "SIGTERM", "SIGINT"):
        number = getattr(signal, name, None)
        if number is not None:
            previous[number] = signal.signal(number, request_stop)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def check_resume_config(saved, current):
    keys = (
        "annotation_sha256",
        "seed",
        "transform",
        "transform_parameters",
        "resolution",
        "learning_rate",
        "lr_schedule",
        "training_order",
        "epochs",
        "native_revision",
        "protocol_version",
        "loss_region",
        "drop_fully_hidden",
        "checkpoint_metric",
    )
    if any(saved.get(key) != current.get(key) for key in keys):
        raise ValueError("Resume requires the same data, saved order, model and training settings")
