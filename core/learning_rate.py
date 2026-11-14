"""Learning-rate schedules indexed by completed optimizer updates for safe resume."""

import math


def learning_rate_for_update(peak, completed, total, schedule=None):
    """Return the rate used for the next (zero-indexed) optimizer update."""
    schedule = schedule or {"name": "constant"}
    name = schedule.get("name", "constant")
    if peak <= 0 or not math.isfinite(peak):
        raise ValueError("Learning rate must be positive and finite")
    if name == "constant":
        return peak
    if name != "warmup_cosine":
        raise ValueError(f"Unknown learning-rate schedule: {name}")
    fraction = schedule.get("warmup_fraction", 0.05)
    start = schedule.get("start_factor", 0.1)
    end = schedule.get("end_factor", 0.1)
    if not 0 < fraction < 1 or not 0 < start <= 1 or not 0 <= end <= 1:
        raise ValueError("Invalid warmup fraction or learning-rate factors")
    if total < 2 or not 0 <= completed < total:
        raise ValueError("Warmup/cosine requires at least two updates and a valid update index")
    warmup = min(total - 1, max(1, round(total * fraction)))
    if completed < warmup:
        progress = completed / max(1, warmup - 1)
        return peak * (start + (1 - start) * progress)
    progress = (completed - warmup + 1) / (total - warmup)
    return peak * (end + (1 - end) * (1 + math.cos(math.pi * progress)) / 2)
