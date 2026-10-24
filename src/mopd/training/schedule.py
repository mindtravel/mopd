"""Learning-rate schedule shared by every training entry (MOPD and SFT).

A method comparison is only fair if both arms step the optimizer under the same
schedule, so the schedule is read from one config field by one function instead of
being re-implemented per script. Absent `lr_schedule` means constant, which is what
every MOPD run before this module used (bare AdamW, no scheduler).

    "lr_schedule": {"type": "constant"}
    "lr_schedule": {"type": "cosine", "warmup_steps": 4, "min_lr_ratio": 0.1}
"""

import math

import torch

SCHEDULES = ("constant", "cosine")


def describe(config):
    spec = dict(config.get("lr_schedule") or {"type": "constant"})
    if spec.get("type") not in SCHEDULES:
        raise ValueError(f"lr_schedule.type must be one of {SCHEDULES}, got {spec.get('type')!r}")
    return spec


def build_scheduler(optimizer, config):
    """One scheduler.step() per optimizer step; multiplier applies to config['lr']."""
    spec = describe(config)
    steps = config["steps"]
    warmup = int(spec.get("warmup_steps", 0))
    floor = float(spec.get("min_lr_ratio", 0.0))

    def multiplier(step):  # step = optimizer steps already taken
        if warmup and step < warmup:
            return (step + 1) / warmup
        if spec["type"] == "constant":
            return 1.0
        progress = (step - warmup) / max(1, steps - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
