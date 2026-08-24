"""Mixed precision, scheduling, and device-specific runtime helpers."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from .config import Config


def configure_acceleration(config: Config, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    if config.runtime.allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def autocast_context(config: Config, device: torch.device):
    if device.type != "cuda" or config.runtime.mixed_precision == "none":
        return nullcontext()
    dtype = torch.float16 if config.runtime.mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(config: Config, device: torch.device):
    enabled = device.type == "cuda" and config.runtime.mixed_precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def optimizer_steps_per_epoch(loader_length: int, accumulation_steps: int) -> int:
    return math.ceil(loader_length / accumulation_steps)


def make_scheduler(optimizer, config: Config, loader_length: int):
    steps_per_epoch = optimizer_steps_per_epoch(
        loader_length, config.training.gradient_accumulation_steps
    )
    total_steps = max(1, steps_per_epoch * config.training.epochs)
    warmup_steps = int(total_steps * config.training.warmup_fraction)
    minimum = config.training.minimum_learning_rate_ratio

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum + (1.0 - minimum) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return scheduler, total_steps, warmup_steps


def learning_rates(optimizer) -> dict[str, float]:
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
