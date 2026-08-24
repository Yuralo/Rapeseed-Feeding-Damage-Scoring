"""Optimizer scheduling and CUDA settings for the small MIL head."""

from __future__ import annotations

import math

import torch

from .config import Config


def configure_acceleration(config: Config, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    if config.runtime.allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def make_scheduler(optimizer, config: Config, loader_length: int):
    total_steps = max(1, loader_length * config.training.epochs)
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


def make_optimizer(model, config: Config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim > 1 and not name.endswith(".bias") else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": config.training.weight_decay,
                "group_name": "head_decay",
            },
            {"params": no_decay, "weight_decay": 0.0, "group_name": "head_no_decay"},
        ],
        lr=config.training.learning_rate,
    )


def learning_rates(optimizer) -> dict[str, float]:
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
