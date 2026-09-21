"""Configuration for the plant-patch residual experiment."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from experiments.dinov3_hierarchical_three_view_mil.config import load_config as load_base_config


@dataclass(frozen=True)
class Config:
    base_config_path: str
    base_checkpoint: str
    run_dir: str
    cache_dir: str
    patches_per_side: int = 2
    overlap_fraction: float = 0.15
    minimum_foreground_fraction: float = 0.005
    hidden_dim: int = 64
    epochs: int = 60
    batch_size: int = 24
    learning_rate: float = 0.0003
    weight_decay: float = 0.001
    residual_penalty: float = 0.01
    patience: int = 12
    num_workers: int = 4
    seed: int = 42
    inspection_images: int = 8

    @cached_property
    def base(self):
        return load_base_config(self.base_config_path)

    def validate(self) -> None:
        if self.patches_per_side not in (2, 3):
            raise ValueError("patches_per_side must be 2 or 3")
        if not 0 <= self.overlap_fraction < 0.5:
            raise ValueError("overlap_fraction must be in [0, 0.5)")
        if not 0 <= self.minimum_foreground_fraction < 1:
            raise ValueError("minimum_foreground_fraction must be in [0, 1)")
        if min(self.hidden_dim, self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("model/training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.residual_penalty < 0:
            raise ValueError("invalid optimization settings")
        if self.num_workers < 0 or self.inspection_images < 0:
            raise ValueError("invalid worker/inspection settings")
        self.base.validate()


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    allowed = set(Config.__dataclass_fields__)
    extra = set(values) - allowed
    if extra:
        raise ValueError(f"Unknown plant-damage config options: {sorted(extra)}")
    config = Config(**values)
    config.validate()
    return config
