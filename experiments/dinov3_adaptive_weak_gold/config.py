"""Reuse the successful adaptive head and the leakage-safe weak/gold manifests."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

from experiments.dinov3_grid_sam_adaptive_mil.config import load_config as load_adaptive
from experiments.dinov3_weak_only_gold_validation.config import load_config as load_weak


@dataclass(frozen=True)
class Config:
    weak_config_path: str
    adaptive_reference_config_path: str
    run_dir: str
    context_cache_dir: str
    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    warmup_fraction: float = 0.1
    minimum_learning_rate_ratio: float = 0.05
    early_stopping_patience: int = 12
    gradient_clip_norm: float = 1.0

    @cached_property
    def weak(self):
        return load_weak(self.weak_config_path)

    @cached_property
    def adaptive(self):
        original = load_adaptive(self.adaptive_reference_config_path)
        training = replace(
            original.training,
            seed=self.seed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_fraction=self.warmup_fraction,
            minimum_learning_rate_ratio=self.minimum_learning_rate_ratio,
            early_stopping_patience=self.early_stopping_patience,
            gradient_clip_norm=self.gradient_clip_norm,
        )
        return replace(original, training=training, output=replace(original.output, run_dir=self.run_dir))

    def validate(self) -> None:
        if self.weak.mode != "all_weak":
            raise ValueError("This comparison requires the all_weak manifests")
        base, adaptive = self.weak.routed_base, self.adaptive
        if (base.features.backbone, base.features.processor, base.features.representation) != (
            adaptive.features.backbone, adaptive.features.processor,
            adaptive.features.representation,
        ):
            raise ValueError("Adaptive reference and routed three-view DINO features differ")
        if base.adaptive_crops != adaptive.adaptive_crops:
            raise ValueError("SAM plant-crop settings differ from the successful adaptive model")
        if (adaptive.context.rows, adaptive.context.columns) != (3, 3):
            raise ValueError("The reference adaptive model must use the original 3x3 context")
        if not self.context_cache_dir or not self.run_dir:
            raise ValueError("Cache and output directories must be configured")
        if min(self.epochs, self.batch_size, self.early_stopping_patience) < 1:
            raise ValueError("Training counts must be positive")
        if self.num_workers < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer or worker settings")
        if not 0 <= self.warmup_fraction < 1 or not 0 <= self.minimum_learning_rate_ratio <= 1:
            raise ValueError("Invalid learning-rate schedule settings")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    unexpected = set(values) - set(Config.__dataclass_fields__)
    if unexpected:
        raise ValueError(f"Unknown adaptive weak/gold settings: {sorted(unexpected)}")
    config = Config(**values)
    config.validate()
    return config
