"""Small, explicit configuration for the weak-to-gold transfer test."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

from experiments.dinov3_hierarchical_three_view_mil.config import load_config as load_base_config


@dataclass(frozen=True)
class Config:
    base_config_path: str
    scored_manifest: str
    manifest_dir: str
    run_dir: str
    mode: str = "all_weak"
    expected_gold_images: int = 470
    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.001
    warmup_fraction: float = 0.1
    minimum_learning_rate_ratio: float = 0.05
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0

    @cached_property
    def base(self):
        return load_base_config(self.base_config_path)

    @cached_property
    def routed_base(self):
        """Reuse the proven three-view feature schema without touching old outputs."""
        return replace(
            self.base,
            data=replace(
                self.base.data,
                manifest_dir=self.manifest_dir,
                finetune_manifest="weak_train.csv",
                validation_manifest="gold_validation.csv",
            ),
            output=replace(
                self.base.output,
                run_dir=str(Path(self.run_dir) / "feature_preparation"),
            ),
        )

    def validate(self) -> None:
        if self.mode not in {"all_weak", "dual_only"}:
            raise ValueError("mode must be all_weak or dual_only")
        if min(self.expected_gold_images, self.epochs, self.batch_size,
               self.early_stopping_patience) < 1:
            raise ValueError("gold count and training counts must be positive")
        if self.num_workers < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer/worker settings")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0 <= self.minimum_learning_rate_ratio <= 1:
            raise ValueError("minimum_learning_rate_ratio must be in [0, 1]")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        self.base.validate()


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    unexpected = set(values) - set(Config.__dataclass_fields__)
    if unexpected:
        raise ValueError(f"Unknown weak-only settings: {sorted(unexpected)}")
    config = Config(**values)
    config.validate()
    return config
