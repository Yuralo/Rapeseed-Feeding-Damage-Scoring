"""Configuration for bite-pair preparation, pretraining, and gold fitting."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path

from experiments.dinov3_plant_damage_mil.config import load_config as load_plant_config


@dataclass(frozen=True)
class Config:
    plant_config_path: str
    adaptation_manifest: str
    run_dir: str
    pair_cache_dir: str
    seed: int = 42
    severity_levels: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    minimum_leaf_pixels: int = 100
    minimum_removed_pixels: int = 8
    minimum_accepted_levels: int = 2
    minimum_coverage_fraction: float = 0.95
    synthetic_holdout_fraction: float = 0.10
    pretrain_epochs: int = 25
    pretrain_batch_size: int = 64
    pretrain_learning_rate: float = 0.0003
    pretrain_patience: int = 7
    pretrain_workers: int = 4
    order_margin: float = 0.02
    order_weight: float = 0.25
    sham_weight: float = 0.5
    delta_huber_beta: float = 0.05
    quality_examples_per_cohort: int = 25
    fitting_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    bootstrap_replicates: int = 5000

    @cached_property
    def plant(self):
        return load_plant_config(self.plant_config_path)

    def validate(self) -> None:
        self.plant.validate()
        if not self.severity_levels or any(not 0 < x < 0.5 for x in self.severity_levels):
            raise ValueError("severity_levels must be positive fractions below 0.5")
        if sorted(set(self.severity_levels)) != list(self.severity_levels):
            raise ValueError("severity_levels must be distinct and increasing")
        if min(self.minimum_leaf_pixels, self.minimum_removed_pixels, self.minimum_accepted_levels) < 1:
            raise ValueError("minimum pixel and level counts must be positive")
        if self.minimum_accepted_levels > len(self.severity_levels):
            raise ValueError("minimum_accepted_levels exceeds severity_levels")
        if not 0 < self.minimum_coverage_fraction <= 1:
            raise ValueError("minimum_coverage_fraction must be in (0, 1]")
        if not 0 < self.synthetic_holdout_fraction < 0.5:
            raise ValueError("synthetic_holdout_fraction must be in (0, 0.5)")
        if min(self.pretrain_epochs, self.pretrain_batch_size, self.pretrain_patience) < 1:
            raise ValueError("pretraining counts must be positive")
        if self.pretrain_workers < 0 or self.pretrain_learning_rate <= 0:
            raise ValueError("invalid pretraining workers or learning rate")
        if any(x < 0 for x in (self.order_margin, self.order_weight, self.sham_weight)):
            raise ValueError("pretraining loss weights and margin cannot be negative")
        if self.delta_huber_beta <= 0 or self.bootstrap_replicates < 1:
            raise ValueError("invalid Huber beta or bootstrap count")
        if not self.fitting_seeds or len(set(self.fitting_seeds)) != len(self.fitting_seeds):
            raise ValueError("fitting_seeds must be nonempty and unique")


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    unexpected = set(values) - {entry.name for entry in fields(Config)}
    if unexpected:
        raise ValueError(f"Unknown counterfactual options: {sorted(unexpected)}")
    for key in ("severity_levels", "fitting_seeds"):
        if key in values:
            values[key] = tuple(values[key])
    config = Config(**values)
    config.validate()
    return config
