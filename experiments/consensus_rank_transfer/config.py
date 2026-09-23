"""Strict, minimal configuration for consensus-safe rank transfer."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    base_config: str
    base_checkpoint: str
    run_dir: str
    target_cohort: str
    seeds: tuple[int, ...]
    cv_seed: int
    cv_folds: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    max_residual: float
    rank_interval_margin: float
    rank_prediction_margin: float
    max_pairs_per_image: int
    rank_weights: tuple[float, ...]
    residual_penalties: tuple[float, ...]
    midpoint_weights: tuple[float, ...]
    bootstrap_replicates: int
    practical_mae_reduction: float
    low_score_max_degradation: float

    def validate(self) -> None:
        if not self.target_cohort or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Target cohort and distinct seeds are required")
        if self.cv_seed not in self.seeds or self.cv_folds < 2:
            raise ValueError("CV seed must be in seeds and cv_folds must be at least two")
        if min(self.max_epochs, self.patience, self.hidden_dim, self.max_pairs_per_image,
               self.bootstrap_replicates) < 1:
            raise ValueError("Counts must be positive")
        if min(self.learning_rate, self.max_residual) <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate, maximum residual, and weight decay are invalid")
        if self.rank_interval_margin < 0 or self.rank_prediction_margin < 0:
            raise ValueError("Ranking margins cannot be negative")
        for name in ("rank_weights", "residual_penalties", "midpoint_weights"):
            values = getattr(self, name)
            if not values or any(value < 0 for value in values):
                raise ValueError(f"{name} must contain nonnegative values")
        if self.practical_mae_reduction < 0 or self.low_score_max_degradation < 0:
            raise ValueError("Decision thresholds cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    expected = {item.name for item in fields(Config)}
    if set(values) != expected:
        raise ValueError(f"Configuration mismatch; missing={sorted(expected-set(values))}, "
                         f"unknown={sorted(set(values)-expected)}")
    for name in ("seeds", "rank_weights", "residual_penalties", "midpoint_weights"):
        values[name] = tuple(values[name])
    result = Config(**values)
    result.validate()
    return result
