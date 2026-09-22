"""Configuration for the strongest plant-damage model trained on paired means."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

from experiments.dinov3_plant_damage_mil.config import load_config as load_plant


@dataclass(frozen=True)
class Config:
    plant_reference_config_path: str
    source_manifest: str
    reference_ood_dir: str
    reference_validation_predictions: str
    run_dir: str
    seed: int = 42
    epochs: int = 60
    batch_size: int = 24
    num_workers: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.001
    residual_penalty: float = 0.01
    early_stopping_patience: int = 12
    gradient_clip_norm: float = 1.0

    @cached_property
    def reference(self):
        return load_plant(self.plant_reference_config_path)

    @cached_property
    def base(self):
        original = self.reference.base
        data = replace(
            original.data,
            manifest_dir=str(Path(self.run_dir) / "manifests"),
            pretrain_manifest="pretrain.csv",
            finetune_manifest="finetune.csv",
            validation_manifest="validation.csv",
            test_manifest="test.csv",
        )
        return replace(original, data=data, output=replace(original.output, run_dir=self.run_dir))

    @property
    def base_checkpoint(self):
        return self.reference.base_checkpoint

    @property
    def cache_dir(self):
        return self.reference.cache_dir

    @property
    def patches_per_side(self):
        return self.reference.patches_per_side

    @property
    def overlap_fraction(self):
        return self.reference.overlap_fraction

    @property
    def minimum_foreground_fraction(self):
        return self.reference.minimum_foreground_fraction

    @property
    def hidden_dim(self):
        return self.reference.hidden_dim

    @property
    def inspection_images(self):
        return self.reference.inspection_images

    def validate(self) -> None:
        self.reference.validate()
        if min(self.epochs, self.batch_size, self.early_stopping_patience) < 1:
            raise ValueError("Training counts must be positive")
        if self.num_workers < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid training settings")
        if self.residual_penalty < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("Invalid penalty or gradient clipping")


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    unexpected = set(values) - set(Config.__dataclass_fields__)
    if unexpected:
        raise ValueError(f"Unknown settings: {sorted(unexpected)}")
    config = Config(**values)
    config.validate()
    return config
