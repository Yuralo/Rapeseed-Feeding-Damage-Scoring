"""Configuration for frozen fixed-tile plus SAM-adaptive hybrid MIL."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from experiments.dinov3_grid_sam_adaptive_mil.config import (
    AdaptiveCropSettings,
    ContextSettings,
    ModelSettings,
    SegmentationSettings,
)
from experiments.dinov3_grid_sam_adaptive_mil.config import (
    Config as AdaptiveConfig,
)
from experiments.dinov3_grid_tiled_mil.config import (
    Config as SingleScaleConfig,
)
from experiments.dinov3_grid_tiled_mil.config import (
    DataSettings,
    FeatureSettings,
    RuntimeSettings,
    TileSettings,
    TrainingSettings,
)
from experiments.dinov3_grid_tiled_mil.config import (
    ExperimentSettings as SingleExperimentSettings,
)
from experiments.dinov3_grid_tiled_mil.config import (
    ModelSettings as SingleModelSettings,
)
from experiments.dinov3_grid_tiled_mil.config import (
    OutputSettings as SingleOutputSettings,
)


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "dinov3_grid_4x4_sam_adaptive_hybrid_mil"


@dataclass(frozen=True)
class FineSettings:
    rows: int = 4
    columns: int = 4
    overlap_fraction: float = 0.25
    cache_dir: str = "cache/dinov3_grid_tiled_mil_features_4x4"


@dataclass(frozen=True)
class OutputSettings:
    run_dir: str = "outputs/dinov3_grid_4x4_sam_adaptive_hybrid_mil_clean_inset075"
    best_checkpoint_name: str = "best_mse.pt"
    best_mae_checkpoint_name: str = "best_mae.pt"
    last_checkpoint_name: str = "last.pt"
    save_plots: bool = True
    example_images: int = 12
    example_columns: int = 4
    attention_inspection_images: int = 6
    attention_arrays_name: str = "hybrid_attention.npz"


@dataclass(frozen=True)
class Config(AdaptiveConfig):
    experiment: ExperimentSettings = ExperimentSettings()
    fine: FineSettings = FineSettings()
    output: OutputSettings = OutputSettings()

    def validate(self) -> None:
        super().validate()
        if self.fine.rows < 1 or self.fine.columns < 1:
            raise ValueError("fine rows/columns must be positive")
        if not 0 <= self.fine.overlap_fraction < 1:
            raise ValueError("fine.overlap_fraction must be in [0, 1)")
        if not self.fine.cache_dir.strip():
            raise ValueError("fine.cache_dir cannot be empty")
        if self.fine.rows * self.fine.columns <= self.context.rows * self.context.columns:
            raise ValueError("fine must contain more tiles than context")
        if self.fine.cache_dir == self.context.cache_dir:
            raise ValueError("fine and context cache directories must differ")

    def fine_config(self) -> SingleScaleConfig:
        return SingleScaleConfig(
            data=self.data,
            experiment=SingleExperimentSettings(name=self.experiment.name),
            tiles=TileSettings(
                rows=self.fine.rows,
                columns=self.fine.columns,
                overlap_fraction=self.fine.overlap_fraction,
                include_global_view=True,
            ),
            features=FeatureSettings(
                backbone=self.features.backbone,
                processor=self.features.processor,
                representation=self.features.representation,
                cache_dir=self.fine.cache_dir,
            ),
            model=SingleModelSettings(
                projection_dim=self.model.projection_dim,
                attention_hidden_dim=self.model.attention_hidden_dim,
                attention_dropout=self.model.attention_dropout,
                attention_temperature=self.model.attention_temperature,
                head_hidden_dim=self.model.head_hidden_dim,
                dropout=self.model.dropout,
            ),
            training=self.training,
            runtime=self.runtime,
            output=SingleOutputSettings(),
        )


SettingsType = TypeVar("SettingsType")


def _settings(settings_type: type[SettingsType], values: dict[str, Any]) -> SettingsType:
    known = {item.name for item in fields(settings_type)}
    unexpected = sorted(set(values) - known)
    if unexpected:
        raise ValueError(f"Unknown {settings_type.__name__} option(s): {', '.join(unexpected)}")
    return settings_type(**values)


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    known = {
        "experiment",
        "data",
        "segmentation",
        "context",
        "fine",
        "adaptive_crops",
        "features",
        "model",
        "training",
        "runtime",
        "output",
    }
    unexpected = sorted(set(raw) - known)
    if unexpected:
        raise ValueError(f"Unknown configuration section(s): {', '.join(unexpected)}")
    if "data" not in raw or "dataset_dir" not in raw["data"]:
        raise ValueError("[data].dataset_dir is required")
    config = Config(
        experiment=_settings(ExperimentSettings, raw.get("experiment", {})),
        data=_settings(DataSettings, raw["data"]),
        segmentation=_settings(SegmentationSettings, raw.get("segmentation", {})),
        context=_settings(ContextSettings, raw.get("context", {})),
        fine=_settings(FineSettings, raw.get("fine", {})),
        adaptive_crops=_settings(AdaptiveCropSettings, raw.get("adaptive_crops", {})),
        features=_settings(FeatureSettings, raw.get("features", {})),
        model=_settings(ModelSettings, raw.get("model", {})),
        training=_settings(TrainingSettings, raw.get("training", {})),
        runtime=_settings(RuntimeSettings, raw.get("runtime", {})),
        output=_settings(OutputSettings, raw.get("output", {})),
    )
    config.validate()
    return config


__all__ = ["Config", "FineSettings", "load_config"]

