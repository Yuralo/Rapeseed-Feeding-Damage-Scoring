"""Configuration for SAM-guided plant-centred frozen-feature MIL."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from experiments.dinov3_grid_lora_patch_attention_sam_fusion.config import (
    SegmentationSettings,
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
    name: str = "dinov3_grid_sam_adaptive_mil"


@dataclass(frozen=True)
class ContextSettings:
    rows: int = 3
    columns: int = 3
    overlap_fraction: float = 0.25
    cache_dir: str = "cache/dinov3_grid_tiled_mil_features"


@dataclass(frozen=True)
class AdaptiveCropSettings:
    grouping_dilation_px: int = 20
    context_scale: float = 2.0
    minimum_crop_size: int = 160
    maximum_crop_size: int = 448
    maximum_instances: int = 20
    minimum_mask_coverage: float = 0.98


@dataclass(frozen=True)
class ModelSettings:
    projection_dim: int = 128
    attention_hidden_dim: int = 64
    attention_dropout: float = 0.1
    attention_temperature: float = 1.0
    head_hidden_dim: int = 128
    dropout: float = 0.35


@dataclass(frozen=True)
class OutputSettings:
    run_dir: str = "outputs/dinov3_grid_sam_adaptive_mil_clean_inset075"
    best_checkpoint_name: str = "best_mse.pt"
    best_mae_checkpoint_name: str = "best_mae.pt"
    last_checkpoint_name: str = "last.pt"
    save_plots: bool = True
    example_images: int = 12
    example_columns: int = 4
    attention_inspection_images: int = 6
    attention_arrays_name: str = "plant_instance_attention.npz"
    feature_failure_log: str = "adaptive_feature_failures.jsonl"
    inspection_dir: str = "adaptive_crop_inspection"


@dataclass(frozen=True)
class Config:
    data: DataSettings
    experiment: ExperimentSettings = ExperimentSettings()
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    context: ContextSettings = ContextSettings()
    adaptive_crops: AdaptiveCropSettings = AdaptiveCropSettings()
    features: FeatureSettings = field(
        default_factory=lambda: FeatureSettings(
            cache_dir="cache/dinov3_grid_sam_adaptive_mil_features"
        )
    )
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = field(default_factory=TrainingSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    output: OutputSettings = OutputSettings()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def context_config(self) -> SingleScaleConfig:
        return SingleScaleConfig(
            data=self.data,
            experiment=SingleExperimentSettings(name=self.experiment.name),
            tiles=TileSettings(
                rows=self.context.rows,
                columns=self.context.columns,
                overlap_fraction=self.context.overlap_fraction,
                include_global_view=True,
            ),
            features=FeatureSettings(
                backbone=self.features.backbone,
                processor=self.features.processor,
                representation=self.features.representation,
                cache_dir=self.context.cache_dir,
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

    def validate(self) -> None:
        if not self.data.dataset_dir.strip() or not self.data.image_extension.startswith("."):
            raise ValueError("data paths/extensions are invalid")
        if not 0 < self.data.validation_fraction < 1:
            raise ValueError("data.validation_fraction must be between 0 and 1")
        if not self.data.normalize_targets:
            raise ValueError("The adaptive MIL experiment requires normalized targets")
        if not 0 <= self.data.grid_inner_margin_fraction < 0.25:
            raise ValueError("grid inner margin must be in [0, 0.25)")
        if not self.segmentation.mask_cache_dir.strip():
            raise ValueError("segmentation.mask_cache_dir cannot be empty")
        if not self.segmentation.prompts:
            raise ValueError("segmentation.prompts cannot be empty")
        if not 0 <= self.segmentation.minimum_foreground_fraction:
            raise ValueError("minimum foreground fraction cannot be negative")
        if self.segmentation.maximum_foreground_fraction > 1:
            raise ValueError("maximum foreground fraction cannot exceed one")
        if self.context.rows < 1 or self.context.columns < 1:
            raise ValueError("context rows/columns must be positive")
        if not 0 <= self.context.overlap_fraction < 1 or not self.context.cache_dir.strip():
            raise ValueError("context cache/layout settings are invalid")
        crops = self.adaptive_crops
        if crops.grouping_dilation_px < 0:
            raise ValueError("grouping dilation cannot be negative")
        if crops.context_scale < 1:
            raise ValueError("adaptive crop context_scale must be at least one")
        if crops.minimum_crop_size < 1 or crops.maximum_crop_size < crops.minimum_crop_size:
            raise ValueError("adaptive crop size bounds are invalid")
        if crops.maximum_instances < 1:
            raise ValueError("adaptive maximum_instances must be positive")
        if not 0 < crops.minimum_mask_coverage <= 1:
            raise ValueError("minimum_mask_coverage must be in (0, 1]")
        if self.features.representation != "cls_mean":
            raise ValueError("features.representation must be 'cls_mean'")
        if self.features.storage_dtype not in {"float16", "float32"}:
            raise ValueError("feature storage dtype must be float16 or float32")
        if not self.features.cache_dir.strip() or self.features.extraction_batch_size < 1:
            raise ValueError("feature cache settings are invalid")
        if self.model.projection_dim < 1 or self.model.attention_hidden_dim < 1:
            raise ValueError("model dimensions must be positive")
        if self.model.head_hidden_dim < 1 or self.model.attention_temperature <= 0:
            raise ValueError("model head/attention settings are invalid")
        if not 0 <= self.model.attention_dropout < 1 or not 0 <= self.model.dropout < 1:
            raise ValueError("model dropout settings are invalid")
        training = self.training
        if training.epochs < 1 or training.batch_size < 1 or training.num_workers < 0:
            raise ValueError("training epoch/batch/worker settings are invalid")
        if training.learning_rate <= 0 or training.weight_decay < 0:
            raise ValueError("training optimizer settings are invalid")
        if training.early_stopping_patience < 1:
            raise ValueError("early stopping patience must be positive")
        if self.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("runtime.device is invalid")
        names = (
            self.output.best_checkpoint_name,
            self.output.best_mae_checkpoint_name,
            self.output.last_checkpoint_name,
        )
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("checkpoint names must be nonempty and distinct")


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
        adaptive_crops=_settings(AdaptiveCropSettings, raw.get("adaptive_crops", {})),
        features=_settings(FeatureSettings, raw.get("features", {})),
        model=_settings(ModelSettings, raw.get("model", {})),
        training=_settings(TrainingSettings, raw.get("training", {})),
        runtime=_settings(RuntimeSettings, raw.get("runtime", {})),
        output=_settings(OutputSettings, raw.get("output", {})),
    )
    config.validate()
    return config
