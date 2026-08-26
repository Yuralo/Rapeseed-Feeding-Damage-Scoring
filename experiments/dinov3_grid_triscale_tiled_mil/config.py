"""Configuration for 3x3-context plus 4x4/5x5-attention DINOv3 MIL."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from experiments.dinov3_grid_tiled_mil.config import Config as SingleScaleConfig
from experiments.dinov3_grid_tiled_mil.config import (
    DataSettings,
    RuntimeSettings,
    TileSettings,
    TrainingSettings,
)
from experiments.dinov3_grid_tiled_mil.config import (
    ExperimentSettings as SingleExperimentSettings,
)
from experiments.dinov3_grid_tiled_mil.config import FeatureSettings as SingleFeatureSettings
from experiments.dinov3_grid_tiled_mil.config import ModelSettings as SingleModelSettings
from experiments.dinov3_grid_tiled_mil.config import OutputSettings as SingleOutputSettings


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "dinov3_grid_triscale_tiled_mil"


@dataclass(frozen=True)
class FeatureSettings:
    backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    representation: str = "cls_mean"


@dataclass(frozen=True)
class ScaleSettings:
    rows: int
    columns: int
    cache_dir: str
    overlap_fraction: float = 0.25
    include_global_view: bool = True


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
    run_dir: str = "outputs/dinov3_grid_triscale_3x3_4x4_5x5_mil_clean_inset075"
    best_checkpoint_name: str = "best_mse.pt"
    best_mae_checkpoint_name: str = "best_mae.pt"
    last_checkpoint_name: str = "last.pt"
    save_plots: bool = True
    example_images: int = 12
    example_columns: int = 4
    attention_inspection_images: int = 6
    attention_arrays_name: str = "triscale_tile_attention.npz"


@dataclass(frozen=True)
class Config:
    data: DataSettings
    context: ScaleSettings
    regional: ScaleSettings
    local: ScaleSettings
    experiment: ExperimentSettings = ExperimentSettings()
    features: FeatureSettings = FeatureSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = field(default_factory=TrainingSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    output: OutputSettings = OutputSettings()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def scales(self) -> tuple[tuple[str, ScaleSettings], ...]:
        return (
            ("context", self.context),
            ("regional", self.regional),
            ("local", self.local),
        )

    def validate(self) -> None:
        if not self.data.dataset_dir.strip():
            raise ValueError("data.dataset_dir cannot be empty")
        if not self.data.image_extension.startswith("."):
            raise ValueError("data.image_extension must start with a dot")
        if not 0 < self.data.validation_fraction < 1:
            raise ValueError("data.validation_fraction must be between 0 and 1")
        if self.data.grid_crop_size < 1 or not self.data.grid_cache_dir.strip():
            raise ValueError("grid crop size and cache directory must be configured")
        if not 0 <= self.data.grid_inner_margin_fraction < 0.25:
            raise ValueError("data.grid_inner_margin_fraction must be in [0, 0.25)")
        if not self.data.normalize_targets:
            raise ValueError("The tri-scale MIL experiment requires normalized targets")
        tile_counts = []
        cache_dirs = []
        for name, scale in self.scales:
            if scale.rows < 1 or scale.columns < 1:
                raise ValueError(f"{name} rows and columns must be positive")
            if not 0 <= scale.overlap_fraction < 1:
                raise ValueError(f"{name}.overlap_fraction must be in [0, 1)")
            if not scale.include_global_view:
                raise ValueError(f"{name} must include its cached global view")
            if not scale.cache_dir.strip():
                raise ValueError(f"{name}.cache_dir cannot be empty")
            tile_counts.append(scale.rows * scale.columns)
            cache_dirs.append(scale.cache_dir)
        if tile_counts != sorted(tile_counts) or len(set(tile_counts)) != 3:
            raise ValueError("context, regional, and local tile counts must strictly increase")
        if len(set(cache_dirs)) != 3:
            raise ValueError("all three feature-cache directories must be distinct")
        if self.features.representation != "cls_mean":
            raise ValueError("features.representation must be 'cls_mean'")
        if self.model.projection_dim < 1 or self.model.attention_hidden_dim < 1:
            raise ValueError("model dimensions must be positive")
        if self.model.head_hidden_dim < 1 or self.model.attention_temperature <= 0:
            raise ValueError("model head/attention settings are invalid")
        if not 0 <= self.model.attention_dropout < 1 or not 0 <= self.model.dropout < 1:
            raise ValueError("model dropout values must be in [0, 1)")
        training = self.training
        if training.epochs < 1 or training.eval_every < 1:
            raise ValueError("training epochs and eval_every must be positive")
        if training.batch_size < 1 or training.num_workers < 0:
            raise ValueError("training batch/worker settings are invalid")
        if training.learning_rate <= 0 or training.weight_decay < 0:
            raise ValueError("training optimizer settings are invalid")
        if not 0 <= training.warmup_fraction < 1:
            raise ValueError("training.warmup_fraction must be in [0, 1)")
        if not 0 <= training.minimum_learning_rate_ratio <= 1:
            raise ValueError("training.minimum_learning_rate_ratio must be in [0, 1]")
        if training.gradient_clip_norm <= 0:
            raise ValueError("training.gradient_clip_norm must be positive")
        if training.early_stopping_patience < 1 or training.early_stopping_min_delta < 0:
            raise ValueError("early stopping settings are invalid")
        if self.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("runtime.device must be auto, cpu, cuda, or mps")
        if self.runtime.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("runtime.mixed_precision must be none, fp16, or bf16")
        if self.output.example_images < 0 or self.output.example_columns < 1:
            raise ValueError("output example settings are invalid")
        if self.output.attention_inspection_images < 1:
            raise ValueError("output.attention_inspection_images must be positive")
        names = (
            self.output.best_checkpoint_name,
            self.output.best_mae_checkpoint_name,
            self.output.last_checkpoint_name,
        )
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("output checkpoint names must be nonempty and distinct")

    def single_scale_config(self, scale: ScaleSettings) -> SingleScaleConfig:
        """Build the exact single-scale config shape used by cache identities."""
        return SingleScaleConfig(
            data=self.data,
            experiment=SingleExperimentSettings(name=self.experiment.name),
            tiles=TileSettings(
                rows=scale.rows,
                columns=scale.columns,
                overlap_fraction=scale.overlap_fraction,
                include_global_view=scale.include_global_view,
            ),
            features=SingleFeatureSettings(
                backbone=self.features.backbone,
                processor=self.features.processor,
                representation=self.features.representation,
                cache_dir=scale.cache_dir,
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
    known = {field.name for field in fields(settings_type)}
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
        "features",
        "context",
        "regional",
        "local",
        "model",
        "training",
        "runtime",
        "output",
    }
    unexpected = sorted(set(raw) - known)
    if unexpected:
        raise ValueError(f"Unknown configuration section(s): {', '.join(unexpected)}")
    required = ("data", "context", "regional", "local")
    if any(section not in raw for section in required):
        raise ValueError("[data], [context], [regional], and [local] sections are required")
    if "dataset_dir" not in raw["data"]:
        raise ValueError("[data].dataset_dir is required")
    config = Config(
        experiment=_settings(ExperimentSettings, raw.get("experiment", {})),
        data=_settings(DataSettings, raw["data"]),
        features=_settings(FeatureSettings, raw.get("features", {})),
        context=_settings(ScaleSettings, raw["context"]),
        regional=_settings(ScaleSettings, raw["regional"]),
        local=_settings(ScaleSettings, raw["local"]),
        model=_settings(ModelSettings, raw.get("model", {})),
        training=_settings(TrainingSettings, raw.get("training", {})),
        runtime=_settings(RuntimeSettings, raw.get("runtime", {})),
        output=_settings(OutputSettings, raw.get("output", {})),
    )
    config.validate()
    return config
