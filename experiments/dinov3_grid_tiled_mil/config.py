"""Configuration for frozen-feature global-plus-tiled DINOv3 MIL."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "dinov3_grid_tiled_mil"


@dataclass(frozen=True)
class DataSettings:
    dataset_dir: str
    scores_file: str = "RSFB-Phenotyping_training_set_scores.csv"
    filename_column: str = "Filename"
    target_column: str = "mean_score"
    image_extension: str = ".jpg"
    validation_fraction: float = 0.33
    split_seed: int = 42
    verify_images: bool = True
    grid_crop_size: int = 1400
    grid_cache_dir: str = "cache/grid_crops_1400_inset075"
    grid_inner_margin_fraction: float = 0.075
    normalize_targets: bool = True


@dataclass(frozen=True)
class TileSettings:
    rows: int = 3
    columns: int = 3
    overlap_fraction: float = 0.25
    include_global_view: bool = True


@dataclass(frozen=True)
class FeatureSettings:
    backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    representation: str = "cls_mean"
    cache_dir: str = "cache/dinov3_grid_tiled_mil_features"
    storage_dtype: str = "float16"
    extraction_batch_size: int = 16
    overwrite: bool = False


@dataclass(frozen=True)
class ModelSettings:
    projection_dim: int = 256
    attention_hidden_dim: int = 128
    attention_dropout: float = 0.1
    attention_temperature: float = 1.0
    head_hidden_dim: int = 256
    dropout: float = 0.2


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 100
    eval_every: int = 1
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    warmup_fraction: float = 0.1
    minimum_learning_rate_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.0001
    seed: int = 42


@dataclass(frozen=True)
class RuntimeSettings:
    device: str = "auto"
    deterministic: bool = True
    pin_memory: bool = True
    mixed_precision: str = "fp16"
    allow_tf32: bool = True


@dataclass(frozen=True)
class OutputSettings:
    run_dir: str = "outputs/dinov3_grid_tiled_mil_clean_inset075"
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    save_plots: bool = True
    example_images: int = 12
    example_columns: int = 4
    attention_inspection_images: int = 6
    attention_arrays_name: str = "tile_attention.npz"
    grid_failure_log: str = "grid_failures.jsonl"
    feature_failure_log: str = "feature_failures.jsonl"


@dataclass(frozen=True)
class Config:
    data: DataSettings
    experiment: ExperimentSettings = ExperimentSettings()
    tiles: TileSettings = TileSettings()
    features: FeatureSettings = FeatureSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    output: OutputSettings = OutputSettings()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
            raise ValueError("The tiled MIL experiment requires normalized targets")
        if self.tiles.rows < 1 or self.tiles.columns < 1:
            raise ValueError("tiles.rows and tiles.columns must be positive")
        if not 0 <= self.tiles.overlap_fraction < 1:
            raise ValueError("tiles.overlap_fraction must be in [0, 1)")
        if not self.tiles.include_global_view:
            raise ValueError("This experiment requires the global view")
        if self.features.representation != "cls_mean":
            raise ValueError("features.representation must be 'cls_mean'")
        if self.features.storage_dtype not in {"float16", "float32"}:
            raise ValueError("features.storage_dtype must be float16 or float32")
        if self.features.extraction_batch_size < 1 or not self.features.cache_dir.strip():
            raise ValueError("feature cache settings are invalid")
        if self.model.projection_dim < 1 or self.model.attention_hidden_dim < 1:
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.model.attention_dropout < 1 or not 0 <= self.model.dropout < 1:
            raise ValueError("model dropout values must be in [0, 1)")
        if self.model.attention_temperature <= 0 or self.model.head_hidden_dim < 1:
            raise ValueError("model attention/head settings are invalid")
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
    known = {"experiment", "data", "tiles", "features", "model", "training", "runtime", "output"}
    unexpected = sorted(set(raw) - known)
    if unexpected:
        raise ValueError(f"Unknown configuration section(s): {', '.join(unexpected)}")
    if "data" not in raw or "dataset_dir" not in raw["data"]:
        raise ValueError("[data].dataset_dir is required")
    config = Config(
        experiment=_settings(ExperimentSettings, raw.get("experiment", {})),
        data=_settings(DataSettings, raw["data"]),
        tiles=_settings(TileSettings, raw.get("tiles", {})),
        features=_settings(FeatureSettings, raw.get("features", {})),
        model=_settings(ModelSettings, raw.get("model", {})),
        training=_settings(TrainingSettings, raw.get("training", {})),
        runtime=_settings(RuntimeSettings, raw.get("runtime", {})),
        output=_settings(OutputSettings, raw.get("output", {})),
    )
    config.validate()
    return config
