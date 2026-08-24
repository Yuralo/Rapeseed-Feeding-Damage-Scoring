"""Configuration for the clean-grid DINOv3 LoRA experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar
import tomllib


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "dinov3_grid_lora"


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
class AugmentationSettings:
    enabled: bool = True
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    color_jitter_strength: float = 0.05


@dataclass(frozen=True)
class ModelSettings:
    backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    train_final_norm: bool = True
    head_hidden_dim: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 30
    eval_every: int = 1
    batch_size: int = 8
    num_workers: int = 4
    gradient_accumulation_steps: int = 2
    head_learning_rate: float = 3e-4
    adapter_learning_rate: float = 1e-4
    head_weight_decay: float = 1e-4
    adapter_weight_decay: float = 0.0
    warmup_fraction: float = 0.1
    minimum_learning_rate_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4
    seed: int = 42


@dataclass(frozen=True)
class RuntimeSettings:
    device: str = "auto"
    deterministic: bool = True
    pin_memory: bool = True
    mixed_precision: str = "fp16"
    allow_tf32: bool = True
    profile_first_n_epochs: int = 1


@dataclass(frozen=True)
class OutputSettings:
    run_dir: str = "outputs/dinov3_grid_lora_clean_inset075"
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    save_plots: bool = True
    example_images: int = 12
    example_columns: int = 4
    grid_failure_log: str = "grid_failures.jsonl"


@dataclass(frozen=True)
class Config:
    data: DataSettings
    experiment: ExperimentSettings = ExperimentSettings()
    augmentation: AugmentationSettings = AugmentationSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    output: OutputSettings = OutputSettings()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        data = self.data
        if not data.dataset_dir.strip():
            raise ValueError("data.dataset_dir cannot be empty")
        if not data.image_extension.startswith("."):
            raise ValueError("data.image_extension must start with a dot")
        if not 0 < data.validation_fraction < 1:
            raise ValueError("data.validation_fraction must be between 0 and 1")
        if data.grid_crop_size < 1 or not data.grid_cache_dir.strip():
            raise ValueError("grid crop size and cache directory must be configured")
        if not 0 <= data.grid_inner_margin_fraction < 0.25:
            raise ValueError("data.grid_inner_margin_fraction must be in [0, 0.25)")
        if not data.normalize_targets:
            raise ValueError("The LoRA experiment requires normalized targets")
        probabilities = (
            self.augmentation.horizontal_flip_probability,
            self.augmentation.vertical_flip_probability,
        )
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("augmentation probabilities must be between 0 and 1")
        if not 0 <= self.augmentation.color_jitter_strength <= 1:
            raise ValueError("augmentation.color_jitter_strength must be between 0 and 1")
        model = self.model
        if model.lora_rank < 1 or model.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= model.lora_dropout < 1:
            raise ValueError("model.lora_dropout must be in [0, 1)")
        if not model.lora_target_modules or any(
            not str(name).strip() for name in model.lora_target_modules
        ):
            raise ValueError("model.lora_target_modules cannot be empty")
        if model.head_hidden_dim < 1 or not 0 <= model.dropout < 1:
            raise ValueError("model head settings are invalid")
        training = self.training
        if training.epochs < 1 or training.eval_every < 1:
            raise ValueError("training epochs and eval_every must be positive")
        if training.batch_size < 1 or training.num_workers < 0:
            raise ValueError("training batch/worker settings are invalid")
        if training.gradient_accumulation_steps < 1:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if training.head_learning_rate <= 0 or training.adapter_learning_rate <= 0:
            raise ValueError("training learning rates must be positive")
        if training.head_weight_decay < 0 or training.adapter_weight_decay < 0:
            raise ValueError("training weight decay cannot be negative")
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
        if self.runtime.profile_first_n_epochs < 0:
            raise ValueError("runtime.profile_first_n_epochs cannot be negative")
        if self.output.example_images < 0 or self.output.example_columns < 1:
            raise ValueError("output example settings are invalid")
        if not self.output.grid_failure_log.strip():
            raise ValueError("output.grid_failure_log cannot be empty")


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
    known = {"experiment", "data", "augmentation", "model", "training", "runtime", "output"}
    unexpected = sorted(set(raw) - known)
    if unexpected:
        raise ValueError(f"Unknown configuration section(s): {', '.join(unexpected)}")
    if "data" not in raw or "dataset_dir" not in raw["data"]:
        raise ValueError("[data].dataset_dir is required")
    config = Config(
        experiment=_settings(ExperimentSettings, raw.get("experiment", {})),
        data=_settings(DataSettings, raw["data"]),
        augmentation=_settings(AugmentationSettings, raw.get("augmentation", {})),
        model=_settings(ModelSettings, raw.get("model", {})),
        training=_settings(TrainingSettings, raw.get("training", {})),
        runtime=_settings(RuntimeSettings, raw.get("runtime", {})),
        output=_settings(OutputSettings, raw.get("output", {})),
    )
    config.validate()
    return config
