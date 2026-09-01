"""Configuration for mixed raw/grid DINOv3 LoRA domain adaptation."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "dinov3_mixed_domain_adaptation"


@dataclass(frozen=True)
class DataSettings:
    manifest: str = "outputs/dataset_manifests/adaptation.csv"
    prepared_manifest: str = "outputs/dinov3_mixed_domain_adaptation/prepared_manifest.csv"
    absolute_path_column: str = "absolute_path"
    relative_path_column: str = "relative_path"
    filename_column: str = "file_name"
    id_column: str = "image_id"
    cohort_column: str = "cohort_id"
    timestamp_pattern: str = r"^\d{8}_\d{6}\.(?:jpg|jpeg)$"
    raw_pattern: str = r"^IMG_.*\.(?:jpg|jpeg)$"
    verify_images: bool = True
    grid_crop_size: int = 1400
    grid_cache_dir: str = "cache/adaptation_grid_crops_1400_inset075"
    grid_inner_margin_fraction: float = 0.075
    maximum_excluded_fraction: float = 0.05
    validation_fraction: float = 0.1
    split_seed: int = 42


@dataclass(frozen=True)
class CropSettings:
    minimum_scale: float = 0.35
    maximum_scale: float = 0.75
    label_overlap_limit: float = 0.02
    candidate_attempts: int = 24
    preview_crops_per_image: int = 4


@dataclass(frozen=True)
class AugmentationSettings:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    color_jitter_strength: float = 0.15
    grayscale_probability: float = 0.05
    blur_probability: float = 0.15
    blur_max_radius: float = 1.5


@dataclass(frozen=True)
class ModelSettings:
    backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    train_final_norm: bool = True


@dataclass(frozen=True)
class ObjectiveSettings:
    cross_view_weight: float = 1.0
    same_view_anchor_weight: float = 0.25


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 20
    eval_every: int = 1
    batch_size: int = 8
    num_workers: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_fraction: float = 0.1
    minimum_learning_rate_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
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
    run_dir: str = "outputs/dinov3_mixed_domain_adaptation"
    source_inspection_dir: str = "source_inspection"
    samples_per_source: int = 8
    inspection_dir: str = "preprocessing_inspection"
    samples_per_mode: int = 12
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    export_dir: str = "adapted_backbone"
    failure_log: str = "preprocessing_failures.jsonl"
    save_plots: bool = True


@dataclass(frozen=True)
class Config:
    experiment: ExperimentSettings = ExperimentSettings()
    data: DataSettings = DataSettings()
    crops: CropSettings = CropSettings()
    augmentation: AugmentationSettings = AugmentationSettings()
    model: ModelSettings = ModelSettings()
    objective: ObjectiveSettings = ObjectiveSettings()
    training: TrainingSettings = TrainingSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    output: OutputSettings = OutputSettings()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        data = self.data
        required_strings = (
            data.manifest,
            data.prepared_manifest,
            data.absolute_path_column,
            data.relative_path_column,
            data.filename_column,
            data.id_column,
            data.cohort_column,
            data.timestamp_pattern,
            data.raw_pattern,
            data.grid_cache_dir,
        )
        if any(not value.strip() for value in required_strings):
            raise ValueError("Data paths, columns, and filename patterns cannot be empty")
        if data.grid_crop_size < 1:
            raise ValueError("data.grid_crop_size must be positive")
        if not 0 <= data.grid_inner_margin_fraction < 0.25:
            raise ValueError("data.grid_inner_margin_fraction must be in [0, 0.25)")
        if not 0 <= data.maximum_excluded_fraction < 1:
            raise ValueError("data.maximum_excluded_fraction must be in [0, 1)")
        if not 0 < data.validation_fraction < 0.5:
            raise ValueError("data.validation_fraction must be between 0 and 0.5")
        crops = self.crops
        if not 0 < crops.minimum_scale <= crops.maximum_scale <= 1:
            raise ValueError("Crop scales must satisfy 0 < minimum <= maximum <= 1")
        if not 0 <= crops.label_overlap_limit <= 1:
            raise ValueError("crops.label_overlap_limit must be in [0, 1]")
        if crops.candidate_attempts < 1 or crops.preview_crops_per_image < 1:
            raise ValueError("Crop attempt and preview counts must be positive")
        augmentation = self.augmentation
        probabilities = (
            augmentation.horizontal_flip_probability,
            augmentation.vertical_flip_probability,
            augmentation.grayscale_probability,
            augmentation.blur_probability,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("Augmentation probabilities must be in [0, 1]")
        if not 0 <= augmentation.color_jitter_strength <= 1:
            raise ValueError("augmentation.color_jitter_strength must be in [0, 1]")
        if augmentation.blur_max_radius < 0:
            raise ValueError("augmentation.blur_max_radius cannot be negative")
        model = self.model
        if model.lora_rank < 1 or model.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= model.lora_dropout < 1:
            raise ValueError("model.lora_dropout must be in [0, 1)")
        if not model.lora_target_modules or any(
            not str(value).strip() for value in model.lora_target_modules
        ):
            raise ValueError("model.lora_target_modules cannot be empty")
        if self.objective.cross_view_weight <= 0:
            raise ValueError("objective.cross_view_weight must be positive")
        if self.objective.same_view_anchor_weight < 0:
            raise ValueError("objective.same_view_anchor_weight cannot be negative")
        training = self.training
        if training.epochs < 1 or training.eval_every < 1:
            raise ValueError("Training epochs and eval_every must be positive")
        if training.batch_size < 1 or training.num_workers < 0:
            raise ValueError("Training batch/worker settings are invalid")
        if training.gradient_accumulation_steps < 1:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if training.learning_rate <= 0 or training.weight_decay < 0:
            raise ValueError("Training optimizer settings are invalid")
        if not 0 <= training.warmup_fraction < 1:
            raise ValueError("training.warmup_fraction must be in [0, 1)")
        if not 0 <= training.minimum_learning_rate_ratio <= 1:
            raise ValueError("training.minimum_learning_rate_ratio must be in [0, 1]")
        if training.gradient_clip_norm <= 0:
            raise ValueError("training.gradient_clip_norm must be positive")
        if training.early_stopping_patience < 1 or training.early_stopping_min_delta < 0:
            raise ValueError("Early stopping settings are invalid")
        if self.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("runtime.device must be auto, cpu, cuda, or mps")
        if self.runtime.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("runtime.mixed_precision must be none, fp16, or bf16")
        if not self.output.source_inspection_dir.strip() or not self.output.inspection_dir.strip():
            raise ValueError("Output inspection directories cannot be empty")
        if self.output.samples_per_source < 1 or self.output.samples_per_mode < 1:
            raise ValueError("Output inspection sample counts must be positive")


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
    sections = {
        "experiment": ExperimentSettings,
        "data": DataSettings,
        "crops": CropSettings,
        "augmentation": AugmentationSettings,
        "model": ModelSettings,
        "objective": ObjectiveSettings,
        "training": TrainingSettings,
        "runtime": RuntimeSettings,
        "output": OutputSettings,
    }
    unexpected = sorted(set(raw) - set(sections))
    if unexpected:
        raise ValueError(f"Unknown configuration section(s): {', '.join(unexpected)}")
    config = Config(
        **{
            name: _settings(settings_type, raw.get(name, {}))
            for name, settings_type in sections.items()
        }
    )
    config.validate()
    return config
