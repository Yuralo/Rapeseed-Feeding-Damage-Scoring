"""Configuration for source-routed DINOv3 domain adaptation."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from experiments.dinov3_mixed_domain_adaptation.config import (
    AugmentationSettings,
    ModelSettings,
    ObjectiveSettings,
    RuntimeSettings,
    TileSettings,
    TrainingSettings,
)


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "dinov3_routed_source_adaptation"


@dataclass(frozen=True)
class DataSettings:
    manifest: str = "outputs/dataset_manifests/adaptation.csv"
    prepared_manifest: str = "outputs/dinov3_routed_source_adaptation/prepared_manifest.csv"
    absolute_path_column: str = "absolute_path"
    relative_path_column: str = "relative_path"
    filename_column: str = "file_name"
    id_column: str = "image_id"
    cohort_column: str = "cohort_id"
    maximum_excluded_fraction: float = 0.05
    validation_fraction: float = 0.10
    split_seed: int = 42


@dataclass(frozen=True)
class PreprocessingSettings:
    """Route only the three audited bad sources around grid detection."""

    raw_source_folders: tuple[str, ...] = (
        "2025_09_15_Re4StRes_T1_DSV",
        "2025_09_12_RSFB_01_NPZi",
        "2025_09_19_RSFB_02_NPZi",
    )
    crop_size: int = 1400
    grid_inner_margin_fraction: float = 0.075
    crop_cache_dir: str = "outputs/dinov3_routed_source_adaptation/grid_crops_1400_inset075"
    crop_jpeg_quality: int = 95
    overwrite_cached_crops: bool = False


@dataclass(frozen=True)
class OutputSettings:
    run_dir: str = "outputs/dinov3_routed_source_adaptation"
    inspection_dir: str = "routed_input_inspection"
    samples_per_source: int = 3
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    export_dir: str = "adapted_backbone"
    failure_log: str = "input_exclusions.jsonl"
    save_plots: bool = True


@dataclass(frozen=True)
class Config:
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    data: DataSettings = field(default_factory=DataSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    tiles: TileSettings = field(default_factory=TileSettings)
    augmentation: AugmentationSettings = field(default_factory=AugmentationSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    objective: ObjectiveSettings = field(default_factory=ObjectiveSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

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
        )
        if any(not value.strip() for value in required_strings):
            raise ValueError("Data paths and columns cannot be empty")
        if not 0 <= data.maximum_excluded_fraction < 1:
            raise ValueError("data.maximum_excluded_fraction must be in [0, 1)")
        if not 0 < data.validation_fraction < 0.5:
            raise ValueError("data.validation_fraction must be between 0 and 0.5")

        preprocessing = self.preprocessing
        if not preprocessing.raw_source_folders:
            raise ValueError("preprocessing.raw_source_folders cannot be empty")
        if len(set(preprocessing.raw_source_folders)) != len(preprocessing.raw_source_folders):
            raise ValueError("preprocessing.raw_source_folders cannot contain duplicates")
        if any(not value.strip() for value in preprocessing.raw_source_folders):
            raise ValueError("Raw source-folder names cannot be empty")
        if preprocessing.crop_size < 64:
            raise ValueError("preprocessing.crop_size must be at least 64")
        if not 0 <= preprocessing.grid_inner_margin_fraction < 0.25:
            raise ValueError("preprocessing.grid_inner_margin_fraction must be in [0, 0.25)")
        if abs(preprocessing.grid_inner_margin_fraction - 0.075) > 1e-12:
            raise ValueError(
                "This controlled experiment requires "
                "preprocessing.grid_inner_margin_fraction = 0.075"
            )
        if not preprocessing.crop_cache_dir.strip():
            raise ValueError("preprocessing.crop_cache_dir cannot be empty")
        if not 1 <= preprocessing.crop_jpeg_quality <= 100:
            raise ValueError("preprocessing.crop_jpeg_quality must be in [1, 100]")

        tiles = self.tiles
        if not tiles.grid_sizes or any(size < 2 for size in tiles.grid_sizes):
            raise ValueError("tiles.grid_sizes must contain integers of at least 2")
        if len(set(tiles.grid_sizes)) != len(tiles.grid_sizes):
            raise ValueError("tiles.grid_sizes cannot contain duplicates")
        if not 0 <= tiles.overlap_fraction < 1:
            raise ValueError("tiles.overlap_fraction must be in [0, 1)")
        if not 0 <= tiles.plant_biased_probability <= 1:
            raise ValueError("tiles.plant_biased_probability must be in [0, 1]")
        if tiles.vegetation_score_power <= 0:
            raise ValueError("tiles.vegetation_score_power must be positive")
        if not 0 <= tiles.label_overlap_limit <= 1:
            raise ValueError("tiles.label_overlap_limit must be in [0, 1]")
        if tiles.mask_analysis_max_side < 64 or tiles.preview_tiles_per_image < 1:
            raise ValueError("Tile mask size and preview count must be positive")

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
        if not self.output.inspection_dir.strip() or self.output.samples_per_source < 1:
            raise ValueError("Output inspection settings are invalid")


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
        "preprocessing": PreprocessingSettings,
        "tiles": TileSettings,
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
