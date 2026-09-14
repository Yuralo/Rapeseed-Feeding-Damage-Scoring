"""Configuration for hierarchical three-view weak-to-gold MIL."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from experiments.dinov3_grid_lora_patch_attention_sam_fusion.config import (
    SegmentationSettings,
)
from experiments.dinov3_grid_sam_adaptive_mil.config import AdaptiveCropSettings
from experiments.dinov3_grid_tiled_mil.config import RuntimeSettings


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "dinov3_hierarchical_three_view_mil"


@dataclass(frozen=True)
class DataSettings:
    manifest_dir: str = "outputs/dataset_manifests"
    pretrain_manifest: str = "pretrain.csv"
    finetune_manifest: str = "finetune.csv"
    validation_manifest: str = "validation.csv"
    test_manifest: str = "test.csv"
    absolute_path_column: str = "absolute_path"
    filename_column: str = "relative_path"
    target_column: str = "target"
    sample_weight_column: str = "sample_weight"
    cohort_column: str = "cohort_id"
    supervision_tier_column: str = "supervision_tier"
    group_column: str = "plot_group_id"
    verify_images: bool = True
    raw_source_folders: list[str] = field(
        default_factory=lambda: [
            "2025_09_15_Re4StRes_T1_DSV",
            "2025_09_12_RSFB_01_NPZi",
            "2025_09_19_RSFB_02_NPZi",
        ]
    )
    crop_size: int = 1400
    grid_inner_margin_fraction: float = 0.075
    cell_inner_margin_fraction: float = 0.02
    processed_cache_dir: str = "cache/hierarchical_three_view_processed"
    processed_jpeg_quality: int = 95
    sam_inference_max_side: int = 1400
    maximum_weak_failure_fraction: float = 0.05

    @property
    def grid_crop_size(self) -> int:
        """Compatibility alias used by the existing versioned SAM cache reader."""
        return self.crop_size


@dataclass(frozen=True)
class FeatureSettings:
    backbone: str = "outputs/dinov3_routed_source_adaptation/adapted_backbone"
    processor: str = "outputs/dinov3_routed_source_adaptation/adapted_backbone"
    representation: str = "cls_mean"
    cache_dir: str = "cache/dinov3_hierarchical_three_view_features_adapted_routed"
    mask_cache_dir: str = "cache/sam3_masks_hierarchical_three_view"
    storage_dtype: str = "float16"
    extraction_batch_size: int = 16
    overwrite: bool = False


@dataclass(frozen=True)
class ModelSettings:
    projection_dim: int = 128
    attention_hidden_dim: int = 64
    attention_dropout: float = 0.10
    attention_temperature: float = 1.0
    head_hidden_dim: int = 128
    dropout: float = 0.30


@dataclass(frozen=True)
class StageSettings:
    epochs: int
    learning_rate: float
    batch_size: int = 32
    num_workers: int = 4
    weight_decay: float = 0.001
    warmup_fraction: float = 0.10
    minimum_learning_rate_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.0001
    huber_delta: float = 1.0


@dataclass(frozen=True)
class TrainingSettings:
    seed: int = 42
    use_weak_pretraining: bool = True
    cohort_balanced_pretraining: bool = True
    pretraining: StageSettings = field(
        default_factory=lambda: StageSettings(epochs=40, learning_rate=0.001)
    )
    finetuning: StageSettings = field(
        default_factory=lambda: StageSettings(epochs=100, learning_rate=0.0003)
    )


@dataclass(frozen=True)
class OutputSettings:
    run_dir: str = "outputs/dinov3_hierarchical_three_view_weak_to_gold"
    pretrain_checkpoint_name: str = "pretrain_best_mse.pt"
    best_checkpoint_name: str = "best_mse.pt"
    best_mae_checkpoint_name: str = "best_mae.pt"
    last_checkpoint_name: str = "last.pt"
    feature_failure_log: str = "feature_failures.jsonl"
    usable_pretrain_manifest: str = "usable_pretrain.csv"
    save_plots: bool = True
    example_images: int = 12
    example_columns: int = 4
    attention_inspection_images: int = 6
    attention_arrays_name: str = "hierarchical_attention.npz"


@dataclass(frozen=True)
class Config:
    data: DataSettings
    experiment: ExperimentSettings = ExperimentSettings()
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    adaptive_crops: AdaptiveCropSettings = field(default_factory=AdaptiveCropSettings)
    features: FeatureSettings = FeatureSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = field(default_factory=TrainingSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    output: OutputSettings = OutputSettings()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def manifest_path(self, split: str) -> Path:
        names = {
            "pretrain": self.data.pretrain_manifest,
            "finetune": self.data.finetune_manifest,
            "validation": self.data.validation_manifest,
            "test": self.data.test_manifest,
        }
        if split not in names:
            raise ValueError(f"Unknown manifest split: {split}")
        return Path(self.data.manifest_dir) / names[split]

    def validate(self) -> None:
        if not self.data.manifest_dir.strip():
            raise ValueError("data.manifest_dir cannot be empty")
        if self.data.crop_size < 1:
            raise ValueError("data.crop_size must be positive")
        if not 0 <= self.data.grid_inner_margin_fraction < 0.25:
            raise ValueError("grid_inner_margin_fraction must be in [0, 0.25)")
        if not 0 <= self.data.cell_inner_margin_fraction < 0.25:
            raise ValueError("cell_inner_margin_fraction must be in [0, 0.25)")
        if not 0 <= self.data.maximum_weak_failure_fraction < 1:
            raise ValueError("maximum_weak_failure_fraction must be in [0, 1)")
        if not 1 <= self.data.processed_jpeg_quality <= 100:
            raise ValueError("processed_jpeg_quality must be in [1, 100]")
        if self.data.sam_inference_max_side < 512:
            raise ValueError("sam_inference_max_side must be at least 512")
        if self.features.representation != "cls_mean":
            raise ValueError("features.representation must be cls_mean")
        if self.features.storage_dtype not in {"float16", "float32"}:
            raise ValueError("features.storage_dtype must be float16 or float32")
        if self.features.extraction_batch_size < 1:
            raise ValueError("features.extraction_batch_size must be positive")
        if not self.segmentation.prompts:
            raise ValueError("segmentation.prompts cannot be empty")
        if not 0 <= self.segmentation.minimum_foreground_fraction:
            raise ValueError("minimum_foreground_fraction cannot be negative")
        if self.segmentation.maximum_foreground_fraction > 1:
            raise ValueError("maximum_foreground_fraction cannot exceed one")
        crops = self.adaptive_crops
        if crops.maximum_instances < 1 or crops.minimum_crop_size < 1:
            raise ValueError("adaptive crop counts/sizes must be positive")
        if crops.maximum_crop_size < crops.minimum_crop_size or crops.context_scale < 1:
            raise ValueError("adaptive crop bounds/context are invalid")
        if min(
            self.model.projection_dim,
            self.model.attention_hidden_dim,
            self.model.head_hidden_dim,
        ) < 1:
            raise ValueError("model dimensions must be positive")
        for name, stage in (
            ("pretraining", self.training.pretraining),
            ("finetuning", self.training.finetuning),
        ):
            if stage.epochs < 1 or stage.learning_rate <= 0 or stage.batch_size < 1:
                raise ValueError(f"{name} epochs/rate/batch settings are invalid")
            if stage.num_workers < 0 or stage.weight_decay < 0 or stage.huber_delta <= 0:
                raise ValueError(f"{name} worker/decay/Huber settings are invalid")
            if not 0 <= stage.warmup_fraction < 1:
                raise ValueError(f"{name}.warmup_fraction must be in [0, 1)")
            if not 0 <= stage.minimum_learning_rate_ratio <= 1:
                raise ValueError(f"{name}.minimum_learning_rate_ratio must be in [0, 1]")
            if stage.gradient_clip_norm <= 0 or stage.early_stopping_patience < 1:
                raise ValueError(f"{name} clipping/patience settings are invalid")
        if self.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("runtime.device is invalid")
        names = (
            self.output.pretrain_checkpoint_name,
            self.output.best_checkpoint_name,
            self.output.best_mae_checkpoint_name,
            self.output.last_checkpoint_name,
        )
        if any(not value.strip() for value in names) or len(set(names)) != len(names):
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
        "experiment", "data", "segmentation", "adaptive_crops", "features", "model",
        "training", "pretraining", "finetuning", "runtime", "output",
    }
    unexpected = sorted(set(raw) - known)
    if unexpected:
        raise ValueError(f"Unknown configuration section(s): {', '.join(unexpected)}")
    if "data" not in raw:
        raise ValueError("[data] is required")
    training_values = raw.get("training", {})
    training = TrainingSettings(
        seed=int(training_values.get("seed", 42)),
        use_weak_pretraining=bool(training_values.get("use_weak_pretraining", True)),
        cohort_balanced_pretraining=bool(
            training_values.get("cohort_balanced_pretraining", True)
        ),
        pretraining=_settings(
            StageSettings, raw.get("pretraining", {"epochs": 40, "learning_rate": 0.001})
        ),
        finetuning=_settings(
            StageSettings, raw.get("finetuning", {"epochs": 100, "learning_rate": 0.0003})
        ),
    )
    config = Config(
        experiment=_settings(ExperimentSettings, raw.get("experiment", {})),
        data=_settings(DataSettings, raw["data"]),
        segmentation=_settings(SegmentationSettings, raw.get("segmentation", {})),
        adaptive_crops=_settings(AdaptiveCropSettings, raw.get("adaptive_crops", {})),
        features=_settings(FeatureSettings, raw.get("features", {})),
        model=_settings(ModelSettings, raw.get("model", {})),
        training=training,
        runtime=_settings(RuntimeSettings, raw.get("runtime", {})),
        output=_settings(OutputSettings, raw.get("output", {})),
    )
    config.validate()
    return config
