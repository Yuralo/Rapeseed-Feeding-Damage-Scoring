"""Configuration for three-representation SAM fusion with DINOv3 LoRA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar
import tomllib


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "dinov3_grid_lora_patch_attention_sam_fusion"


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
class SegmentationSettings:
    model_name: str = "facebook/sam3"
    prompts: tuple[str, ...] = ("green leaf",)
    score_threshold: float = 0.25
    mask_threshold: float = 0.5
    mask_cache_dir: str = "cache/sam3_masks_grid1400_inset075"
    device: str = "auto"
    background_value: int = 255
    minimum_foreground_fraction: float = 0.0001
    maximum_foreground_fraction: float = 0.6


@dataclass(frozen=True)
class ModelSettings:
    backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    train_final_norm: bool = True
    attention_hidden_dim: int = 128
    attention_dropout: float = 0.1
    attention_temperature: float = 1.0
    mask_input_size: int = 56
    mask_embedding_dim: int = 128
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.2
    head_hidden_dim: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 30
    eval_every: int = 1
    batch_size: int = 4
    num_workers: int = 4
    gradient_accumulation_steps: int = 4
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
    run_dir: str = "outputs/dinov3_grid_lora_patch_attention_sam_fusion_clean_inset075"
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    save_plots: bool = True
    example_images: int = 12
    example_columns: int = 4
    attention_inspection_images: int = 6
    attention_top_fraction: float = 0.1
    attention_ratio_min: float = 0.5
    attention_ratio_max: float = 2.0
    attention_arrays_name: str = "sam_fusion_attention.npz"
    grid_failure_log: str = "grid_failures.jsonl"
    sam_failure_log: str = "sam_failures.jsonl"
    sam_inspection_images: int = 24


@dataclass(frozen=True)
class Config:
    data: DataSettings
    experiment: ExperimentSettings = ExperimentSettings()
    augmentation: AugmentationSettings = AugmentationSettings()
    segmentation: SegmentationSettings = SegmentationSettings()
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
            raise ValueError("The LoRA patch-attention experiment requires normalized targets")
        probabilities = (
            self.augmentation.horizontal_flip_probability,
            self.augmentation.vertical_flip_probability,
        )
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("augmentation probabilities must be between 0 and 1")
        if not 0 <= self.augmentation.color_jitter_strength <= 1:
            raise ValueError("augmentation.color_jitter_strength must be between 0 and 1")
        segmentation = self.segmentation
        if not segmentation.model_name.strip():
            raise ValueError("segmentation.model_name cannot be empty")
        if not segmentation.prompts or any(
            not str(prompt).strip() for prompt in segmentation.prompts
        ):
            raise ValueError("segmentation.prompts cannot be empty")
        if not 0 <= segmentation.score_threshold <= 1:
            raise ValueError("segmentation.score_threshold must be in [0, 1]")
        if not 0 <= segmentation.mask_threshold <= 1:
            raise ValueError("segmentation.mask_threshold must be in [0, 1]")
        if not segmentation.mask_cache_dir.strip():
            raise ValueError("segmentation.mask_cache_dir cannot be empty")
        if segmentation.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("segmentation.device must be auto, cpu, cuda, or mps")
        if not 0 <= segmentation.background_value <= 255:
            raise ValueError("segmentation.background_value must be in [0, 255]")
        if not (
            0 <= segmentation.minimum_foreground_fraction
            < segmentation.maximum_foreground_fraction
            <= 1
        ):
            raise ValueError("segmentation foreground fraction bounds are invalid")
        model = self.model
        if model.lora_rank < 1 or model.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= model.lora_dropout < 1:
            raise ValueError("model.lora_dropout must be in [0, 1)")
        if not model.lora_target_modules or any(
            not str(name).strip() for name in model.lora_target_modules
        ):
            raise ValueError("model.lora_target_modules cannot be empty")
        if model.attention_hidden_dim < 1:
            raise ValueError("model.attention_hidden_dim must be positive")
        if not 0 <= model.attention_dropout < 1:
            raise ValueError("model.attention_dropout must be in [0, 1)")
        if model.attention_temperature <= 0:
            raise ValueError("model.attention_temperature must be positive")
        if model.mask_input_size < 8 or model.mask_embedding_dim < 1:
            raise ValueError("model mask encoder settings are invalid")
        if model.fusion_hidden_dim < 1 or not 0 <= model.fusion_dropout < 1:
            raise ValueError("model fusion settings are invalid")
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
        if self.output.attention_inspection_images < 1:
            raise ValueError("output.attention_inspection_images must be positive")
        if not 0 < self.output.attention_top_fraction <= 1:
            raise ValueError("output.attention_top_fraction must be in (0, 1]")
        if not 0 <= self.output.attention_ratio_min < 1 < self.output.attention_ratio_max:
            raise ValueError("attention ratio range must contain the uniform value 1")
        if not self.output.attention_arrays_name.strip():
            raise ValueError("output.attention_arrays_name cannot be empty")
        if not self.output.grid_failure_log.strip():
            raise ValueError("output.grid_failure_log cannot be empty")
        if not self.output.sam_failure_log.strip():
            raise ValueError("output.sam_failure_log cannot be empty")
        if self.output.sam_inspection_images < 1:
            raise ValueError("output.sam_inspection_images must be positive")


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
        "augmentation",
        "segmentation",
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
        augmentation=_settings(AugmentationSettings, raw.get("augmentation", {})),
        segmentation=_settings(SegmentationSettings, raw.get("segmentation", {})),
        model=_settings(ModelSettings, raw.get("model", {})),
        training=_settings(TrainingSettings, raw.get("training", {})),
        runtime=_settings(RuntimeSettings, raw.get("runtime", {})),
        output=_settings(OutputSettings, raw.get("output", {})),
    )
    config.validate()
    return config

