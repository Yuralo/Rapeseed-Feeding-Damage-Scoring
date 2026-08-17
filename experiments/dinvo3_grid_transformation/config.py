"""Configuration owned only by the DINOv3 regression experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar
import tomllib


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "dinov3_baseline"


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


@dataclass(frozen=True)
class ModelSettings:
    backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    freeze_backbone: bool = True
    head_hidden_dim: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 30
    eval_every: int = 6
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42


@dataclass(frozen=True)
class RuntimeSettings:
    device: str = "auto"
    deterministic: bool = True
    pin_memory: bool = True


@dataclass(frozen=True)
class OutputSettings:
    run_dir: str = "outputs/dinov3_baseline"
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
        if self.training.epochs < 1 or self.training.eval_every < 1:
            raise ValueError("training epochs and eval_every must be positive")
        if self.training.batch_size < 1 or self.training.num_workers < 0:
            raise ValueError("training batch/worker settings are invalid")
        if self.training.learning_rate <= 0 or self.training.weight_decay < 0:
            raise ValueError("training optimizer settings are invalid")
        if self.model.head_hidden_dim < 1 or not 0 <= self.model.dropout < 1:
            raise ValueError("model head settings are invalid")
        if self.output.example_images < 0 or self.output.example_columns < 1:
            raise ValueError("output example settings are invalid")
        if not self.output.grid_failure_log.strip():
            raise ValueError("output.grid_failure_log cannot be empty")
        if self.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("runtime.device must be auto, cpu, cuda, or mps")


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
    known = {"experiment", "data", "model", "training", "runtime", "output"}
    unexpected = sorted(set(raw) - known)
    if unexpected:
        raise ValueError(f"Unknown configuration section(s): {', '.join(unexpected)}")
    if "data" not in raw or "dataset_dir" not in raw["data"]:
        raise ValueError("[data].dataset_dir is required")
    config = Config(
        experiment=_settings(ExperimentSettings, raw.get("experiment", {})),
        data=_settings(DataSettings, raw["data"]),
        model=_settings(ModelSettings, raw.get("model", {})),
        training=_settings(TrainingSettings, raw.get("training", {})),
        runtime=_settings(RuntimeSettings, raw.get("runtime", {})),
        output=_settings(OutputSettings, raw.get("output", {})),
    )
    config.validate()
    return config
