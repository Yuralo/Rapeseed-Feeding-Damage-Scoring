"""Configuration for multi-cohort 3x3+4x4 frozen-feature MIL."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from experiments.dinov3_grid_multiscale_tiled_mil.config import (
    ScaleSettings,
)
from experiments.dinov3_grid_tiled_mil.config import (
    Config as SingleScaleConfig,
)
from experiments.dinov3_grid_tiled_mil.config import (
    DataSettings as SingleDataSettings,
)
from experiments.dinov3_grid_tiled_mil.config import (
    ExperimentSettings as SingleExperimentSettings,
)
from experiments.dinov3_grid_tiled_mil.config import (
    FeatureSettings as SingleFeatureSettings,
)
from experiments.dinov3_grid_tiled_mil.config import (
    ModelSettings,
    RuntimeSettings,
    TileSettings,
)
from experiments.dinov3_grid_tiled_mil.config import (
    OutputSettings as SingleOutputSettings,
)
from experiments.dinov3_grid_tiled_mil.config import (
    TrainingSettings as SingleTrainingSettings,
)


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "dinov3_grid_multicohort_multiscale_mil"


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
    verify_images: bool = True
    grid_crop_size: int = 1400
    grid_cache_dir: str = "cache/grid_crops_multicohort_1400_inset075"
    grid_inner_margin_fraction: float = 0.075
    normalize_targets: bool = True


@dataclass(frozen=True)
class FeatureSettings:
    backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    representation: str = "cls_mean"
    storage_dtype: str = "float16"
    extraction_batch_size: int = 16
    overwrite: bool = False


@dataclass(frozen=True)
class StageSettings:
    epochs: int
    learning_rate: float
    batch_size: int = 32
    num_workers: int = 4
    weight_decay: float = 0.001
    warmup_fraction: float = 0.1
    minimum_learning_rate_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.0001


@dataclass(frozen=True)
class TrainingSettings:
    seed: int = 42
    pretraining: StageSettings = field(
        default_factory=lambda: StageSettings(
            epochs=40,
            learning_rate=0.001,
            early_stopping_patience=10,
        )
    )
    finetuning: StageSettings = field(
        default_factory=lambda: StageSettings(
            epochs=80,
            learning_rate=0.0003,
            early_stopping_patience=12,
        )
    )


@dataclass(frozen=True)
class OutputSettings:
    run_dir: str = "outputs/dinov3_grid_multicohort_multiscale_mil"
    pretrain_checkpoint_name: str = "pretrain_best_mse.pt"
    best_checkpoint_name: str = "best_mse.pt"
    best_mae_checkpoint_name: str = "best_mae.pt"
    last_checkpoint_name: str = "last.pt"
    save_plots: bool = True
    example_images: int = 12
    example_columns: int = 4
    attention_inspection_images: int = 6
    attention_arrays_name: str = "multiscale_tile_attention.npz"
    grid_failure_log: str = "grid_failures.jsonl"
    feature_failure_log: str = "feature_failures.jsonl"


@dataclass(frozen=True)
class Config:
    data: DataSettings
    coarse: ScaleSettings
    fine: ScaleSettings
    experiment: ExperimentSettings = ExperimentSettings()
    features: FeatureSettings = FeatureSettings()
    model: ModelSettings = field(
        default_factory=lambda: ModelSettings(
            projection_dim=128,
            attention_hidden_dim=64,
            attention_dropout=0.1,
            attention_temperature=1.0,
            head_hidden_dim=128,
            dropout=0.35,
        )
    )
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
        try:
            return Path(self.data.manifest_dir) / names[split]
        except KeyError as error:
            raise ValueError(f"Unknown manifest split: {split}") from error

    def validate(self) -> None:
        data = self.data
        if not data.manifest_dir.strip():
            raise ValueError("data.manifest_dir cannot be empty")
        if not all(
            value.strip()
            for value in (
                data.pretrain_manifest,
                data.finetune_manifest,
                data.validation_manifest,
                data.test_manifest,
            )
        ):
            raise ValueError("all manifest filenames must be configured")
        if not data.normalize_targets:
            raise ValueError("multicohort training requires normalized targets")
        if data.grid_crop_size < 1 or not data.grid_cache_dir.strip():
            raise ValueError("grid cache settings are invalid")
        if not 0 <= data.grid_inner_margin_fraction < 0.25:
            raise ValueError("grid_inner_margin_fraction must be in [0, 0.25)")
        for name, scale in (("coarse", self.coarse), ("fine", self.fine)):
            if scale.rows < 1 or scale.columns < 1 or not scale.cache_dir.strip():
                raise ValueError(f"{name} scale settings are invalid")
            if not 0 <= scale.overlap_fraction < 1 or not scale.include_global_view:
                raise ValueError(f"{name} overlap/global-view settings are invalid")
        if self.coarse.rows * self.coarse.columns >= self.fine.rows * self.fine.columns:
            raise ValueError("coarse must contain fewer tiles than fine")
        if self.coarse.cache_dir == self.fine.cache_dir:
            raise ValueError("coarse and fine cache directories must differ")
        if self.features.representation != "cls_mean":
            raise ValueError("features.representation must be cls_mean")
        if self.features.storage_dtype not in {"float16", "float32"}:
            raise ValueError("features.storage_dtype must be float16 or float32")
        if self.features.extraction_batch_size < 1:
            raise ValueError("features.extraction_batch_size must be positive")
        if (
            min(
                self.model.projection_dim,
                self.model.attention_hidden_dim,
                self.model.head_hidden_dim,
            )
            < 1
        ):
            raise ValueError("model dimensions must be positive")
        for name, stage in (
            ("pretraining", self.training.pretraining),
            ("finetuning", self.training.finetuning),
        ):
            if stage.epochs < 1 or stage.learning_rate <= 0 or stage.batch_size < 1:
                raise ValueError(f"{name} epochs/rate/batch settings are invalid")
            if stage.num_workers < 0 or stage.weight_decay < 0:
                raise ValueError(f"{name} worker/decay settings are invalid")
            if not 0 <= stage.warmup_fraction < 1:
                raise ValueError(f"{name}.warmup_fraction must be in [0, 1)")
            if not 0 <= stage.minimum_learning_rate_ratio <= 1:
                raise ValueError(f"{name}.minimum_learning_rate_ratio must be in [0, 1]")
            if stage.gradient_clip_norm <= 0 or stage.early_stopping_patience < 1:
                raise ValueError(f"{name} clipping/patience settings are invalid")
        if self.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("runtime.device must be auto, cpu, cuda, or mps")
        if self.runtime.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("runtime.mixed_precision must be none, fp16, or bf16")
        checkpoints = (
            self.output.pretrain_checkpoint_name,
            self.output.best_checkpoint_name,
            self.output.best_mae_checkpoint_name,
            self.output.last_checkpoint_name,
        )
        if len(set(checkpoints)) != len(checkpoints) or any(not name for name in checkpoints):
            raise ValueError("checkpoint filenames must be nonempty and distinct")

    def single_scale_config(self, scale: ScaleSettings) -> SingleScaleConfig:
        stage = self.training.pretraining
        return SingleScaleConfig(
            data=SingleDataSettings(
                dataset_dir=".",
                scores_file="unused.csv",
                filename_column=self.data.filename_column,
                target_column=self.data.target_column,
                validation_fraction=0.2,
                split_seed=self.training.seed,
                verify_images=self.data.verify_images,
                grid_crop_size=self.data.grid_crop_size,
                grid_cache_dir=self.data.grid_cache_dir,
                grid_inner_margin_fraction=self.data.grid_inner_margin_fraction,
                normalize_targets=True,
            ),
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
                storage_dtype=self.features.storage_dtype,
                extraction_batch_size=self.features.extraction_batch_size,
                overwrite=self.features.overwrite,
            ),
            model=self.model,
            training=SingleTrainingSettings(
                epochs=stage.epochs,
                eval_every=1,
                batch_size=stage.batch_size,
                num_workers=stage.num_workers,
                learning_rate=stage.learning_rate,
                weight_decay=stage.weight_decay,
                warmup_fraction=stage.warmup_fraction,
                minimum_learning_rate_ratio=stage.minimum_learning_rate_ratio,
                gradient_clip_norm=stage.gradient_clip_norm,
                early_stopping_patience=stage.early_stopping_patience,
                early_stopping_min_delta=stage.early_stopping_min_delta,
                seed=self.training.seed,
            ),
            runtime=self.runtime,
            output=SingleOutputSettings(
                run_dir=self.output.run_dir,
                save_plots=self.output.save_plots,
                example_images=self.output.example_images,
                example_columns=self.output.example_columns,
                attention_inspection_images=self.output.attention_inspection_images,
                grid_failure_log=self.output.grid_failure_log,
                feature_failure_log=self.output.feature_failure_log,
            ),
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
        "features",
        "coarse",
        "fine",
        "model",
        "training",
        "pretraining",
        "finetuning",
        "runtime",
        "output",
    }
    unexpected = sorted(set(raw) - known)
    if unexpected:
        raise ValueError(f"Unknown configuration section(s): {', '.join(unexpected)}")
    if not {"data", "coarse", "fine"}.issubset(raw):
        raise ValueError("[data], [coarse], and [fine] are required")
    training_values = raw.get("training", {})
    training = TrainingSettings(
        seed=int(training_values.get("seed", 42)),
        pretraining=_settings(
            StageSettings, raw.get("pretraining", {"epochs": 40, "learning_rate": 0.001})
        ),
        finetuning=_settings(
            StageSettings, raw.get("finetuning", {"epochs": 80, "learning_rate": 0.0003})
        ),
    )
    config = Config(
        experiment=_settings(ExperimentSettings, raw.get("experiment", {})),
        data=_settings(DataSettings, raw["data"]),
        features=_settings(FeatureSettings, raw.get("features", {})),
        coarse=_settings(ScaleSettings, raw["coarse"]),
        fine=_settings(ScaleSettings, raw["fine"]),
        model=_settings(ModelSettings, raw.get("model", {})),
        training=training,
        runtime=_settings(RuntimeSettings, raw.get("runtime", {})),
        output=_settings(OutputSettings, raw.get("output", {})),
    )
    config.validate()
    return config
