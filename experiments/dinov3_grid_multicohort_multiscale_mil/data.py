"""Manifest-backed cached-feature datasets for two-stage multicohort training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.dinov3_grid_tiled_mil.data import EpochSampler, TargetScaler
from experiments.dinov3_grid_tiled_mil.features import (
    cache_identity,
    feature_cache_path,
    load_feature_record,
)
from rapeseed_damage.reproducibility import seed_worker

from .config import Config, StageSettings


def load_manifest(config: Config, split: str) -> pd.DataFrame:
    path = config.manifest_path(split)
    if not path.is_file():
        raise FileNotFoundError(
            f"{split} manifest not found: {path}. Run analysis.build_supervised_manifests first."
        )
    table = pd.read_csv(path)
    required = {
        config.data.absolute_path_column,
        config.data.filename_column,
        config.data.target_column,
        config.data.sample_weight_column,
        config.data.cohort_column,
        config.data.supervision_tier_column,
        "plot_group_id",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    if table.empty:
        raise ValueError(f"{split} manifest is empty: {path}")
    table = table.copy()
    table[config.data.target_column] = pd.to_numeric(
        table[config.data.target_column], errors="raise"
    )
    table[config.data.sample_weight_column] = pd.to_numeric(
        table[config.data.sample_weight_column], errors="raise"
    )
    targets = table[config.data.target_column].to_numpy(dtype=np.float64)
    weights = table[config.data.sample_weight_column].to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError(f"{split} manifest contains non-finite targets")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError(f"{split} manifest weights must be finite and positive")
    paths = table[config.data.absolute_path_column].astype(str)
    if paths.duplicated().any():
        raise ValueError(f"{split} manifest contains duplicate image paths")
    if config.data.verify_images:
        absent = [value for value in paths if not Path(value).is_file()]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} {split} image(s) are absent. First paths:\n" + "\n".join(absent[:5])
            )
    return table.reset_index(drop=True)


def validate_split_isolation(
    pretrain: pd.DataFrame,
    finetune: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: Config,
) -> None:
    path_column = config.data.absolute_path_column
    groups = {
        "pretrain": set(pretrain["plot_group_id"].astype(str)),
        "finetune": set(finetune["plot_group_id"].astype(str)),
        "validation": set(validation["plot_group_id"].astype(str)),
        "test": set(test["plot_group_id"].astype(str)),
    }
    holdout = groups["validation"] | groups["test"]
    if groups["validation"] & groups["test"]:
        raise ValueError("validation and test manifests share physical plot groups")
    if holdout & (groups["pretrain"] | groups["finetune"]):
        raise ValueError("training and holdout manifests share physical plot groups")
    paths = {
        name: set(table[path_column].astype(str))
        for name, table in (
            ("pretrain", pretrain),
            ("finetune", finetune),
            ("validation", validation),
            ("test", test),
        )
    }
    for left, right in (
        ("pretrain", "validation"),
        ("pretrain", "test"),
        ("finetune", "validation"),
        ("finetune", "test"),
        ("validation", "test"),
    ):
        if paths[left] & paths[right]:
            raise ValueError(f"{left} and {right} manifests share exact image paths")


def source_path(row: pd.Series, config: Config) -> Path:
    return Path(str(row[config.data.absolute_path_column]))


def feature_name(row: pd.Series, config: Config) -> str:
    return str(row[config.data.filename_column])


class MultiCohortFeatureDataset(Dataset):
    def __init__(self, table: pd.DataFrame, targets: np.ndarray, config: Config):
        if len(table) != len(targets):
            raise ValueError("Table and target lengths differ")
        self.table = table.reset_index(drop=True)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.config = config
        self.coarse_config = config.single_scale_config(config.coarse)
        self.fine_config = config.single_scale_config(config.fine)

    def __len__(self) -> int:
        return len(self.table)

    def _record(self, name: str, source: Path, scale_config):
        identity = cache_identity(scale_config, name, source)
        path = feature_cache_path(scale_config, name, source)
        if not path.is_file():
            raise FileNotFoundError(
                f"Frozen features missing for {name}: {path}. Run prepare_features first."
            )
        return path, load_feature_record(path, expected_identity=identity)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.table.iloc[index]
        name = feature_name(row, self.config)
        source = source_path(row, self.config)
        coarse_path, coarse = self._record(name, source, self.coarse_config)
        fine_path, fine = self._record(name, source, self.fine_config)
        expected_coarse = self.config.coarse.rows * self.config.coarse.columns
        expected_fine = self.config.fine.rows * self.config.fine.columns
        if coarse["tile_boxes"].shape[0] != expected_coarse:
            raise ValueError(f"Expected {expected_coarse} coarse tiles for {name}")
        if fine["tile_boxes"].shape[0] != expected_fine:
            raise ValueError(f"Expected {expected_fine} fine tiles for {name}")
        if coarse["features"].shape[1] != fine["features"].shape[1]:
            raise ValueError(f"Coarse/fine feature dimensions differ for {name}")
        return {
            "coarse_features": torch.from_numpy(coarse["features"]),
            "fine_features": torch.from_numpy(fine["features"]),
            "coarse_tile_boxes": torch.from_numpy(coarse["tile_boxes"]),
            "fine_tile_boxes": torch.from_numpy(fine["tile_boxes"]),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "sample_weight": torch.tensor(
                float(row[self.config.data.sample_weight_column]), dtype=torch.float32
            ),
            "filename": name,
            "source_image_path": str(source),
            "processed_image_path": coarse["processed_image_path"],
            "coarse_feature_cache_path": str(coarse_path),
            "fine_feature_cache_path": str(fine_path),
            "cohort_id": str(row[self.config.data.cohort_column]),
            "supervision_tier": str(row[self.config.data.supervision_tier_column]),
            "plot_group_id": str(row["plot_group_id"]),
        }


def verify_feature_caches(tables: list[pd.DataFrame], config: Config) -> int:
    combined = pd.concat(tables, ignore_index=True)
    combined = combined.drop_duplicates(subset=[config.data.absolute_path_column])
    dimensions: set[int] = set()
    missing = []
    for label, scale in (("coarse", config.coarse), ("fine", config.fine)):
        scale_config = config.single_scale_config(scale)
        expected_tiles = scale.rows * scale.columns
        for _, row in combined.iterrows():
            name, source = feature_name(row, config), source_path(row, config)
            path = feature_cache_path(scale_config, name, source)
            if not path.is_file():
                missing.append((label, name, path))
                continue
            record = load_feature_record(
                path, expected_identity=cache_identity(scale_config, name, source)
            )
            if record["tile_boxes"].shape[0] != expected_tiles:
                raise ValueError(f"Expected {expected_tiles} {label} tiles for {name}")
            dimensions.add(int(record["features"].shape[1]))
    if missing:
        preview = "\n".join(f"{scale} {name}: {path}" for scale, name, path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} multicohort feature record(s) are missing. First records:\n"
            f"{preview}\nRun the package prepare_features command."
        )
    if len(dimensions) != 1:
        raise ValueError(f"Expected one frozen feature dimension, got {sorted(dimensions)}")
    return dimensions.pop()


def make_loader(
    table: pd.DataFrame,
    scaler: TargetScaler,
    config: Config,
    stage: StageSettings,
    *,
    training: bool,
    seed_offset: int = 0,
) -> DataLoader:
    dataset = MultiCohortFeatureDataset(
        table,
        scaler.transform(table[config.data.target_column]),
        config,
    )
    arguments = {
        "batch_size": stage.batch_size,
        "num_workers": stage.num_workers,
        "pin_memory": config.runtime.pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": stage.num_workers > 0,
        "generator": torch.Generator().manual_seed(config.training.seed + seed_offset),
    }
    if training:
        return DataLoader(
            dataset,
            sampler=EpochSampler(dataset, config.training.seed + seed_offset),
            **arguments,
        )
    return DataLoader(dataset, shuffle=False, **arguments)


def prepare_data(config: Config, *, verify_caches: bool = True):
    pretrain = load_manifest(config, "pretrain")
    finetune = load_manifest(config, "finetune")
    validation = load_manifest(config, "validation")
    test = load_manifest(config, "test")
    validate_split_isolation(pretrain, finetune, validation, test, config)
    scaler = TargetScaler.fit(finetune[config.data.target_column])
    dimension = (
        verify_feature_caches([pretrain, finetune, validation, test], config)
        if verify_caches
        else None
    )
    return pretrain, finetune, validation, test, scaler, dimension
