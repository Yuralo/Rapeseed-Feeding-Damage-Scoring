"""Manifest-backed padded three-view datasets and deterministic samplers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from experiments.dinov3_grid_tiled_mil.data import EpochSampler, TargetScaler
from rapeseed_damage.reproducibility import seed_worker

from .config import Config, StageSettings
from .features import cache_identity, feature_cache_path, load_record


def load_manifest(config: Config, split: str) -> pd.DataFrame:
    path = config.manifest_path(split)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} manifest: {path}")
    table = pd.read_csv(path)
    required = {
        config.data.absolute_path_column,
        config.data.filename_column,
        config.data.target_column,
        config.data.sample_weight_column,
        config.data.cohort_column,
        config.data.supervision_tier_column,
        config.data.group_column,
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    if table.empty:
        raise ValueError(f"{split} manifest is empty")
    table = table.copy()
    table[config.data.target_column] = pd.to_numeric(table[config.data.target_column], errors="raise")
    table[config.data.sample_weight_column] = pd.to_numeric(
        table[config.data.sample_weight_column], errors="raise"
    )
    if not np.isfinite(table[config.data.target_column]).all():
        raise ValueError(f"{split} contains non-finite targets")
    weights = table[config.data.sample_weight_column].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError(f"{split} sample weights must be finite and positive")
    paths = table[config.data.absolute_path_column].astype(str)
    if paths.duplicated().any():
        raise ValueError(f"{split} contains duplicate absolute paths")
    if config.data.verify_images:
        absent = [value for value in paths if not Path(value).is_file()]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} {split} image(s) are absent; first paths:\n" + "\n".join(absent[:5])
            )
    return table.reset_index(drop=True)


def validate_split_isolation(tables: dict[str, pd.DataFrame], config: Config) -> None:
    group = config.data.group_column
    groups = {name: set(table[group].astype(str)) for name, table in tables.items()}
    if groups["validation"] & groups["test"]:
        raise ValueError("Validation and test share physical plot groups")
    holdout = groups["validation"] | groups["test"]
    if holdout & (groups["pretrain"] | groups["finetune"]):
        raise ValueError("Training and holdout manifests share physical plot groups")
    path_column = config.data.absolute_path_column
    paths = {name: set(table[path_column].astype(str)) for name, table in tables.items()}
    names = list(paths)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if paths[left] & paths[right]:
                raise ValueError(f"{left} and {right} share exact image paths")


def _row_cache(row: pd.Series, config: Config) -> tuple[Path, str, Path]:
    relative = str(row[config.data.filename_column])
    source = Path(str(row[config.data.absolute_path_column]))
    return feature_cache_path(config, relative, source), relative, source


def filter_usable_pretrain(table: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, int]:
    usable = []
    for _, row in table.iterrows():
        path, relative, source = _row_cache(row, config)
        if not path.is_file():
            usable.append(False)
            continue
        try:
            load_record(path, expected_identity=cache_identity(config, relative, source))
            usable.append(True)
        except (OSError, ValueError, RuntimeError, KeyError):
            # Stale/corrupt weak records are treated exactly like missing records.
            usable.append(False)
    filtered = table.loc[np.asarray(usable, dtype=bool)].reset_index(drop=True)
    missing = len(table) - len(filtered)
    if missing / len(table) > config.data.maximum_weak_failure_fraction:
        raise FileNotFoundError(
            f"{missing}/{len(table)} weak feature records are missing or invalid; run prepare_features"
        )
    return filtered, missing


def verify_features(table: pd.DataFrame, config: Config) -> int:
    dimensions, missing = set(), []
    for _, row in table.iterrows():
        path, relative, source = _row_cache(row, config)
        if not path.is_file():
            missing.append((relative, path))
            continue
        record = load_record(path, expected_identity=cache_identity(config, relative, source))
        dimensions.add(int(record["global_feature"].shape[0]))
    if missing:
        preview = "\n".join(f"{name}: {path}" for name, path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} required feature records are missing; run prepare_features. First:\n{preview}"
        )
    if len(dimensions) != 1:
        raise ValueError(f"Expected one feature dimension, got {sorted(dimensions)}")
    return dimensions.pop()


class ThreeViewFeatureDataset(Dataset):
    def __init__(self, table: pd.DataFrame, targets: np.ndarray, config: Config):
        self.table = table.reset_index(drop=True)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.config = config
        if len(self.table) != len(self.targets):
            raise ValueError("Table and target lengths differ")

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.table.iloc[index]
        path, relative, source = _row_cache(row, self.config)
        record = load_record(path, expected_identity=cache_identity(self.config, relative, source))
        count = len(record["plant_features"])
        maximum = self.config.adaptive_crops.maximum_instances
        if count > maximum:
            raise ValueError(f"{relative} has {count} plant instances, configured maximum is {maximum}")
        dimension = record["global_feature"].shape[0]
        plants = np.zeros((maximum, dimension), dtype=np.float32)
        boxes = np.full((maximum, 4), -1, dtype=np.int32)
        cell_indices = np.full(maximum, -1, dtype=np.int64)
        foreground = np.zeros(maximum, dtype=np.float32)
        valid = np.zeros(maximum, dtype=bool)
        plants[:count] = record["plant_features"]
        boxes[:count] = record["plant_boxes"]
        cell_indices[:count] = record["plant_cell_indices"]
        foreground[:count] = record["foreground_pixels"]
        valid[:count] = True
        return {
            "global_feature": torch.from_numpy(record["global_feature"]),
            "cell_features": torch.from_numpy(record["cell_features"]),
            "cell_boxes": torch.from_numpy(record["cell_boxes"]),
            "plant_features": torch.from_numpy(plants),
            "plant_boxes": torch.from_numpy(boxes),
            "plant_cell_indices": torch.from_numpy(cell_indices),
            "plant_foreground_pixels": torch.from_numpy(foreground),
            "plant_valid": torch.from_numpy(valid),
            "plant_count": torch.tensor(count),
            "mask_coverage": torch.tensor(record["mask_coverage"], dtype=torch.float32),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "sample_weight": torch.tensor(
                float(row[self.config.data.sample_weight_column]), dtype=torch.float32
            ),
            "filename": relative,
            "source_image_path": str(source),
            "processed_image_path": record["processed_image_path"],
            "mask_path": record["mask_path"],
            "feature_cache_path": str(path),
            "input_mode": record["input_mode"],
            "cohort_id": str(row[self.config.data.cohort_column]),
            "supervision_tier": str(row[self.config.data.supervision_tier_column]),
        }


class CohortBalancedSampler(Sampler[int]):
    """Sample equally across cohorts while preserving deterministic epoch variation."""

    def __init__(self, cohorts, seed: int):
        values = pd.Series(cohorts, dtype=str)
        counts = values.value_counts()
        self.weights = torch.tensor([1.0 / counts[value] for value in values], dtype=torch.double)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(self.weights, len(self.weights), replacement=True, generator=generator)
        return iter(sampled.tolist())

    def __len__(self) -> int:
        return len(self.weights)


def make_loader(
    table: pd.DataFrame,
    scaler: TargetScaler,
    config: Config,
    settings: StageSettings,
    *,
    training: bool,
    cohort_balanced: bool = False,
    seed_offset: int = 0,
) -> DataLoader:
    dataset = ThreeViewFeatureDataset(
        table, scaler.transform(table[config.data.target_column]), config
    )
    common = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "pin_memory": config.runtime.pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": settings.num_workers > 0,
        "generator": torch.Generator().manual_seed(config.training.seed + seed_offset),
    }
    if not training:
        return DataLoader(dataset, shuffle=False, **common)
    if cohort_balanced:
        sampler = CohortBalancedSampler(
            table[config.data.cohort_column].astype(str), config.training.seed + seed_offset
        )
    else:
        sampler = EpochSampler(dataset, config.training.seed + seed_offset)
    return DataLoader(dataset, sampler=sampler, **common)


def prepare_data(config: Config):
    tables = {name: load_manifest(config, name) for name in ("pretrain", "finetune", "validation", "test")}
    validate_split_isolation(tables, config)
    scaler = TargetScaler.fit(tables["finetune"][config.data.target_column])
    required = [tables["finetune"], tables["validation"]]
    if config.training.use_weak_pretraining:
        tables["pretrain"], missing = filter_usable_pretrain(tables["pretrain"], config)
        required.append(tables["pretrain"])
    else:
        missing = 0
    dimension = verify_features(pd.concat(required, ignore_index=True), config)
    return tables, scaler, dimension, missing


__all__ = [
    "CohortBalancedSampler", "ThreeViewFeatureDataset", "filter_usable_pretrain", "load_manifest",
    "make_loader", "prepare_data", "validate_split_isolation", "verify_features",
]
