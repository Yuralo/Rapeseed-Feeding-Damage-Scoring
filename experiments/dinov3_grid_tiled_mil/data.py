"""Score splits and loaders backed by per-image frozen feature records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

from rapeseed_damage.reproducibility import seed_worker

from .config import Config
from .features import cache_identity, feature_cache_path, load_feature_record


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    std: float
    training_mean: float | None = None

    @classmethod
    def fit(cls, values: Sequence[float]) -> TargetScaler:
        array = np.asarray(values, dtype=np.float32)
        mean, std = float(array.mean()), float(array.std())
        if not len(array) or not np.isfinite(array).all() or std <= 0:
            raise ValueError("Training targets require a positive finite standard deviation")
        return cls(mean, std, training_mean=mean)

    def transform(self, values: Sequence[float]) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse(self, values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean

    @property
    def baseline_mean(self) -> float:
        return self.mean if self.training_mean is None else self.training_mean


def image_path(config: Config, filename: str) -> Path:
    path = Path(config.data.dataset_dir) / filename
    return path if path.suffix else path.with_suffix(config.data.image_extension)


def load_scores(config: Config) -> pd.DataFrame:
    path = Path(config.data.dataset_dir) / config.data.scores_file
    if not path.is_file():
        raise FileNotFoundError(f"Scores file not found: {path}")
    table = pd.read_csv(path)
    required = {config.data.filename_column, config.data.target_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Scores CSV is missing column(s): {', '.join(sorted(missing))}")
    table = table.copy()
    table[config.data.filename_column] = table[config.data.filename_column].astype(str)
    table[config.data.target_column] = pd.to_numeric(
        table[config.data.target_column], errors="raise"
    )
    if table.empty or not np.isfinite(table[config.data.target_column]).all():
        raise ValueError("Scores CSV has no usable finite targets")
    if config.data.verify_images:
        missing_images = [
            str(image_path(config, name))
            for name in table[config.data.filename_column]
            if not image_path(config, name).is_file()
        ]
        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} image(s) are missing. First paths:\n"
                + "\n".join(missing_images[:5])
            )
    return table


def split_scores(
    table: pd.DataFrame,
    config: Config,
    training_filenames: Sequence[str] | None = None,
    validation_filenames: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if training_filenames is None or validation_filenames is None:
        train, validation = train_test_split(
            table,
            test_size=config.data.validation_fraction,
            random_state=config.data.split_seed,
        )
    else:
        names = table[config.data.filename_column].astype(str)
        if names.duplicated().any():
            raise ValueError("Checkpoint split restoration requires unique filenames")
        indexed = table.copy()
        indexed.index = names
        train_names = list(map(str, training_filenames))
        validation_names = list(map(str, validation_filenames))
        requested = train_names + validation_names
        unknown = set(requested) - set(names)
        if unknown:
            raise ValueError("Checkpoint images are absent: " + ", ".join(sorted(unknown)[:5]))
        if len(requested) != len(set(requested)):
            raise ValueError("Checkpoint train/validation manifests overlap")
        train, validation = indexed.loc[train_names], indexed.loc[validation_names]
    if train.empty or validation.empty:
        raise ValueError("Training and validation splits must both contain samples")
    return train.reset_index(drop=True), validation.reset_index(drop=True)


class FeatureDataset(Dataset):
    def __init__(self, table, targets, config: Config):
        if len(table) != len(targets):
            raise ValueError("Table and target lengths differ")
        self.table = table.reset_index(drop=True)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.config = config

    def __len__(self) -> int:
        return len(self.table)

    def record_path(self, index: int) -> tuple[str, Path, Path, str]:
        filename = str(self.table.iloc[index][self.config.data.filename_column])
        source = image_path(self.config, filename)
        identity = cache_identity(self.config, filename, source)
        return filename, source, feature_cache_path(self.config, filename, source), identity

    def __getitem__(self, index: int) -> dict[str, object]:
        filename, source, cache_path, identity = self.record_path(index)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Frozen features missing for {filename}: {cache_path}. "
                "Run the package's prepare_features command first."
            )
        record = load_feature_record(cache_path, expected_identity=identity)
        expected_tiles = self.config.tiles.rows * self.config.tiles.columns
        if record["tile_boxes"].shape[0] != expected_tiles:
            raise ValueError(
                f"Expected {expected_tiles} tiles for {filename}, got "
                f"{record['tile_boxes'].shape[0]}"
            )
        return {
            "features": torch.from_numpy(record["features"]),
            "tile_boxes": torch.from_numpy(record["tile_boxes"]),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "filename": filename,
            "source_image_path": str(source),
            "processed_image_path": record["processed_image_path"],
            "feature_cache_path": str(cache_path),
        }


class EpochSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, seed: int):
        self.dataset, self.seed, self.epoch = dataset, seed, 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())


def verify_feature_cache(table: pd.DataFrame, config: Config) -> int:
    missing = []
    feature_dim = None
    for index, row in table.iterrows():
        filename = str(row[config.data.filename_column])
        source = image_path(config, filename)
        path = feature_cache_path(config, filename, source)
        if not path.is_file():
            missing.append((filename, path))
            continue
        record = load_feature_record(
            path, expected_identity=cache_identity(config, filename, source)
        )
        current_dim = int(record["features"].shape[1])
        if feature_dim is None:
            feature_dim = current_dim
        elif current_dim != feature_dim:
            raise ValueError(f"Mixed frozen feature dimensions: {feature_dim} and {current_dim}")
    if missing:
        preview = "\n".join(f"{name}: {path}" for name, path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} frozen feature record(s) are missing. First records:\n{preview}\n"
            "Run: python -m experiments.dinov3_grid_tiled_mil.prepare_features --config ..."
        )
    if feature_dim is None:
        raise ValueError("Feature cache is empty")
    return feature_dim


def make_loaders(train, validation, scaler: TargetScaler, config: Config):
    train_dataset = FeatureDataset(
        train, scaler.transform(train[config.data.target_column]), config
    )
    validation_dataset = FeatureDataset(
        validation, scaler.transform(validation[config.data.target_column]), config
    )
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.runtime.pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": config.training.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=EpochSampler(train_dataset, config.training.seed),
        generator=torch.Generator().manual_seed(config.training.seed),
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.training.seed + 1),
        **common,
    )
    return train_loader, validation_loader


def prepare_data(config: Config, checkpoint=None):
    table = load_scores(config)
    train, validation = split_scores(
        table,
        config,
        training_filenames=(checkpoint or {}).get("training_filenames"),
        validation_filenames=(checkpoint or {}).get("validation_filenames"),
    )
    feature_dim = verify_feature_cache(pd.concat([train, validation], ignore_index=True), config)
    scaler = (
        scaler_from_checkpoint(checkpoint)
        if checkpoint is not None
        else TargetScaler.fit(train[config.data.target_column])
    )
    train_loader, validation_loader = make_loaders(train, validation, scaler, config)
    return table, train, validation, scaler, feature_dim, train_loader, validation_loader


def scaler_from_checkpoint(state) -> TargetScaler:
    if state.get("target_mean") is None or state.get("target_std") is None:
        raise ValueError("Checkpoint has no target normalization statistics")
    return TargetScaler(
        float(state["target_mean"]),
        float(state["target_std"]),
        training_mean=float(state.get("target_training_mean", state["target_mean"])),
    )
