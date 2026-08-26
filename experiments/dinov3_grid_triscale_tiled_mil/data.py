"""Aligned 3x3, 4x4, and 5x5 frozen-feature datasets and loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.dinov3_grid_tiled_mil.data import (
    EpochSampler,
    TargetScaler,
    image_path,
    load_scores,
    scaler_from_checkpoint,
    split_scores,
)
from experiments.dinov3_grid_tiled_mil.features import (
    cache_identity,
    feature_cache_path,
    load_feature_record,
)
from rapeseed_damage.reproducibility import seed_worker

from .config import Config


class TriScaleFeatureDataset(Dataset):
    def __init__(self, table, targets, config: Config):
        if len(table) != len(targets):
            raise ValueError("Table and target lengths differ")
        self.table = table.reset_index(drop=True)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.config = config
        self.scale_configs = {
            name: config.single_scale_config(scale) for name, scale in config.scales
        }

    def __len__(self) -> int:
        return len(self.table)

    def _record(self, filename: str, source: Path, scale_config):
        identity = cache_identity(scale_config, filename, source)
        path = feature_cache_path(scale_config, filename, source)
        if not path.is_file():
            raise FileNotFoundError(
                f"Frozen features missing for {filename}: {path}. "
                "Run the corresponding single-scale prepare_features command first."
            )
        return path, load_feature_record(path, expected_identity=identity)

    def __getitem__(self, index: int) -> dict[str, object]:
        filename = str(self.table.iloc[index][self.config.data.filename_column])
        source = image_path(self.config, filename)
        records = {
            name: self._record(filename, source, self.scale_configs[name])
            for name, _ in self.config.scales
        }
        dimensions = set()
        processed_paths = set()
        item: dict[str, object] = {
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "filename": filename,
            "source_image_path": str(source),
        }
        for name, scale in self.config.scales:
            path, record = records[name]
            expected_tiles = scale.rows * scale.columns
            if record["tile_boxes"].shape[0] != expected_tiles:
                raise ValueError(f"Expected {expected_tiles} {name} tiles for {filename}")
            dimensions.add(int(record["features"].shape[1]))
            processed_paths.add(str(record["processed_image_path"]))
            item[f"{name}_features"] = torch.from_numpy(record["features"])
            item[f"{name}_tile_boxes"] = torch.from_numpy(record["tile_boxes"])
            item[f"{name}_feature_cache_path"] = str(path)
        if len(dimensions) != 1:
            raise ValueError(f"Feature dimensions differ across scales for {filename}")
        if len(processed_paths) != 1:
            raise ValueError(f"Processed-image paths differ across scales for {filename}")
        item["processed_image_path"] = processed_paths.pop()
        return item


def verify_feature_caches(table: pd.DataFrame, config: Config) -> int:
    dimensions: set[int] = set()
    missing: list[tuple[str, str, Path]] = []
    for label, scale in config.scales:
        scale_config = config.single_scale_config(scale)
        expected_tiles = scale.rows * scale.columns
        for _, row in table.iterrows():
            filename = str(row[config.data.filename_column])
            source = image_path(config, filename)
            path = feature_cache_path(scale_config, filename, source)
            if not path.is_file():
                missing.append((label, filename, path))
                continue
            record = load_feature_record(
                path, expected_identity=cache_identity(scale_config, filename, source)
            )
            if record["tile_boxes"].shape[0] != expected_tiles:
                raise ValueError(
                    f"Expected {expected_tiles} {label} tiles for {filename}, got "
                    f"{record['tile_boxes'].shape[0]}"
                )
            dimensions.add(int(record["features"].shape[1]))
    if missing:
        preview = "\n".join(f"{label} {name}: {path}" for label, name, path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} tri-scale feature record(s) are missing. First records:\n"
            f"{preview}\nCreate the 3x3, 4x4, and 5x5 caches before training."
        )
    if len(dimensions) != 1:
        raise ValueError(f"Expected one shared frozen feature dimension, got {sorted(dimensions)}")
    return dimensions.pop()


def make_loaders(train, validation, scaler: TargetScaler, config: Config):
    train_dataset = TriScaleFeatureDataset(
        train, scaler.transform(train[config.data.target_column]), config
    )
    validation_dataset = TriScaleFeatureDataset(
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
    feature_dim = verify_feature_caches(pd.concat([train, validation], ignore_index=True), config)
    scaler = (
        scaler_from_checkpoint(checkpoint)
        if checkpoint is not None
        else TargetScaler.fit(train[config.data.target_column])
    )
    train_loader, validation_loader = make_loaders(train, validation, scaler, config)
    return table, train, validation, scaler, feature_dim, train_loader, validation_loader
