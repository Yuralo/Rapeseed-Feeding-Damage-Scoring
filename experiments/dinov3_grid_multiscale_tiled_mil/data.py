"""Paired 3x3 and 4x4 frozen-feature datasets and deterministic loaders."""

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


class MultiScaleFeatureDataset(Dataset):
    def __init__(self, table, targets, config: Config):
        if len(table) != len(targets):
            raise ValueError("Table and target lengths differ")
        self.table = table.reset_index(drop=True)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.config = config
        self.coarse_config = config.single_scale_config(config.coarse)
        self.fine_config = config.single_scale_config(config.fine)

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
        coarse_path, coarse = self._record(filename, source, self.coarse_config)
        fine_path, fine = self._record(filename, source, self.fine_config)
        expected_coarse = self.config.coarse.rows * self.config.coarse.columns
        expected_fine = self.config.fine.rows * self.config.fine.columns
        if coarse["tile_boxes"].shape[0] != expected_coarse:
            raise ValueError(f"Expected {expected_coarse} coarse tiles for {filename}")
        if fine["tile_boxes"].shape[0] != expected_fine:
            raise ValueError(f"Expected {expected_fine} fine tiles for {filename}")
        if coarse["features"].shape[1] != fine["features"].shape[1]:
            raise ValueError(f"Coarse/fine feature dimensions differ for {filename}")
        return {
            "coarse_features": torch.from_numpy(coarse["features"]),
            "fine_features": torch.from_numpy(fine["features"]),
            "coarse_tile_boxes": torch.from_numpy(coarse["tile_boxes"]),
            "fine_tile_boxes": torch.from_numpy(fine["tile_boxes"]),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "filename": filename,
            "source_image_path": str(source),
            "processed_image_path": coarse["processed_image_path"],
            "coarse_feature_cache_path": str(coarse_path),
            "fine_feature_cache_path": str(fine_path),
        }


def verify_feature_caches(table: pd.DataFrame, config: Config) -> int:
    dimensions: set[int] = set()
    missing: list[tuple[str, str, Path]] = []
    for label, scale in (("coarse", config.coarse), ("fine", config.fine)):
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
            f"{len(missing)} multi-scale feature record(s) are missing. First records:\n"
            f"{preview}\nCreate the 3x3 and 4x4 caches before training."
        )
    if len(dimensions) != 1:
        raise ValueError(f"Expected one shared frozen feature dimension, got {sorted(dimensions)}")
    return dimensions.pop()


def make_loaders(train, validation, scaler: TargetScaler, config: Config):
    train_dataset = MultiScaleFeatureDataset(
        train, scaler.transform(train[config.data.target_column]), config
    )
    validation_dataset = MultiScaleFeatureDataset(
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
