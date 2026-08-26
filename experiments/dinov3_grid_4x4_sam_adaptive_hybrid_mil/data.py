"""Dataset joining 3x3, 4x4, and variable SAM-adaptive frozen features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.dinov3_grid_sam_adaptive_mil.features import (
    cache_identity as adaptive_identity,
)
from experiments.dinov3_grid_sam_adaptive_mil.features import (
    feature_cache_path as adaptive_path,
)
from experiments.dinov3_grid_sam_adaptive_mil.features import (
    load_feature_record as load_adaptive,
)
from experiments.dinov3_grid_tiled_mil.data import (
    EpochSampler,
    TargetScaler,
    image_path,
    load_scores,
    scaler_from_checkpoint,
    split_scores,
)
from experiments.dinov3_grid_tiled_mil.features import cache_identity as tiled_identity
from experiments.dinov3_grid_tiled_mil.features import feature_cache_path as tiled_path
from experiments.dinov3_grid_tiled_mil.features import load_feature_record as load_tiled
from rapeseed_damage.reproducibility import seed_worker

from .config import Config


class HybridFeatureDataset(Dataset):
    def __init__(self, table, targets, config: Config):
        if len(table) != len(targets):
            raise ValueError("Table and target lengths differ")
        self.table = table.reset_index(drop=True)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.config = config
        self.context_config = config.context_config()
        self.fine_config = config.fine_config()

    def __len__(self):
        return len(self.table)

    def _tiled_record(self, filename, source, scale_config):
        path = tiled_path(scale_config, filename, source)
        record = load_tiled(
            path, expected_identity=tiled_identity(scale_config, filename, source)
        )
        return path, record

    def __getitem__(self, index):
        filename = str(self.table.iloc[index][self.config.data.filename_column])
        source = image_path(self.config, filename)
        context_file, context = self._tiled_record(filename, source, self.context_config)
        fine_file, fine = self._tiled_record(filename, source, self.fine_config)
        adaptive_file = adaptive_path(self.config, filename, source)
        adaptive = load_adaptive(
            adaptive_file,
            expected_identity=adaptive_identity(self.config, filename, source),
        )
        count = len(adaptive["features"])
        maximum = self.config.adaptive_crops.maximum_instances
        if count > maximum:
            raise ValueError(f"Adaptive cache has {count} instances, configured maximum is {maximum}")
        dimensions = {
            int(context["features"].shape[1]),
            int(fine["features"].shape[1]),
            int(adaptive["features"].shape[1]),
        }
        if len(dimensions) != 1:
            raise ValueError(f"Feature dimensions differ for {filename}: {sorted(dimensions)}")
        feature_dim = dimensions.pop()
        instances = np.zeros((maximum, feature_dim), dtype=np.float32)
        boxes = np.full((maximum, 4), -1, dtype=np.int32)
        pixels = np.zeros(maximum, dtype=np.float32)
        valid = np.zeros(maximum, dtype=bool)
        instances[:count] = adaptive["features"]
        boxes[:count] = adaptive["boxes"]
        pixels[:count] = adaptive["foreground_pixels"]
        valid[:count] = True
        return {
            "context_features": torch.from_numpy(context["features"]),
            "fine_features": torch.from_numpy(fine["features"]),
            "fine_tile_boxes": torch.from_numpy(fine["tile_boxes"]),
            "instance_features": torch.from_numpy(instances),
            "instance_boxes": torch.from_numpy(boxes),
            "instance_foreground_pixels": torch.from_numpy(pixels),
            "instance_valid": torch.from_numpy(valid),
            "instance_count": torch.tensor(count),
            "mask_coverage": torch.tensor(adaptive["mask_coverage"], dtype=torch.float32),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "filename": filename,
            "source_image_path": str(source),
            "processed_image_path": adaptive["processed_image_path"],
            "mask_path": adaptive["mask_path"],
            "context_feature_cache_path": str(context_file),
            "fine_feature_cache_path": str(fine_file),
            "adaptive_feature_cache_path": str(adaptive_file),
        }


def verify_caches(table: pd.DataFrame, config: Config) -> int:
    context_config = config.context_config()
    fine_config = config.fine_config()
    dimensions = set()
    missing = []
    for _, row in table.iterrows():
        filename = str(row[config.data.filename_column])
        source = image_path(config, filename)
        context_file = tiled_path(context_config, filename, source)
        fine_file = tiled_path(fine_config, filename, source)
        adaptive_file = adaptive_path(config, filename, source)
        absent = [path for path in (context_file, fine_file, adaptive_file) if not path.is_file()]
        if absent:
            missing.append((filename, absent))
            continue
        context = load_tiled(
            context_file, expected_identity=tiled_identity(context_config, filename, source)
        )
        fine = load_tiled(fine_file, expected_identity=tiled_identity(fine_config, filename, source))
        adaptive = load_adaptive(
            adaptive_file, expected_identity=adaptive_identity(config, filename, source)
        )
        if context["tile_boxes"].shape[0] != config.context.rows * config.context.columns:
            raise ValueError(f"Unexpected 3x3 tile count for {filename}")
        if fine["tile_boxes"].shape[0] != config.fine.rows * config.fine.columns:
            raise ValueError(f"Unexpected 4x4 tile count for {filename}")
        dimensions.update(
            (
                int(context["features"].shape[1]),
                int(fine["features"].shape[1]),
                int(adaptive["features"].shape[1]),
            )
        )
    if missing:
        preview = "\n".join(
            f"{filename}: {', '.join(map(str, paths))}" for filename, paths in missing[:5]
        )
        raise FileNotFoundError(
            f"{len(missing)} hybrid cache set(s) are incomplete. First records:\n{preview}"
        )
    if len(dimensions) != 1:
        raise ValueError(f"Expected one shared feature dimension, got {sorted(dimensions)}")
    return dimensions.pop()


def make_loaders(train, validation, scaler, config):
    train_dataset = HybridFeatureDataset(
        train, scaler.transform(train[config.data.target_column]), config
    )
    validation_dataset = HybridFeatureDataset(
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
    feature_dim = verify_caches(pd.concat([train, validation], ignore_index=True), config)
    scaler = (
        scaler_from_checkpoint(checkpoint)
        if checkpoint is not None
        else TargetScaler.fit(train[config.data.target_column])
    )
    train_loader, validation_loader = make_loaders(train, validation, scaler, config)
    return table, train, validation, scaler, feature_dim, train_loader, validation_loader


__all__ = ["HybridFeatureDataset", "prepare_data", "verify_caches"]
