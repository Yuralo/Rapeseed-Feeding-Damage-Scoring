"""Context plus padded variable-length adaptive-instance datasets."""

from __future__ import annotations

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
from experiments.dinov3_grid_tiled_mil.features import cache_identity as context_identity
from experiments.dinov3_grid_tiled_mil.features import feature_cache_path as context_path
from experiments.dinov3_grid_tiled_mil.features import load_feature_record as load_context
from rapeseed_damage.reproducibility import seed_worker

from .config import Config
from .features import cache_identity, feature_cache_path, load_feature_record


class AdaptiveFeatureDataset(Dataset):
    def __init__(self, table, targets, config: Config):
        self.table = table.reset_index(drop=True)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.config = config
        self.context_config = config.context_config()
        if len(self.table) != len(self.targets):
            raise ValueError("Table and target lengths differ")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        filename = str(self.table.iloc[index][self.config.data.filename_column])
        source = image_path(self.config, filename)
        context_file = context_path(self.context_config, filename, source)
        adaptive_file = feature_cache_path(self.config, filename, source)
        context = load_context(
            context_file, expected_identity=context_identity(self.context_config, filename, source)
        )
        adaptive = load_feature_record(
            adaptive_file, expected_identity=cache_identity(self.config, filename, source)
        )
        count, maximum = len(adaptive["features"]), self.config.adaptive_crops.maximum_instances
        if count > maximum:
            raise ValueError(
                f"Adaptive cache has {count} instances, configured maximum is {maximum}"
            )
        feature_dim = adaptive["features"].shape[1]
        features = np.zeros((maximum, feature_dim), dtype=np.float32)
        boxes = np.full((maximum, 4), -1, dtype=np.int32)
        pixels = np.zeros(maximum, dtype=np.float32)
        valid = np.zeros(maximum, dtype=bool)
        features[:count] = adaptive["features"]
        boxes[:count] = adaptive["boxes"]
        pixels[:count] = adaptive["foreground_pixels"]
        valid[:count] = True
        return {
            "context_features": torch.from_numpy(context["features"]),
            "instance_features": torch.from_numpy(features),
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
            "adaptive_feature_cache_path": str(adaptive_file),
        }


def verify_caches(table: pd.DataFrame, config: Config) -> int:
    context_config = config.context_config()
    dimensions = set()
    missing = []
    for _, row in table.iterrows():
        filename = str(row[config.data.filename_column])
        source = image_path(config, filename)
        context_file = context_path(context_config, filename, source)
        adaptive_file = feature_cache_path(config, filename, source)
        if not context_file.is_file() or not adaptive_file.is_file():
            missing.append((filename, context_file, adaptive_file))
            continue
        context = load_context(
            context_file, expected_identity=context_identity(context_config, filename, source)
        )
        adaptive = load_feature_record(
            adaptive_file, expected_identity=cache_identity(config, filename, source)
        )
        dimensions.update((context["features"].shape[1], adaptive["features"].shape[1]))
    if missing:
        preview = "\n".join(
            f"{name}: context={left} adaptive={right}" for name, left, right in missing[:5]
        )
        raise FileNotFoundError(f"{len(missing)} cache pair(s) missing:\n{preview}")
    if len(dimensions) != 1:
        raise ValueError(f"Mixed feature dimensions: {sorted(dimensions)}")
    return dimensions.pop()


def make_loaders(train, validation, scaler, config):
    train_dataset = AdaptiveFeatureDataset(
        train, scaler.transform(train[config.data.target_column]), config
    )
    validation_dataset = AdaptiveFeatureDataset(
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
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
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
        if checkpoint
        else TargetScaler.fit(train[config.data.target_column])
    )
    loaders = make_loaders(train, validation, scaler, config)
    return table, train, validation, scaler, feature_dim, *loaders
