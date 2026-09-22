"""Exact adaptive-MIL tensor interface from routed, leakage-safe cached views."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.dinov3_grid_tiled_mil.data import EpochSampler, TargetScaler
from experiments.dinov3_hierarchical_three_view_mil.features import (
    cache_identity as base_identity,
)
from experiments.dinov3_hierarchical_three_view_mil.features import (
    feature_cache_path as base_cache_path,
)
from experiments.dinov3_hierarchical_three_view_mil.features import (
    load_record as load_base,
)
from experiments.dinov3_weak_only_gold_validation.data import (
    prepare_data as prepare_weak_data,
)
from experiments.dinov3_weak_only_gold_validation.data import (
    prepare_gold_data as prepare_weak_gold,
)
from rapeseed_damage.reproducibility import seed_worker

from .config import Config
from .features import cache_path, identity, load


def _paths(config: Config, row) -> tuple[str, Path, Path, Path]:
    base = config.weak.routed_base
    relative = str(row[base.data.filename_column])
    source = Path(str(row[base.data.absolute_path_column]))
    return (relative, source, base_cache_path(base, relative, source),
            cache_path(config, relative, source))


def _verify_context(table: pd.DataFrame, config: Config, dimension: int, *, strict: bool):
    valid, omitted = [], []
    for _, row in table.iterrows():
        relative, source, _, path = _paths(config, row)
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            record = load(path, identity(config, relative, source))
            if record["tile_features"].shape[1] != dimension:
                raise ValueError(f"DINO feature dimension differs in {path}")
        except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
            if strict:
                raise FileNotFoundError(
                    f"Gold image lacks valid 3x3 context features: {relative}: {error}"
                ) from error
            omitted.append((row, str(error)))
            continue
        valid.append(row)
    selected = pd.DataFrame(valid, columns=table.columns).reset_index(drop=True)
    omitted_table = pd.DataFrame([dict(row) | {"reason": reason} for row, reason in omitted])
    return selected, omitted_table


def prepare_data(config: Config):
    weak, gold, _, dimension, missing_base = prepare_weak_data(config.weak)
    selected, missing_context = _verify_context(weak, config, dimension, strict=False)
    _verify_context(gold, config, dimension, strict=True)
    if selected.empty:
        raise ValueError("No weak images have complete adaptive features")
    base_missing = missing_base.assign(reason="missing_or_invalid_three_view_features")
    omitted = pd.concat([base_missing, missing_context], ignore_index=True)
    eligible = len(weak) + len(missing_base)
    if len(omitted) / eligible > config.weak.routed_base.data.maximum_weak_failure_fraction:
        raise FileNotFoundError(
            f"{len(omitted)}/{eligible} weak images lack usable adaptive features; "
            "inspect feature-preparation failures"
        )
    scaler = TargetScaler.fit(selected[config.weak.routed_base.data.target_column])
    return selected, gold, scaler, dimension, omitted


def prepare_gold_data(config: Config):
    gold, dimension = prepare_weak_gold(config.weak)
    _verify_context(gold, config, dimension, strict=True)
    return gold, dimension


class AdaptiveDataset(Dataset):
    def __init__(self, table, scaler: TargetScaler, config: Config):
        self.table = table.reset_index(drop=True)
        self.targets = scaler.transform(
            self.table[config.weak.routed_base.data.target_column]
        ).astype(np.float32)
        self.config = config

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        relative, source, base_path, context_path = _paths(self.config, row)
        base = load_base(
            base_path,
            expected_identity=base_identity(self.config.weak.routed_base, relative, source),
        )
        context = load(context_path, identity(self.config, relative, source))
        maximum = self.config.adaptive.adaptive_crops.maximum_instances
        count = len(base["plant_features"])
        if count > maximum:
            raise ValueError(f"{relative} has {count} plants; maximum is {maximum}")
        dimension = len(base["global_feature"])
        plants = np.zeros((maximum, dimension), dtype=np.float32)
        boxes = np.full((maximum, 4), -1, dtype=np.int32)
        foreground = np.zeros(maximum, dtype=np.float32)
        valid = np.zeros(maximum, dtype=bool)
        plants[:count] = base["plant_features"]
        boxes[:count] = base["plant_boxes"]
        foreground[:count] = base["foreground_pixels"]
        valid[:count] = True
        views = np.concatenate(
            [base["global_feature"][None], context["tile_features"]], axis=0
        )
        return {
            "context_features": torch.from_numpy(views),
            "instance_features": torch.from_numpy(plants),
            "instance_boxes": torch.from_numpy(boxes),
            "instance_foreground_pixels": torch.from_numpy(foreground),
            "instance_valid": torch.from_numpy(valid),
            "instance_count": torch.tensor(count),
            "mask_coverage": torch.tensor(base["mask_coverage"], dtype=torch.float32),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "filename": relative,
            "source_image_path": str(source),
            "processed_image_path": base["processed_image_path"],
            "mask_path": base["mask_path"],
            "context_feature_cache_path": str(context_path),
            "adaptive_feature_cache_path": str(base_path),
        }


def make_loader(table, scaler, config: Config, *, training: bool, seed_offset: int = 0):
    dataset = AdaptiveDataset(table, scaler, config)
    arguments = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.adaptive.runtime.pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": config.num_workers > 0,
        "generator": torch.Generator().manual_seed(config.seed + seed_offset),
    }
    if training:
        return DataLoader(
            dataset, sampler=EpochSampler(dataset, config.seed + seed_offset), **arguments
        )
    return DataLoader(dataset, shuffle=False, **arguments)
