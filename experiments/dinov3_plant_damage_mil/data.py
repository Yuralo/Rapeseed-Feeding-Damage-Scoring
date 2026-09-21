"""Gold split datasets augmented with versioned plant-patch embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.dinov3_grid_tiled_mil.data import EpochSampler
from experiments.dinov3_hierarchical_three_view_mil.data import (
    ThreeViewFeatureDataset,
    load_manifest,
    validate_split_isolation,
    verify_features,
)
from rapeseed_damage.reproducibility import seed_worker

from .config import Config
from .features import cache_path, identity, load


class DamageFeatureDataset(ThreeViewFeatureDataset):
    def __init__(self, table, targets, config: Config):
        super().__init__(table, targets, config.base)
        self.damage_config = config

    def __getitem__(self, index):
        example = super().__getitem__(index)
        source = Path(example["source_image_path"])
        relative = example["filename"]
        patch_path = cache_path(self.damage_config, relative, source)
        record = load(patch_path, identity(self.damage_config, relative, source))
        count, patches, dimension = record["patch_features"].shape
        if count != int(example["plant_count"]):
            raise ValueError(f"Plant/patch count mismatch for {relative}")
        if dimension != example["plant_features"].shape[-1]:
            raise ValueError(f"DINO feature dimension mismatch for {relative}")
        maximum = self.config.adaptive_crops.maximum_instances
        values = np.zeros((maximum, patches, dimension), dtype=np.float32)
        boxes = np.full((maximum, patches, 4), -1, dtype=np.int32)
        coverage = np.zeros((maximum, patches), dtype=np.float32)
        valid = np.zeros((maximum, patches), dtype=bool)
        values[:count] = record["patch_features"]
        boxes[:count] = record["patch_boxes"]
        coverage[:count] = record["patch_foreground_fraction"]
        valid[:count] = record["patch_valid"]
        example.update({
            "patch_features": torch.from_numpy(values),
            "patch_boxes": torch.from_numpy(boxes),
            "patch_foreground_fraction": torch.from_numpy(coverage),
            "patch_valid": torch.from_numpy(valid),
            "patch_cache_path": str(patch_path),
        })
        return example


def prepare_tables(config: Config, checkpoint: dict):
    base = config.base
    tables = {split: load_manifest(base, split) for split in ("pretrain", "finetune", "validation", "test")}
    validate_split_isolation(tables, base)
    saved = checkpoint.get("manifests") or {}
    for split in ("finetune", "validation", "test"):
        names = tables[split][base.data.filename_column].astype(str).tolist()
        if names != list(map(str, saved.get(split, []))):
            raise ValueError(f"Base checkpoint {split} manifest differs; cannot compare on the same split")
    dimension = verify_features(
        pd.concat([tables["finetune"], tables["validation"]], ignore_index=True), base
    )
    if int(checkpoint["feature_dim"]) != dimension:
        raise ValueError("Base checkpoint and three-view cache feature dimensions differ")
    verify_patch_features(config, tables["finetune"])
    verify_patch_features(config, tables["validation"])
    return tables, dimension


def verify_patch_features(config: Config, table) -> None:
    base = config.base
    missing = []
    for _, row in table.iterrows():
        relative = str(row[base.data.filename_column])
        source = Path(str(row[base.data.absolute_path_column]))
        path = cache_path(config, relative, source)
        if not path.is_file():
            missing.append((relative, path))
            continue
        load(path, identity(config, relative, source))
    if missing:
        preview = "\n".join(f"{name}: {path}" for name, path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} plant-patch cache records are missing; run "
            "python -m experiments.dinov3_plant_damage_mil.prepare_features first.\n" + preview
        )


def make_loader(table, scaler, config: Config, *, training: bool, offset: int = 0):
    base = config.base
    dataset = DamageFeatureDataset(
        table, scaler.transform(table[base.data.target_column]), config
    )
    sampler = EpochSampler(dataset, config.seed + offset) if training else None
    loader = DataLoader(
        dataset, batch_size=config.batch_size, sampler=sampler, shuffle=False,
        num_workers=config.num_workers, pin_memory=base.runtime.pin_memory,
        worker_init_fn=seed_worker, persistent_workers=config.num_workers > 0,
        generator=torch.Generator().manual_seed(config.seed + offset),
    )
    return loader
