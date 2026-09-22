"""Plot-isolated paired labels using the plant-damage model tensor interface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_hierarchical_three_view_mil.checkpoint import validate_for
from experiments.dinov3_hierarchical_three_view_mil.data import (
    filter_usable_pretrain,
    load_manifest,
    validate_split_isolation,
    verify_features,
)
from experiments.dinov3_plant_damage_mil.data import make_loader, verify_patch_features
from experiments.dinov3_plant_damage_mil.features import cache_path, identity, load


def tables(config):
    result = {
        split: load_manifest(config.base, split)
        for split in ("pretrain", "finetune", "validation", "test")
    }
    validate_split_isolation(result, config.base)
    return result


def _usable_patches(config, table):
    valid, omitted = [], []
    for _, row in table.iterrows():
        relative = str(row[config.base.data.filename_column])
        source = Path(str(row[config.base.data.absolute_path_column]))
        path = cache_path(config, relative, source)
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            load(path, identity(config, relative, source))
            valid.append(row)
        except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
            omitted.append(dict(row) | {"reason": str(error)})
    return (
        pd.DataFrame(valid, columns=table.columns).reset_index(drop=True),
        pd.DataFrame(omitted),
    )


def prepare_data(config, base_state):
    validate_for(base_state, config.base)
    if base_state.get("stage") != "finetune":
        raise ValueError("The frozen three-view base must be gold-finetuned")
    split = tables(config)
    saved = base_state.get("manifests") or {}
    for name in ("finetune", "validation", "test"):
        actual = split[name][config.base.data.filename_column].astype(str).tolist()
        if actual != list(map(str, saved.get(name, []))):
            raise ValueError(f"Frozen base checkpoint {name} rows differ")
    strong = pd.concat([split["finetune"], split["validation"], split["test"]], ignore_index=True)
    dimension = verify_features(strong, config.base)
    validate_for(base_state, config.base, dimension)
    verify_patch_features(config, strong)
    weak, missing_base = filter_usable_pretrain(split["pretrain"], config.base)
    weak, missing_patches = _usable_patches(config, weak)
    missing_total = missing_base + len(missing_patches)
    if missing_total / len(split["pretrain"]) > config.base.data.maximum_weak_failure_fraction:
        raise FileNotFoundError(
            f"{missing_total}/{len(split['pretrain'])} discordant paired images lack features"
        )
    train = pd.concat([split["finetune"], weak], ignore_index=True)
    train = train.sort_values(config.base.data.filename_column).reset_index(drop=True)
    scaler = TargetScaler(
        mean=float(base_state["target_mean"]),
        std=float(base_state["target_std"]),
        training_mean=float(base_state.get("target_training_mean", base_state["target_mean"])),
    )
    return (
        train,
        split["validation"],
        split["test"],
        scaler,
        dimension,
        missing_base,
        missing_patches,
    )


__all__ = ["make_loader", "prepare_data", "tables"]
