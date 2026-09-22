"""Inventory the image store, model weights, and prerequisite feature caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from experiments.dinov3_hierarchical_three_view_mil.features import feature_cache_path
from experiments.dinov3_plant_damage_mil.features import cache_path as patch_cache_path

from .config import Config, load_config
from .prepare_pairs import _manifest

WEIGHT_NAMES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                "pytorch_model.bin.index.json")


def _images(table: pd.DataFrame, available: set[str]) -> dict:
    paths = [Path(value) for value in table["absolute_path"].astype(str)]
    missing = [str(path) for path in paths if str(path) not in available]
    return {"rows": len(paths), "images_present": len(paths) - len(missing),
            "first_missing_images": missing[:5]}


def _features(table: pd.DataFrame, config: Config, available: set[str]) -> dict:
    base = config.plant.base
    totals = {"three_view_present": 0, "patch_present": 0, "rows": len(table)}
    for _, row in table.iterrows():
        source = Path(str(row[base.data.absolute_path_column]))
        if str(source) not in available:
            continue
        relative = str(row[base.data.filename_column])
        totals["three_view_present"] += feature_cache_path(base, relative, source).is_file()
        totals["patch_present"] += patch_cache_path(config.plant, relative, source).is_file()
    return totals


def run(config: Config) -> dict:
    adaptation = _manifest(config)
    gold = {split: pd.read_csv(Path(config.plant.base.data.manifest_dir) / f"{split}.csv")
            for split in ("finetune", "validation", "test")}
    pretrain = pd.read_csv(Path(config.plant.base.data.manifest_dir) / "pretrain.csv")
    ood = pretrain.loc[pretrain["cohort_id"].isin(
        ("wg_insects_t1_bbch10", "dsv_asendorf_t1_bbch11"))].reset_index(drop=True)
    all_paths = set().union(*(set(table["absolute_path"].astype(str)) for table in
                              [adaptation, *gold.values(), ood]))
    image_root = Path(os.path.commonpath(all_paths))
    available = ({path for path in all_paths if Path(path).is_file()}
                 if image_root.is_dir() else set())
    backbone = Path(config.plant.base.features.backbone)
    processor = Path(config.plant.base.features.processor)
    weights = backbone.is_dir() and any((backbone / name).is_file() for name in WEIGHT_NAMES)
    report = {
        "image_store_root": str(image_root),
        "image_store_root_present": image_root.is_dir(),
        "adaptation": _images(adaptation, available),
        "gold_images": {split: _images(table, available) for split, table in gold.items()},
        "ood_images": _images(ood, available),
        "backbone_weight_files_present": bool(weights),
        "processor_config_present": (processor / "preprocessor_config.json").is_file(),
        "gold_base_checkpoint_present": Path(config.plant.base_checkpoint).is_file(),
        "gold_feature_caches": {
            split: _features(gold[split], config, available)
            for split in ("finetune", "validation")
        },
        "ood_feature_caches": _features(ood, config, available),
        "test_feature_caches": _features(gold["test"], config, available),
    }
    report["pair_preparation_ready"] = bool(
        report["adaptation"]["images_present"] == report["adaptation"]["rows"] and
        report["backbone_weight_files_present"] and report["processor_config_present"]
    )
    report["gold_fit_ready_after_pretraining"] = bool(
        report["gold_base_checkpoint_present"] and
        all(value["three_view_present"] == value["rows"] and
            value["patch_present"] == value["rows"]
            for value in report["gold_feature_caches"].values())
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
