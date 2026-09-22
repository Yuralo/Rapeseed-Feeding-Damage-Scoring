"""Leakage checks and existing three-view feature loaders for the new split."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_hierarchical_three_view_mil.data import (
    filter_usable_pretrain,
    load_manifest,
    verify_features,
)
from experiments.dinov3_hierarchical_three_view_mil.data import (
    make_loader as make_base_loader,
)

from .build_manifests import _target
from .config import Config


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def manifest_hashes(config: Config) -> dict[str, str]:
    directory = Path(config.manifest_dir)
    return {
        name: _digest(directory / name)
        for name in ("weak_train.csv", "gold_validation.csv")
    }


def _validated_rows(path: Path, expected_gold: bool, config: Config) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty experiment manifest: {path}")
    for row in rows:
        gold = row["is_gold_standard"].casefold() == "true"
        if gold != expected_gold:
            raise ValueError(f"Wrong supervision tier in {path}: {row['image_id']}")
        target, source = _target(row)
        if not math.isclose(target, float(row["target"]), abs_tol=1e-5):
            raise ValueError(f"Scorer mean differs from target in {path}: {row['image_id']}")
        if row.get("target_source") != source:
            raise ValueError(f"Target source differs from scores in {path}: {row['image_id']}")
        if not expected_gold and config.mode == "dual_only" and source != "mean_of_two_scorers":
            raise ValueError("Dual-only training manifest contains a single-scorer row")
    return rows


def _validate_manifests(config: Config) -> None:
    directory = Path(config.manifest_dir)
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Run build_manifests before training")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["mode"] != config.mode:
        raise ValueError("Manifest mode does not match config")
    if _digest(Path(config.scored_manifest)) != summary["scored_manifest_sha256"]:
        raise ValueError("Source scored_manifest.csv changed; rebuild manifests")
    weak = _validated_rows(directory / "weak_train.csv", False, config)
    gold = _validated_rows(directory / "gold_validation.csv", True, config)
    if len(gold) != config.expected_gold_images:
        raise ValueError("Gold validation count changed")
    if len(weak) != summary["weak_train_images"] or len(gold) != summary["gold_validation_images"]:
        raise ValueError("Manifest counts changed since build")
    for field in ("plot_group_id", "sha256", "relative_path", "image_id"):
        if {row[field] for row in weak} & {row[field] for row in gold}:
            raise ValueError(f"Weak training and gold validation overlap by {field}")


def prepare_gold_data(config: Config):
    """Gold evaluation needs no weak feature cache once a checkpoint exists."""
    _validate_manifests(config)
    base = config.routed_base
    validation = load_manifest(base, "validation")
    dimension = verify_features(validation, base)
    return validation, dimension


def prepare_data(config: Config):
    _validate_manifests(config)
    base = config.routed_base
    eligible_train = load_manifest(base, "finetune")
    validation = load_manifest(base, "validation")
    train, _ = filter_usable_pretrain(eligible_train, base)
    selected_names = set(train[base.data.filename_column].astype(str))
    omitted = eligible_train.loc[
        ~eligible_train[base.data.filename_column].astype(str).isin(selected_names)
    ].reset_index(drop=True)
    dimension = verify_features(pd.concat([train, validation], ignore_index=True), base)
    scaler = TargetScaler.fit(train[base.data.target_column])
    return train, validation, scaler, dimension, omitted


def make_loader(table, scaler, config: Config, *, training: bool, seed_offset: int):
    from dataclasses import replace

    base = config.routed_base
    settings = replace(
        base.training.finetuning,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    routed = replace(base, training=replace(base.training, seed=config.seed))
    return make_base_loader(
        table, scaler, routed, settings, training=training,
        cohort_balanced=False, seed_offset=seed_offset,
    )
