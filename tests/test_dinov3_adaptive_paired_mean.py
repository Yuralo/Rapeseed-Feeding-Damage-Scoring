from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.dinov3_adaptive_paired_mean.config import load_config
from experiments.dinov3_adaptive_paired_mean.manifests import make_manifests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/dinov3_adaptive_paired_mean/config.toml"


def _config(tmp_path: Path):
    config = load_config(CONFIG)
    return replace(
        config,
        source_manifest=str(ROOT / config.source_manifest),
        reference_ood_dir=str(ROOT / config.reference_ood_dir),
        run_dir=str(tmp_path / "run"),
    )


def test_manifests_use_paired_means_and_plot_isolated_gold_holdouts(tmp_path):
    config = _config(tmp_path)
    report = make_manifests(config)
    assert report["paired_total"] == 928
    assert report["paired_training"] == 706
    assert report["gold_validation"] == 73
    assert report["gold_test"] == 72
    assert report["excluded_holdout_related"] == 77
    directory = Path(config.run_dir) / "manifests"
    with (directory / "pretrain.csv").open(newline="") as handle:
        train = list(csv.DictReader(handle))
    with (directory / "validation.csv").open(newline="") as handle:
        validation = list(csv.DictReader(handle))
    with (directory / "test.csv").open(newline="") as handle:
        test = list(csv.DictReader(handle))
    assert all(
        float(row["target"])
        == pytest.approx((float(row["score_jlu"]) + float(row["score_gau"])) / 2)
        for row in train + validation + test
    )
    assert not (
        {row["plot_group_id"] for row in train}
        & {row["plot_group_id"] for row in validation + test}
    )


def test_manifest_creation_is_reproducible(tmp_path):
    config = _config(tmp_path)
    assert make_manifests(config)["manifest_sha256"] == make_manifests(config)["manifest_sha256"]


def test_config_reuses_exact_previous_best_architecture(tmp_path):
    config = _config(tmp_path)
    reference = config.reference
    assert config.base_checkpoint == reference.base_checkpoint
    assert config.cache_dir == reference.cache_dir
    assert config.patches_per_side == reference.patches_per_side
    assert config.overlap_fraction == reference.overlap_fraction
    assert config.minimum_foreground_fraction == reference.minimum_foreground_fraction
    assert config.hidden_dim == reference.hidden_dim
    assert config.base.model == reference.base.model
    assert config.base.features == reference.base.features
