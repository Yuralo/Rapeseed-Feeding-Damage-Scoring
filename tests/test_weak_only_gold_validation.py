from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.dinov3_weak_only_gold_validation.build_manifests import run
from experiments.dinov3_weak_only_gold_validation.config import load_config

CONFIG = "experiments/dinov3_weak_only_gold_validation/config_all_weak.toml"
FIELDS = [
    "image_id", "absolute_path", "relative_path", "sha256", "cohort_id",
    "plot_group_id", "is_gold_standard", "supervision_tier", "target",
    "score_single", "score_jlu", "score_gau", "sample_weight", "split",
]


def _row(name, group, *, gold=False, single=None, jlu=None, gau=None, sha=None):
    scores = [score for score in (single, jlu, gau) if score is not None]
    target = sum(scores) / len(scores)
    return {
        "image_id": name, "absolute_path": f"/images/{name}.jpg",
        "relative_path": f"folder/{name}.jpg", "sha256": sha or f"hash_{name}",
        "cohort_id": "gg" if gold or jlu is not None else "wg",
        "plot_group_id": group, "is_gold_standard": str(gold),
        "supervision_tier": "gold" if gold else ("dual_weak" if jlu is not None else "single_weak"),
        "target": str(target), "score_single": "" if single is None else str(single),
        "score_jlu": "" if jlu is None else str(jlu),
        "score_gau": "" if gau is None else str(gau),
        "sample_weight": "1", "split": "old_split",
    }


def _fixture(tmp_path):
    scored = tmp_path / "scored.csv"
    rows = [
        _row("gold_a", "plot_a", gold=True, jlu=4, gau=6),
        _row("gold_b", "plot_b", gold=True, jlu=10, gau=12),
        _row("weak_overlap", "plot_a", jlu=2, gau=20),
        _row("weak_dual", "plot_c", jlu=10, gau=20),
        _row("weak_single", "plot_d", single=4),
        _row("weak_duplicate", "plot_e", single=2, sha="hash_gold_b"),
    ]
    with scored.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return replace(load_config(CONFIG), scored_manifest=str(scored),
                   manifest_dir=str(tmp_path / "manifests"), run_dir=str(tmp_path / "run"),
                   expected_gold_images=2)


def test_all_weak_uses_mean_or_single_and_excludes_gold_plots(tmp_path):
    config = _fixture(tmp_path)
    summary = run(config)
    assert summary["weak_train_images"] == 2
    assert summary["gold_validation_images"] == 2
    assert summary["gold_training_images"] == 0
    assert summary["exclusion_reasons"] == {"same_plot_as_gold": 1, "duplicate_of_gold": 1}
    with (tmp_path / "manifests" / "weak_train.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["image_id"]: float(row["target"]) for row in rows} == {
        "weak_dual": 15.0, "weak_single": 4.0,
    }
    assert {row["target_source"] for row in rows} == {
        "mean_of_two_scorers", "single_scorer",
    }


def test_dual_only_is_an_explicit_small_control(tmp_path):
    config = replace(_fixture(tmp_path), mode="dual_only")
    summary = run(config)
    assert summary["weak_train_images"] == 1
    assert summary["weak_train_target_sources"] == {"mean_of_two_scorers": 1}
    assert summary["exclusion_reasons"]["not_dual_scored"] == 1


def test_gold_count_is_strict(tmp_path):
    config = replace(_fixture(tmp_path), expected_gold_images=470)
    with pytest.raises(ValueError, match="Expected 470 gold"):
        run(config)


def test_end_to_end_head_training_on_synthetic_cache(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    import numpy as np
    from PIL import Image

    from experiments.dinov3_hierarchical_three_view_mil.features import (
        cache_identity,
        feature_cache_path,
        save_record,
    )
    from experiments.dinov3_weak_only_gold_validation.train import run as train

    config = _fixture(tmp_path)
    scored = tmp_path / "scored.csv"
    with scored.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    for row in source_rows:
        source = tmp_path / "images" / f"{row['image_id']}.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), "green").save(source)
        row["absolute_path"] = str(source)
    with scored.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(source_rows)
    original_base = Path(
        "experiments/dinov3_hierarchical_three_view_mil/config_gold_only.toml"
    ).read_text(encoding="utf-8")
    cache = tmp_path / "feature_cache"
    base_text = original_base.replace(
        'cache_dir = "cache/dinov3_hierarchical_three_view_features_adapted_routed"',
        f'cache_dir = "{cache}"',
    ).replace('device = "auto"', 'device = "cpu"').replace("save_plots = true", "save_plots = false")
    base_path = tmp_path / "base.toml"
    base_path.write_text(base_text, encoding="utf-8")
    config = replace(config, base_config_path=str(base_path), epochs=1, batch_size=2,
                     num_workers=0, early_stopping_patience=1)
    run(config)
    for index, row in enumerate(source_rows):
        source = tmp_path / "images" / f"{row['image_id']}.jpg"
        mask = tmp_path / "images" / f"{row['image_id']}.png"
        Image.new("L", (100, 100), 255).save(mask)
        relative = row["relative_path"]
        identity = cache_identity(config.routed_base, relative, source)
        rng = np.random.default_rng(index)
        save_record(
            feature_cache_path(config.routed_base, relative, source),
            global_feature=rng.normal(size=24).astype(np.float32),
            cell_features=rng.normal(size=(4, 24)).astype(np.float32),
            cell_boxes=np.asarray([[0, 0, 50, 50], [50, 0, 100, 50],
                                   [0, 50, 50, 100], [50, 50, 100, 100]]),
            plant_features=rng.normal(size=(2, 24)).astype(np.float32),
            plant_boxes=np.asarray([[10, 10, 30, 30], [60, 60, 80, 80]]),
            plant_cell_indices=np.asarray([0, 3]),
            foreground_pixels=np.asarray([100, 100]),
            mask_coverage=0.1,
            processed_image_path=str(source),
            mask_path=str(mask),
            mode="grid_crop_inset075",
            identity=identity,
        )
    summary = train(config)
    assert summary["weak_train_images"] == 2
    assert summary["gold_validation_images"] == 2
    assert summary["gold_training_images"] == 0
    assert (tmp_path / "run" / "best_mae.pt").is_file()
    assert (tmp_path / "run" / "predictions.csv").is_file()
