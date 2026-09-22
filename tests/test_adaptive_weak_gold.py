from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.dinov3_adaptive_weak_gold.config import load_config
from experiments.dinov3_adaptive_weak_gold.features import extract

CONFIG = "experiments/dinov3_adaptive_weak_gold/config.toml"
FIELDS = [
    "image_id", "absolute_path", "relative_path", "sha256", "cohort_id",
    "plot_group_id", "is_gold_standard", "supervision_tier", "target",
    "score_single", "score_jlu", "score_gau", "sample_weight", "split",
]


def test_config_reuses_exact_adaptive_context_and_backbone():
    config = load_config(CONFIG)
    assert config.adaptive.context.rows == config.adaptive.context.columns == 3
    assert config.adaptive.features.backbone == config.weak.routed_base.features.backbone
    assert config.adaptive.model.dropout == 0.35


def test_context_tiles_use_exif_oriented_raw_image(tmp_path):
    class Extractor:
        def extract(self, views):
            assert len(views) == 9
            return np.ones((9, 8), dtype=np.float32)

    image_path = tmp_path / "raw.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (80, 120), "green").save(image_path, exif=exif)
    record = {"processed_image_path": str(image_path),
              "global_feature": np.zeros(8, dtype=np.float32)}
    result = extract(Extractor(), load_config(CONFIG), record)
    assert result["tile_features"].shape == (9, 8)
    assert result["tile_boxes"][:, 2].max() == 120
    assert result["tile_boxes"][:, 3].max() == 80


def test_synthetic_adaptive_training_and_gold_only_reevaluation(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    from experiments.dinov3_adaptive_weak_gold.evaluate import run as evaluate
    from experiments.dinov3_adaptive_weak_gold.features import (
        cache_path as context_path,
    )
    from experiments.dinov3_adaptive_weak_gold.features import (
        identity as context_identity,
    )
    from experiments.dinov3_adaptive_weak_gold.features import (
        save as save_context,
    )
    from experiments.dinov3_adaptive_weak_gold.train import run as train
    from experiments.dinov3_hierarchical_three_view_mil.features import (
        cache_identity as base_identity,
    )
    from experiments.dinov3_hierarchical_three_view_mil.features import (
        feature_cache_path as base_path,
    )
    from experiments.dinov3_hierarchical_three_view_mil.features import (
        save_record as save_base,
    )
    from experiments.dinov3_weak_only_gold_validation.build_manifests import run as build
    from experiments.dinov3_weak_only_gold_validation.config import load_config as load_weak

    rows = []
    for name, group, gold, score in (
        ("gold_a", "plot_a", True, 5.0),
        ("gold_b", "plot_b", True, 11.0),
        ("weak_a", "plot_c", False, 15.0),
        ("weak_b", "plot_d", False, 4.0),
    ):
        source = tmp_path / "images" / f"{name}.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), "green").save(source)
        rows.append({
            "image_id": name, "absolute_path": str(source),
            "relative_path": f"folder/{name}.jpg", "sha256": f"hash_{name}",
            "cohort_id": "gg", "plot_group_id": group,
            "is_gold_standard": str(gold),
            "supervision_tier": "gold" if gold else "single_weak",
            "target": str(score), "score_single": "" if gold else str(score),
            "score_jlu": str(score - 1) if gold else "",
            "score_gau": str(score + 1) if gold else "",
            "sample_weight": "1", "split": "old",
        })
    scored = tmp_path / "scored.csv"
    with scored.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    base_text = Path(
        "experiments/dinov3_hierarchical_three_view_mil/config_gold_only.toml"
    ).read_text(encoding="utf-8")
    base_text = base_text.replace(
        'cache_dir = "cache/dinov3_hierarchical_three_view_features_adapted_routed"',
        f'cache_dir = "{tmp_path / "base_cache"}"',
    ).replace('device = "auto"', 'device = "cpu"').replace(
        "save_plots = true", "save_plots = false"
    )
    base_config = tmp_path / "base.toml"
    base_config.write_text(base_text, encoding="utf-8")
    weak_text = Path(
        "experiments/dinov3_weak_only_gold_validation/config_all_weak.toml"
    ).read_text(encoding="utf-8")
    weak_text = weak_text.replace(
        'base_config_path = "experiments/dinov3_hierarchical_three_view_mil/config_gold_only.toml"',
        f'base_config_path = "{base_config}"',
    ).replace(
        'scored_manifest = "outputs/dataset_manifests/scored_manifest.csv"',
        f'scored_manifest = "{scored}"',
    ).replace(
        'manifest_dir = "outputs/dinov3_weak_only_gold_validation_all_weak/manifests"',
        f'manifest_dir = "{tmp_path / "manifests"}"',
    ).replace("expected_gold_images = 470", "expected_gold_images = 2")
    weak_path = tmp_path / "weak.toml"
    weak_path.write_text(weak_text, encoding="utf-8")
    adaptive_text = Path(
        "experiments/dinov3_grid_sam_adaptive_mil/config_adapted_routed.toml"
    ).read_text(encoding="utf-8")
    adaptive_text = adaptive_text.replace('device = "auto"', 'device = "cpu"').replace(
        "save_plots = true", "save_plots = false"
    )
    adaptive_path = tmp_path / "adaptive.toml"
    adaptive_path.write_text(adaptive_text, encoding="utf-8")
    config = replace(
        load_config(CONFIG), weak_config_path=str(weak_path),
        adaptive_reference_config_path=str(adaptive_path),
        run_dir=str(tmp_path / "run"), context_cache_dir=str(tmp_path / "context_cache"),
        epochs=1, batch_size=2, num_workers=0,
    )
    config.validate()
    build(load_weak(weak_path))
    for index, row in enumerate(rows):
        source = Path(row["absolute_path"])
        mask_path = source.with_suffix(".png")
        Image.new("L", (100, 100), 255).save(mask_path)
        relative = row["relative_path"]
        key = base_identity(config.weak.routed_base, relative, source)
        rng = np.random.default_rng(index)
        save_base(
            base_path(config.weak.routed_base, relative, source),
            global_feature=rng.normal(size=24).astype(np.float32),
            cell_features=rng.normal(size=(4, 24)).astype(np.float32),
            cell_boxes=np.asarray([[0, 0, 50, 50], [50, 0, 100, 50],
                                   [0, 50, 50, 100], [50, 50, 100, 100]]),
            plant_features=rng.normal(size=(2, 24)).astype(np.float32),
            plant_boxes=np.asarray([[10, 10, 30, 30], [60, 60, 80, 80]]),
            plant_cell_indices=np.asarray([0, 3]),
            foreground_pixels=np.asarray([100, 100]),
            mask_coverage=1.0,
            processed_image_path=str(source), mask_path=str(mask_path),
            mode="grid_crop_inset075", identity=key,
        )
        from experiments.dinov3_grid_tiled_mil.tiling import make_tile_layout

        save_context(
            context_path(config, relative, source),
            {"tile_features": rng.normal(size=(9, 24)).astype(np.float32),
             "tile_boxes": make_tile_layout(100, 100, 3, 3, 0.25).boxes,
             "processed_image_path": str(source)},
            context_identity(config, relative, source),
        )
    summary = train(config)
    assert summary["actual_weak_training_images"] == 2
    assert summary["gold_validation_images"] == 2
    assert summary["gold_training_images"] == 0
    assert (tmp_path / "run" / "predictions.csv").is_file()
    assert (tmp_path / "run" / "best_mae.pt").is_file()
    # A saved model needs only the gold caches, not weak training features.
    for row in rows[2:]:
        context_path(config, row["relative_path"], Path(row["absolute_path"])).unlink()
    report = evaluate(config, tmp_path / "run" / "best_mae.pt")
    assert report["samples"] == 2
