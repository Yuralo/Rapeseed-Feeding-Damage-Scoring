from pathlib import Path

import pytest

from experiments.dinov3_grid_multicohort_multiscale_mil.config import load_config

CONFIG = "experiments/dinov3_grid_multicohort_multiscale_mil/config.toml"


def test_multicohort_config_preserves_proven_architecture_and_gold_stage():
    config = load_config(CONFIG)
    assert (config.coarse.rows, config.coarse.columns) == (3, 3)
    assert (config.fine.rows, config.fine.columns) == (4, 4)
    assert config.model.projection_dim == 128
    assert config.training.pretraining.learning_rate == pytest.approx(0.001)
    assert config.training.finetuning.learning_rate == pytest.approx(0.0003)
    assert config.manifest_path("finetune") == Path("outputs/dataset_manifests/finetune.csv")


def test_single_scale_adapters_use_distinct_multicohort_caches():
    config = load_config(CONFIG)
    coarse = config.single_scale_config(config.coarse)
    fine = config.single_scale_config(config.fine)
    assert coarse.features.cache_dir != fine.features.cache_dir
    assert (coarse.tiles.rows, coarse.tiles.columns) == (3, 3)
    assert (fine.tiles.rows, fine.tiles.columns) == (4, 4)
    assert coarse.data.grid_inner_margin_fraction == pytest.approx(0.075)


def test_multicohort_model_shape_is_unchanged():
    torch = pytest.importorskip("torch")
    from experiments.dinov3_grid_multicohort_multiscale_mil.model import (
        MultiScaleTiledMILRegressor,
    )

    config = load_config(CONFIG)
    model = MultiScaleTiledMILRegressor(24, config).eval()
    prediction, coarse, fine = model(
        torch.randn(3, 10, 24), torch.randn(3, 17, 24), return_attention=True
    )
    assert prediction.shape == (3,)
    assert coarse.shape == (3, 9)
    assert fine.shape == (3, 16)
