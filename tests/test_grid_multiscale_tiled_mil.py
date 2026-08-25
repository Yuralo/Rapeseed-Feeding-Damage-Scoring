from pathlib import Path

import pytest

from experiments.dinov3_grid_multiscale_tiled_mil.config import load_config

CONFIG_PATH = "experiments/dinov3_grid_multiscale_tiled_mil/config.toml"


def test_multiscale_config_matches_existing_seed_42_caches():
    config = load_config(CONFIG_PATH)
    assert config.data.split_seed == 42
    assert (config.coarse.rows, config.coarse.columns) == (3, 3)
    assert (config.fine.rows, config.fine.columns) == (4, 4)
    assert config.coarse.overlap_fraction == pytest.approx(0.25)
    assert config.fine.overlap_fraction == pytest.approx(0.25)
    assert config.coarse.cache_dir == "cache/dinov3_grid_tiled_mil_features"
    assert config.fine.cache_dir == "cache/dinov3_grid_tiled_mil_features_4x4"
    assert config.model.projection_dim == 128
    assert config.model.attention_hidden_dim == 64
    assert config.model.head_hidden_dim == 128
    assert config.output.best_checkpoint_name == "best_mse.pt"
    assert config.output.best_mae_checkpoint_name == "best_mae.pt"


def test_scale_adapters_preserve_cache_identity_settings():
    config = load_config(CONFIG_PATH)
    coarse = config.single_scale_config(config.coarse)
    fine = config.single_scale_config(config.fine)
    assert (coarse.tiles.rows, coarse.tiles.columns) == (3, 3)
    assert (fine.tiles.rows, fine.tiles.columns) == (4, 4)
    assert coarse.features.cache_dir == config.coarse.cache_dir
    assert fine.features.cache_dir == config.fine.cache_dir
    assert coarse.features.backbone == fine.features.backbone == config.features.backbone
    assert coarse.features.processor == fine.features.processor == config.features.processor


def test_multiscale_config_rejects_reversed_scales(tmp_path: Path):
    source = Path(CONFIG_PATH).read_text()
    path = tmp_path / "invalid.toml"
    source = source.replace("rows = 3\ncolumns = 3", "rows = 5\ncolumns = 5", 1)
    path.write_text(source)
    with pytest.raises(ValueError, match="coarse must contain fewer tiles"):
        load_config(path)


def test_multiscale_head_shapes_and_uniform_initial_attention():
    torch = pytest.importorskip("torch")
    from experiments.dinov3_grid_multiscale_tiled_mil.model import (
        MultiScaleTiledMILRegressor,
    )

    config = load_config(CONFIG_PATH)
    model = MultiScaleTiledMILRegressor(32, config).eval()
    predictions, coarse, fine = model(
        torch.randn(4, 10, 32), torch.randn(4, 17, 32), return_attention=True
    )
    assert predictions.shape == (4,)
    assert coarse.shape == (4, 9)
    assert fine.shape == (4, 16)
    assert torch.allclose(coarse, torch.full_like(coarse, 1 / 9), atol=1e-6)
    assert torch.allclose(fine, torch.full_like(fine, 1 / 16), atol=1e-6)
    summary = model.parameter_summary()
    assert summary["coarse_tiles_per_image"] == 9
    assert summary["fine_tiles_per_image"] == 16
