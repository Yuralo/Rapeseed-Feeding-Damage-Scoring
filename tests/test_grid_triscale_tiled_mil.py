from pathlib import Path

import pytest

from experiments.dinov3_grid_triscale_tiled_mil.config import load_config

CONFIG_PATH = "experiments/dinov3_grid_triscale_tiled_mil/config.toml"


def test_triscale_config_matches_the_three_existing_caches():
    config = load_config(CONFIG_PATH)
    assert config.data.split_seed == 42
    assert (config.context.rows, config.context.columns) == (3, 3)
    assert (config.regional.rows, config.regional.columns) == (4, 4)
    assert (config.local.rows, config.local.columns) == (5, 5)
    assert config.context.cache_dir == "cache/dinov3_grid_tiled_mil_features"
    assert config.regional.cache_dir == "cache/dinov3_grid_tiled_mil_features_4x4"
    assert config.local.cache_dir == "cache/dinov3_grid_tiled_mil_features_5x5"
    assert config.model.projection_dim == 128
    assert config.model.attention_hidden_dim == 64
    assert config.model.head_hidden_dim == 128
    assert config.output.best_checkpoint_name == "best_mse.pt"
    assert config.output.best_mae_checkpoint_name == "best_mae.pt"


def test_scale_adapters_preserve_cache_identity_settings():
    config = load_config(CONFIG_PATH)
    adapters = [config.single_scale_config(scale) for _, scale in config.scales]
    assert [(item.tiles.rows, item.tiles.columns) for item in adapters] == [
        (3, 3),
        (4, 4),
        (5, 5),
    ]
    assert [item.features.cache_dir for item in adapters] == [
        config.context.cache_dir,
        config.regional.cache_dir,
        config.local.cache_dir,
    ]
    assert len({item.features.backbone for item in adapters}) == 1
    assert len({item.features.processor for item in adapters}) == 1


def test_triscale_config_rejects_non_increasing_scales(tmp_path: Path):
    source = Path(CONFIG_PATH).read_text()
    path = tmp_path / "invalid.toml"
    local_section = source.index("[local]")
    source = source[:local_section] + source[local_section:].replace("rows = 5", "rows = 3", 1)
    path.write_text(source)
    with pytest.raises(ValueError, match="must strictly increase"):
        load_config(path)


def test_triscale_head_shapes_and_uniform_initial_attention():
    torch = pytest.importorskip("torch")
    from experiments.dinov3_grid_triscale_tiled_mil.model import (
        TriScaleTiledMILRegressor,
    )

    config = load_config(CONFIG_PATH)
    model = TriScaleTiledMILRegressor(32, config).eval()
    prediction, regional, local = model(
        torch.randn(4, 10, 32),
        torch.randn(4, 17, 32),
        torch.randn(4, 26, 32),
        return_attention=True,
    )
    assert prediction.shape == (4,)
    assert regional.shape == (4, 16)
    assert local.shape == (4, 25)
    assert torch.allclose(regional, torch.full_like(regional, 1 / 16), atol=1e-6)
    assert torch.allclose(local, torch.full_like(local, 1 / 25), atol=1e-6)
    summary = model.parameter_summary()
    assert summary["context_tiles_per_image"] == 9
    assert summary["regional_tiles_per_image"] == 16
    assert summary["local_tiles_per_image"] == 25
    assert "3x3_mean" in summary["pooled_representations"]
