from pathlib import Path

import pytest

from experiments.dinov3_grid_tiled_mil.config import load_config
from experiments.dinov3_grid_tiled_mil.tiling import make_tile_layout


def test_default_config_is_the_controlled_global_plus_nine_tile_experiment():
    config = load_config("experiments/dinov3_grid_tiled_mil/config.toml")
    assert config.data.grid_crop_size == 1400
    assert config.data.grid_inner_margin_fraction == pytest.approx(0.075)
    assert config.tiles.rows == config.tiles.columns == 3
    assert config.tiles.overlap_fraction == pytest.approx(0.25)
    assert config.tiles.include_global_view
    assert config.data.normalize_targets
    assert "tiled_mil" in config.output.run_dir


def test_three_by_three_layout_covers_1400_with_expected_tiles():
    layout = make_tile_layout(1400, 1400, 3, 3, 0.25)
    assert layout.tile_width == layout.tile_height == 560
    assert layout.boxes.shape == (9, 4)
    assert layout.boxes[0].tolist() == [0, 0, 560, 560]
    assert layout.boxes[-1].tolist() == [840, 840, 1400, 1400]
    assert sorted(set(layout.boxes[:, 0].tolist())) == [0, 420, 840]
    assert sorted(set(layout.boxes[:, 1].tolist())) == [0, 420, 840]


def test_four_by_four_small_regularized_config_and_layout():
    config = load_config("experiments/dinov3_grid_tiled_mil/config_4x4_small_regularized.toml")
    assert config.data.split_seed == 42
    assert config.tiles.rows == config.tiles.columns == 4
    assert config.model.projection_dim == 128
    assert config.model.attention_hidden_dim == 64
    assert config.model.head_hidden_dim == 128
    assert config.model.dropout == pytest.approx(0.35)
    assert config.training.weight_decay == pytest.approx(0.001)
    assert config.output.best_checkpoint_name == "best_mse.pt"
    assert config.output.best_mae_checkpoint_name == "best_mae.pt"
    assert "4x4" in config.features.cache_dir
    assert "4x4" in config.output.run_dir

    layout = make_tile_layout(1400, 1400, 4, 4, 0.25)
    assert layout.tile_width == layout.tile_height == 431
    assert layout.boxes.shape == (16, 4)
    assert layout.boxes[0].tolist() == [0, 0, 431, 431]
    assert layout.boxes[-1].tolist() == [969, 969, 1400, 1400]
    assert sorted(set(layout.boxes[:, 0].tolist())) == [0, 323, 646, 969]
    assert sorted(set(layout.boxes[:, 1].tolist())) == [0, 323, 646, 969]


def test_config_rejects_non_normalized_targets(tmp_path: Path):
    source = Path("experiments/dinov3_grid_tiled_mil/config.toml").read_text()
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace("normalize_targets = true", "normalize_targets = false"))
    with pytest.raises(ValueError, match="normalized targets"):
        load_config(path)


def test_config_rejects_duplicate_checkpoint_names(tmp_path: Path):
    source = Path("experiments/dinov3_grid_tiled_mil/config.toml").read_text()
    path = tmp_path / "duplicate_checkpoints.toml"
    path.write_text(
        source.replace('last_checkpoint_name = "last.pt"', 'last_checkpoint_name = "best.pt"')
    )
    with pytest.raises(ValueError, match="checkpoint names must be distinct"):
        load_config(path)


def test_mil_head_starts_with_uniform_tile_attention():
    torch = pytest.importorskip("torch")
    from experiments.dinov3_grid_tiled_mil.model import GlobalTiledMILRegressor

    config = load_config("experiments/dinov3_grid_tiled_mil/config.toml")
    model = GlobalTiledMILRegressor(32, config).eval()
    predictions, weights = model(torch.randn(4, 10, 32), return_attention=True)
    assert predictions.shape == (4,)
    assert weights.shape == (4, 9)
    assert torch.allclose(weights, torch.full_like(weights, 1 / 9), atol=1e-6)
