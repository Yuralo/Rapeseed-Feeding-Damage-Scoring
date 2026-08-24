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


def test_config_rejects_non_normalized_targets(tmp_path: Path):
    source = Path("experiments/dinov3_grid_tiled_mil/config.toml").read_text()
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace("normalize_targets = true", "normalize_targets = false"))
    with pytest.raises(ValueError, match="normalized targets"):
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
