from pathlib import Path

import pytest

from experiments.dinov3_grid_4x4_sam_adaptive_hybrid_mil.config import load_config
from experiments.dinov3_grid_multiscale_tiled_mil.config import (
    load_config as load_multiscale_config,
)
from experiments.dinov3_grid_sam_adaptive_mil.config import load_config as load_adaptive_config
from experiments.dinov3_grid_sam_adaptive_mil.features import (
    cache_identity as adaptive_identity,
)
from experiments.dinov3_grid_tiled_mil.features import cache_identity

HYBRID_CONFIG = "experiments/dinov3_grid_4x4_sam_adaptive_hybrid_mil/config.toml"


def test_hybrid_config_reuses_all_selected_feature_caches(tmp_path):
    hybrid = load_config(HYBRID_CONFIG)
    adaptive = load_adaptive_config("experiments/dinov3_grid_sam_adaptive_mil/config.toml")
    multiscale = load_multiscale_config(
        "experiments/dinov3_grid_multiscale_tiled_mil/config.toml"
    )
    source = tmp_path / "sample.jpg"
    source.write_bytes(b"cache identity fixture")
    filename = "sample"
    assert cache_identity(hybrid.context_config(), filename, source) == cache_identity(
        adaptive.context_config(), filename, source
    )
    assert cache_identity(hybrid.fine_config(), filename, source) == cache_identity(
        multiscale.single_scale_config(multiscale.fine), filename, source
    )
    assert adaptive_identity(hybrid, filename, source) == adaptive_identity(
        adaptive, filename, source
    )
    assert hybrid.features.cache_dir == "cache/dinov3_grid_sam_adaptive_mil_features"
    assert hybrid.training.seed == 42


def test_hybrid_model_returns_normalized_masked_attention():
    torch = pytest.importorskip("torch")
    from experiments.dinov3_grid_4x4_sam_adaptive_hybrid_mil.model import HybridMILRegressor

    config = load_config(HYBRID_CONFIG)
    model = HybridMILRegressor(32, config).eval()
    context = torch.randn(2, 10, 32)
    fine = torch.randn(2, 17, 32)
    instances = torch.randn(2, 20, 32)
    valid = torch.zeros(2, 20, dtype=torch.bool)
    valid[0, :3] = True
    valid[1, :7] = True
    prediction, fine_weights, plant_weights = model(
        context, fine, instances, valid, return_attention=True
    )
    assert prediction.shape == (2,)
    assert fine_weights.shape == (2, 16)
    assert plant_weights.shape == (2, 20)
    assert torch.allclose(fine_weights.sum(1), torch.ones(2))
    assert torch.allclose(plant_weights.sum(1), torch.ones(2))
    assert torch.all(plant_weights[~valid] == 0)


def test_hybrid_model_rejects_wrong_view_count():
    torch = pytest.importorskip("torch")
    from experiments.dinov3_grid_4x4_sam_adaptive_hybrid_mil.model import HybridMILRegressor

    model = HybridMILRegressor(16, load_config(HYBRID_CONFIG))
    with pytest.raises(ValueError, match="4x4"):
        model(
            torch.randn(1, 10, 16),
            torch.randn(1, 16, 16),
            torch.randn(1, 20, 16),
            torch.ones(1, 20, dtype=torch.bool),
        )


def test_checkpoint_validation_rejects_other_experiments():
    from experiments.dinov3_grid_4x4_sam_adaptive_hybrid_mil.checkpoint import validate_for

    with pytest.raises(ValueError, match="Incompatible"):
        validate_for({"experiment": "something_else"}, load_config(HYBRID_CONFIG))


def test_default_output_is_separate_from_source_experiments():
    output = Path(load_config(HYBRID_CONFIG).output.run_dir)
    assert output.name == "dinov3_grid_4x4_sam_adaptive_hybrid_mil_clean_inset075"
