from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from experiments.dinov3_plant_damage_mil.config import load_config
from experiments.dinov3_plant_damage_mil.features import extract, load, patch_boxes, save

CONFIG = "experiments/dinov3_plant_damage_mil/config.toml"


def test_config_and_patch_boxes():
    config = load_config(CONFIG)
    assert not config.base.training.use_weak_pretraining
    boxes = patch_boxes(np.asarray([10, 20, 110, 220]), 2, 0.15)
    assert boxes.shape == (4, 4)
    assert boxes.min() >= 10
    assert boxes[:, 2].max() <= 110
    assert boxes[:, 3].max() <= 220


def test_local_patch_extraction_and_cache(tmp_path):
    class Extractor:
        def extract(self, views):
            return np.arange(len(views) * 8, dtype=np.float32).reshape(len(views), 8)

    image_path = tmp_path / "processed.jpg"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (100, 100), "green").save(image_path)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    Image.fromarray(mask).save(mask_path)
    record = {"processed_image_path": str(image_path), "mask_path": str(mask_path),
              "plant_boxes": np.asarray([[0, 0, 50, 50]])}
    config = load_config(CONFIG)
    result = extract(Extractor(), config, record)
    assert result["patch_features"].shape == (1, 4, 8)
    assert result["patch_valid"].any()
    path = tmp_path / "patch.npz"
    save(path, result, "synthetic")
    restored = load(path, "synthetic")
    np.testing.assert_allclose(restored["patch_features"], result["patch_features"])
    with pytest.raises(ValueError, match="Stale"):
        load(path, "other")


def test_residual_model_starts_at_baseline_and_receives_gradients():
    torch = pytest.importorskip("torch")
    from experiments.dinov3_hierarchical_three_view_mil.model import HierarchicalThreeViewRegressor
    from experiments.dinov3_plant_damage_mil.model import PlantDamageRegressor

    config = replace(load_config(CONFIG), hidden_dim=16)
    base = HierarchicalThreeViewRegressor(24, config.base)
    model = PlantDamageRegressor(24, config, base.state_dict())
    batch = {
        "global_feature": torch.randn(2, 24),
        "cell_features": torch.randn(2, 4, 24),
        "plant_features": torch.randn(2, 3, 24),
        "plant_valid": torch.tensor([[True, True, False], [True, True, True]]),
        "plant_cell_indices": torch.tensor([[0, 1, -1], [0, 2, 3]]),
        "patch_features": torch.randn(2, 3, 4, 24),
        "patch_foreground_fraction": torch.ones(2, 3, 4) * 0.3,
        "patch_valid": torch.tensor([[[True] * 4, [True] * 4, [False] * 4],
                                      [[True] * 4, [True] * 4, [True] * 4]]),
    }
    prediction, diagnostics = model(batch, return_attention=True)
    assert prediction.shape == (2,)
    assert torch.allclose(prediction, diagnostics["base_prediction"])
    assert torch.allclose(diagnostics["patch_weights"][0, 0].sum(), torch.tensor(1.0))
    assert diagnostics["patch_weights"][0, 2].sum() == 0
    model.train()
    assert not model.base.training
    prediction.sum().backward()
    assert model.residual_head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.base.parameters())
