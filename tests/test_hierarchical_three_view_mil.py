from pathlib import Path

import numpy as np
import pytest

from experiments.dinov3_hierarchical_three_view_mil.config import load_config
from experiments.dinov3_hierarchical_three_view_mil.features import assign_cells, cell_boxes

WEAK_CONFIG = "experiments/dinov3_hierarchical_three_view_mil/config_weak_then_gold.toml"
GOLD_CONFIG = "experiments/dinov3_hierarchical_three_view_mil/config_gold_only.toml"


def test_control_configs_change_training_mode_but_share_features():
    weak = load_config(WEAK_CONFIG)
    gold = load_config(GOLD_CONFIG)
    assert weak.training.use_weak_pretraining
    assert not gold.training.use_weak_pretraining
    assert weak.features == gold.features
    assert weak.model == gold.model
    assert weak.output.run_dir != gold.output.run_dir
    assert weak.manifest_path("finetune") == Path("outputs/dataset_manifests/finetune.csv")


def test_cell_layout_and_assignment_are_row_major():
    boxes = cell_boxes(1400, 1400, 0.02)
    assert boxes.shape == (4, 4)
    plants = np.asarray([[100, 100, 200, 200], [800, 100, 900, 200],
                         [100, 800, 200, 900], [800, 800, 900, 900]])
    assert assign_cells(plants, 1400, 1400).tolist() == [0, 1, 2, 3]


def test_hierarchical_model_shapes_and_attention():
    torch = pytest.importorskip("torch")
    from experiments.dinov3_hierarchical_three_view_mil.model import (
        HierarchicalThreeViewRegressor,
    )

    config = load_config(WEAK_CONFIG)
    model = HierarchicalThreeViewRegressor(24, config).eval()
    valid = torch.tensor([[True, True, False], [True, True, True]])
    cell_indices = torch.tensor([[0, 3, -1], [0, 1, 1]])
    prediction, attention = model(
        torch.randn(2, 24),
        torch.randn(2, 4, 24),
        torch.randn(2, 3, 24),
        valid,
        cell_indices,
        return_attention=True,
    )
    assert prediction.shape == (2,)
    assert attention["cell_weights"].shape == (2, 4)
    assert attention["plant_weights"].shape == (2, 3)
    assert torch.allclose(attention["cell_weights"].sum(1), torch.ones(2))
    assert torch.allclose(attention["plant_weights"].sum(1), torch.ones(2))
    assert torch.all(attention["plant_weights"][~valid] == 0)


def test_sam_inference_is_bounded_and_mask_is_restored(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    from PIL import Image

    from experiments.dinov3_hierarchical_three_view_mil import prepare_features

    observed = []

    def fake_generate_mask(segmenter, image, config):
        observed.append(image.size)
        return np.ones((image.height, image.width), dtype=bool)

    monkeypatch.setattr(prepare_features, "generate_mask", fake_generate_mask)
    config = load_config(WEAK_CONFIG)
    image = Image.new("RGB", (4000, 2000))
    mask, limit, retries = prepare_features.generate_memory_bounded_mask(
        object(), image, config
    )
    assert observed == [(1400, 700)]
    assert mask.shape == (2000, 4000)
    assert limit == 1400
    assert retries == 0


def test_three_view_feature_cache_round_trip(tmp_path):
    from experiments.dinov3_hierarchical_three_view_mil.features import (
        load_record,
        save_record,
    )

    destination = tmp_path / "features.npz"
    save_record(
        destination,
        global_feature=np.arange(6, dtype=np.float32),
        cell_features=np.ones((4, 6), dtype=np.float32),
        cell_boxes=np.asarray(
            [[0, 0, 10, 10], [10, 0, 20, 10], [0, 10, 10, 20], [10, 10, 20, 20]]
        ),
        plant_features=np.ones((2, 6), dtype=np.float32),
        plant_boxes=np.asarray([[1, 1, 5, 5], [12, 12, 18, 18]]),
        plant_cell_indices=np.asarray([0, 3]),
        foreground_pixels=np.asarray([8, 12]),
        mask_coverage=1.0,
        processed_image_path="processed.jpg",
        mask_path="mask.png",
        mode="grid_crop_inset075",
        identity="test-identity",
    )
    record = load_record(destination, expected_identity="test-identity")
    assert record["cell_boxes"].shape == (4, 4)
    assert record["plant_features"].shape == (2, 6)
    assert record["plant_cell_indices"].tolist() == [0, 3]
