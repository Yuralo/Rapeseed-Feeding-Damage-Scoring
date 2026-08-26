import numpy as np
import pytest

from experiments.dinov3_grid_sam_adaptive_mil.config import AdaptiveCropSettings, load_config
from experiments.dinov3_grid_sam_adaptive_mil.crops import make_adaptive_crop_layout
from experiments.dinov3_grid_sam_adaptive_mil.metrics import Predictions
from experiments.dinov3_grid_sam_adaptive_mil.reporting import (
    error_analysis,
    target_range_metrics,
)


def _predictions():
    return Predictions(
        targets=np.asarray([1.0, 5.0, 10.0, 20.0]),
        predictions=np.asarray([2.0, 4.0, 12.0, 16.0]),
        filenames=[f"sample_{index}.jpg" for index in range(4)],
        source_image_paths=[""] * 4,
        processed_image_paths=[""] * 4,
        mask_paths=[""] * 4,
        context_feature_cache_paths=[""] * 4,
        adaptive_feature_cache_paths=[""] * 4,
        objective_mse=0.5,
        weights=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.2, 0.3, 0.5],
                [0.8, 0.2, 0.0],
            ]
        ),
        valid=np.asarray(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
                [True, True, False],
            ]
        ),
        boxes=np.zeros((4, 3, 4), dtype=np.int32),
        foreground_pixels=np.ones((4, 3), dtype=np.float32),
        instance_counts=np.asarray([1, 2, 3, 2]),
        mask_coverages=np.ones(4),
    )


def test_config_reuses_sam_and_three_by_three_caches():
    config = load_config("experiments/dinov3_grid_sam_adaptive_mil/config.toml")
    assert config.data.split_seed == 42
    assert config.segmentation.mask_cache_dir == "cache/sam3_masks_grid1400_inset075"
    assert config.context.cache_dir == "cache/dinov3_grid_tiled_mil_features"
    assert config.adaptive_crops.maximum_instances == 20


def test_adaptive_layout_groups_nearby_leaves_and_covers_foreground():
    mask = np.zeros((400, 400), dtype=bool)
    mask[80:100, 80:100] = True
    mask[110:130, 105:125] = True
    mask[280:310, 290:320] = True
    settings = AdaptiveCropSettings(
        grouping_dilation_px=20,
        minimum_crop_size=100,
        maximum_crop_size=200,
        context_scale=1.5,
        maximum_instances=10,
        minimum_mask_coverage=1.0,
    )
    layout = make_adaptive_crop_layout(mask, settings)
    assert len(layout.boxes) == 2
    assert layout.mask_coverage == pytest.approx(1.0)
    assert (
        layout.boxes[:, 2] - layout.boxes[:, 0] == layout.boxes[:, 3] - layout.boxes[:, 1]
    ).all()


def test_instance_cap_merges_without_dropping_coverage():
    mask = np.zeros((500, 500), dtype=bool)
    for index in range(12):
        y = 20 + (index // 4) * 150
        x = 20 + (index % 4) * 120
        mask[y : y + 10, x : x + 10] = True
    settings = AdaptiveCropSettings(
        grouping_dilation_px=0,
        minimum_crop_size=40,
        maximum_crop_size=300,
        context_scale=1,
        maximum_instances=5,
        minimum_mask_coverage=1.0,
    )
    layout = make_adaptive_crop_layout(mask, settings)
    assert len(layout.boxes) == 5
    assert layout.component_count_before_merge == 12
    assert layout.mask_coverage == pytest.approx(1.0)


def test_model_masks_padded_instances():
    torch = pytest.importorskip("torch")
    from experiments.dinov3_grid_sam_adaptive_mil.model import SamAdaptiveMILRegressor

    config = load_config("experiments/dinov3_grid_sam_adaptive_mil/config.toml")
    model = SamAdaptiveMILRegressor(32, config).eval()
    context = torch.randn(2, 10, 32)
    instances = torch.randn(2, 20, 32)
    valid = torch.zeros(2, 20, dtype=torch.bool)
    valid[0, :3] = True
    valid[1, :7] = True
    prediction, weights = model(context, instances, valid, return_attention=True)
    assert prediction.shape == (2,)
    assert torch.all(weights[~valid] == 0)
    assert torch.allclose(weights.sum(1), torch.ones(2))


def test_attention_statistics_handle_variable_instance_counts():
    statistics = _predictions().attention_statistics()
    assert statistics["normalized_entropy"][0] == pytest.approx(0.0)
    assert statistics["normalized_entropy"][1] == pytest.approx(1.0)
    assert statistics["top_index"].tolist() == [0, 0, 2, 0]
    assert statistics["effective_instance_count"][1] == pytest.approx(2.0)


def test_error_report_has_target_ranges_and_worst_filename():
    predictions = _predictions()
    ranges = target_range_metrics(predictions)
    assert all(value["samples"] == 1 for value in ranges.values())
    analysis = error_analysis(predictions)
    assert analysis["worst_sample"]["filename"] == "sample_3.jpg"
    assert analysis["worst_sample"]["absolute_error"] == pytest.approx(4.0)
