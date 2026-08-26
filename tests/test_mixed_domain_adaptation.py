from pathlib import Path
from random import Random

import pytest

from experiments.dinov3_mixed_domain_adaptation.config import load_config

CONFIG_PATH = "experiments/dinov3_mixed_domain_adaptation/config.toml"


def test_mixed_adaptation_config_preserves_explicit_routing_and_3090_defaults():
    config = load_config(CONFIG_PATH)
    assert config.data.grid_inner_margin_fraction == pytest.approx(0.075)
    assert config.data.grid_crop_size == 1400
    assert config.training.batch_size == 8
    assert config.training.gradient_accumulation_steps == 2
    assert config.model.lora_rank == 8
    assert tuple(config.model.lora_target_modules) == ("q_proj", "v_proj")


def test_filename_routing_never_falls_back():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from experiments.dinov3_mixed_domain_adaptation.preprocessing import (
        GRID_CROP_MODE,
        RAW_MODE,
        preprocessing_mode,
    )

    config = load_config(CONFIG_PATH)
    assert preprocessing_mode("20251021_153843.jpg", config) == GRID_CROP_MODE
    assert preprocessing_mode("IMG_2533.JPG", config) == RAW_MODE
    with pytest.raises(ValueError, match="Unsupported adaptation filename"):
        preprocessing_mode("mystery-camera-1.jpg", config)


def test_local_crop_selection_returns_valid_box_and_avoids_supplied_label_mask():
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    from experiments.dinov3_mixed_domain_adaptation.preprocessing import select_local_crop

    config = load_config(CONFIG_PATH)
    image = image_module.new("RGB", (1000, 800), "green")
    mask = np.zeros((800, 1000), dtype=np.uint8)
    mask[250:550, 350:650] = 1
    selection = select_local_crop(image, config, Random(42), label_mask=mask)
    left, top, right, bottom = selection.box
    assert 0 <= left < right <= 1000
    assert 0 <= top < bottom <= 800
    assert right - left == bottom - top
    assert selection.label_overlap_fraction <= config.crops.label_overlap_limit


def test_readme_requires_inspection_before_training():
    text = Path("experiments/dinov3_mixed_domain_adaptation/README.md").read_text()
    assert "Do not start training until these previews look correct" in text
    assert "IMG_*" in text
    assert "no homography" in text
