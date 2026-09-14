import csv
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.dinov3_routed_source_adaptation.config import load_config
from experiments.dinov3_routed_source_adaptation.preprocessing import (
    GRID_INSET_TILED_MODE,
    RAW_FULL_TILED_MODE,
    route_for_source,
)

CONFIG_PATH = "experiments/dinov3_routed_source_adaptation/config.toml"


def test_config_routes_only_the_three_audited_bad_sources_to_raw():
    config = load_config(CONFIG_PATH)
    assert set(config.preprocessing.raw_source_folders) == {
        "2025_09_15_Re4StRes_T1_DSV",
        "2025_09_12_RSFB_01_NPZi",
        "2025_09_19_RSFB_02_NPZi",
    }
    for source in config.preprocessing.raw_source_folders:
        assert route_for_source(source, config) == RAW_FULL_TILED_MODE
    assert route_for_source("2025_09_30_Res4StRes_T2_DSV", config) == GRID_INSET_TILED_MODE
    assert config.preprocessing.grid_inner_margin_fraction == pytest.approx(0.075)
    assert config.preprocessing.crop_size == 1400


def test_routing_uses_source_folder_not_img_filename():
    config = load_config(CONFIG_PATH)
    assert route_for_source("2025_09_15_Re4StRes_T1_DSV", config) == RAW_FULL_TILED_MODE
    assert route_for_source("2025_09_30_Res4StRes_T2_DSV", config) == GRID_INSET_TILED_MODE


def test_prepare_inputs_preserves_raw_source_and_caches_grid_crop(tmp_path, monkeypatch):
    image_module = pytest.importorskip("PIL.Image")
    from experiments.dinov3_routed_source_adaptation import preprocessing
    from experiments.dinov3_routed_source_adaptation.prepare_inputs import run

    raw_source = tmp_path / "raw.jpg"
    cropped_source = tmp_path / "cropped.jpg"
    image_module.new("RGB", (320, 240), "green").save(raw_source)
    image_module.new("RGB", (320, 240), "brown").save(cropped_source)
    raw_bytes = raw_source.read_bytes()
    cropped_bytes = cropped_source.read_bytes()

    manifest = tmp_path / "adaptation.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "file_name",
                "cohort_id",
                "relative_path",
                "absolute_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "raw",
                "file_name": "IMG_0001.jpg",
                "cohort_id": "raw-cohort",
                "relative_path": "2025_09_15_Re4StRes_T1_DSV/IMG_0001.jpg",
                "absolute_path": raw_source,
            }
        )
        writer.writerow(
            {
                "image_id": "grid",
                "file_name": "IMG_0002.jpg",
                "cohort_id": "grid-cohort",
                "relative_path": "2025_09_30_Res4StRes_T2_DSV/IMG_0002.jpg",
                "absolute_path": cropped_source,
            }
        )

    def fake_crop(image, config):
        return image_module.new("RGB", (128, 128), "green"), "[[synthetic-grid]]"

    monkeypatch.setattr(preprocessing, "_detect_and_crop", fake_crop)
    config = load_config(CONFIG_PATH)
    prepared_manifest = tmp_path / "prepared.csv"
    config = replace(
        config,
        data=replace(
            config.data,
            manifest=str(manifest),
            prepared_manifest=str(prepared_manifest),
            maximum_excluded_fraction=0.0,
        ),
        preprocessing=replace(
            config.preprocessing,
            crop_size=128,
            crop_cache_dir=str(tmp_path / "grid-cache"),
        ),
        output=replace(config.output, run_dir=str(tmp_path / "run")),
    )

    report = run(config)
    with prepared_manifest.open(newline="", encoding="utf-8") as handle:
        rows = {row["image_id"]: row for row in csv.DictReader(handle)}

    assert report["input_modes"] == {
        RAW_FULL_TILED_MODE: 1,
        GRID_INSET_TILED_MODE: 1,
    }
    assert rows["raw"]["prepared_path"] == str(raw_source.resolve())
    assert rows["raw"]["grid_valid"] == "not_applicable"
    assert rows["grid"]["prepared_path"] != str(cropped_source.resolve())
    assert Path(rows["grid"]["prepared_path"]).is_file()
    assert rows["grid"]["grid_valid"] == "true"
    assert rows["grid"]["width"] == "128"
    assert raw_source.read_bytes() == raw_bytes
    assert cropped_source.read_bytes() == cropped_bytes


def test_downstream_routed_configs_are_valid_and_use_dedicated_caches():
    from experiments.dinov3_grid_multiscale_tiled_mil.config import (
        load_config as load_multiscale,
    )
    from experiments.dinov3_grid_tiled_mil.config import load_config as load_single

    coarse = load_single("experiments/dinov3_grid_tiled_mil/config_adapted_routed_3x3.toml")
    fine = load_single("experiments/dinov3_grid_tiled_mil/config_adapted_routed_4x4.toml")
    multiscale = load_multiscale(
        "experiments/dinov3_grid_multiscale_tiled_mil/config_adapted_routed.toml"
    )
    expected = "outputs/dinov3_routed_source_adaptation/adapted_backbone"
    assert coarse.features.backbone == expected
    assert fine.features.backbone == expected
    assert multiscale.features.backbone == expected
    assert coarse.features.cache_dir == multiscale.coarse.cache_dir
    assert fine.features.cache_dir == multiscale.fine.cache_dir
