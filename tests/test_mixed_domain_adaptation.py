import csv
from dataclasses import replace
from pathlib import Path
from random import Random

import pytest

from experiments.dinov3_mixed_domain_adaptation.config import load_config

CONFIG_PATH = "experiments/dinov3_mixed_domain_adaptation/config.toml"


def test_mixed_adaptation_config_preserves_raw_tiling_and_3090_defaults():
    config = load_config(CONFIG_PATH)
    assert tuple(config.tiles.grid_sizes) == (3, 4)
    assert config.tiles.overlap_fraction == pytest.approx(0.15)
    assert config.tiles.plant_biased_probability == pytest.approx(0.70)
    assert config.data.maximum_excluded_fraction == pytest.approx(0.05)
    assert config.output.samples_per_source == 8
    assert config.training.batch_size == 8
    assert config.training.gradient_accumulation_steps == 2
    assert config.model.lora_rank == 8
    assert tuple(config.model.lora_target_modules) == ("q_proj", "v_proj")


def test_raw_tile_grids_cover_both_configured_scales():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from experiments.dinov3_mixed_domain_adaptation.preprocessing import tile_candidates

    config = load_config(CONFIG_PATH)
    candidates = tile_candidates((4000, 3000), config)
    assert len(candidates) == 3**2 + 4**2
    assert {candidate.grid_size for candidate in candidates} == {3, 4}
    assert all(
        0 <= candidate.box[0] < candidate.box[2] <= 4000
        and 0 <= candidate.box[1] < candidate.box[3] <= 3000
        for candidate in candidates
    )


def test_source_audit_sampling_spans_the_capture_sequence():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from experiments.dinov3_mixed_domain_adaptation.inspect_sources import _evenly_spaced

    rows = [{"file_name": f"IMG_{index:04d}.JPG"} for index in range(10)]
    selected = _evenly_spaced(rows, 3, "file_name")
    assert [row["file_name"] for row in selected] == [
        "IMG_0000.JPG",
        "IMG_0004.JPG",
        "IMG_0009.JPG",
    ]


def test_source_audit_recovers_and_marks_truncated_jpeg(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    from experiments.dinov3_mixed_domain_adaptation.inspect_sources import (
        _load_source_image,
    )

    path = tmp_path / "truncated.jpg"
    image_module.new("RGB", (128, 128), "green").save(path, quality=90)
    encoded = path.read_bytes()
    path.write_bytes(encoded[:-10])

    image, status, warning = _load_source_image(path)

    assert image.size == (128, 128)
    assert status == "recovered_truncated"
    assert "truncated" in warning.casefold()


def test_tile_selection_is_deterministic_and_avoids_supplied_label_mask():
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    from experiments.dinov3_mixed_domain_adaptation.preprocessing import select_adaptation_tile

    config = load_config(CONFIG_PATH)
    image = image_module.new("RGB", (1000, 800), "green")
    label_mask = np.zeros((80, 100), dtype=np.uint8)
    label_mask[:, :45] = 1
    vegetation_mask = np.ones((80, 100), dtype=np.uint8)
    first = select_adaptation_tile(
        image,
        config,
        Random(42),
        label_mask=label_mask,
        vegetation_mask=vegetation_mask,
    )
    second = select_adaptation_tile(
        image,
        config,
        Random(42),
        label_mask=label_mask,
        vegetation_mask=vegetation_mask,
    )
    assert first == second
    selection = first
    left, top, right, bottom = selection.box
    assert 0 <= left < right <= 1000
    assert 0 <= top < bottom <= 800
    assert right - left == bottom - top
    assert selection.label_overlap_fraction <= config.tiles.label_overlap_limit


def test_plant_biased_tile_selection_chooses_detected_vegetation():
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    from experiments.dinov3_mixed_domain_adaptation.preprocessing import select_adaptation_tile

    config = load_config(CONFIG_PATH)
    config = replace(
        config,
        tiles=replace(config.tiles, grid_sizes=(3,), plant_biased_probability=1.0),
    )
    image = image_module.new("RGB", (1000, 800), "brown")
    label_mask = np.zeros((80, 100), dtype=np.uint8)
    vegetation_mask = np.zeros((80, 100), dtype=np.uint8)
    vegetation_mask[:, 70:] = 1

    selection = select_adaptation_tile(
        image,
        config,
        Random(7),
        label_mask=label_mask,
        vegetation_mask=vegetation_mask,
    )

    assert selection.sampling_strategy == "plant_biased"
    assert selection.vegetation_fraction > 0


def test_prepare_inputs_validates_raw_files_without_creating_image_derivatives(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    from experiments.dinov3_mixed_domain_adaptation.prepare_inputs import run

    valid = tmp_path / "valid.jpg"
    invalid = tmp_path / "invalid.jpg"
    image_module.new("RGB", (320, 240), "green").save(valid)
    invalid.write_bytes(b"not a decodable image")
    original_bytes = valid.read_bytes()
    manifest = tmp_path / "adaptation.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "file_name", "cohort_id", "absolute_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "valid",
                "file_name": valid.name,
                "cohort_id": "cohort",
                "absolute_path": valid,
            }
        )
        writer.writerow(
            {
                "image_id": "invalid",
                "file_name": invalid.name,
                "cohort_id": "cohort",
                "absolute_path": invalid,
            }
        )
    prepared = tmp_path / "prepared.csv"
    config = load_config(CONFIG_PATH)
    config = replace(
        config,
        data=replace(
            config.data,
            manifest=str(manifest),
            prepared_manifest=str(prepared),
            maximum_excluded_fraction=0.6,
        ),
        output=replace(
            config.output,
            run_dir=str(tmp_path / "run"),
            failure_log="exclusions.jsonl",
        ),
    )

    report = run(config)
    with prepared.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert report["prepared_images"] == 1
    assert report["excluded_images"] == 1
    assert report["images_written_or_modified"] == 0
    assert rows[0]["source_path"] == str(valid.resolve())
    assert rows[0]["input_mode"] == "raw_tiled"
    assert rows[0]["tile_candidates"].startswith("[")
    assert "processed_path" not in rows[0]
    assert valid.read_bytes() == original_bytes


def test_tile_inspection_writes_one_preview_without_mutating_source(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    from experiments.dinov3_mixed_domain_adaptation.inspect_preprocessing import _save_preview
    from experiments.dinov3_mixed_domain_adaptation.preprocessing import (
        score_tile_candidates,
        serialize_tile_candidates,
    )

    source = tmp_path / "source.jpg"
    image = image_module.new("RGB", (1200, 900), "#6f5840")
    draw = draw_module.Draw(image)
    draw.rectangle((700, 250, 1050, 700), fill="#2fa83f")
    draw.rectangle((50, 50, 250, 150), fill="white")
    image.save(source, quality=95)
    image.close()
    original_bytes = source.read_bytes()
    destination = tmp_path / "previews"
    destination.mkdir()
    config = load_config(CONFIG_PATH)
    with image_module.open(source) as handle:
        candidates = serialize_tile_candidates(score_tile_candidates(handle, config))
    record = {
        "image_id": "source",
        "file_name": source.name,
        "cohort_id": "cohort",
        "source_path": str(source),
        "input_mode": "raw_tiled",
        "tile_candidates": candidates,
    }

    preview, selections, _, _ = _save_preview(record, config, destination, 1)

    assert preview.is_file()
    assert len(selections) == config.tiles.preview_tiles_per_image
    assert source.read_bytes() == original_bytes


def test_paired_dataset_returns_two_views_of_a_deterministic_raw_tile(tmp_path):
    torch = pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    from experiments.dinov3_mixed_domain_adaptation.data import PairedViewDataset
    from experiments.dinov3_mixed_domain_adaptation.preprocessing import (
        score_tile_candidates,
        serialize_tile_candidates,
    )

    source = tmp_path / "source.jpg"
    image_module.new("RGB", (1000, 800), "green").save(source)
    config = load_config(CONFIG_PATH)
    with image_module.open(source) as handle:
        candidates = serialize_tile_candidates(score_tile_candidates(handle, config))
    record = {
        "image_id": "source",
        "file_name": source.name,
        "cohort_id": "cohort",
        "source_path": str(source),
        "input_mode": "raw_tiled",
        "tile_candidates": candidates,
    }

    class Processor:
        def __call__(self, *, images, return_tensors):
            assert len(images) == 2
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros((2, 3, 224, 224))}

    dataset = PairedViewDataset(
        [record], Processor(), config, training=True
    )

    first = dataset[(3, 0)]
    second = dataset[(3, 0)]

    assert first["view_a"].shape == (3, 224, 224)
    assert first["view_b"].shape == (3, 224, 224)
    assert first["tile_grid_size"].item() in {3, 4}
    assert first["tile_row"].item() == second["tile_row"].item()
    assert first["tile_column"].item() == second["tile_column"].item()
    assert first["tile_sampling_strategy"] == second["tile_sampling_strategy"]


def test_readme_requires_inspection_before_training():
    text = Path("experiments/dinov3_mixed_domain_adaptation/README.md").read_text()
    assert "inspect_sources" in text
    assert "does not detect grids, crop" in text
    assert "Do not train until these previews look correct" in text
    assert "3x3 and 4x4" in text
    assert "untouched raw" in text
