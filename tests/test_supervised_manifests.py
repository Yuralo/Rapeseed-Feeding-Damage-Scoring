import csv
import json
from pathlib import Path

from analysis.build_supervised_manifests import BuildSettings, _plot_group, run

INVENTORY_COLUMNS = (
    "image_id",
    "absolute_path",
    "relative_path",
    "file_name",
    "sha256",
    "canonical_relative_path",
    "cohort_id",
    "duplicate_role",
    "partner",
    "location",
    "experiment",
    "sampling_date",
    "timepoint",
    "bbch",
    "qr_payload",
)


def _write_csv(path: Path, rows, columns, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _image(root: Path, cohort: str, filename: str, index: int, *, duplicate="unique"):
    location = {
        "gg_insects_t1_bbch10": "Gross-Gerau",
        "gg_insects_t2_bbch13": "Gross-Gerau",
        "dsv_asendorf_t1_bbch11": "Asendorf",
        "wg_insects_t1_bbch10": "Weilburger Grenze",
    }[cohort]
    relative = f"{cohort}/{filename}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"image-{index}".encode())
    return {
        "image_id": f"image-{index}",
        "absolute_path": str(path),
        "relative_path": relative,
        "file_name": filename,
        "sha256": f"hash-{index}",
        "canonical_relative_path": relative,
        "cohort_id": cohort,
        "duplicate_role": duplicate,
        "partner": "JLU",
        "location": location,
        "experiment": "Insects",
        "sampling_date": "2025-10-21",
        "timepoint": "T1",
        "bbch": "BBCH10",
        "qr_payload": "",
    }


def test_builds_gold_only_holdouts_and_weak_pretraining_manifest(tmp_path):
    root = tmp_path / "dataset"
    images = []
    gg_rows = []
    gold_rows = []
    for index in range(6):
        filename = f"gg_{index}.jpg"
        images.append(_image(root, "gg_insects_t1_bbch10", filename, index))
        gg_rows.append(
            {
                "Filename": filename,
                "QR-Code": f"plot-{index}",
                "Plotnr": index,
                "Genotyp": f"genotype-{index}",
                "Score_JLU": 5 + index / 10,
                "Score_GAU": 6 + index / 10,
                "diff": 1,
            }
        )
        gold_rows.append(
            {
                "Filename": filename,
                "Score_JLU": 5 + index / 10,
                "Score_GAU": 6 + index / 10,
                "mean_score": 5.5 + index / 10,
            }
        )
    images.append(_image(root, "gg_insects_t1_bbch10", "gg_weak.jpg", 7))
    gg_rows.append(
        {
            "Filename": "gg_weak.jpg",
            "QR-Code": "plot-weak-gg",
            "Plotnr": 7,
            "Genotyp": "genotype-weak",
            "Score_JLU": 5,
            "Score_GAU": 25,
            "diff": 20,
        }
    )
    weak_sources = (
        ("dsv_asendorf_t1_bbch11", "2025_09_15_Res4StRes_T1_DSV_scores.csv"),
        ("wg_insects_t1_bbch10", "2025_10_07_RSFB-Phenotyping_WG1_JLU_scores.csv"),
    )
    for source_index, (cohort, score_name) in enumerate(weak_sources, start=1):
        rows = []
        for view in range(2):
            index = 10 * source_index + view
            filename = f"weak_{source_index}_{view}.jpg"
            images.append(_image(root, cohort, filename, index))
            rows.append({"Filename": filename, "QR-Code": f"weak-{source_index}", "Score": 3})
        _write_csv(root / score_name, rows, ("Filename", "QR-Code", "Score"), delimiter=";")
    images.append(_image(root, "gg_insects_t2_bbch13", "future.jpg", 99))

    inventory = tmp_path / "dataset_images.csv"
    _write_csv(inventory, images, INVENTORY_COLUMNS)
    _write_csv(
        root / "2025_10_21_RSFB-Phenotyping_GG1_scores.csv",
        gg_rows,
        (
            "Filename",
            "QR-Code",
            "Plotnr",
            "Genotyp",
            "Score_JLU",
            "Score_GAU",
            "diff",
        ),
    )
    _write_csv(
        root / "RSFB-Phenotyping_training_set" / "RSFB-Phenotyping_training_set_scores.csv",
        gold_rows,
        ("Filename", "Score_JLU", "Score_GAU", "mean_score"),
    )

    output = tmp_path / "manifests"
    summary = run(root, inventory, output, BuildSettings(seed=7))
    assert summary["gold_images"] == 6
    assert summary["weak_images"] == 5
    assert summary["score_join_issues"] == 0
    assert summary["splits"]["pretrain"]["samples"] == 5
    assert summary["splits"]["finetune"]["samples"] > 0
    assert summary["splits"]["validation"]["samples"] > 0
    assert summary["splits"]["test"]["samples"] > 0

    manifests = {}
    for name in ("pretrain", "finetune", "validation", "test"):
        with (output / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
            manifests[name] = list(csv.DictReader(handle))
    assert {row["supervision_tier"] for row in manifests["pretrain"]} == {
        "dual_weak",
        "single_weak",
    }
    dual = next(row for row in manifests["pretrain"] if row["supervision_tier"] == "dual_weak")
    assert float(dual["sample_weight"]) == 0.15
    assert all(row["is_gold_standard"] == "True" for row in manifests["finetune"])
    assert all(row["is_gold_standard"] == "True" for row in manifests["validation"])
    assert all(row["is_gold_standard"] == "True" for row in manifests["test"])
    groups = {name: {row["plot_group_id"] for row in rows} for name, rows in manifests.items()}
    assert not groups["validation"] & groups["test"]
    assert not (groups["validation"] | groups["test"]) & (groups["pretrain"] | groups["finetune"])
    stored = json.loads((output / "manifest_summary.json").read_text())
    assert stored["gold_images"] == 6
    assert "gg_insects_t2_bbch13" not in stored["adaptation"]["cohorts"]


def test_gold_threshold_violations_are_reported_but_curated_rows_remain_gold(tmp_path):
    root = tmp_path / "dataset"
    images = []
    full, gold = [], []
    for index in range(6):
        filename = f"image_{index}.jpg"
        images.append(_image(root, "gg_insects_t1_bbch10", filename, index))
        full.append(
            {
                "Filename": filename,
                "QR-Code": f"plot-{index}",
                "Score_JLU": 1,
                "Score_GAU": 2,
            }
        )
        gold.append(
            {
                "Filename": filename,
                "Score_JLU": 1,
                "Score_GAU": 7 if index == 0 else 2,
                "mean_score": 4 if index == 0 else 1.5,
            }
        )
    weak_filename = "weak.jpg"
    images.append(_image(root, "dsv_asendorf_t1_bbch11", weak_filename, 100))
    inventory = tmp_path / "inventory.csv"
    _write_csv(inventory, images, INVENTORY_COLUMNS)
    _write_csv(
        root / "2025_10_21_RSFB-Phenotyping_GG1_scores.csv",
        full,
        ("Filename", "QR-Code", "Score_JLU", "Score_GAU"),
    )
    _write_csv(
        root / "RSFB-Phenotyping_training_set_scores.csv",
        gold,
        ("Filename", "Score_JLU", "Score_GAU", "mean_score"),
    )
    _write_csv(
        root / "2025_09_15_Res4StRes_T1_DSV_scores.csv",
        [{"Filename": weak_filename, "QR-Code": "weak-plot", "Score": 2}],
        ("Filename", "QR-Code", "Score"),
        delimiter=";",
    )
    output = tmp_path / "output"
    summary = run(root, inventory, output)
    validation = next(
        row for row in summary["score_sources"] if row["score_file"] == "__gold_validation__"
    )
    assert validation["threshold_violations"] == 1
    assert summary["gold_images"] == 6


def test_score_and_image_decoded_qr_create_the_same_scoped_plot_identity():
    image = {
        "location": "Gross-Gerau",
        "experiment": "Insects",
        "qr_payload": "Plot_17",
        "sha256": "hash",
        "relative_path": "image.jpg",
    }
    from_score, score_source = _plot_group(image, score_qr="plot_17")
    from_image, image_source = _plot_group(image)
    assert from_score == from_image
    assert score_source == "score_qr"
    assert image_source == "image_qr"
