import csv
import json
from pathlib import Path

from analysis.dataset_inventory import (
    COHORTS,
    PDF_REPORTED_TOTAL,
    classify,
    run,
)


def _write(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_documented_table_total_is_explicitly_different_from_pdf_headline():
    assert sum(cohort.expected_images for cohort in COHORTS) == 9456
    assert PDF_REPORTED_TOTAL == 8946


def test_classification_handles_nested_and_training_subset_paths():
    raw = Path(
        "2025_09_15_Re4StRes_T1_DSV/2025_09_15_Res4StRes_T1_DSV/example.jpg"
    )
    training = Path(
        "RSFB-Phenotyping_training_set/RSFB-Phenotyping_training_set/20251021_1.jpg"
    )
    assert classify(raw).cohort_id == "dsv_asendorf_t1_bbch11"
    assert classify(training).cohort_id == "gg_reliable_training_subset_bbch10"
    assert classify(Path("20251020_113029.jpg")).cohort_id == "wg_insects_t2_first_bbch15"


def test_inventory_deduplicates_nested_dataset_and_audits_scores(tmp_path):
    root = tmp_path / "dataset"
    gg = root / "2025_10_21_RSFB-Phenotyping_GG1_JLU"
    _write(gg / gg.name / "20251021_120001.jpg", b"same-image-content")
    _write(gg / gg.name / "20251021_120002.JPG", b"second-image")
    _write(gg / "__MACOSX" / "ignored.jpg", b"ignored")
    training = root / "RSFB-Phenotyping_training_set" / "RSFB-Phenotyping_training_set"
    _write(training / "20251021_120001.jpg", b"same-image-content")
    dsv_a = root / "2025_09_15_Re4StRes_T1_DSV" / "a.jpg"
    dsv_b = root / "2025_09_15_Res4StRes_T1_DSV" / "a-copy.jpg"
    _write(dsv_a, b"dsv-duplicate")
    _write(dsv_b, b"dsv-duplicate")
    _write(root / "standalone.jpg", b"standalone")
    (root / "scores.csv").write_text("Filename,score\na,1\nb,2\n", encoding="utf-8")
    (root / "notes.md").write_text("notes", encoding="utf-8")

    output = tmp_path / "inventory"
    summary = run(root, output, progress_every=0)

    assert summary["totals"]["discovered_image_paths"] == 6
    assert summary["totals"]["unique_content_hashes"] == 4
    assert summary["totals"]["duplicate_groups"] == 2
    assert summary["totals"]["duplicate_copies"] == 2
    assert summary["totals"]["score_csv_files"] == 1
    assert "notes.md" in summary["other_root_files"]
    assert any("8946" in warning and "9456" in warning for warning in summary["warnings"])

    with (output / "dataset_images.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert not any("__MACOSX" in row["relative_path"] for row in rows)
    copied = next(row for row in rows if row["is_training_subset"] == "True")
    assert copied["duplicate_role"] == "copy"
    assert copied["canonical_relative_path"].startswith(
        "2025_10_21_RSFB-Phenotyping_GG1_JLU/"
    )
    dsv_rows = [row for row in rows if row["cohort_id"] == "dsv_asendorf_t1_bbch11"]
    assert len(dsv_rows) == 2
    assert {row["duplicate_role"] for row in dsv_rows} == {"canonical", "copy"}

    with (output / "dataset_score_files.csv").open(newline="", encoding="utf-8") as handle:
        score = next(csv.DictReader(handle))
    assert score["row_count"] == "2"
    assert score["columns"] == "Filename|score"
    stored_summary = json.loads((output / "dataset_summary.json").read_text())
    assert stored_summary["totals"] == summary["totals"]


def test_limit_is_only_for_smoke_testing(tmp_path):
    root = tmp_path / "dataset"
    _write(root / "2025_09_12_RSFB_01_NPZi" / "one.jpg", b"one")
    _write(root / "2025_09_12_RSFB_01_NPZi" / "two.jpg", b"two")
    summary = run(root, tmp_path / "output", limit=1, progress_every=0)
    assert summary["totals"]["discovered_image_paths"] == 1
    assert any("scanned 1 of 2" in warning for warning in summary["warnings"])
