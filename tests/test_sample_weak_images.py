import csv
from pathlib import Path

import pytest

from analysis.sample_weak_images import sample_weak_images

FIELDS = (
    "image_id", "absolute_path", "relative_path", "cohort_id", "target",
    "target_source", "score_single", "score_jlu", "score_gau",
)


def _manifest(path: Path, source_dir: Path):
    rows = []
    for cohort in ("dsv", "gg", "wg"):
        for number in range(4):
            image = source_dir / cohort / f"image_{number}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"{cohort} {number}".encode())
            dual = cohort == "gg"
            rows.append({
                "image_id": f"{cohort}_{number}",
                "absolute_path": str(image),
                "relative_path": f"{cohort}/image_{number}.jpg",
                "cohort_id": cohort,
                "target": str(number + 2 if dual else number + 1),
                "target_source": "mean_of_two_scorers" if dual else "single_scorer",
                "score_single": "" if dual else str(number + 1),
                "score_jlu": str(number + 1) if dual else "",
                "score_gau": str(number + 3) if dual else "",
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_cohort_balanced_reproducible_gallery_with_scores(tmp_path):
    manifest = tmp_path / "weak_train.csv"
    source = _manifest(manifest, tmp_path / "sources")
    selected = sample_weak_images(manifest, tmp_path / "sample", per_cohort=2, seed=7)
    repeated = sample_weak_images(manifest, tmp_path / "repeat", per_cohort=2, seed=7)
    assert [row["image_id"] for row in selected] == [row["image_id"] for row in repeated]
    assert len(selected) == 6
    assert {cohort: sum(row["cohort_id"] == cohort for row in selected)
            for cohort in ("dsv", "gg", "wg")} == {"dsv": 2, "gg": 2, "wg": 2}
    originals = {row["image_id"]: row for row in source}
    for row in selected:
        original = originals[row["image_id"]]
        assert row["score"] == original["target"]
        assert (tmp_path / "sample" / row["sample_image"]).read_bytes() == (
            Path(original["absolute_path"]).read_bytes()
        )
    with (tmp_path / "sample" / "sampled_rows.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle)) == selected
    gallery = (tmp_path / "sample" / "index.html").read_text()
    assert "A single-scorer score is not an average" in gallery
    assert "JLU:" in gallery


def test_existing_output_requires_explicit_overwrite(tmp_path):
    manifest = tmp_path / "weak_train.csv"
    _manifest(manifest, tmp_path / "sources")
    output = tmp_path / "sample"
    sample_weak_images(manifest, output, per_cohort=1)
    with pytest.raises(FileExistsError, match="--overwrite"):
        sample_weak_images(manifest, output, per_cohort=1)
    assert len(sample_weak_images(manifest, output, per_cohort=1, seed=11,
                                  overwrite=True)) == 3


def test_missing_image_is_reported_before_output_is_created(tmp_path):
    manifest = tmp_path / "weak_train.csv"
    _manifest(manifest, tmp_path / "sources")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    Path(rows[0]["absolute_path"]).unlink()
    output = tmp_path / "sample"
    with pytest.raises(FileNotFoundError, match="sampled source image"):
        sample_weak_images(manifest, output, per_cohort=4)
    assert not output.exists()
