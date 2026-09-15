import csv
from pathlib import Path

import pytest

from analysis.sample_gold_images import sample_gold_images

FIELDS = ("image_id", "absolute_path", "file_name", "is_gold_standard", "target", "split")


def _write_manifest(path: Path, image_dir: Path) -> list[dict[str, str]]:
    rows = []
    for index in range(8):
        image = image_dir / f"image_{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image bytes {index}".encode())
        rows.append(
            {
                "image_id": f"id-{index}",
                "absolute_path": str(image),
                "file_name": image.name,
                "is_gold_standard": "True" if index < 7 else "False",
                "target": str(index + 0.5),
                "split": "gold_train" if index < 7 else "weak_pretrain",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_samples_only_gold_and_copies_exact_rows_and_images(tmp_path):
    manifest = tmp_path / "scored_manifest.csv"
    source_rows = _write_manifest(manifest, tmp_path / "source")

    first_output = tmp_path / "sample_one"
    first = sample_gold_images(manifest, first_output, count=5, seed=17)
    second = sample_gold_images(manifest, tmp_path / "sample_two", count=5, seed=17)

    assert [row["image_id"] for row in first] == [row["image_id"] for row in second]
    assert all(row["is_gold_standard"] == "True" for row in first)
    assert all(row in source_rows for row in first)

    with (first_output / "sampled_rows.csv").open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written == first
    assert list(written[0]) == list(FIELDS)

    copied = sorted((first_output / "images").iterdir())
    assert len(copied) == 5
    for row in first:
        assert (first_output / "images" / row["file_name"]).read_bytes() == Path(
            row["absolute_path"]
        ).read_bytes()


def test_refuses_to_mix_with_an_existing_sample_unless_overwrite_is_explicit(tmp_path):
    manifest = tmp_path / "scored_manifest.csv"
    _write_manifest(manifest, tmp_path / "source")
    output = tmp_path / "sample"
    sample_gold_images(manifest, output, count=5, seed=1)

    with pytest.raises(FileExistsError, match="--overwrite"):
        sample_gold_images(manifest, output, count=5, seed=2)

    replacement = sample_gold_images(manifest, output, count=5, seed=2, overwrite=True)
    with (output / "sampled_rows.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == replacement
