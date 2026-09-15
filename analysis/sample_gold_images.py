"""Copy a small reproducible random sample from the gold-labelled dataset."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

DEFAULT_MANIFEST = Path("outputs/dataset_manifests/scored_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/random_gold_sample")
TRUTHY_VALUES = {"1", "true", "yes"}


def _is_gold(row: dict[str, str]) -> bool:
    return row.get("is_gold_standard", "").strip().casefold() in TRUTHY_VALUES


def sample_gold_images(
    manifest_path: Path,
    output_dir: Path,
    *,
    count: int = 5,
    seed: int = 42,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    """Sample gold rows, copy their images, and write image IDs and scores to CSV."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        required = {"absolute_path", "image_id", "is_gold_standard", "target"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"Manifest is missing required column(s): {', '.join(sorted(missing))}"
            )
        gold_rows = [dict(row) for row in reader if _is_gold(row)]

    if len(gold_rows) < count:
        raise ValueError(
            f"Requested {count} images, but the manifest contains only {len(gold_rows)} gold rows"
        )

    selected = random.Random(seed).sample(gold_rows, count)
    sources = [Path(row["absolute_path"]) for row in selected]
    missing_sources = [source for source in sources if not source.is_file()]
    if missing_sources:
        preview = "\n".join(str(path) for path in missing_sources[:5])
        raise FileNotFoundError(
            f"{len(missing_sources)} selected image(s) do not exist. First missing path(s):\n{preview}"
        )

    output_dir = output_dir.resolve()
    images_dir = output_dir / "images"
    rows_path = output_dir / "sampled_rows.csv"
    if (images_dir.exists() or rows_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Sample output already exists in {output_dir}. Use --overwrite to replace it."
        )

    if overwrite and images_dir.exists():
        if not images_dir.is_dir():
            raise NotADirectoryError(f"Expected a directory: {images_dir}")
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=False)

    for row, source in zip(selected, sources, strict=True):
        image_id = row.get("image_id", "").strip()
        if not image_id:
            raise ValueError("A selected gold row has an empty image_id")
        name = f"{image_id}{source.suffix.lower()}"
        shutil.copy2(source, images_dir / name)

    output_dir.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "score"))
        writer.writeheader()
        writer.writerows(
            {"image_id": row["image_id"], "score": row["target"]} for row in selected
        )

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy random gold-labelled images and export their image IDs and scores."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a sample already present in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    selected = sample_gold_images(
        arguments.manifest,
        arguments.output_dir,
        count=arguments.count,
        seed=arguments.seed,
        overwrite=arguments.overwrite,
    )
    destination = arguments.output_dir.resolve()
    print(f"Copied {len(selected)} gold images to {destination / 'images'}")
    print(f"Saved their image IDs and scores to {destination / 'sampled_rows.csv'}")


if __name__ == "__main__":
    main()
