"""Make a small, cohort-balanced gallery of the weak training images and scores."""

from __future__ import annotations

import argparse
import csv
import html
import random
import shutil
from collections import defaultdict
from pathlib import Path

DEFAULT_MANIFEST = Path(
    "outputs/dinov3_weak_only_gold_validation_all_weak/manifests/weak_train.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/random_weak_sample")
REQUIRED = {
    "image_id", "absolute_path", "relative_path", "cohort_id", "target",
    "target_source", "score_single", "score_jlu", "score_gau",
}
CSV_COLUMNS = (
    "image_id", "score", "target_source", "cohort_id", "score_single",
    "score_jlu", "score_gau", "relative_path", "sample_image",
)


def _rows(manifest: Path) -> list[dict[str, str]]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Weak training manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Weak manifest lacks columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Weak training manifest is empty: {manifest}")
    return rows


def _select(rows: list[dict[str, str]], per_cohort: int, seed: int) -> list[dict[str, str]]:
    if per_cohort < 1:
        raise ValueError("per_cohort must be at least 1")
    cohorts: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cohorts[row["cohort_id"]].append(row)
    generator = random.Random(seed)
    selected = []
    for cohort in sorted(cohorts):
        members = sorted(cohorts[cohort], key=lambda row: row["image_id"])
        selected.extend(generator.sample(members, min(per_cohort, len(members))))
    return selected


def _gallery(rows: list[dict[str, str]]) -> str:
    cards = []
    for row in rows:
        image = html.escape(row["sample_image"], quote=True)
        label = html.escape(row["image_id"])
        cohort = html.escape(row["cohort_id"])
        score = html.escape(row["score"])
        source = html.escape(row["target_source"])
        raw = ""
        if source == "mean_of_two_scorers":
            raw = (
                f"<p>JLU: {html.escape(row['score_jlu'])} · "
                f"GAU: {html.escape(row['score_gau'])}</p>"
            )
        cards.append(
            f'<article><img src="{image}" alt="{label}" loading="lazy">'
            f"<h2>{label}</h2><p>Score: <strong>{score}</strong></p>"
            f"<p>{cohort} · {source}</p>{raw}</article>"
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Weak training image sample</title><style>'
        'body{font:16px system-ui;margin:2rem;background:#f5f5f1;color:#222}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}'
        'article{background:white;padding:1rem;border-radius:8px;overflow:hidden}'
        'img{display:block;width:100%;height:330px;object-fit:contain;background:#222}'
        'h2{font-size:1rem;overflow-wrap:anywhere}p{margin:.35rem 0}'
        '</style><h1>Weak training image sample</h1>'
        '<p>Original source images. Scores are training targets, not model predictions. '
        'A single-scorer score is not an average.</p><main>'
        + "".join(cards) + "</main></html>"
    )


def sample_weak_images(
    manifest_path: Path,
    output_dir: Path,
    *,
    per_cohort: int = 3,
    seed: int = 42,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    """Copy a reproducible sample from each cohort and create CSV + HTML gallery."""
    selected = _select(_rows(manifest_path), per_cohort, seed)
    sources = [Path(row["absolute_path"]) for row in selected]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} sampled source image(s) are missing on this machine:\n{preview}"
        )
    for row in selected:
        image_id = row["image_id"].strip()
        if not image_id or Path(image_id).name != image_id:
            raise ValueError(f"Invalid image_id in weak manifest: {image_id!r}")
    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite or a new path.")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    images_dir = output_dir / "images"
    if images_dir.exists() and not images_dir.is_dir():
        raise NotADirectoryError(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for row, source in zip(selected, sources, strict=True):
        image_id = row["image_id"].strip()
        destination = images_dir / f"{image_id}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        exported.append({
            "image_id": image_id,
            "score": row["target"],
            "target_source": row["target_source"],
            "cohort_id": row["cohort_id"],
            "score_single": row["score_single"],
            "score_jlu": row["score_jlu"],
            "score_gau": row["score_gau"],
            "relative_path": row["relative_path"],
            "sample_image": f"images/{destination.name}",
        })
    with (output_dir / "sampled_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(exported)
    (output_dir / "index.html").write_text(_gallery(exported), encoding="utf-8")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-cohort", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = sample_weak_images(
        args.manifest, args.output_dir, per_cohort=args.per_cohort,
        seed=args.seed, overwrite=args.overwrite,
    )
    print(f"Sampled {len(rows)} weak images across {len({r['cohort_id'] for r in rows})} cohorts")
    print(f"Gallery: {(args.output_dir / 'index.html').resolve()}")
    print(f"Scores: {(args.output_dir / 'sampled_rows.csv').resolve()}")


if __name__ == "__main__":
    main()
