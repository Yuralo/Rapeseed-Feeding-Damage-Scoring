"""Prepare the explicit grid-crop/raw adaptation manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from experiments.dinov3_grid_lora_patch_attention.preprocessing import (
    load_or_create_grid_crop,
)
from rapeseed_damage.artifacts import append_jsonl, write_json

from .config import Config, load_config
from .preprocessing import GRID_CROP_MODE, PREPARED_SCHEMA_VERSION, preprocessing_mode

OUTPUT_FIELDS = (
    "image_id",
    "file_name",
    "cohort_id",
    "source_path",
    "processed_path",
    "preprocessing_mode",
    "prepared_schema_version",
)


def _read_source_manifest(config: Config) -> list[dict[str, str]]:
    path = Path(config.data.manifest)
    if not path.is_file():
        raise FileNotFoundError(
            f"Adaptation manifest not found: {path}. Run analysis.build_supervised_manifests."
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        config.data.absolute_path_column,
        config.data.filename_column,
        config.data.id_column,
        config.data.cohort_column,
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    if not rows:
        raise ValueError(f"Adaptation manifest is empty: {path}")
    return rows


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(config: Config, *, overwrite: bool = False, limit: int | None = None) -> dict:
    source_rows = _read_source_manifest(config)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        source_rows = source_rows[:limit]
    destination = Path(config.data.prepared_manifest)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_log = run_dir / config.output.failure_log
    # Each preparation report describes this run only; do not accumulate stale
    # exclusions from earlier attempts.
    failure_log.unlink(missing_ok=True)
    prepared: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for index, row in enumerate(source_rows):
        image_id = row[config.data.id_column]
        filename = row[config.data.filename_column]
        source = Path(row[config.data.absolute_path_column])
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image_id in adaptation manifest: {image_id}")
        seen_ids.add(image_id)
        try:
            mode = preprocessing_mode(filename, config)
            if config.data.verify_images and not source.is_file():
                raise FileNotFoundError(f"Source image is missing: {source}")
            if mode == GRID_CROP_MODE:
                image, processed, _ = load_or_create_grid_crop(
                    source,
                    config.data.grid_cache_dir,
                    size=config.data.grid_crop_size,
                    inner_margin_fraction=config.data.grid_inner_margin_fraction,
                    overwrite=overwrite,
                )
                image.close()
            else:
                processed = source.resolve()
            prepared.append(
                {
                    "image_id": image_id,
                    "file_name": filename,
                    "cohort_id": row[config.data.cohort_column],
                    "source_path": str(source.resolve()),
                    "processed_path": str(processed),
                    "preprocessing_mode": mode,
                    "prepared_schema_version": PREPARED_SCHEMA_VERSION,
                }
            )
            counts[mode] += 1
            print(f"[{index + 1:04d}/{len(source_rows):04d}] {mode:9s} {filename}", flush=True)
        except Exception as error:  # noqa: BLE001 - preserve complete batch audit.
            counts["excluded"] += 1
            append_jsonl(
                failure_log,
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "image_id": image_id,
                    "filename": filename,
                    "source_path": str(source),
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"EXCLUDED {filename}: {error}", flush=True)
    _write_csv_atomic(destination, prepared)
    excluded_images = counts["excluded"]
    excluded_fraction = excluded_images / len(source_rows)
    summary = {
        "source_manifest": str(Path(config.data.manifest).resolve()),
        "prepared_manifest": str(destination.resolve()),
        "requested_images": len(source_rows),
        "prepared_images": len(prepared),
        "excluded_images": excluded_images,
        "excluded_fraction": excluded_fraction,
        "maximum_excluded_fraction": config.data.maximum_excluded_fraction,
        "status": "completed_with_exclusions" if excluded_images else "completed",
        "counts": dict(sorted(counts.items())),
        "grid_crop_size": config.data.grid_crop_size,
        "grid_inner_margin_fraction": config.data.grid_inner_margin_fraction,
        "failure_log": str(failure_log.resolve()),
        "routing": {
            "timestamp_pattern": GRID_CROP_MODE,
            "IMG_pattern": "raw",
            "silent_fallback": False,
        },
    }
    write_json(run_dir / "preparation_summary.json", summary)
    if not prepared:
        raise RuntimeError("Preparation produced no usable images; inspect the failure log")
    if excluded_fraction > config.data.maximum_excluded_fraction:
        raise RuntimeError(
            f"Excluded {excluded_images}/{len(source_rows)} images "
            f"({excluded_fraction:.2%}), above the configured "
            f"{config.data.maximum_excluded_fraction:.2%} safety limit; inspect {failure_log}"
        )
    if excluded_images:
        print(
            f"Preparation completed with {excluded_images} excluded image(s) "
            f"({excluded_fraction:.2%}); details: {failure_log}",
            flush=True,
        )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args(argv)
    report = run(
        load_config(arguments.config), overwrite=arguments.overwrite, limit=arguments.limit
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
