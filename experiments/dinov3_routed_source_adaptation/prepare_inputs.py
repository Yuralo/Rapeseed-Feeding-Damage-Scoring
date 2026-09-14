"""Build routed adaptation inputs and cache only validated grid crops."""

from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from rapeseed_damage.artifacts import append_jsonl, write_json

from .config import Config, load_config
from .preprocessing import (
    GRID_INSET_TILED_MODE,
    PREPARED_SCHEMA_VERSION,
    RAW_FULL_TILED_MODE,
    prepare_image,
    route_for_source,
    score_tile_candidates,
    serialize_tile_candidates,
    source_folder,
)

OUTPUT_FIELDS = (
    "image_id",
    "file_name",
    "cohort_id",
    "source_folder",
    "source_path",
    "prepared_path",
    "source_width",
    "source_height",
    "width",
    "height",
    "input_mode",
    "grid_valid",
    "grid_points_json",
    "tile_candidates",
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
        config.data.relative_path_column,
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


def run(config: Config, *, limit: int | None = None) -> dict:
    source_rows = _read_source_manifest(config)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        source_rows = source_rows[:limit]
    destination = Path(config.data.prepared_manifest)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_log = run_dir / config.output.failure_log
    failure_log.unlink(missing_ok=True)
    prepared: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for index, row in enumerate(source_rows):
        image_id = row[config.data.id_column]
        filename = row[config.data.filename_column]
        source = Path(row[config.data.absolute_path_column])
        folder = source_folder(row, config)
        intended_mode = route_for_source(folder, config)
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image_id in adaptation manifest: {image_id}")
        seen_ids.add(image_id)
        routed = None
        try:
            if not source.is_file():
                raise FileNotFoundError(f"Source image is missing: {source}")
            routed = prepare_image(source, folder, config)
            candidates = score_tile_candidates(routed.image, config)
            prepared.append(
                {
                    "image_id": image_id,
                    "file_name": filename,
                    "cohort_id": row[config.data.cohort_column],
                    "source_folder": folder,
                    "source_path": str(source.resolve()),
                    "prepared_path": str(routed.prepared_path),
                    "source_width": routed.source_width,
                    "source_height": routed.source_height,
                    "width": routed.image.width,
                    "height": routed.image.height,
                    "input_mode": routed.input_mode,
                    "grid_valid": (
                        "true" if routed.input_mode == GRID_INSET_TILED_MODE else "not_applicable"
                    ),
                    "grid_points_json": routed.grid_points_json,
                    "tile_candidates": serialize_tile_candidates(candidates),
                    "prepared_schema_version": PREPARED_SCHEMA_VERSION,
                }
            )
            counts[routed.input_mode] += 1
            if routed.input_mode == GRID_INSET_TILED_MODE:
                counts["grid_crops_created" if routed.cache_created else "grid_crops_reused"] += 1
            else:
                counts["raw_source_references"] += 1
            source_counts[f"{folder}|{routed.input_mode}"] += 1
            print(
                f"[{index + 1:04d}/{len(source_rows):04d}] "
                f"{routed.input_mode:22s} {folder} / {filename}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - preserve the complete batch audit.
            counts["excluded"] += 1
            append_jsonl(
                failure_log,
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "image_id": image_id,
                    "filename": filename,
                    "source_folder": folder,
                    "intended_input_mode": intended_mode,
                    "source_path": str(source),
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"EXCLUDED {folder} / {filename}: {error}", flush=True)
        finally:
            if routed is not None:
                routed.image.close()

    _write_csv_atomic(destination, prepared)
    excluded_images = counts["excluded"]
    excluded_fraction = excluded_images / len(source_rows)
    raw_images = counts[RAW_FULL_TILED_MODE]
    cropped_images = counts[GRID_INSET_TILED_MODE]
    summary = {
        "source_manifest": str(Path(config.data.manifest).resolve()),
        "prepared_manifest": str(destination.resolve()),
        "requested_images": len(source_rows),
        "prepared_images": len(prepared),
        "excluded_images": excluded_images,
        "excluded_fraction": excluded_fraction,
        "maximum_excluded_fraction": config.data.maximum_excluded_fraction,
        "status": "completed_with_exclusions" if excluded_images else "completed",
        "input_modes": {
            RAW_FULL_TILED_MODE: raw_images,
            GRID_INSET_TILED_MODE: cropped_images,
        },
        "source_routes": dict(sorted(source_counts.items())),
        "raw_source_folders": list(config.preprocessing.raw_source_folders),
        "grid_crop": {
            "size": config.preprocessing.crop_size,
            "inner_margin_fraction": config.preprocessing.grid_inner_margin_fraction,
            "cache_dir": str(Path(config.preprocessing.crop_cache_dir).resolve()),
            "new_crops_written": counts["grid_crops_created"],
            "existing_crops_reused": counts["grid_crops_reused"],
        },
        "source_images_written_or_modified": 0,
        "failure_log": str(failure_log.resolve()),
        "tiling": {
            "grid_sizes": list(config.tiles.grid_sizes),
            "overlap_fraction": config.tiles.overlap_fraction,
            "selection": "one deterministic tile per routed image and epoch",
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
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Explicitly approve preprocessing the complete routed dataset.",
    )
    parser.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if not arguments.full and arguments.limit is None:
        parser.error("Refusing a full-dataset pass without --full")
    report = run(load_config(arguments.config), limit=arguments.limit)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
