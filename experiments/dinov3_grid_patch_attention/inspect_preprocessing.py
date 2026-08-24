"""Visually audit the outer-grid crop against the configured clean inset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rapeseed_damage.artifacts import write_json

from .config import Config, load_config
from .data import image_path, load_scores
from .preprocessing import (
    CACHE_SCHEMA_VERSION,
    load_grid_crop,
    load_or_create_grid_crop,
    log_grid_failure,
)


def _representative_rows(table, config: Config, count: int):
    count = min(count, len(table))
    ordered = table.sort_values(config.data.target_column).reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return ordered.iloc[positions].reset_index(drop=True)


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in stem
    )
    return cleaned or "image"


def _save_sample_inspection(
    path: Path,
    *,
    original: Image.Image,
    outer_crop: Image.Image,
    clean_crop: Image.Image,
    filename: str,
    target: float,
    margin: float,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    axes = axes[0]
    for axis, image in zip(axes, (original, outer_crop, clean_crop), strict=True):
        axis.imshow(image)
        axis.axis("off")
    axes[0].set_title(f"Source | {filename}\nTarget {target:.2f}")
    axes[1].set_title("Old crop | outer grid corners")
    axes[2].set_title(f"Clean crop | {100 * margin:.1f}% inset per edge")
    figure.tight_layout()
    figure.savefig(
        path,
        format="jpeg",
        dpi=110,
        bbox_inches="tight",
        pil_kwargs={"quality": 88, "optimize": True},
    )
    plt.close(figure)


def run(
    config: Config,
    *,
    count: int = 12,
    output_dir: str | Path | None = None,
    filenames: list[str] | None = None,
) -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    if config.data.grid_inner_margin_fraction <= 0:
        raise ValueError(
            "The inspection config must set data.grid_inner_margin_fraction above zero"
        )

    table = load_scores(config)
    if filenames:
        filename_column = config.data.filename_column
        indexed = table.set_index(filename_column, drop=False)
        unknown = [filename for filename in filenames if filename not in indexed.index]
        if unknown:
            raise ValueError("Unknown requested filename(s): " + ", ".join(unknown))
        selected = indexed.loc[filenames].reset_index(drop=True)
    else:
        selected = _representative_rows(table, config, count)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    inspection_dir = Path(output_dir or run_dir / "preprocessing_inspection")
    inspection_dir.mkdir(parents=True, exist_ok=True)
    failure_log = run_dir / config.output.grid_failure_log

    records: list[dict] = []
    successful_samples = 0
    margin = config.data.grid_inner_margin_fraction
    for selected_index, (_, row) in enumerate(selected.iterrows()):
        filename = str(row[config.data.filename_column])
        target = float(row[config.data.target_column])
        source = image_path(config, filename)
        try:
            with Image.open(source) as image:
                original = image.convert("RGB").copy()
            original.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
            outer_crop = load_grid_crop(
                source,
                size=config.data.grid_crop_size,
                inner_margin_fraction=0.0,
            )
            clean_crop, clean_path, was_created = load_or_create_grid_crop(
                source,
                config.data.grid_cache_dir,
                size=config.data.grid_crop_size,
                inner_margin_fraction=config.data.grid_inner_margin_fraction,
            )
            inspection_path = (
                inspection_dir / f"{selected_index + 1:03d}_{_safe_stem(filename)}.jpg"
            )
            _save_sample_inspection(
                inspection_path,
                original=original,
                outer_crop=outer_crop,
                clean_crop=clean_crop,
                filename=filename,
                target=target,
                margin=margin,
            )
            original.close()
            outer_crop.close()
            clean_crop.close()
            successful_samples += 1
            records.append(
                {
                    "filename": filename,
                    "target": target,
                    "source_image_path": str(source.resolve()),
                    "clean_crop_path": str(clean_path),
                    "clean_crop_created": was_created,
                    "inspection_path": str(inspection_path.resolve()),
                    "status": "ok",
                }
            )
        except Exception as error:
            log_grid_failure(
                failure_log,
                error=error,
                image_path=source,
                filename=filename,
                dataset_index=selected_index,
            )
            records.append(
                {
                    "filename": filename,
                    "target": target,
                    "source_image_path": str(source.resolve()),
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            )

    report = {
        "requested_samples": len(selected),
        "successful_samples": successful_samples,
        "failed_samples": sum(record["status"] == "failed" for record in records),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "grid_inner_margin_fraction": margin,
        "approximate_grid_area_retained_fraction": math.pow(1.0 - 2.0 * margin, 2),
        "inspection_directory": str(inspection_dir.resolve()),
        "failure_log": str(failure_log.resolve()),
        "samples": records,
    }
    write_json(run_dir / "preprocessing_inspection.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--filename",
        action="append",
        dest="filenames",
        help="Audit a specific dataset filename; may be supplied more than once.",
    )
    parser.add_argument("--output-dir", "--output", dest="output_dir")
    arguments = parser.parse_args(argv)
    report = run(
        load_config(arguments.config),
        count=arguments.count,
        output_dir=arguments.output_dir,
        filenames=arguments.filenames,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failed_samples"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
