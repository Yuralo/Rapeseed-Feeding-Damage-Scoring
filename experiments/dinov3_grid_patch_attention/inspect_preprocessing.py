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


def run(
    config: Config,
    *,
    count: int = 12,
    output_path: str | Path | None = None,
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
    destination = Path(output_path or run_dir / "preprocessing_inspection.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    failure_log = run_dir / config.output.grid_failure_log

    records: list[dict] = []
    panels: list[tuple[Image.Image, Image.Image, Image.Image, str, float]] = []
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
            panels.append((original, outer_crop, clean_crop, filename, target))
            records.append(
                {
                    "filename": filename,
                    "target": target,
                    "source_image_path": str(source.resolve()),
                    "clean_crop_path": str(clean_path),
                    "clean_crop_created": was_created,
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

    if panels:
        figure, axes = plt.subplots(
            len(panels),
            3,
            figsize=(14, 4.5 * len(panels)),
            squeeze=False,
        )
        margin = config.data.grid_inner_margin_fraction
        for row_index, (original, outer, clean, filename, target) in enumerate(panels):
            for column, image in enumerate((original, outer, clean)):
                axes[row_index, column].imshow(image)
                axes[row_index, column].axis("off")
            axes[row_index, 0].set_title(f"Source | {filename}\nTarget {target:.2f}")
            axes[row_index, 1].set_title("Old crop | outer grid corners")
            axes[row_index, 2].set_title(f"Clean crop | {100 * margin:.1f}% inset per edge")
        figure.suptitle(
            "Preprocessing audit: confirm collector labels are removed and plants remain",
            fontsize=14,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.995))
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(figure)

    margin = config.data.grid_inner_margin_fraction
    report = {
        "requested_samples": len(selected),
        "successful_samples": len(panels),
        "failed_samples": sum(record["status"] == "failed" for record in records),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "grid_inner_margin_fraction": margin,
        "approximate_grid_area_retained_fraction": math.pow(1.0 - 2.0 * margin, 2),
        "inspection_image": str(destination.resolve()) if panels else None,
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
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    report = run(
        load_config(arguments.config),
        count=arguments.count,
        output_path=arguments.output,
        filenames=arguments.filenames,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failed_samples"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
