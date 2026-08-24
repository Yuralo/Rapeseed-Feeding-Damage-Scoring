"""Save one compact inspection image per cached SAM mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rapeseed_damage.artifacts import write_json

from .config import Config, load_config
from .data import image_path, load_scores
from .preprocessing import load_or_create_grid_crop
from .segmentation import load_cached_mask, make_masked_image


def _representative_rows(table, config: Config, count: int):
    count = min(count, len(table))
    ordered = table.sort_values(config.data.target_column).reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return ordered.iloc[positions].reset_index(drop=True)


def _safe_stem(filename: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in Path(filename).stem
    )
    return cleaned or "image"


def _save_inspection(
    path: Path,
    *,
    grid_image: Image.Image,
    mask: Image.Image,
    masked_image: Image.Image,
    filename: str,
    target: float,
    foreground_fraction: float,
    valid: bool,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    axes = axes[0]
    axes[0].imshow(grid_image)
    axes[0].set_title(f"Grid crop | {filename}\nTarget {target:.2f}")
    axes[1].imshow(grid_image)
    mask_array = np.asarray(mask)
    axes[1].imshow(
        np.ma.masked_where(mask_array < 128, mask_array),
        cmap="spring",
        alpha=0.48,
        vmin=0,
        vmax=255,
    )
    axes[1].set_title(
        f"SAM mask | foreground {100 * foreground_fraction:.2f}%\n"
        f"quality {'valid' if valid else 'INVALID'}"
    )
    axes[2].imshow(masked_image)
    axes[2].set_title("Masked DINOv3 input")
    for axis in axes:
        axis.axis("off")
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
    count: int | None = None,
    output_dir: str | Path | None = None,
    filenames: list[str] | None = None,
) -> dict:
    count = config.output.sam_inspection_images if count is None else count
    if count < 1:
        raise ValueError("count must be positive")
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
    destination = Path(output_dir or run_dir / "sam_mask_inspection")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        filename = str(row[config.data.filename_column])
        target = float(row[config.data.target_column])
        source = image_path(config, filename)
        grid_image = mask = masked_image = None
        try:
            grid_image, grid_path, _ = load_or_create_grid_crop(
                source,
                config.data.grid_cache_dir,
                size=config.data.grid_crop_size,
                inner_margin_fraction=config.data.grid_inner_margin_fraction,
            )
            mask, mask_path, metadata = load_cached_mask(
                source,
                config,
                require_valid=False,
            )
            masked_image = make_masked_image(
                grid_image,
                mask,
                background_value=config.segmentation.background_value,
            )
            quality = metadata["quality"]
            inspection_path = destination / f"{index:03d}_{_safe_stem(filename)}.jpg"
            _save_inspection(
                inspection_path,
                grid_image=grid_image,
                mask=mask,
                masked_image=masked_image,
                filename=filename,
                target=target,
                foreground_fraction=float(quality["foreground_fraction"]),
                valid=bool(quality["valid"]),
            )
            records.append(
                {
                    "filename": filename,
                    "target": target,
                    "status": "ok",
                    "valid": bool(quality["valid"]),
                    "quality_reasons": quality["quality_reasons"],
                    "foreground_fraction": quality["foreground_fraction"],
                    "grid_crop_path": str(grid_path),
                    "mask_path": str(mask_path),
                    "inspection_path": str(inspection_path.resolve()),
                }
            )
        except Exception as error:
            records.append(
                {
                    "filename": filename,
                    "target": target,
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            )
        finally:
            for image in (grid_image, mask, masked_image):
                if image is not None:
                    image.close()

    report = {
        "requested_samples": len(selected),
        "successful_samples": sum(record["status"] == "ok" for record in records),
        "failed_samples": sum(record["status"] == "failed" for record in records),
        "invalid_masks": sum(
            record["status"] == "ok" and not record["valid"] for record in records
        ),
        "inspection_directory": str(destination.resolve()),
        "samples": records,
    }
    write_json(run_dir / "sam_mask_inspection.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--output-dir", "--output", dest="output_dir")
    parser.add_argument("--filename", action="append", dest="filenames")
    arguments = parser.parse_args(argv)
    report = run(
        load_config(arguments.config),
        count=arguments.count,
        output_dir=arguments.output_dir,
        filenames=arguments.filenames,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failed_samples"] or report["invalid_masks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
