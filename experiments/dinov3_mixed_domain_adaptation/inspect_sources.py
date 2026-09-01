"""Audit representative raw images from every adaptation source."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps

from rapeseed_damage.artifacts import write_json

from .config import Config, load_config
from .preprocessing import RAW_TILED_MODE

INDEX_FIELDS = (
    "cohort_id",
    "source_folder",
    "image_id",
    "file_name",
    "relative_path",
    "absolute_path",
    "width",
    "height",
    "decode_status",
    "decode_warning",
    "adaptation_input",
    "sheet_path",
)


def _read_manifest(config: Config) -> list[dict[str, str]]:
    path = Path(config.data.manifest)
    if not path.is_file():
        raise FileNotFoundError(
            f"Adaptation manifest not found: {path}. Run analysis.build_supervised_manifests first."
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


def source_folder(row: dict[str, str], config: Config) -> str:
    relative = Path(row[config.data.relative_path_column])
    return relative.parts[0] if len(relative.parts) > 1 else "<dataset-root>"


def _evenly_spaced(rows: list[dict[str, str]], count: int, filename_column: str):
    ordered = sorted(rows, key=lambda row: row[filename_column].casefold())
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[position] for position in positions]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "source"


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = image.copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(result, ((size[0] - result.width) // 2, (size[1] - result.height) // 2))
    return canvas


def _load_source_image(path: Path) -> tuple[Image.Image, str, str]:
    """Decode an audit image, retrying recoverable truncated files explicitly."""
    try:
        with Image.open(path) as handle:
            return ImageOps.exif_transpose(handle).convert("RGB"), "ok", ""
    except OSError as strict_error:
        previous_setting = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
        except Exception as tolerant_error:
            raise OSError(
                f"Could not decode source audit image '{path}'. "
                f"Strict decoder: {strict_error}. "
                f"Truncated-image recovery: {tolerant_error}."
            ) from tolerant_error
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous_setting
        warning = f"Recovered with Pillow truncated-image mode: {strict_error}"
        print(f"WARNING: {path}: {warning}", flush=True)
        return image, "recovered_truncated", warning


def _save_source_sheet(
    cohort: str,
    folder: str,
    rows: list[dict[str, str]],
    config: Config,
    destination: Path,
) -> tuple[Path, list[dict[str, object]]]:
    columns = min(4, max(1, len(rows)))
    rows_count = math.ceil(len(rows) / columns)
    tile_width, image_height, caption_height = 430, 330, 76
    header_height = 70
    canvas = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows_count * (image_height + caption_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 12), f"Cohort: {cohort}", fill="black", font=font)
    draw.text(
        (14, 34),
        f"Source folder: {folder} | RAW images only — no crop or warp has been applied",
        fill="black",
        font=font,
    )
    records = []
    for index, row in enumerate(rows):
        path = Path(row[config.data.absolute_path_column])
        if not path.is_file():
            raise FileNotFoundError(f"Source audit image is missing: {path}")
        image, decode_status, decode_warning = _load_source_image(path)
        width, height = image.size
        preview = _thumbnail(image, (tile_width, image_height))
        column, grid_row = index % columns, index // columns
        x = column * tile_width
        y = header_height + grid_row * (image_height + caption_height)
        canvas.paste(preview, (x, y))
        filename = row[config.data.filename_column]
        draw.text((x + 7, y + image_height + 5), filename, fill="black", font=font)
        draw.text(
            (x + 7, y + image_height + 25),
            f"{width}x{height} | adaptation input: {RAW_TILED_MODE}",
            fill="black",
            font=font,
        )
        draw.text(
            (x + 7, y + image_height + 45),
            f"Decision: visual review required | decode: {decode_status}",
            fill="#b00020" if decode_warning else "#9a3b00",
            font=font,
        )
        records.append(
            {
                "cohort_id": cohort,
                "source_folder": folder,
                "image_id": row[config.data.id_column],
                "file_name": filename,
                "relative_path": row[config.data.relative_path_column],
                "absolute_path": str(path),
                "width": width,
                "height": height,
                "decode_status": decode_status,
                "decode_warning": decode_warning,
                "adaptation_input": RAW_TILED_MODE,
            }
        )
    sheet = destination / f"{_safe_name(cohort)}__{_safe_name(folder)}.jpg"
    canvas.save(sheet, format="JPEG", quality=88, optimize=True)
    for record in records:
        record["sheet_path"] = str(sheet.resolve())
    return sheet, records


def run(config: Config, *, samples_per_source: int | None = None) -> dict:
    count = samples_per_source or config.output.samples_per_source
    if count < 1:
        raise ValueError("samples_per_source must be positive")
    rows = _read_manifest(config)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cohort = row[config.data.cohort_column]
        groups[(cohort, source_folder(row, config))].append(row)
    destination = Path(config.output.run_dir) / config.output.source_inspection_dir
    destination.mkdir(parents=True, exist_ok=True)
    index_records: list[dict[str, object]] = []
    group_records = []
    for number, ((cohort, folder), source_rows) in enumerate(sorted(groups.items()), start=1):
        selected = _evenly_spaced(source_rows, count, config.data.filename_column)
        sheet, records = _save_source_sheet(cohort, folder, selected, config, destination)
        index_records.extend(records)
        inputs = Counter(record["adaptation_input"] for record in records)
        decode_statuses = Counter(record["decode_status"] for record in records)
        group_records.append(
            {
                "cohort_id": cohort,
                "source_folder": folder,
                "total_images": len(source_rows),
                "sampled_images": len(selected),
                "adaptation_inputs": json.dumps(dict(inputs), sort_keys=True),
                "decode_statuses": json.dumps(dict(decode_statuses), sort_keys=True),
                "decision": "raw_tiled",
                "sheet_path": str(sheet.resolve()),
            }
        )
        print(f"[{number:02d}/{len(groups):02d}] {sheet}", flush=True)
    index_path = destination / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_records)
    source_summary_path = destination / "sources.csv"
    with source_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_records[0]))
        writer.writeheader()
        writer.writerows(group_records)
    summary = {
        "source_groups": len(groups),
        "dataset_images": len(rows),
        "sampled_images": len(index_records),
        "recovered_truncated_images": sum(
            record["decode_status"] == "recovered_truncated" for record in index_records
        ),
        "samples_per_source_requested": count,
        "directory": str(destination.resolve()),
        "index_csv": str(index_path.resolve()),
        "sources_csv": str(source_summary_path.resolve()),
        "sheets": [record["sheet_path"] for record in group_records],
        "preprocessing_applied": False,
        "next_step": "Run prepare_inputs to validate raw files, then inspect sampled tiles.",
    }
    write_json(destination / "summary.json", summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-source", type=int)
    arguments = parser.parse_args(argv)
    report = run(load_config(arguments.config), samples_per_source=arguments.samples_per_source)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
