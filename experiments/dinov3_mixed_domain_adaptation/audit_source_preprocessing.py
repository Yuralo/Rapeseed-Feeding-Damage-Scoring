"""Sample every adaptation source and audit the existing grid preprocessing.

The command deliberately works on a small, reproducible sample.  It copies each
selected source file unchanged and writes separate grid-overlay, outer-crop,
7.5%-inset, and comparison images.  Detection failures are recorded instead of
stopping the remaining source groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rapeseed_damage.artifacts import append_jsonl, write_json
from rapeseed_damage.grid import detect_grid, image_to_ndarray, to_rgb, warp_big_square

from .config import Config, load_config
from .inspect_sources import (
    _load_source_image,
    _read_manifest,
    _safe_name,
    family_aware_random_sample,
    filename_family,
    source_folder,
)


DEFAULT_INSET_FRACTION = 0.075
DEFAULT_CROP_SIZE = 1400

INDEX_FIELDS = (
    "cohort_id",
    "source_folder",
    "filename_family",
    "image_id",
    "file_name",
    "relative_path",
    "absolute_path",
    "source_copy_path",
    "raw_width",
    "raw_height",
    "decode_status",
    "decode_warning",
    "grid_status",
    "grid_error",
    "grid_area_fraction",
    "outside_corner_count",
    "grid_points_json",
    "grid_overlay_path",
    "outer_crop_path",
    "inset_crop_path",
    "comparison_path",
    # These empty columns make the CSV directly usable as a manual review sheet.
    "review_complete_quadrat_visible",
    "review_grid_detection_correct",
    "review_inset_correct",
    "review_recommended_route",
    "review_notes",
)


def _save_rgb(array_bgr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        array_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )
    if not ok:
        raise OSError(f"OpenCV could not write {path}")


def _draw_grid_overlay(image_bgr: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    scale = max(2, round(min(image_bgr.shape[:2]) * 0.004))
    points = np.rint(grid_points).astype(np.int32)
    for row in range(3):
        cv2.line(
            overlay,
            tuple(points[row, 0]),
            tuple(points[row, 2]),
            (30, 30, 240),
            scale,
            cv2.LINE_AA,
        )
    for column in range(3):
        cv2.line(
            overlay,
            tuple(points[0, column]),
            tuple(points[2, column]),
            (30, 30, 240),
            scale,
            cv2.LINE_AA,
        )
    for row in range(3):
        for column in range(3):
            cv2.circle(
                overlay,
                tuple(points[row, column]),
                2 * scale,
                (20, 220, 20),
                -1,
                cv2.LINE_AA,
            )
    return overlay


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    preview = image.convert("RGB").copy()
    preview.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(preview, ((size[0] - preview.width) // 2, (size[1] - preview.height) // 2))
    return canvas


def _save_comparison(
    *,
    raw: Image.Image,
    overlay_bgr: np.ndarray,
    outer_bgr: np.ndarray,
    inset_bgr: np.ndarray,
    title: str,
    destination: Path,
    inset_fraction: float,
) -> None:
    width, height, caption = 520, 390, 34
    canvas = Image.new("RGB", (2 * width, 2 * (height + caption) + 48), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), title, fill="black", font=font)
    panels = (
        (raw, "Raw source (unchanged copy is saved separately)"),
        (Image.fromarray(to_rgb(overlay_bgr)), "Detected 3x3 intersections / 4 cells"),
        (Image.fromarray(to_rgb(outer_bgr)), "Perspective crop at outer grid bars"),
        (
            Image.fromarray(to_rgb(inset_bgr)),
            f"Perspective crop with {100 * inset_fraction:.1f}% inset per edge",
        ),
    )
    for index, (image, label) in enumerate(panels):
        column, row = index % 2, index // 2
        x = column * width
        y = 48 + row * (height + caption)
        canvas.paste(_thumbnail(image, (width, height)), (x, y))
        draw.text((x + 8, y + height + 8), label, fill="black", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=90, optimize=True)


def _geometry(grid_points: np.ndarray, image_bgr: np.ndarray) -> tuple[float, int]:
    corners = np.float32(
        [grid_points[0, 0], grid_points[0, 2], grid_points[2, 2], grid_points[2, 0]]
    )
    height, width = image_bgr.shape[:2]
    area_fraction = abs(float(cv2.contourArea(corners))) / float(width * height)
    outside = sum(
        x < 0 or x >= width or y < 0 or y >= height
        for x, y in corners.tolist()
    )
    return area_fraction, int(outside)


def _copy_source(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def run(
    config: Config,
    *,
    samples_per_source: int = 8,
    seed: int | None = None,
    inset_fraction: float = DEFAULT_INSET_FRACTION,
    crop_size: int = DEFAULT_CROP_SIZE,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    if samples_per_source < 1:
        raise ValueError("samples_per_source must be positive")
    if not 0.0 <= inset_fraction < 0.5:
        raise ValueError("inset_fraction must be in [0, 0.5)")
    if crop_size < 1:
        raise ValueError("crop_size must be positive")

    selected_seed = config.training.seed if seed is None else seed
    rows = _read_manifest(config)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cohort = row[config.data.cohort_column]
        groups[(cohort, source_folder(row, config))].append(row)

    destination = Path(
        output_dir
        or Path(config.output.run_dir)
        / "source_preprocessing_audit"
        / f"seed_{selected_seed}_n{samples_per_source}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    failure_log = destination / "failures.jsonl"
    index_records: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []

    for group_index, ((cohort, folder), source_rows) in enumerate(
        sorted(groups.items()), start=1
    ):
        group_name = f"{_safe_name(cohort)}__{_safe_name(folder)}"
        selected = family_aware_random_sample(
            source_rows,
            samples_per_source,
            seed=selected_seed,
            group_key=group_name,
            filename_column=config.data.filename_column,
        )
        statuses = Counter()
        families = Counter()
        for sample_index, row in enumerate(selected, start=1):
            source = Path(row[config.data.absolute_path_column])
            filename = row[config.data.filename_column]
            family = filename_family(filename)
            families[family] += 1
            stem = f"{sample_index:02d}_{_safe_name(source.stem)}_{row[config.data.id_column][-8:]}"
            group_dir = destination / group_name
            source_copy = group_dir / "raw" / f"{stem}{source.suffix.lower()}"
            overlay_path = group_dir / "grid_overlay" / f"{stem}.jpg"
            outer_path = group_dir / "outer_crop" / f"{stem}.jpg"
            inset_label = _safe_name(f"inset_{100 * inset_fraction:.3f}_percent")
            inset_path = group_dir / f"{inset_label}_crop" / f"{stem}.jpg"
            comparison_path = group_dir / "comparison" / f"{stem}.jpg"
            record: dict[str, object] = {
                "cohort_id": cohort,
                "source_folder": folder,
                "filename_family": family,
                "image_id": row[config.data.id_column],
                "file_name": filename,
                "relative_path": row[config.data.relative_path_column],
                "absolute_path": str(source),
                "source_copy_path": str(source_copy.resolve()),
                "raw_width": "",
                "raw_height": "",
                "decode_status": "",
                "decode_warning": "",
                "grid_status": "failed",
                "grid_error": "",
                "grid_area_fraction": "",
                "outside_corner_count": "",
                "grid_points_json": "",
                "grid_overlay_path": "",
                "outer_crop_path": "",
                "inset_crop_path": "",
                "comparison_path": "",
                "review_complete_quadrat_visible": "",
                "review_grid_detection_correct": "",
                "review_inset_correct": "",
                "review_recommended_route": "",
                "review_notes": "",
            }
            try:
                if not source.is_file():
                    raise FileNotFoundError(f"Source image is missing: {source}")
                _copy_source(source, source_copy)
                raw, decode_status, decode_warning = _load_source_image(source)
                record.update(
                    raw_width=raw.width,
                    raw_height=raw.height,
                    decode_status=decode_status,
                    decode_warning=decode_warning,
                )
                image_bgr = image_to_ndarray(str(source))
                if image_bgr is None:
                    raise OSError("OpenCV could not decode the source image")
                grid_points, _, _ = detect_grid(image_bgr)
                overlay_bgr = _draw_grid_overlay(image_bgr, grid_points)
                outer_bgr = warp_big_square(
                    image_bgr, grid_points, size=crop_size, inner_margin_fraction=0.0
                )
                inset_bgr = warp_big_square(
                    image_bgr,
                    grid_points,
                    size=crop_size,
                    inner_margin_fraction=inset_fraction,
                )
                area_fraction, outside_count = _geometry(grid_points, image_bgr)
                _save_rgb(overlay_bgr, overlay_path)
                _save_rgb(outer_bgr, outer_path)
                _save_rgb(inset_bgr, inset_path)
                _save_comparison(
                    raw=raw,
                    overlay_bgr=overlay_bgr,
                    outer_bgr=outer_bgr,
                    inset_bgr=inset_bgr,
                    title=f"{filename} | cohort={cohort} | source={folder}",
                    destination=comparison_path,
                    inset_fraction=inset_fraction,
                )
                record.update(
                    grid_status="detected_needs_manual_review",
                    grid_area_fraction=f"{area_fraction:.6f}",
                    outside_corner_count=outside_count,
                    grid_points_json=json.dumps(np.asarray(grid_points).round(2).tolist()),
                    grid_overlay_path=str(overlay_path.resolve()),
                    outer_crop_path=str(outer_path.resolve()),
                    inset_crop_path=str(inset_path.resolve()),
                    comparison_path=str(comparison_path.resolve()),
                )
                statuses["detected_needs_manual_review"] += 1
            except Exception as error:  # noqa: BLE001 - every sampled failure belongs in the audit
                record["grid_error"] = f"{type(error).__name__}: {error}"
                statuses["failed"] += 1
                append_jsonl(
                    failure_log,
                    {
                        "cohort_id": cohort,
                        "source_folder": folder,
                        "image_id": row[config.data.id_column],
                        "file_name": filename,
                        "source_path": str(source),
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
            index_records.append(record)

        source_records.append(
            {
                "cohort_id": cohort,
                "source_folder": folder,
                "dataset_images": len(source_rows),
                "sampled_images": len(selected),
                "filename_families": json.dumps(dict(sorted(families.items()))),
                "grid_statuses": json.dumps(dict(sorted(statuses.items()))),
                "output_directory": str((destination / group_name).resolve()),
            }
        )
        print(
            f"[{group_index:02d}/{len(groups):02d}] {cohort} / {folder}: "
            f"sampled={len(selected)}, statuses={dict(statuses)}",
            flush=True,
        )

    index_path = destination / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_records)
    source_summary_path = destination / "sources.csv"
    with source_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_records[0]))
        writer.writeheader()
        writer.writerows(source_records)

    overall_status = Counter(str(record["grid_status"]) for record in index_records)
    summary = {
        "dataset_images": len(rows),
        "source_groups": len(groups),
        "sampled_images": len(index_records),
        "samples_per_source_requested": samples_per_source,
        "seed": selected_seed,
        "crop_size": crop_size,
        "inset_fraction": inset_fraction,
        "grid_statuses": dict(sorted(overall_status.items())),
        "directory": str(destination.resolve()),
        "index_csv": str(index_path.resolve()),
        "sources_csv": str(source_summary_path.resolve()),
        "failure_log": str(failure_log.resolve()),
        "source_images_modified": False,
        "manual_review_required": True,
        "review_instruction": (
            "Open each comparison image, then fill the review_* columns in index.csv. "
            "A successful detector call is not evidence that its geometry is correct."
        ),
    }
    write_json(destination / "summary.json", summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-source", type=int, default=8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inset-fraction", type=float, default=DEFAULT_INSET_FRACTION)
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--output-dir")
    arguments = parser.parse_args(argv)
    report = run(
        load_config(arguments.config),
        samples_per_source=arguments.samples_per_source,
        seed=arguments.seed,
        inset_fraction=arguments.inset_fraction,
        crop_size=arguments.crop_size,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
