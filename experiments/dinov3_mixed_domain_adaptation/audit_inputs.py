"""Build a representative visual audit before any full adaptation-data pass."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from rapeseed_damage.artifacts import write_json

from .config import Config, load_config
from .inspect_preprocessing import _save_preview
from .inspect_sources import _evenly_spaced, _load_source_image, source_folder
from .prepare_inputs import _read_source_manifest
from .preprocessing import RAW_TILED_MODE, score_tile_candidates, serialize_tile_candidates


def filename_family(filename: str) -> str:
    """Keep camera-style IMG files and timestamp files represented separately."""
    return "img" if Path(filename).stem.upper().startswith("IMG") else "timestamp"


def representative_sample(
    rows: list[dict[str, str]], count: int, config: Config
) -> list[dict[str, str]]:
    """Allocate an exact-size, source-stratified sample and span each capture sequence."""
    if count < 1:
        raise ValueError("Audit sample size must be positive")
    if count >= len(rows):
        return rows.copy()
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row[config.data.cohort_column],
            source_folder(row, config),
            filename_family(row[config.data.filename_column]),
        )
        groups[key].append(row)
    if count < len(groups):
        raise ValueError(
            f"Audit size {count} is smaller than the {len(groups)} source/family groups"
        )

    # Start with one image from every group, then use a deterministic divisor
    # allocation so larger sources receive proportionally more of the remaining slots.
    quotas = {key: 1 for key in groups}
    for _ in range(count - len(groups)):
        eligible = [key for key, values in groups.items() if quotas[key] < len(values)]
        if not eligible:
            break
        chosen = max(
            eligible,
            key=lambda key: (len(groups[key]) / (quotas[key] + 1), key),
        )
        quotas[chosen] += 1

    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        selected.extend(
            _evenly_spaced(
                groups[key], quotas[key], config.data.filename_column
            )
        )
    return selected


def _clear_previous_audit(destination: Path) -> None:
    """Remove only artifacts owned by this audit so stale previews cannot survive."""
    destination.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.jpg", "index.csv", "summary.json", "failures.jsonl"):
        for path in destination.glob(pattern):
            if path.is_file():
                path.unlink()


def run(config: Config, *, sample_size: int | None = None) -> dict:
    rows = _read_source_manifest(config)
    requested = sample_size or config.output.audit_sample_size
    selected = representative_sample(rows, requested, config)
    destination = Path(config.output.run_dir) / config.output.audit_dir
    _clear_previous_audit(destination)

    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    decode_statuses: Counter[str] = Counter()
    for position, row in enumerate(selected, start=1):
        path = Path(row[config.data.absolute_path_column])
        filename = row[config.data.filename_column]
        try:
            image, decode_status, decode_warning = _load_source_image(path)
            decode_statuses[decode_status] += 1
            candidates = score_tile_candidates(image, config)
            record = {
                "image_id": row[config.data.id_column],
                "file_name": filename,
                "cohort_id": row[config.data.cohort_column],
                "source_path": str(path),
                "input_mode": RAW_TILED_MODE,
                "tile_candidates": serialize_tile_candidates(candidates),
            }
            preview, selections, label_fraction, vegetation_fraction = _save_preview(
                record,
                config,
                destination,
                position,
                image=image,
            )
            records.append(
                {
                    "position": position,
                    "cohort_id": record["cohort_id"],
                    "source_folder": source_folder(row, config),
                    "filename_family": filename_family(filename),
                    "image_id": record["image_id"],
                    "file_name": filename,
                    "source_path": str(path),
                    "decode_status": decode_status,
                    "decode_warning": decode_warning,
                    "probable_label_fraction": label_fraction,
                    "probable_vegetation_fraction": vegetation_fraction,
                    "sampled_grids": json.dumps(
                        [selection.grid_size for selection in selections]
                    ),
                    "preview_path": str(preview.resolve()),
                }
            )
            print(
                f"[{position:03d}/{len(selected):03d}] {decode_status:19s} {preview.name}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - audit must report every bad sample.
            decode_statuses["failed"] += 1
            failure = {
                "position": str(position),
                "cohort_id": row[config.data.cohort_column],
                "source_folder": source_folder(row, config),
                "filename_family": filename_family(filename),
                "file_name": filename,
                "source_path": str(path),
                "error_type": type(error).__name__,
                "message": str(error),
            }
            failures.append(failure)
            print(
                f"[{position:03d}/{len(selected):03d}] FAILED {filename}: {error}",
                flush=True,
            )

    index_path = destination / "index.csv"
    if records:
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    failure_path = destination / "failures.jsonl"
    if failures:
        with failure_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, sort_keys=True) + "\n")

    summary = {
        "dataset_images": len(rows),
        "requested_sample_size": requested,
        "selected_images": len(selected),
        "preview_files": len(records),
        "decode_statuses": dict(sorted(decode_statuses.items())),
        "source_groups": len(
            {
                (record["cohort_id"], record["source_folder"], record["filename_family"])
                for record in records
            }
        ),
        "directory": str(destination.resolve()),
        "index_csv": str(index_path.resolve()),
        "failures_jsonl": str(failure_path.resolve()) if failures else None,
        "prepared_manifest_modified": False,
        "full_dataset_processed": False,
        "status": "visual_review_required",
        "next_step": (
            "Open all preview JPEGs and review index.csv. Only after approval run "
            "prepare_inputs with --full."
        ),
    }
    write_json(destination / "summary.json", summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-size", type=int)
    arguments = parser.parse_args(argv)
    report = run(load_config(arguments.config), sample_size=arguments.sample_size)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
