"""Read-only inventory builder for the complete CSFB image collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage")
DEFAULT_OUTPUT = Path("outputs/dataset_inventory")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
IGNORED_DIRECTORY_NAMES = {"__MACOSX"}
PDF_REPORTED_TOTAL = 8_946


@dataclass(frozen=True)
class Cohort:
    cohort_id: str
    date_token: str
    partner: str
    location: str
    experiment: str
    sampling_date: str
    timepoint: str
    bbch: str
    expected_images: int
    label_status: str
    label_sources: str
    training_subset: bool = False


COHORTS = (
    Cohort(
        "gg_sulfur_bbch15",
        "2025_10_14",
        "JLU",
        "Gross-Gerau",
        "Sulfur",
        "2025-10-14",
        "single",
        "BBCH15",
        1271,
        "unlabeled",
        "",
    ),
    Cohort(
        "gg_insects_t1_bbch10",
        "2025_10_21",
        "JLU",
        "Gross-Gerau",
        "Insects",
        "2025-10-21",
        "T1",
        "BBCH10",
        937,
        "labeled_twice",
        "JLU+GAU",
    ),
    Cohort(
        "gg_insects_t2_bbch13",
        "2025_11_10",
        "JLU",
        "Gross-Gerau",
        "Insects",
        "2025-11-10",
        "T2",
        "BBCH13",
        937,
        "unlabeled",
        "",
    ),
    Cohort(
        "npzi_malchow_t1_bbch10",
        "2025_09_12",
        "NPZi",
        "Malchow",
        "Insects",
        "2025-09-12",
        "T1",
        "BBCH10",
        900,
        "unlabeled",
        "",
    ),
    Cohort(
        "npzi_malchow_t2_bbch13",
        "2025_09_19",
        "NPZi",
        "Malchow",
        "Insects",
        "2025-09-19",
        "T2",
        "BBCH13",
        900,
        "unlabeled",
        "",
    ),
    Cohort(
        "dsv_asendorf_t1_bbch11",
        "2025_09_15",
        "DSV",
        "Asendorf",
        "Insects",
        "2025-09-15_to_2025-09-17",
        "T1",
        "BBCH11",
        910,
        "labeled",
        "visual_score",
    ),
    Cohort(
        "dsv_asendorf_t2_bbch15",
        "2025_09_30",
        "DSV",
        "Asendorf",
        "Insects",
        "2025-09-30_to_2025-10-01",
        "T2",
        "BBCH15",
        900,
        "unlabeled",
        "",
    ),
    Cohort(
        "wg_insects_t1_bbch10",
        "2025_10_07",
        "JLU",
        "Weilburger Grenze",
        "Insects",
        "2025-10-07",
        "T1",
        "BBCH10",
        900,
        "labeled",
        "visual_score",
    ),
    Cohort(
        "wg_insects_t2_first_bbch15",
        "2025_10_20",
        "JLU",
        "Weilburger Grenze",
        "Insects",
        "2025-10-20",
        "T2_first_half",
        "BBCH15",
        360,
        "unlabeled",
        "",
    ),
    Cohort(
        "wg_insects_t2_second_bbch15",
        "2025_10_24",
        "JLU",
        "Weilburger Grenze",
        "Insects",
        "2025-10-24",
        "T2_second_half",
        "BBCH15",
        541,
        "unlabeled",
        "",
    ),
    Cohort(
        "rhh_insects_bbch13",
        "2025_09_29",
        "JLU",
        "Rauischholzhausen",
        "Insects",
        "2025-09-29",
        "single",
        "BBCH13",
        900,
        "unlabeled",
        "",
    ),
)

TRAINING_SUBSET = Cohort(
    "gg_reliable_training_subset_bbch10",
    "",
    "JLU+GAU",
    "Gross-Gerau",
    "Insects",
    "2025-10-21",
    "T1",
    "BBCH10",
    470,
    "reliable_calibration_subset",
    "JLU+GAU_less_than_5_percent_difference",
    True,
)


IMAGE_COLUMNS = (
    "image_id",
    "absolute_path",
    "relative_path",
    "top_level_folder",
    "file_name",
    "extension",
    "file_size_bytes",
    "modified_time_ns",
    "sha256",
    "duplicate_group_id",
    "duplicate_count",
    "duplicate_role",
    "canonical_relative_path",
    "cohort_id",
    "partner",
    "location",
    "experiment",
    "sampling_date",
    "timepoint",
    "bbch",
    "expected_cohort_images",
    "label_status",
    "label_sources",
    "is_training_subset",
    "width",
    "height",
    "image_mode",
    "image_read_status",
    "image_read_error",
    "qr_payload",
    "qr_status",
    "plot_group_id",
    "plot_group_size_all_dates",
    "views_in_same_cohort_and_plot",
    "view_index_within_cohort_and_plot",
)


def _normalized(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def classify(relative_path: Path) -> Cohort | None:
    normalized = _normalized(relative_path.as_posix())
    compact = normalized.replace("_", "")
    if "phenotyping_training_set" in normalized:
        return TRAINING_SUBSET
    for cohort in COHORTS:
        token = cohort.date_token.lower()
        if token in normalized or token.replace("_", "") in compact:
            return cohort
    return None


def discover_files(root: Path, extensions: set[str] | None = None):
    extensions = extensions or IMAGE_EXTENSIONS
    images, score_files, other_root_files = [], [], []
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = [
            name
            for name in names
            if name not in IGNORED_DIRECTORY_NAMES and not name.startswith("._")
        ]
        base = Path(directory)
        for filename in filenames:
            if filename.startswith("._"):
                continue
            path = base / filename
            suffix = path.suffix.lower()
            if suffix in extensions:
                images.append(path)
            elif suffix == ".csv":
                score_files.append(path)
            elif base == root:
                other_root_files.append(path)
    return sorted(images), sorted(score_files), sorted(other_root_files)


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def image_metadata(path: Path) -> tuple[str, str, str, str, str]:
    try:
        from PIL import Image
    except ImportError:
        return "", "", "", "metadata_backend_unavailable", "Pillow is not installed"
    try:
        with Image.open(path) as image:
            width, height, mode = image.width, image.height, image.mode
            image.verify()
        return str(width), str(height), str(mode), "ok", ""
    except (OSError, ValueError) as error:
        return "", "", "", "unreadable", f"{type(error).__name__}: {error}"


class QRDecoder:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.error = ""
        self.cv2 = None
        self.detector = None
        if not enabled:
            return
        try:
            import cv2

            self.cv2 = cv2
            self.detector = cv2.QRCodeDetector()
        except ImportError as error:
            self.error = f"{type(error).__name__}: {error}"

    @property
    def available(self) -> bool:
        return self.detector is not None

    def decode(self, path: Path) -> tuple[str, str]:
        if not self.enabled:
            return "", "not_requested"
        if not self.available:
            return "", "backend_unavailable"
        try:
            image = self.cv2.imread(str(path), self.cv2.IMREAD_COLOR)
            if image is None:
                return "", "image_unreadable"
            payload, points, _ = self.detector.detectAndDecode(image)
            if payload:
                return payload.strip(), "decoded"
            return "", "detected_not_decoded" if points is not None else "not_detected"
        except Exception as error:  # noqa: BLE001 - a corrupt image must not abort the inventory
            return "", f"error:{type(error).__name__}"


def _canonical_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row["is_training_subset"]),
        len(Path(row["relative_path"]).parts),
        row["relative_path"],
    )


def _cohort_values(cohort: Cohort | None) -> dict[str, Any]:
    if cohort is None:
        return {
            "cohort_id": "unclassified",
            "partner": "",
            "location": "",
            "experiment": "",
            "sampling_date": "",
            "timepoint": "",
            "bbch": "",
            "expected_cohort_images": "",
            "label_status": "unknown",
            "label_sources": "",
            "is_training_subset": False,
        }
    values = asdict(cohort)
    values.pop("date_token")
    values["expected_cohort_images"] = values.pop("expected_images")
    values["is_training_subset"] = values.pop("training_subset")
    return values


def scan_images(
    root: Path,
    paths: list[Path],
    *,
    decode_qr: bool,
    skip_hash: bool,
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    decoder = QRDecoder(decode_qr)
    warnings = []
    if decode_qr and not decoder.available:
        warnings.append(f"QR decoding requested but OpenCV is unavailable: {decoder.error}")
    rows = []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(root)
        stat = path.stat()
        digest = "" if skip_hash else file_sha256(path)
        width, height, mode, image_status, image_error = image_metadata(path)
        qr_payload, qr_status = decoder.decode(path)
        path_identity = hashlib.sha256(relative.as_posix().encode()).hexdigest()
        row = {
            "image_id": f"img_{(digest or path_identity)[:20]}",
            "absolute_path": str(path.resolve()),
            "relative_path": relative.as_posix(),
            "top_level_folder": relative.parts[0] if len(relative.parts) > 1 else "__ROOT__",
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns,
            "sha256": digest,
            "duplicate_group_id": "",
            "duplicate_count": 1,
            "duplicate_role": "not_checked" if skip_hash else "unique",
            "canonical_relative_path": relative.as_posix(),
            **_cohort_values(classify(relative)),
            "width": width,
            "height": height,
            "image_mode": mode,
            "image_read_status": image_status,
            "image_read_error": image_error,
            "qr_payload": qr_payload,
            "qr_status": qr_status,
            "plot_group_id": (
                f"plot_{hashlib.sha256(qr_payload.encode()).hexdigest()[:20]}"
                if qr_payload
                else ""
            ),
            "plot_group_size_all_dates": "",
            "views_in_same_cohort_and_plot": "",
            "view_index_within_cohort_and_plot": "",
        }
        rows.append(row)
        if progress_every and (index % progress_every == 0 or index == total):
            print(f"Scanned {index:,}/{total:,} images", file=sys.stderr, flush=True)
    _annotate_duplicates(rows, skip_hash)
    _annotate_plot_groups(rows)
    return rows, warnings


def _annotate_duplicates(rows: list[dict[str, Any]], skip_hash: bool) -> None:
    if skip_hash:
        return
    groups = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)
    for digest, members in groups.items():
        if len(members) == 1:
            continue
        members.sort(key=_canonical_key)
        canonical = members[0]["relative_path"]
        group_id = f"dup_{digest[:20]}"
        for index, row in enumerate(members):
            row["duplicate_group_id"] = group_id
            row["duplicate_count"] = len(members)
            row["duplicate_role"] = "canonical" if index == 0 else "copy"
            row["canonical_relative_path"] = canonical


def _annotate_plot_groups(rows: list[dict[str, Any]]) -> None:
    plot_groups = defaultdict(list)
    cohort_plot_groups = defaultdict(list)
    for row in rows:
        if row["plot_group_id"]:
            plot_groups[row["plot_group_id"]].append(row)
            cohort_plot_groups[(row["cohort_id"], row["plot_group_id"])].append(row)
    for members in plot_groups.values():
        for row in members:
            row["plot_group_size_all_dates"] = len(members)
    for members in cohort_plot_groups.values():
        members.sort(key=lambda row: row["relative_path"])
        for index, row in enumerate(members, start=1):
            row["views_in_same_cohort_and_plot"] = len(members)
            row["view_index_within_cohort_and_plot"] = index


def inspect_score_file(root: Path, path: Path) -> dict[str, Any]:
    result = {
        "absolute_path": str(path.resolve()),
        "relative_path": path.relative_to(root).as_posix(),
        "file_size_bytes": path.stat().st_size,
        "row_count": "",
        "column_count": "",
        "columns": "",
        "delimiter": "",
        "read_status": "",
        "read_error": "",
    }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(64 * 1024)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(handle, dialect)
            header = next(reader, [])
            rows = sum(1 for row in reader if any(cell.strip() for cell in row))
        result.update(
            {
                "row_count": rows,
                "column_count": len(header),
                "columns": "|".join(header),
                "delimiter": repr(dialect.delimiter),
                "read_status": "ok",
            }
        )
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        result.update(
            {
                "read_status": "error",
                "read_error": f"{type(error).__name__}: {error}",
            }
        )
    return result


def _aggregate_cohorts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["cohort_id"]].append(row)
    output = []
    for cohort_id, members in sorted(groups.items()):
        expected = members[0]["expected_cohort_images"]
        unique_hashes = {row["sha256"] for row in members if row["sha256"]}
        canonical = sum(row["duplicate_role"] != "copy" for row in members)
        decoded = sum(row["qr_status"] == "decoded" for row in members)
        output.append(
            {
                "cohort_id": cohort_id,
                "partner": members[0]["partner"],
                "location": members[0]["location"],
                "experiment": members[0]["experiment"],
                "sampling_date": members[0]["sampling_date"],
                "timepoint": members[0]["timepoint"],
                "bbch": members[0]["bbch"],
                "label_status": members[0]["label_status"],
                "expected_images_from_pdf_table": expected,
                "discovered_image_paths": len(members),
                "canonical_images": canonical,
                "unique_content_hashes": len(unique_hashes) if unique_hashes else "",
                "duplicate_copies": sum(row["duplicate_role"] == "copy" for row in members),
                "readable_images": sum(row["image_read_status"] == "ok" for row in members),
                "qr_decoded_images": decoded,
                "unique_plot_groups": len(
                    {row["plot_group_id"] for row in members if row["plot_group_id"]}
                ),
                "path_count_minus_expected": (
                    len(members) - int(expected) if expected != "" else ""
                ),
            }
        )
    return output


def _duplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        if row["duplicate_group_id"]:
            groups[row["duplicate_group_id"]].append(row)
    output = []
    for group_id, members in sorted(groups.items()):
        members.sort(key=_canonical_key)
        output.append(
            {
                "duplicate_group_id": group_id,
                "sha256": members[0]["sha256"],
                "copies": len(members),
                "canonical_relative_path": members[0]["canonical_relative_path"],
                "cohort_ids": "|".join(sorted({row["cohort_id"] for row in members})),
                "relative_paths": "|".join(row["relative_path"] for row in members),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], columns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    root: Path,
    rows: list[dict[str, Any]],
    cohorts: list[dict[str, Any]],
    score_files: list[dict[str, Any]],
    other_root_files: list[Path],
    warnings: list[str],
    *,
    hashing_enabled: bool,
    qr_requested: bool,
) -> dict[str, Any]:
    raw_cohort_expected_total = sum(cohort.expected_images for cohort in COHORTS)
    duplicate_groups = {row["duplicate_group_id"] for row in rows if row["duplicate_group_id"]}
    hashes = {row["sha256"] for row in rows if row["sha256"]}
    filename_counts = Counter(row["file_name"].lower() for row in rows)
    top_level = defaultdict(lambda: {"image_paths": 0, "canonical_images": 0})
    for row in rows:
        group = top_level[row["top_level_folder"]]
        group["image_paths"] += 1
        group["canonical_images"] += int(row["duplicate_role"] != "copy")
    generated_warnings = list(warnings)
    if raw_cohort_expected_total != PDF_REPORTED_TOTAL:
        generated_warnings.append(
            f"PDF headline total {PDF_REPORTED_TOTAL} differs from its acquisition table sum "
            f"{raw_cohort_expected_total}."
        )
    if any(row["cohort_id"] == "unclassified" for row in rows):
        generated_warnings.append("Some images could not be assigned to a documented cohort.")
    if any(row["image_read_status"] == "unreadable" for row in rows):
        generated_warnings.append("One or more images could not be opened by Pillow.")
    if any(row["image_read_status"] == "metadata_backend_unavailable" for row in rows):
        generated_warnings.append(
            "Pillow is unavailable, so dimensions and image integrity were not checked."
        )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_root": str(root.resolve()),
        "settings": {
            "content_hashing_enabled": hashing_enabled,
            "qr_decoding_requested": qr_requested,
            "ignored_directory_names": sorted(IGNORED_DIRECTORY_NAMES),
            "image_extensions": sorted(IMAGE_EXTENSIONS),
        },
        "documentation": {
            "pdf_reported_total_images": PDF_REPORTED_TOTAL,
            "pdf_acquisition_table_sum": raw_cohort_expected_total,
            "pdf_training_subset_images": TRAINING_SUBSET.expected_images,
        },
        "totals": {
            "discovered_image_paths": len(rows),
            "canonical_image_paths": sum(row["duplicate_role"] != "copy" for row in rows),
            "unique_content_hashes": len(hashes) if hashing_enabled else None,
            "duplicate_groups": len(duplicate_groups) if hashing_enabled else None,
            "duplicate_copies": sum(row["duplicate_role"] == "copy" for row in rows),
            "readable_images": sum(row["image_read_status"] == "ok" for row in rows),
            "unreadable_images": sum(row["image_read_status"] == "unreadable" for row in rows),
            "qr_decoded_images": sum(row["qr_status"] == "decoded" for row in rows),
            "unique_qr_plot_groups": len(
                {row["plot_group_id"] for row in rows if row["plot_group_id"]}
            ),
            "score_csv_files": len(score_files),
            "case_insensitive_filename_collision_groups": sum(
                count > 1 for count in filename_counts.values()
            ),
        },
        "top_level_folders": [
            {"name": name, **counts} for name, counts in sorted(top_level.items())
        ],
        "cohorts": cohorts,
        "score_files": score_files,
        "other_root_files": [path.name for path in other_root_files],
        "warnings": generated_warnings,
    }


def run(
    root: Path,
    output_dir: Path,
    *,
    decode_qr: bool = False,
    skip_hash: bool = False,
    limit: int | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    root = root.expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist or is not a directory: {root}")
    images, score_paths, other_root_files = discover_files(root)
    discovered_before_limit = len(images)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"No supported images found below {root}")
    rows, warnings = scan_images(
        root,
        images,
        decode_qr=decode_qr,
        skip_hash=skip_hash,
        progress_every=progress_every,
    )
    if limit is not None:
        warnings.append(
            f"Smoke-test limit active: scanned {len(rows)} of {discovered_before_limit} paths; "
            "cohort totals are not a complete dataset audit."
        )
    score_files = [inspect_score_file(root, path) for path in score_paths]
    cohorts = _aggregate_cohorts(rows)
    duplicates = _duplicate_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "dataset_images.csv", rows, IMAGE_COLUMNS)
    _write_csv(
        output_dir / "dataset_cohorts.csv",
        cohorts,
        tuple(cohorts[0]) if cohorts else (),
    )
    duplicate_columns = (
        "duplicate_group_id",
        "sha256",
        "copies",
        "canonical_relative_path",
        "cohort_ids",
        "relative_paths",
    )
    _write_csv(output_dir / "dataset_duplicates.csv", duplicates, duplicate_columns)
    score_columns = (
        "absolute_path",
        "relative_path",
        "file_size_bytes",
        "row_count",
        "column_count",
        "columns",
        "delimiter",
        "read_status",
        "read_error",
    )
    _write_csv(output_dir / "dataset_score_files.csv", score_files, score_columns)
    summary = build_summary(
        root,
        rows,
        cohorts,
        score_files,
        other_root_files,
        warnings,
        hashing_enabled=not skip_hash,
        qr_requested=decode_qr,
    )
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a read-only CSV/JSON inventory of the full CSFB image dataset."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--decode-qr",
        action="store_true",
        help="Attempt OpenCV QR decoding for plot grouping (slower and optional).",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip SHA-256 content hashing and exact duplicate detection (not recommended).",
    )
    parser.add_argument("--limit", type=int, help="Scan only the first N images for a smoke test.")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)
    summary = run(
        args.root,
        args.output_dir,
        decode_qr=args.decode_qr,
        skip_hash=args.skip_hash,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
