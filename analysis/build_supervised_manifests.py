"""Join CSFB score tables to canonical images and build leakage-safe manifests.

The 470-image calibration folder is a copied subset of the Gross-Gerau T1
cohort.  It is treated as gold supervision.  The remaining double-scored
Gross-Gerau images and the single-scored DSV/WG images are retained as weaker
supervision for a separate pretraining stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage")
DEFAULT_INVENTORY = Path("outputs/dataset_inventory/dataset_images.csv")
DEFAULT_OUTPUT = Path("outputs/dataset_manifests")
DEFAULT_ADAPTATION_EXCLUDED_COHORTS = (
    "gg_insects_t1_bbch10",
    "gg_insects_t2_bbch13",
)

MANIFEST_COLUMNS = (
    "image_id",
    "absolute_path",
    "relative_path",
    "file_name",
    "sha256",
    "cohort_id",
    "partner",
    "location",
    "experiment",
    "sampling_date",
    "timepoint",
    "bbch",
    "target",
    "score_single",
    "score_jlu",
    "score_gau",
    "annotator_difference",
    "is_gold_standard",
    "gold_threshold_points",
    "supervision_tier",
    "sample_weight",
    "score_sources",
    "score_qr_code",
    "plot_number",
    "genotype",
    "plot_group_id",
    "plot_group_source",
    "split",
)


@dataclass(frozen=True)
class BuildSettings:
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    gold_difference_threshold: float = 5.0
    gold_weight: float = 1.0
    dual_weak_weight: float = 0.6
    single_weak_weight: float = 0.4
    adaptation_excluded_cohorts: tuple[str, ...] = DEFAULT_ADAPTATION_EXCLUDED_COHORTS

    def validate(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be below 1")
        if self.gold_difference_threshold <= 0:
            raise ValueError("gold_difference_threshold must be positive")
        if min(self.gold_weight, self.dual_weak_weight, self.single_weak_weight) <= 0:
            raise ValueError("all supervision weights must be positive")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def _number(value: Any) -> float | None:
    text = _clean(value).replace("%", "").replace(" ", "")
    if not text:
        return None
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _truth(value: Any) -> bool:
    return _clean(value).casefold() in {"true", "1", "yes"}


def _normalized_filename(value: Any) -> str:
    return Path(_clean(value).replace("\\", "/")).name.casefold()


def _stem(value: Any) -> str:
    return Path(_normalized_filename(value)).stem


def _normalized_code(value: Any) -> str:
    return " ".join(_clean(value).split()).casefold()


def _hash_id(prefix: str, *parts: str) -> str:
    material = "|".join(parts)
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:20]}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(64 * 1024)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"Score CSV has no header: {path}")
        return [
            {str(key).strip(): _clean(value) for key, value in row.items() if key is not None}
            for row in reader
            if any(_clean(value) for value in row.values())
        ]


def _field(row: dict[str, str], *names: str) -> str:
    normalized = {key.casefold().replace("-", "_"): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name.casefold().replace("-", "_"), "")
        if _clean(value):
            return _clean(value)
    return ""


def _score_source(path: Path) -> tuple[str, str]:
    name = path.name.casefold()
    if "training_set_scores" in name:
        return "gold", "gg_insects_t1_bbch10"
    if "2025_10_21" in name and "gg1" in name:
        return "dual_weak", "gg_insects_t1_bbch10"
    if "2025_09_15" in name and "dsv" in name:
        return "single_weak", "dsv_asendorf_t1_bbch11"
    if "2025_10_07" in name and "wg1" in name:
        return "single_weak", "wg_insects_t1_bbch10"
    raise ValueError(f"Unrecognized score-table naming scheme: {path.name}")


def discover_score_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.csv"):
        if "__MACOSX" in path.parts:
            continue
        try:
            _score_source(path)
        except ValueError:
            continue
        files.append(path)
    return sorted(files, key=lambda path: (_score_source(path)[0] == "gold", str(path)))


class ImageIndex:
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows
        self.by_relative = {row["relative_path"]: row for row in rows}
        self.by_cohort_name: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        self.by_cohort_stem: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            self.by_cohort_name[(row["cohort_id"], _normalized_filename(row["file_name"]))].append(
                row
            )
            self.by_cohort_stem[(row["cohort_id"], _stem(row["file_name"]))].append(row)

    def canonical(self, row: dict[str, str]) -> dict[str, str]:
        path = row.get("canonical_relative_path") or row["relative_path"]
        try:
            return self.by_relative[path]
        except KeyError as error:
            raise ValueError(f"Canonical inventory path is absent: {path}") from error

    def match(self, cohort_id: str, filename: str) -> tuple[dict[str, str] | None, str]:
        candidates = self.by_cohort_name.get((cohort_id, _normalized_filename(filename)), [])
        if not candidates:
            candidates = self.by_cohort_stem.get((cohort_id, _stem(filename)), [])
        canonical = {
            row.get("canonical_relative_path") or row["relative_path"] for row in candidates
        }
        if len(canonical) == 1:
            return self.by_relative[canonical.pop()], ""
        if not canonical:
            return None, "unmatched_filename"
        return None, "ambiguous_filename"


def load_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "absolute_path",
        "relative_path",
        "file_name",
        "sha256",
        "canonical_relative_path",
        "cohort_id",
        "duplicate_role",
    }
    if not rows or not required.issubset(rows[0]):
        missing = required - (set(rows[0]) if rows else set())
        raise ValueError(f"Inventory is empty or missing columns: {', '.join(sorted(missing))}")
    return rows


def _score_values(row: dict[str, str], source_tier: str) -> dict[str, Any]:
    single = _number(_field(row, "Score", "visual_score"))
    jlu = _number(_field(row, "Score_JLU"))
    gau = _number(_field(row, "Score_GAU"))
    supplied_mean = _number(_field(row, "mean_score"))
    supplied_difference = _number(_field(row, "diff", "difference"))
    if supplied_mean is not None:
        target = supplied_mean
    elif jlu is not None and gau is not None:
        target = 0.5 * (jlu + gau)
    elif single is not None:
        target = single
    elif jlu is not None:
        target = jlu
    elif gau is not None:
        target = gau
    else:
        target = None
    difference = abs(jlu - gau) if jlu is not None and gau is not None else supplied_difference
    return {
        "target": target,
        "score_single": single,
        "score_jlu": jlu,
        "score_gau": gau,
        "annotator_difference": difference,
        "source_tier": source_tier,
    }


def _plot_group(
    image: dict[str, str], *, score_qr: str = "", plot_number: str = ""
) -> tuple[str, str]:
    scope = "|".join(
        (_normalized_code(image.get("location")), _normalized_code(image.get("experiment")))
    )
    qr = _normalized_code(score_qr)
    if qr:
        return _hash_id("plot", scope, "qr", qr), "score_qr"
    plot = _normalized_code(plot_number)
    if plot:
        return _hash_id("plot", scope, "plot_number", plot), "score_plot_number"
    decoded = _normalized_code(image.get("qr_payload"))
    if decoded:
        return _hash_id("plot", scope, "qr", decoded), "image_qr"
    identity = image.get("sha256") or image["relative_path"]
    return _hash_id("image", identity), "individual_image"


def join_scores(
    root: Path,
    inventory_rows: list[dict[str, str]],
    settings: BuildSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    index = ImageIndex(inventory_rows)
    joined: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = []
    source_reports = []
    seen_score_images: set[tuple[str, str]] = set()
    for path in discover_score_files(root):
        tier, cohort_id = _score_source(path)
        matched = usable = 0
        rows = _read_csv(path)
        for position, raw in enumerate(rows, start=2):
            filename = _field(raw, "Filename", "file_name", "Image", "image_name")
            image, error = index.match(cohort_id, filename)
            if image is None:
                issues.append(
                    {
                        "score_file": str(path.relative_to(root)),
                        "row_number": str(position),
                        "filename": filename,
                        "issue": error,
                    }
                )
                continue
            matched += 1
            values = _score_values(raw, tier)
            if values["target"] is None:
                issues.append(
                    {
                        "score_file": str(path.relative_to(root)),
                        "row_number": str(position),
                        "filename": filename,
                        "issue": "missing_numeric_target",
                    }
                )
                continue
            canonical_key = image["canonical_relative_path"]
            score_image_key = (str(path.resolve()), canonical_key)
            if score_image_key in seen_score_images:
                issues.append(
                    {
                        "score_file": str(path.relative_to(root)),
                        "row_number": str(position),
                        "filename": filename,
                        "issue": "duplicate_score_row_for_image",
                    }
                )
                continue
            seen_score_images.add(score_image_key)
            usable += 1
            record = joined.setdefault(
                canonical_key,
                {
                    **{key: image.get(key, "") for key in MANIFEST_COLUMNS if key in image},
                    "target": values["target"],
                    "score_single": "",
                    "score_jlu": "",
                    "score_gau": "",
                    "annotator_difference": "",
                    "is_gold_standard": False,
                    "gold_threshold_points": settings.gold_difference_threshold,
                    "supervision_tier": tier,
                    "sample_weight": "",
                    "score_sources": "",
                    "score_qr_code": "",
                    "plot_number": "",
                    "genotype": "",
                    "plot_group_id": "",
                    "plot_group_source": "",
                    "split": "",
                },
            )
            for key in ("score_single", "score_jlu", "score_gau", "annotator_difference"):
                if values[key] is not None:
                    record[key] = values[key]
            score_qr = _field(raw, "QR-Code", "QR_Code", "qr", "qrcode")
            plot_number = _field(raw, "Plotnr", "plot_number", "plot")
            genotype = _field(raw, "Genotyp", "genotype", "variety")
            if score_qr:
                record["score_qr_code"] = score_qr
            if plot_number:
                record["plot_number"] = plot_number
            if genotype:
                record["genotype"] = genotype
            sources = {part for part in str(record["score_sources"]).split("|") if part}
            sources.add(str(path.relative_to(root)))
            record["score_sources"] = "|".join(sorted(sources))
            if tier == "gold":
                record["target"] = values["target"]
                record["is_gold_standard"] = True
                record["supervision_tier"] = "gold"
            elif record["supervision_tier"] != "gold":
                record["target"] = values["target"]
                record["supervision_tier"] = (
                    "single_weak"
                    if tier == "dual_weak"
                    and (values["score_jlu"] is None or values["score_gau"] is None)
                    else tier
                )
        source_reports.append(
            {
                "score_file": str(path.relative_to(root)),
                "source_tier": tier,
                "rows": len(rows),
                "matched_images": matched,
                "usable_targets": usable,
            }
        )

    weights = {
        "gold": settings.gold_weight,
        "dual_weak": settings.dual_weak_weight,
        "single_weak": settings.single_weak_weight,
    }
    gold_threshold_violations = 0
    for record in joined.values():
        tier = record["supervision_tier"]
        weight = weights[tier]
        if tier == "dual_weak" and record["annotator_difference"] != "":
            difference = float(record["annotator_difference"])
            reliability = min(
                1.0,
                settings.gold_difference_threshold
                / max(difference, settings.gold_difference_threshold),
            )
            weight = max(0.1, weight * reliability)
        record["sample_weight"] = weight
        image = index.by_relative[record["relative_path"]]
        group, source = _plot_group(
            image,
            score_qr=str(record["score_qr_code"]),
            plot_number=str(record["plot_number"]),
        )
        record["plot_group_id"] = group
        record["plot_group_source"] = source
        if (
            record["is_gold_standard"]
            and record["annotator_difference"] != ""
            and float(record["annotator_difference"]) > settings.gold_difference_threshold
        ):
            gold_threshold_violations += 1
    source_reports.append(
        {
            "score_file": "__gold_validation__",
            "source_tier": "gold",
            "rows": sum(bool(record["is_gold_standard"]) for record in joined.values()),
            "matched_images": "",
            "usable_targets": "",
            "threshold_violations": gold_threshold_violations,
        }
    )
    return list(joined.values()), issues, source_reports


def _target_bin(value: float) -> str:
    if value <= 2.5:
        return "0_to_2_5"
    if value <= 7.5:
        return "over_2_5_to_7_5"
    if value <= 15:
        return "over_7_5_to_15"
    return "over_15"


def split_gold_groups(records: list[dict[str, Any]], settings: BuildSettings) -> dict[str, str]:
    gold = [record for record in records if record["is_gold_standard"]]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in gold:
        groups[str(record["plot_group_id"])].append(record)
    if len(groups) < 3:
        raise ValueError("At least three gold plot groups are required for train/validation/test")

    by_bin: dict[str, list[tuple[str, list[dict[str, Any]], float]]] = defaultdict(list)
    for group_id, members in groups.items():
        mean_target = sum(float(member["target"]) for member in members) / len(members)
        by_bin[_target_bin(mean_target)].append((group_id, members, mean_target))

    assignment: dict[str, str] = {}
    rng = random.Random(settings.seed)
    for bin_name in sorted(by_bin):
        items = by_bin[bin_name]
        rng.shuffle(items)
        # Process the hardest end of every target range first. The deficit-based
        # assignment then places the first three groups into train/validation/test,
        # preventing both holdouts from accidentally missing the severe-damage tail.
        items.sort(key=lambda item: (item[2], len(item[1])), reverse=True)
        samples = sum(len(members) for _, members, _ in items)
        targets = {
            "validation": samples * settings.validation_fraction,
            "test": samples * settings.test_fraction,
            "gold_train": samples * (1 - settings.validation_fraction - settings.test_fraction),
        }
        counts = Counter()
        priority = {"gold_train": 2, "validation": 1, "test": 0}
        for group_id, members, _ in items:
            choices = sorted(
                targets,
                key=lambda split: (
                    (targets[split] - counts[split]) / max(targets[split], 1.0),
                    priority[split],
                ),
                reverse=True,
            )
            selected = choices[0]
            assignment[group_id] = selected
            counts[selected] += len(members)
    if set(assignment.values()) != {"gold_train", "validation", "test"}:
        raise ValueError("Gold split construction did not produce all three partitions")
    return assignment


def assign_splits(records: list[dict[str, Any]], settings: BuildSettings) -> set[str]:
    gold_assignment = split_gold_groups(records, settings)
    holdout_groups = {
        group for group, split in gold_assignment.items() if split in {"validation", "test"}
    }
    for record in records:
        group = str(record["plot_group_id"])
        if record["is_gold_standard"]:
            record["split"] = gold_assignment[group]
        elif group in holdout_groups:
            record["split"] = "excluded_holdout_related"
        else:
            record["split"] = "weak_pretrain"
    return holdout_groups


def adaptation_rows(
    inventory_rows: list[dict[str, str]],
    scored: list[dict[str, Any]],
    holdout_groups: set[str],
    settings: BuildSettings,
) -> tuple[list[dict[str, Any]], Counter]:
    scored_by_relative = {str(record["relative_path"]): record for record in scored}
    output, exclusions = [], Counter()
    for image in inventory_rows:
        if image["duplicate_role"] == "copy":
            exclusions["duplicate_copy"] += 1
            continue
        if image["cohort_id"] in settings.adaptation_excluded_cohorts:
            exclusions["conservative_target_cohort_exclusion"] += 1
            continue
        scored_record = scored_by_relative.get(image["relative_path"])
        if scored_record is not None:
            group = str(scored_record["plot_group_id"])
        else:
            group, _ = _plot_group(image)
        if group in holdout_groups:
            exclusions["holdout_plot"] += 1
            continue
        output.append(
            {
                "image_id": image["image_id"],
                "absolute_path": image["absolute_path"],
                "relative_path": image["relative_path"],
                "file_name": image["file_name"],
                "sha256": image["sha256"],
                "cohort_id": image["cohort_id"],
                "partner": image.get("partner", ""),
                "location": image.get("location", ""),
                "experiment": image.get("experiment", ""),
                "sampling_date": image.get("sampling_date", ""),
                "timepoint": image.get("timepoint", ""),
                "bbch": image.get("bbch", ""),
                "plot_group_id": group,
            }
        )
    return output, exclusions


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"samples": 0}
    targets = [float(record["target"]) for record in records]
    weights = [float(record["sample_weight"]) for record in records]
    return {
        "samples": len(records),
        "plot_groups": len({record["plot_group_id"] for record in records}),
        "target_minimum": min(targets),
        "target_maximum": max(targets),
        "target_mean": sum(targets) / len(targets),
        "sample_weight_minimum": min(weights),
        "sample_weight_maximum": max(weights),
        "sample_weight_mean": sum(weights) / len(weights),
        "target_bins": dict(Counter(_target_bin(value) for value in targets)),
        "supervision_tiers": dict(Counter(record["supervision_tier"] for record in records)),
        "cohorts": dict(Counter(record["cohort_id"] for record in records)),
    }


def run(
    root: Path,
    inventory_path: Path,
    output_dir: Path,
    settings: BuildSettings | None = None,
) -> dict[str, Any]:
    settings = settings or BuildSettings()
    settings.validate()
    root, inventory_path, output_dir = Path(root), Path(inventory_path), Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Inventory CSV not found: {inventory_path}")
    inventory = load_inventory(inventory_path)
    scored, issues, sources = join_scores(root, inventory, settings)
    if not scored:
        raise ValueError("No score rows could be joined to the inventory")
    holdout_groups = assign_splits(scored, settings)

    split_rows = {
        "pretrain": [record for record in scored if record["split"] == "weak_pretrain"],
        "finetune": [record for record in scored if record["split"] == "gold_train"],
        "validation": [record for record in scored if record["split"] == "validation"],
        "test": [record for record in scored if record["split"] == "test"],
    }
    if any(not rows for rows in split_rows.values()):
        empty = [name for name, rows in split_rows.items() if not rows]
        raise ValueError(f"Generated empty supervised manifest(s): {', '.join(empty)}")
    adaptation, adaptation_exclusions = adaptation_rows(inventory, scored, holdout_groups, settings)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "scored_manifest.csv", scored, MANIFEST_COLUMNS)
    for name, rows in split_rows.items():
        _write_csv(output_dir / f"{name}.csv", rows, MANIFEST_COLUMNS)
    adaptation_columns = (
        tuple(adaptation[0])
        if adaptation
        else (
            "image_id",
            "absolute_path",
            "relative_path",
        )
    )
    _write_csv(output_dir / "adaptation.csv", adaptation, adaptation_columns)
    issue_columns = ("score_file", "row_number", "filename", "issue")
    _write_csv(output_dir / "score_join_issues.csv", issues, issue_columns)
    source_columns = sorted({key for row in sources for key in row})
    _write_csv(output_dir / "score_source_summary.csv", sources, source_columns)

    summary = {
        "dataset_root": str(root.resolve()),
        "inventory": str(inventory_path.resolve()),
        "settings": {
            "validation_fraction": settings.validation_fraction,
            "test_fraction": settings.test_fraction,
            "seed": settings.seed,
            "gold_difference_threshold": settings.gold_difference_threshold,
            "supervision_weights": {
                "gold": settings.gold_weight,
                "dual_weak": settings.dual_weak_weight,
                "single_weak": settings.single_weak_weight,
            },
            "adaptation_excluded_cohorts": list(settings.adaptation_excluded_cohorts),
        },
        "canonical_inventory_images": sum(row["duplicate_role"] != "copy" for row in inventory),
        "joined_scored_images": len(scored),
        "gold_images": sum(bool(record["is_gold_standard"]) for record in scored),
        "weak_images": sum(not bool(record["is_gold_standard"]) for record in scored),
        "score_join_issues": len(issues),
        "splits": {name: _distribution(rows) for name, rows in split_rows.items()},
        "adaptation": {
            "images": len(adaptation),
            "cohorts": dict(Counter(row["cohort_id"] for row in adaptation)),
            "exclusions": dict(adaptation_exclusions),
        },
        "score_sources": sources,
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gold-difference-threshold", type=float, default=5.0)
    parser.add_argument("--gold-weight", type=float, default=1.0)
    parser.add_argument("--dual-weak-weight", type=float, default=0.6)
    parser.add_argument("--single-weak-weight", type=float, default=0.4)
    parser.add_argument(
        "--adaptation-exclude-cohort",
        action="append",
        default=None,
        help="Repeatable. Defaults to excluding both Gross-Gerau insect timepoints.",
    )
    arguments = parser.parse_args(argv)
    settings = BuildSettings(
        validation_fraction=arguments.validation_fraction,
        test_fraction=arguments.test_fraction,
        seed=arguments.seed,
        gold_difference_threshold=arguments.gold_difference_threshold,
        gold_weight=arguments.gold_weight,
        dual_weak_weight=arguments.dual_weak_weight,
        single_weak_weight=arguments.single_weak_weight,
        adaptation_excluded_cohorts=tuple(
            arguments.adaptation_exclude_cohort or DEFAULT_ADAPTATION_EXCLUDED_COHORTS
        ),
    )
    summary = run(arguments.root, arguments.inventory, arguments.output_dir, settings)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
