"""Make weak-train / all-gold-validation manifests with plot-level isolation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from rapeseed_damage.artifacts import write_json

from .config import Config, load_config

REQUIRED = {
    "image_id", "absolute_path", "relative_path", "sha256", "cohort_id",
    "plot_group_id", "is_gold_standard", "supervision_tier", "target",
    "score_single", "score_jlu", "score_gau", "sample_weight",
}


def _number(value: str) -> float | None:
    if not str(value).strip():
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite score: {value}")
    return number


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = REQUIRED - set(columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Scored manifest is empty: {path}")
    ids = [row["image_id"] for row in rows]
    paths = [row["relative_path"] for row in rows]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("Scored manifest has duplicate image IDs or canonical paths")
    if any(not row["plot_group_id"] or not row["sha256"] for row in rows):
        raise ValueError("Every scored image needs a plot group and content hash")
    return columns, rows


def _gold(row: dict[str, str]) -> bool:
    return str(row["is_gold_standard"]).strip().casefold() == "true"


def _target(row: dict[str, str]) -> tuple[float, str]:
    jlu, gau = _number(row["score_jlu"]), _number(row["score_gau"])
    single = _number(row["score_single"])
    if jlu is not None and gau is not None and single is None:
        value, source = (jlu + gau) / 2, "mean_of_two_scorers"
    else:
        available = [score for score in (single, jlu, gau) if score is not None]
        if len(available) != 1:
            raise ValueError(f"Expected exactly one or two usable scores for {row['image_id']}")
        value, source = available[0], "single_scorer"
    prior = _number(row["target"])
    if prior is None or not math.isclose(value, prior, abs_tol=1e-5):
        raise ValueError(f"Scorer-derived target disagrees with manifest for {row['image_id']}")
    return value, source


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(config: Config) -> dict:
    source = Path(config.scored_manifest)
    columns, scored = _read(source)
    gold = [row for row in scored if _gold(row)]
    weak = [row for row in scored if not _gold(row)]
    if len(gold) != config.expected_gold_images:
        raise ValueError(f"Expected {config.expected_gold_images} gold images; found {len(gold)}")
    if any(row["supervision_tier"] != "gold" for row in gold):
        raise ValueError("Gold flag and supervision tier disagree")
    gold_groups = {row["plot_group_id"] for row in gold}
    gold_hashes = {row["sha256"] for row in gold}
    gold_paths = {row["relative_path"] for row in gold}
    selected, rejected = [], []
    for row in weak:
        reason = None
        if row["plot_group_id"] in gold_groups:
            reason = "same_plot_as_gold"
        elif row["sha256"] in gold_hashes or row["relative_path"] in gold_paths:
            reason = "duplicate_of_gold"
        elif row["supervision_tier"] not in {"dual_weak", "single_weak"}:
            reason = "unknown_supervision_tier"
        elif config.mode == "dual_only" and row["supervision_tier"] != "dual_weak":
            reason = "not_dual_scored"
        if reason:
            rejected.append({**row, "exclusion_reason": reason})
            continue
        value, source_type = _target(row)
        if row["supervision_tier"] == "dual_weak" and source_type != "mean_of_two_scorers":
            raise ValueError(f"Dual-weak row lacks two scores: {row['image_id']}")
        if row["supervision_tier"] == "single_weak" and source_type != "single_scorer":
            raise ValueError(f"Single-weak row unexpectedly has two scores: {row['image_id']}")
        selected.append({**row, "target": str(value), "target_source": source_type,
                         "split": "weak_train"})
    for row in gold:
        value, source_type = _target(row)
        if source_type != "mean_of_two_scorers":
            raise ValueError(f"Gold row lacks two scores: {row['image_id']}")
        row["target"] = str(value)
        row["target_source"] = source_type
        row["split"] = "gold_validation"
    if not selected:
        raise ValueError("No weak images remain after leakage filtering")
    if ({row["plot_group_id"] for row in selected} & gold_groups or
            {row["sha256"] for row in selected} & gold_hashes):
        raise RuntimeError("Gold/weak leakage survived filtering")
    destination = Path(config.manifest_dir)
    output_columns = list(dict.fromkeys([*columns, "target_source"]))
    _write_csv(destination / "weak_train.csv", output_columns, selected)
    _write_csv(destination / "gold_validation.csv", output_columns, gold)
    _write_csv(destination / "excluded_weak.csv", [*output_columns, "exclusion_reason"], rejected)
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "mode": config.mode,
        "scored_manifest": str(source),
        "scored_manifest_sha256": digest,
        "gold_validation_images": len(gold),
        "gold_training_images": 0,
        "gold_validation_plot_groups": len(gold_groups),
        "weak_scored_images": len(weak),
        "weak_train_images": len(selected),
        "weak_train_plot_groups": len({row["plot_group_id"] for row in selected}),
        "weak_train_cohorts": dict(Counter(row["cohort_id"] for row in selected)),
        "weak_train_target_sources": dict(Counter(row["target_source"] for row in selected)),
        "excluded_weak_images": len(rejected),
        "exclusion_reasons": dict(Counter(row["exclusion_reason"] for row in rejected)),
        "plot_group_overlap": 0,
        "content_hash_overlap": 0,
        "manifests": {
            "weak_train": str(destination / "weak_train.csv"),
            "gold_validation": str(destination / "gold_validation.csv"),
            "excluded_weak": str(destination / "excluded_weak.csv"),
        },
    }
    write_json(destination / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
