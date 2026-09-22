"""Freeze paired-rater splits and the exact earlier OOD probe images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

from rapeseed_damage.artifacts import write_json

OOD_COHORTS = ("dsv_asendorf_t1_bbch11", "wg_insects_t1_bbch10")


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(row: dict[str, str]) -> float:
    scores = [float(row[key]) for key in ("score_jlu", "score_gau")]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError(f"Non-finite paired scores: {row['relative_path']}")
    result = sum(scores) / 2
    if not math.isclose(result, float(row["target"]), abs_tol=1e-6):
        raise ValueError(f"Existing target differs from two-scorer mean: {row['relative_path']}")
    return result


def _write_table(path: Path, rows: list[dict[str, str]], fields: list[str]) -> str:
    if not rows:
        raise ValueError(f"Cannot create empty manifest: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue()
    if path.is_file() and path.read_text() != content:
        raise ValueError(f"Manifest would change after creation: {path}. Use a new run directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def make_manifests(config) -> dict:
    source = Path(config.source_manifest)
    rows = _read(source)
    if not rows:
        raise ValueError("Empty scored manifest")
    fields = list(rows[0])
    required = {
        "relative_path",
        "absolute_path",
        "sha256",
        "plot_group_id",
        "split",
        "supervision_tier",
        "score_jlu",
        "score_gau",
        "target",
        "cohort_id",
    }
    if required - set(fields):
        raise ValueError(f"Scored manifest lacks: {sorted(required - set(fields))}")
    selected: dict[str, list[dict[str, str]]] = {
        "pretrain": [],
        "finetune": [],
        "validation": [],
        "test": [],
    }
    excluded = []
    by_name = {}
    for row in rows:
        name = row["relative_path"]
        if name in by_name:
            raise ValueError(f"Duplicate relative_path: {name}")
        by_name[name] = row
        if not (row["score_jlu"] and row["score_gau"]):
            continue
        row = dict(row)
        row["target"] = str(_mean(row))
        row["sample_weight"] = "1.0"
        split, tier = row["split"], row["supervision_tier"]
        if split == "gold_train" and tier == "gold":
            selected["finetune"].append(row)
        elif split == "weak_pretrain" and tier == "dual_weak":
            selected["pretrain"].append(row)
        elif split in ("validation", "test") and tier == "gold":
            selected[split].append(row)
        elif split == "excluded_holdout_related" and tier == "dual_weak":
            excluded.append(row)
        else:
            raise ValueError(f"Unexpected paired-score assignment: {name}: {split}/{tier}")
    if sum(map(len, selected.values())) + len(excluded) != sum(
        bool(row["score_jlu"] and row["score_gau"]) for row in rows
    ):
        raise ValueError("Some paired-score images were not assigned")
    # The train pool combines pretrain and finetune. Neither may share a plot,
    # exact image, or hash with validation/test, even across different filenames.
    train = selected["pretrain"] + selected["finetune"]
    holdout = selected["validation"] + selected["test"]
    for field in ("relative_path", "absolute_path", "sha256", "plot_group_id"):
        a = {row[field] for row in train if row[field]}
        b = {row[field] for row in holdout if row[field]}
        if a & b:
            raise ValueError(f"Train/holdout leakage by {field}: {sorted(a & b)[:3]}")
    if {row["plot_group_id"] for row in selected["validation"]} & {
        row["plot_group_id"] for row in selected["test"]
    }:
        raise ValueError("Gold validation and test share plot groups")
    directory = Path(config.run_dir) / "manifests"
    hashes = {
        name: _write_table(directory / f"{name}.csv", table, fields)
        for name, table in selected.items()
    }
    reference = Path(config.reference_ood_dir)
    ood = {}
    used = {row["relative_path"] for table in selected.values() for row in table}
    for cohort in OOD_COHORTS:
        old = _read(reference / cohort / "predictions.csv")
        names = [row["filename"] for row in old]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Invalid previous OOD sample list: {cohort}")
        samples = []
        for old_row in old:
            name = old_row["filename"]
            row = by_name.get(name)
            if row is None or row["cohort_id"] != cohort or name in used:
                raise ValueError(
                    f"OOD sample is missing, wrong-cohort or in gold experiment: {name}"
                )
            if not math.isclose(float(row["target"]), float(old_row["target"]), abs_tol=1e-5):
                raise ValueError(f"OOD target changed since prior probe: {name}")
            sample = dict(row)
            sample["split"] = "weak_train"  # feature preparation permits limited weak failures
            samples.append(sample)
        ood[cohort] = {
            "samples": len(samples),
            "sha256": _write_table(directory / f"ood_{cohort}.csv", samples, fields),
        }
    report = {
        "paired_total": len(train) + len(holdout) + len(excluded),
        "paired_training": len(train),
        "gold_training": len(selected["finetune"]),
        "discordant_training": len(selected["pretrain"]),
        "gold_validation": len(selected["validation"]),
        "gold_test": len(selected["test"]),
        "excluded_holdout_related": len(excluded),
        "manifest_sha256": hashes,
        "ood_reference": str(reference),
        "ood": ood,
        "training_cohorts": dict(Counter(row["cohort_id"] for row in train)),
        "target": (
            "(score_jlu + score_gau) / 2; the frozen base model's scaler was fitted "
            "only on the 325 gold-train rows"
        ),
    }
    write_json(Path(config.run_dir) / "manifest_summary.json", report)
    return report


def main(argv=None):
    from .config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(make_manifests(load_config(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
