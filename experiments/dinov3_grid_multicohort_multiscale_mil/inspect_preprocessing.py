"""Audit grid cropping on a deterministic sample from every supervised cohort."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from experiments.dinov3_grid_lora_patch_attention.preprocessing import (
    load_or_create_grid_crop,
    log_grid_failure,
)
from rapeseed_damage.artifacts import write_json

from .config import load_config
from .data import feature_name, load_manifest, source_path


def _sample(config, count: int) -> pd.DataFrame:
    tables = [
        load_manifest(config, split).assign(manifest_split=split)
        for split in ("pretrain", "finetune", "validation", "test")
    ]
    table = pd.concat(tables, ignore_index=True).drop_duplicates(
        subset=[config.data.absolute_path_column]
    )
    pieces = []
    for _, cohort in table.groupby(config.data.cohort_column, sort=True):
        pieces.append(
            cohort.sample(
                n=min(count, len(cohort)),
                random_state=config.training.seed,
            )
        )
    return pd.concat(pieces, ignore_index=True)


def run(config, *, samples_per_cohort: int = 12, overwrite: bool = False) -> dict:
    if samples_per_cohort < 1:
        raise ValueError("samples_per_cohort must be positive")
    table = _sample(config, samples_per_cohort)
    destination = Path(config.output.run_dir) / "preprocessing_audit"
    destination.mkdir(parents=True, exist_ok=True)
    failure_log = destination / config.output.grid_failure_log
    records = []
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        name, source = feature_name(row, config), source_path(row, config)
        try:
            image, crop_path, created = load_or_create_grid_crop(
                source,
                config.data.grid_cache_dir,
                size=config.data.grid_crop_size,
                inner_margin_fraction=config.data.grid_inner_margin_fraction,
                overwrite=overwrite,
            )
            records.append(
                {
                    "filename": name,
                    "cohort_id": row[config.data.cohort_column],
                    "manifest_split": row["manifest_split"],
                    "source_image_path": str(source),
                    "processed_image_path": str(crop_path),
                    "processed_width": image.width,
                    "processed_height": image.height,
                    "created": created,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as error:  # noqa: BLE001 - record every cohort failure
            log_grid_failure(
                failure_log,
                error=error,
                image_path=source,
                filename=name,
                dataset_index=position - 1,
            )
            records.append(
                {
                    "filename": name,
                    "cohort_id": row[config.data.cohort_column],
                    "manifest_split": row["manifest_split"],
                    "source_image_path": str(source),
                    "processed_image_path": "",
                    "processed_width": "",
                    "processed_height": "",
                    "created": False,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    report = pd.DataFrame(records)
    report.to_csv(destination / "preprocessing_audit.csv", index=False)
    status = Counter(report["status"])
    cohort_status = {
        cohort: dict(Counter(group["status"]))
        for cohort, group in report.groupby("cohort_id", sort=True)
    }
    summary = {
        "sampled_images": len(report),
        "samples_per_cohort_requested": samples_per_cohort,
        "status": dict(status),
        "cohorts": cohort_status,
        "audit_csv": str((destination / "preprocessing_audit.csv").resolve()),
        "failure_log": str(failure_log.resolve()),
        "note": "Each processed_image_path is a separate crop; no mega-contact-sheet is created.",
    }
    write_json(destination / "summary.json", summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-cohort", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    report = run(
        load_config(arguments.config),
        samples_per_cohort=arguments.samples_per_cohort,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
