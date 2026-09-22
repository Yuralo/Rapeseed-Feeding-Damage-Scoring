"""Prepare routed three-view and high-resolution plant-patch features."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path

import pandas as pd

from experiments.dinov3_grid_tiled_mil.features import FrozenDinoExtractor
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.data import load_manifest
from experiments.dinov3_hierarchical_three_view_mil.prepare_features import run as prepare_base
from experiments.dinov3_plant_damage_mil.features import (
    cache_path,
    extract,
    identity,
    load,
    load_base,
    save,
)
from rapeseed_damage.artifacts import append_jsonl, write_json
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .manifests import OOD_COHORTS, make_manifests


def _patches(config, table, *, label, limit=None, overwrite=False):
    if limit is not None:
        table = table.iloc[:limit]
    device = resolve_device(config.base.runtime.device)
    configure_acceleration(config.base, device)
    seed_everything(config.seed, config.base.runtime.deterministic)
    extractor = None
    created = skipped = 0
    failures = []
    log = Path(config.run_dir) / "feature_preparation" / f"{label}_patch_failures.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        log.unlink(missing_ok=True)
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        relative = str(row[config.base.data.filename_column])
        source = Path(str(row[config.base.data.absolute_path_column]))
        path = cache_path(config, relative, source)
        key = identity(config, relative, source)
        try:
            if path.is_file() and not overwrite:
                load(path, key)
                skipped += 1
            else:
                base_record = load_base(config, relative, source)
                if extractor is None:
                    extractor = FrozenDinoExtractor(config.base, device)
                save(path, extract(extractor, config, base_record), key)
                created += 1
            print(
                f"[patch {label} {position}/{len(table)}] {relative} | "
                f"created={created} skipped={skipped} failed={len(failures)}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - audit every image failure
            failures.append((str(row.get("split", "")), relative))
            append_jsonl(
                log,
                {
                    "split": row.get("split"),
                    "filename": relative,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[patch {label} {position}/{len(table)}] FAILED {relative}: {error}", flush=True)
    weak_failures = sum(split in {"weak_pretrain", "weak_train"} for split, _ in failures)
    strong_failures = len(failures) - weak_failures
    report = {
        "samples": len(table),
        "created": created,
        "skipped": skipped,
        "weak_failures": weak_failures,
        "strong_failures": strong_failures,
        "failure_log": str(log),
    }
    write_json(log.parent / f"{label}_patch_summary.json", report)
    if strong_failures:
        raise RuntimeError(f"{strong_failures} required gold patch records failed; inspect {log}")
    if table.empty or weak_failures / len(table) > config.base.data.maximum_weak_failure_fraction:
        raise RuntimeError(f"Weak patch failure fraction is too high; inspect {log}")
    return report


def _ood_base(config, cohort):
    directory = Path(config.run_dir) / "manifests"
    return replace(
        config.base,
        data=replace(
            config.base.data,
            manifest_dir=str(directory),
            pretrain_manifest=f"ood_{cohort}.csv",
        ),
        output=replace(
            config.base.output,
            run_dir=str(Path(config.run_dir) / "ood_preparation" / cohort),
        ),
    )


def run(config: Config, *, include_ood=False, limit=None, overwrite=False):
    make_manifests(config)
    if include_ood:
        reports = {}
        for cohort in OOD_COHORTS:
            base = _ood_base(config, cohort)
            base_report = prepare_base(base, splits=["pretrain"], limit=limit, overwrite=overwrite)
            table = load_manifest(base, "pretrain")
            reports[cohort] = {
                "base": base_report,
                "patches": _patches(
                    config, table, label=f"ood_{cohort}", limit=limit, overwrite=overwrite
                ),
            }
        return reports
    base_report = prepare_base(
        config.base,
        splits=["pretrain", "finetune", "validation", "test"],
        limit=limit,
        overwrite=overwrite,
    )
    table = pd.concat(
        [
            load_manifest(config.base, split)
            for split in ("pretrain", "finetune", "validation", "test")
        ],
        ignore_index=True,
    )
    return {
        "base": base_report,
        "patches": _patches(config, table, label="paired", limit=limit, overwrite=overwrite),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--include-ood", action="store_true", help="Prepare the exact earlier OOD samples"
    )
    parser.add_argument("--limit", type=int, help="Smoke test only; full cache needed for training")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                load_config(args.config),
                include_ood=args.include_ood,
                limit=args.limit,
                overwrite=args.overwrite,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
