"""Prepare resumable high-resolution plant-patch features without rerunning SAM."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from experiments.dinov3_grid_tiled_mil.features import FrozenDinoExtractor
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.data import load_manifest
from rapeseed_damage.artifacts import append_jsonl, write_json
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import select_cohorts
from .features import cache_path, extract, identity, load, load_base, save


def run(config: Config, splits: list[str], *, limit: int | None = None,
        cohorts: list[str] | None = None, limit_per_cohort: int | None = None,
        overwrite: bool = False):
    base = config.base
    seed_everything(config.seed, base.runtime.deterministic)
    device = resolve_device(base.runtime.device)
    configure_acceleration(base, device)
    tables = [load_manifest(base, split) for split in splits]
    table = pd.concat(tables, ignore_index=True).drop_duplicates(
        subset=[base.data.absolute_path_column]
    )
    if cohorts is not None:
        table = select_cohorts(table, config, cohorts, limit_per_cohort)
    elif limit_per_cohort is not None:
        raise ValueError("--limit-per-cohort requires --cohorts")
    if limit is not None:
        table = table.iloc[:limit]
    destination = Path(config.run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    failure_log = destination / "patch_feature_failures.jsonl"
    if overwrite:
        failure_log.unlink(missing_ok=True)
    extractor = None
    created = skipped = failed = 0
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        relative = str(row[base.data.filename_column])
        source = Path(str(row[base.data.absolute_path_column]))
        path = cache_path(config, relative, source)
        key = identity(config, relative, source)
        try:
            if path.is_file() and not overwrite:
                load(path, key)
                skipped += 1
            else:
                base_record = load_base(config, relative, source)
                if extractor is None:
                    extractor = FrozenDinoExtractor(base, device)
                save(path, extract(extractor, config, base_record), key)
                created += 1
            print(f"[patch {position:04d}/{len(table):04d}] {relative} | "
                  f"created={created} skipped={skipped} failed={failed}", flush=True)
        except Exception as error:  # noqa: BLE001 - record per-image failures and continue
            failed += 1
            append_jsonl(failure_log, {
                "filename": relative, "error_type": type(error).__name__,
                "error": str(error), "traceback": traceback.format_exc(),
            })
            print(f"[patch {position:04d}/{len(table):04d}] FAILED {relative}: {error}", flush=True)
    report = {"samples": len(table), "created": created, "skipped": skipped,
              "failed": failed, "splits": splits, "cohorts": cohorts,
              "limit_per_cohort": limit_per_cohort, "failure_log": str(failure_log)}
    write_json(destination / "patch_feature_summary.json", report)
    if failed:
        raise RuntimeError(f"Patch extraction failed for {failed} image(s); inspect {failure_log}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", choices=("finetune", "validation", "test", "pretrain"),
                        default=["finetune", "validation"])
    parser.add_argument("--cohorts", nargs="+")
    parser.add_argument("--limit-per-cohort", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.splits, limit=args.limit,
                         cohorts=args.cohorts, limit_per_cohort=args.limit_per_cohort,
                         overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
