"""Compute only the missing 3x3 context tiles; never rerun SAM or grid routing."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from experiments.dinov3_grid_tiled_mil.features import FrozenDinoExtractor
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_weak_only_gold_validation.data import prepare_data as prepare_weak_data
from rapeseed_damage.artifacts import append_jsonl, write_json
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import _paths
from .features import extract, identity, load, save


def run(config: Config, *, limit: int | None = None, overwrite: bool = False):
    seed_everything(config.seed, config.adaptive.runtime.deterministic)
    device = resolve_device(config.adaptive.runtime.device)
    configure_acceleration(config.adaptive.context_config(), device)
    weak, gold, _, _, omitted_base = prepare_weak_data(config.weak)
    table = pd.concat([weak.assign(_split="weak"), gold.assign(_split="gold")], ignore_index=True)
    if limit is not None:
        table = table.iloc[:limit]
    output = Path(config.run_dir) / "feature_preparation"
    output.mkdir(parents=True, exist_ok=True)
    log = output / "context_feature_failures.jsonl"
    if overwrite:
        log.unlink(missing_ok=True)
    extractor = None
    created = skipped = failed_weak = failed_gold = 0
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        relative, source, base_path, path = _paths(config, row)
        key = identity(config, relative, source)
        try:
            if path.is_file() and not overwrite:
                load(path, key)
                skipped += 1
            else:
                from experiments.dinov3_hierarchical_three_view_mil.features import (
                    cache_identity as base_identity,
                )
                from experiments.dinov3_hierarchical_three_view_mil.features import (
                    load_record as load_base,
                )

                base_record = load_base(
                    base_path,
                    expected_identity=base_identity(config.weak.routed_base, relative, source),
                )
                if extractor is None:
                    extractor = FrozenDinoExtractor(config.adaptive.context_config(), device)
                save(path, extract(extractor, config, base_record), key)
                created += 1
            print(
                f"[context {position:04d}/{len(table):04d}] {relative} | "
                f"created={created} skipped={skipped} weak_failed={failed_weak} "
                f"gold_failed={failed_gold}", flush=True,
            )
        except Exception as error:  # noqa: BLE001 - audit all image failures
            if row["_split"] == "gold":
                failed_gold += 1
            else:
                failed_weak += 1
            append_jsonl(log, {
                "split": row["_split"], "filename": relative,
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(),
            })
            print(f"[context {position:04d}/{len(table):04d}] FAILED {relative}: {error}", flush=True)
    eligible = len(weak) + len(omitted_base)
    total_weak_missing = len(omitted_base) + failed_weak
    report = {
        "images": len(table), "created": created, "skipped": skipped,
        "failed_weak_context": failed_weak, "failed_gold_context": failed_gold,
        "omitted_weak_base_features": len(omitted_base),
        "weak_missing_total": total_weak_missing, "weak_eligible_total": eligible,
        "failure_log": str(log),
    }
    write_json(output / "summary.json", report)
    if failed_gold:
        raise RuntimeError(f"{failed_gold} gold context feature(s) failed; inspect {log}")
    if limit is None and total_weak_missing / eligible > config.weak.routed_base.data.maximum_weak_failure_fraction:
        raise RuntimeError(
            f"{total_weak_missing}/{eligible} weak features missing; inspect {log}"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="Smoke test only; training needs the full cache")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), limit=args.limit,
                         overwrite=args.overwrite), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
