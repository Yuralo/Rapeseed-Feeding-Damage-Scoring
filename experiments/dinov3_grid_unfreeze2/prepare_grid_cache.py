"""Precompute and validate every grid crop before GPU training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from rapeseed_damage.artifacts import write_json

from .config import Config, load_config
from .data import image_path, load_scores
from .preprocessing import load_or_create_grid_crop, log_grid_failure


def run(config: Config, *, overwrite: bool = False) -> dict:
    table = load_scores(config)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_log = run_dir / config.output.grid_failure_log
    created, reused, failures = 0, 0, []
    started = perf_counter()
    total = len(table)
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        filename = str(row[config.data.filename_column])
        source = image_path(config, filename)
        try:
            _, destination, was_created = load_or_create_grid_crop(
                source,
                config.data.grid_cache_dir,
                size=config.data.grid_crop_size,
                overwrite=overwrite,
            )
            created += int(was_created)
            reused += int(not was_created)
        except Exception as error:
            destination = None
            failures.append(filename)
            log_grid_failure(
                failure_log,
                error=error,
                image_path=source,
                filename=filename,
                dataset_index=position - 1,
            )
        if position % 25 == 0 or position == total:
            print(
                f"Grid cache {position}/{total} | created {created} | "
                f"reused {reused} | failed {len(failures)}",
                flush=True,
            )
    report = {
        "images": total,
        "created": created,
        "reused": reused,
        "failures": len(failures),
        "failed_filenames": failures,
        "cache_dir": str(Path(config.data.grid_cache_dir).resolve()),
        "failure_log": str(failure_log.resolve()),
        "seconds": perf_counter() - started,
    }
    write_json(run_dir / "grid_cache_summary.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing cached crops after a preprocessing-code change.",
    )
    arguments = parser.parse_args(argv)
    report = run(load_config(arguments.config), overwrite=arguments.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
