"""Check grid detection across the dataset before starting model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from rapeseed_damage.artifacts import write_json

from .config import load_config
from .data import image_path, load_scores
from .preprocessing import load_grid_crop, log_grid_failure


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="Only validate the first N rows; omit to scan the entire dataset",
    )
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    table = load_scores(config)
    if arguments.limit is not None:
        table = table.head(arguments.limit)

    run_dir = Path(config.output.run_dir)
    failure_log = run_dir / config.output.grid_failure_log
    successes = 0
    failures = 0
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        filename = str(row[config.data.filename_column])
        path = image_path(config, filename)
        try:
            load_grid_crop(path)
            successes += 1
        except Exception as error:
            failures += 1
            log_grid_failure(
                failure_log,
                error=error,
                image_path=path,
                filename=filename,
                dataset_index=position - 1,
            )
            print(f"FAILED {filename}: {type(error).__name__}: {error}")
        if position % 25 == 0 or position == len(table):
            print(
                f"Checked {position}/{len(table)} | "
                f"successes={successes} failures={failures}",
                flush=True,
            )

    summary = {
        "checked": len(table),
        "successes": successes,
        "failures": failures,
        "failure_log": str(failure_log),
    }
    write_json(run_dir / "grid_validation_summary.json", summary)
    if failures:
        raise SystemExit(
            f"Grid validation found {failures} failure(s). See {failure_log}."
        )


if __name__ == "__main__":
    main()
