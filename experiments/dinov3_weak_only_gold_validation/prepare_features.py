"""Build missing three-view caches for the new leakage-safe weak/gold split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.dinov3_hierarchical_three_view_mil.prepare_features import run as prepare_base

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke-test only; training requires all records")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    required = [Path(config.manifest_dir) / name for name in ("weak_train.csv", "gold_validation.csv")]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Build the weak/gold manifests before preparing features")
    report = prepare_base(
        config.routed_base,
        splits=["finetune", "validation"],
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
