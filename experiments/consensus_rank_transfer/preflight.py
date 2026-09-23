"""Fast training-machine inventory before frozen embedding extraction."""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from pathlib import Path

from .config import load_config


def run(config_path: str) -> dict:
    config = load_config(config_path)
    if not Path(config.base_config).is_file():
        return {"base_config_exists": False, "base_checkpoint_exists":
                Path(config.base_checkpoint).is_file(), "ready_for_prepare": False}
    with Path(config.base_config).open("rb") as handle:
        base = tomllib.load(handle)
    manifests = Path(base["data"]["manifest_dir"])
    report = {"base_config_exists": Path(config.base_config).is_file(),
              "base_checkpoint_exists": Path(config.base_checkpoint).is_file(),
              "manifests": {}, "feature_record_count": 0,
              "run_dir": config.run_dir}
    for name in ("pretrain", "finetune", "validation", "test"):
        path = manifests / base["data"][f"{name}_manifest"]
        if not path.is_file():
            report["manifests"][name] = {"exists": False}
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        report["manifests"][name] = {
            "exists": True, "images": len(rows),
            "missing_original_images": sum(not Path(row["absolute_path"]).is_file() for row in rows),
            "plot_groups": len({row["plot_group_id"] for row in rows}),
        }
    # This is an inventory only; prepare validates each record's cache identity.
    cache = Path(base["features"]["cache_dir"])
    report["feature_record_count"] = sum(1 for _ in cache.glob("*.npz")) if cache.is_dir() else 0
    report["ready_for_prepare"] = (report["base_config_exists"] and
                                   report["base_checkpoint_exists"] and
                                   all(item.get("exists") and item["missing_original_images"] == 0
                                       for item in report["manifests"].values()))
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/consensus_rank_transfer/config.toml")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
