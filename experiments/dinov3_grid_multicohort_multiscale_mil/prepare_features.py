"""Create resumable 3x3 and 4x4 DINO feature caches from absolute-path manifests."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from experiments.dinov3_grid_lora_patch_attention.preprocessing import (
    load_or_create_grid_crop,
    log_grid_failure,
)
from experiments.dinov3_grid_tiled_mil.features import (
    FrozenDinoExtractor,
    cache_identity,
    feature_cache_path,
    load_feature_record,
    save_feature_record,
)
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from rapeseed_damage.artifacts import append_jsonl, write_json
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import feature_name, load_manifest, source_path


def _table(config: Config, splits: list[str]) -> pd.DataFrame:
    tables = [load_manifest(config, split) for split in splits]
    combined = pd.concat(tables, ignore_index=True)
    return combined.drop_duplicates(subset=[config.data.absolute_path_column]).reset_index(
        drop=True
    )


def run(
    config: Config,
    *,
    scales: list[str],
    splits: list[str],
    overwrite: bool = False,
    limit: int | None = None,
) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config.single_scale_config(config.coarse), device)
    table = _table(config, splits)
    if limit is not None:
        table = table.iloc[:limit]
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_log = run_dir / config.output.feature_failure_log
    grid_log = run_dir / config.output.grid_failure_log
    reports = {}
    scale_settings = {"coarse": config.coarse, "fine": config.fine}
    for scale_name in scales:
        scale = scale_settings[scale_name]
        scale_config = config.single_scale_config(scale)
        extractor = FrozenDinoExtractor(scale_config, device)
        created = skipped = failed = 0
        feature_dim = None
        for position, (_, row) in enumerate(table.iterrows(), start=1):
            name, source = feature_name(row, config), source_path(row, config)
            identity = cache_identity(scale_config, name, source)
            destination = feature_cache_path(scale_config, name, source)
            replace = overwrite or config.features.overwrite
            try:
                if destination.is_file() and not replace:
                    record = load_feature_record(destination, expected_identity=identity)
                    feature_dim = int(record["features"].shape[1])
                    skipped += 1
                else:
                    try:
                        image, processed, _ = load_or_create_grid_crop(
                            source,
                            config.data.grid_cache_dir,
                            size=config.data.grid_crop_size,
                            inner_margin_fraction=config.data.grid_inner_margin_fraction,
                        )
                    except Exception as error:
                        log_grid_failure(
                            grid_log,
                            error=error,
                            image_path=source,
                            filename=name,
                            dataset_index=position - 1,
                        )
                        raise
                    features, boxes = extractor.extract_image(image)
                    feature_dim = int(features.shape[1])
                    save_feature_record(
                        destination,
                        features=features,
                        tile_boxes=boxes,
                        processed_image_path=str(processed),
                        identity=identity,
                    )
                    created += 1
                print(
                    f"[{scale_name} {position:04d}/{len(table):04d}] {name} | "
                    f"created={created} skipped={skipped} failed={failed}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - finish cache audit and preserve traceback
                failed += 1
                append_jsonl(
                    feature_log,
                    {
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "scale": scale_name,
                        "filename": name,
                        "source_image_path": str(source),
                        "feature_cache_path": str(destination),
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"[{scale_name}] FAILED {name}: {error}", flush=True)
        reports[scale_name] = {
            "images": len(table),
            "created": created,
            "skipped": skipped,
            "failed": failed,
            "feature_dim": feature_dim,
            "views_per_image": 1 + scale.rows * scale.columns,
            "cache_dir": str(Path(scale.cache_dir).resolve()),
        }
        del extractor
    summary = {
        "device": str(device),
        "splits": splits,
        "unique_images": len(table),
        "scales": reports,
    }
    write_json(run_dir / "feature_cache_summary.json", summary)
    failures = sum(report["failed"] for report in reports.values())
    if failures:
        raise RuntimeError(f"Feature extraction failed {failures} time(s); inspect {feature_log}")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale", choices=("coarse", "fine", "both"), default="both")
    parser.add_argument(
        "--split",
        action="append",
        choices=("pretrain", "finetune", "validation", "test"),
        help="Repeatable; defaults to all supervised splits.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args(argv)
    scales = ["coarse", "fine"] if arguments.scale == "both" else [arguments.scale]
    splits = arguments.split or ["pretrain", "finetune", "validation", "test"]
    report = run(
        load_config(arguments.config),
        scales=scales,
        splits=splits,
        overwrite=arguments.overwrite,
        limit=arguments.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
