"""Create resumable per-image global and tiled frozen DINOv3 features."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from experiments.dinov3_grid_lora_patch_attention.preprocessing import (
    load_or_create_grid_crop,
    log_grid_failure,
)
from rapeseed_damage.artifacts import append_jsonl, write_json
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import load_config
from .data import image_path, load_scores
from .features import (
    FrozenDinoExtractor,
    cache_identity,
    feature_cache_path,
    load_feature_record,
    save_feature_record,
)
from .runtime import configure_acceleration


def run(config, *, overwrite: bool = False, limit: int | None = None) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config, device)
    table = load_scores(config)
    if limit is not None:
        table = table.iloc[:limit]
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_log = run_dir / config.output.feature_failure_log
    grid_log = run_dir / config.output.grid_failure_log
    extractor = FrozenDinoExtractor(config, device)
    created = skipped = failed = 0
    feature_dim = None
    total = len(table)
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        filename = str(row[config.data.filename_column])
        source = image_path(config, filename)
        identity = cache_identity(config, filename, source)
        destination = feature_cache_path(config, filename, source)
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
                        filename=filename,
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
                f"[{position:04d}/{total:04d}] {filename} | "
                f"created={created} skipped={skipped} failed={failed}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - log one bad image and finish the cache audit
            failed += 1
            append_jsonl(
                feature_log,
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "filename": filename,
                    "source_image_path": str(source),
                    "feature_cache_path": str(destination),
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[{position:04d}/{total:04d}] FAILED {filename}: {error}", flush=True)
    summary = {
        "images": total,
        "created": created,
        "skipped": skipped,
        "failed": failed,
        "feature_dim": feature_dim,
        "views_per_image": 1 + config.tiles.rows * config.tiles.columns,
        "tiles_per_image": config.tiles.rows * config.tiles.columns,
        "feature_cache_dir": str(Path(config.features.cache_dir).resolve()),
        "device": str(device),
    }
    write_json(run_dir / "feature_cache_summary.json", summary)
    if failed:
        raise RuntimeError(
            f"Feature extraction failed for {failed} image(s); inspect {feature_log}"
        )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args(argv)
    report = run(
        load_config(arguments.config), overwrite=arguments.overwrite, limit=arguments.limit
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
