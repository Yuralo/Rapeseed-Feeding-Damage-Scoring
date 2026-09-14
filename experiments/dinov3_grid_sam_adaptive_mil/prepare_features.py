"""Create resumable frozen DINO features for SAM-guided plant-centred crops."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from experiments.dinov3_grid_lora_patch_attention_sam_fusion.preprocessing import (
    load_or_create_grid_crop,
)
from experiments.dinov3_grid_lora_patch_attention_sam_fusion.segmentation import load_cached_mask
from experiments.dinov3_grid_tiled_mil.data import image_path, load_scores
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from rapeseed_damage.artifacts import append_jsonl, write_json
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import load_config
from .features import (
    FrozenDinoExtractor,
    cache_identity,
    extract_adaptive_features,
    extract_features_from_layout,
    feature_cache_path,
    load_cached_layout,
    load_feature_record,
    save_feature_record,
)


def run(config, *, overwrite: bool = False, limit: int | None = None) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config, device)
    table = load_scores(config)
    if limit is not None:
        table = table.iloc[:limit]
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_log = run_dir / config.output.feature_failure_log
    failure_log.unlink(missing_ok=True)
    extractor = FrozenDinoExtractor(config, device)
    created = skipped = failed = 0
    layout_sources = {"sam_mask": 0, "cached_layout": 0}
    instance_counts, coverages = [], []
    total = len(table)
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        filename = str(row[config.data.filename_column])
        source = image_path(config, filename)
        identity = cache_identity(config, filename, source)
        destination = feature_cache_path(config, filename, source)
        image = mask = None
        try:
            if destination.is_file() and not (overwrite or config.features.overwrite):
                record = load_feature_record(destination, expected_identity=identity)
                skipped += 1
            else:
                image, processed, _ = load_or_create_grid_crop(
                    source,
                    config.data.grid_cache_dir,
                    size=config.data.grid_crop_size,
                    inner_margin_fraction=config.data.grid_inner_margin_fraction,
                )
                try:
                    mask, mask_path, _ = load_cached_mask(source, config)
                    features, layout = extract_adaptive_features(extractor, image, mask, config)
                    boxes = layout.boxes
                    foreground_pixels = layout.foreground_pixels
                    mask_coverage = layout.mask_coverage
                    components_before_merge = layout.component_count_before_merge
                    layout_sources["sam_mask"] += 1
                except FileNotFoundError as mask_error:
                    try:
                        old_layout, old_layout_path = load_cached_layout(
                            config, filename, source
                        )
                    except FileNotFoundError as layout_error:
                        raise FileNotFoundError(
                            f"Neither a SAM mask nor a reusable plant layout exists for "
                            f"{filename}. SAM error: {mask_error} Layout error: {layout_error}"
                        ) from layout_error
                    features = extract_features_from_layout(
                        extractor, image, old_layout, config
                    )
                    boxes = old_layout["boxes"]
                    foreground_pixels = old_layout["foreground_pixels"]
                    mask_coverage = old_layout["mask_coverage"]
                    components_before_merge = old_layout["components_before_merge"]
                    mask_path = Path(old_layout["mask_path"])
                    layout_sources["cached_layout"] += 1
                    print(
                        f"[{position:04d}/{total:04d}] reusing plant layout "
                        f"{old_layout_path.name}",
                        flush=True,
                    )
                save_feature_record(
                    destination,
                    features=features,
                    boxes=boxes,
                    foreground_pixels=foreground_pixels,
                    mask_coverage=mask_coverage,
                    components_before_merge=components_before_merge,
                    processed_image_path=str(processed),
                    mask_path=str(mask_path),
                    identity=identity,
                )
                record = load_feature_record(destination, expected_identity=identity)
                created += 1
            instance_counts.append(len(record["features"]))
            coverages.append(record["mask_coverage"])
            print(
                f"[{position:04d}/{total:04d}] {filename} | instances={len(record['features'])} "
                f"coverage={record['mask_coverage']:.3f} | created={created} "
                f"skipped={skipped} failed={failed}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            failed += 1
            append_jsonl(
                failure_log,
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
        finally:
            if image is not None:
                image.close()
            if mask is not None:
                mask.close()
    summary = {
        "images": total,
        "created": created,
        "skipped": skipped,
        "failed": failed,
        "instances": {
            "minimum": min(instance_counts) if instance_counts else None,
            "mean": sum(instance_counts) / len(instance_counts) if instance_counts else None,
            "maximum": max(instance_counts) if instance_counts else None,
        },
        "mask_coverage": {
            "minimum": min(coverages) if coverages else None,
            "mean": sum(coverages) / len(coverages) if coverages else None,
        },
        "layout_sources": layout_sources,
        "feature_cache_dir": str(Path(config.features.cache_dir).resolve()),
        "device": str(device),
    }
    write_json(run_dir / "adaptive_feature_cache_summary.json", summary)
    if failed:
        raise RuntimeError(
            f"Feature extraction failed for {failed} image(s); inspect {failure_log}"
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
