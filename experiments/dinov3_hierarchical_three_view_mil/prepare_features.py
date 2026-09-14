"""Build resumable routed global/cell/SAM-plant features for all supervised splits."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from experiments.dinov3_grid_lora_patch_attention_sam_fusion.segmentation import (
    create_segmenter,
    generate_mask,
    load_cached_mask,
    validate_mask,
)
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from rapeseed_damage.artifacts import append_jsonl, write_json
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import load_manifest
from .features import (
    GRID_MODE,
    FrozenDinoExtractor,
    cache_identity,
    extract_record,
    feature_cache_path,
    input_mode,
    load_record,
    mask_cache_path,
    prepare_image,
    save_mask,
    save_record,
)


def _table(config: Config, splits: list[str]) -> pd.DataFrame:
    combined = pd.concat([load_manifest(config, split) for split in splits], ignore_index=True)
    return combined.drop_duplicates(subset=[config.data.absolute_path_column]).reset_index(drop=True)


def run(
    config: Config,
    *,
    splits: list[str],
    overwrite: bool = False,
    limit: int | None = None,
) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config, device)
    table = _table(config, splits)
    if limit is not None:
        table = table.iloc[:limit]
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_log = run_dir / config.output.feature_failure_log
    if overwrite:
        failure_log.unlink(missing_ok=True)

    # Pass one: SAM only. Keeping SAM3 and DINOv3 off the GPU at the same time
    # materially lowers preparation memory on a 24 GB 3090.
    segmenter = create_segmenter(config, device)
    mask_created = mask_skipped = mask_reused = 0
    failed_keys: set[str] = set()
    failures_by_split: dict[str, int] = {name: 0 for name in splits}
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        relative = str(row[config.data.filename_column])
        source = Path(str(row[config.data.absolute_path_column]))
        split = str(row.get("split", "unknown"))
        destination = mask_cache_path(config, relative, source).resolve()
        image = None
        try:
            if destination.is_file() and not overwrite:
                with Image.open(destination) as stored:
                    mask = np.asarray(stored.convert("L")) >= 128
                quality = validate_mask(mask, config)
                if not quality["valid"]:
                    raise ValueError(
                        "Cached SAM mask failed quality checks: "
                        + "; ".join(quality["quality_reasons"])
                    )
                mask_skipped += 1
            else:
                mask = None
                generated = False
                if input_mode(relative, config) == GRID_MODE:
                    try:
                        cached_mask, _, _ = load_cached_mask(source, config)
                        try:
                            mask = np.asarray(cached_mask, dtype=np.uint8) >= 128
                        finally:
                            cached_mask.close()
                        mask_reused += 1
                    except (FileNotFoundError, ValueError, RuntimeError, OSError):
                        mask = None
                if mask is None:
                    image, _, _ = prepare_image(source, relative, config)
                    mask = generate_mask(segmenter, image, config)
                    generated = True
                quality = validate_mask(mask, config)
                if not quality["valid"]:
                    raise ValueError(
                        "SAM mask failed quality checks: "
                        + "; ".join(quality["quality_reasons"])
                    )
                save_mask(mask, destination)
                if generated:
                    mask_created += 1
            print(
                f"[mask {position:04d}/{len(table):04d}] {relative} | "
                f"created={mask_created} reused={mask_reused} skipped={mask_skipped} "
                f"failed={len(failed_keys)}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - audit the full dataset in one run
            key = str(source.resolve())
            failed_keys.add(key)
            failures_by_split[split] = failures_by_split.get(split, 0) + 1
            append_jsonl(
                failure_log,
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "stage": "sam_mask",
                    "split": split,
                    "filename": relative,
                    "source_image_path": str(source),
                    "mask_cache_path": str(destination),
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[mask {position:04d}/{len(table):04d}] FAILED {relative}: {error}", flush=True)
        finally:
            if image is not None:
                image.close()
    del segmenter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Pass two: DINO only, using the cached masks from pass one.
    extractor = FrozenDinoExtractor(config, device)
    created = skipped = feature_failed = 0
    counts, coverages, modes = [], [], {}
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        relative = str(row[config.data.filename_column])
        source = Path(str(row[config.data.absolute_path_column]))
        split = str(row.get("split", "unknown"))
        identity = cache_identity(config, relative, source)
        destination = feature_cache_path(config, relative, source)
        image = None
        if str(source.resolve()) in failed_keys:
            continue
        try:
            if destination.is_file() and not (overwrite or config.features.overwrite):
                record = load_record(destination, expected_identity=identity)
                skipped += 1
            else:
                image, processed_path, mode = prepare_image(source, relative, config)
                mask_path = mask_cache_path(config, relative, source).resolve()
                with Image.open(mask_path) as stored:
                    mask = np.asarray(stored.convert("L")) >= 128
                extracted = extract_record(extractor, image, mask, config)
                save_record(
                    destination,
                    **extracted,
                    processed_image_path=str(processed_path),
                    mask_path=str(mask_path),
                    mode=mode,
                    identity=identity,
                )
                record = load_record(destination, expected_identity=identity)
                created += 1
            counts.append(len(record["plant_features"]))
            coverages.append(record["mask_coverage"])
            modes[record["input_mode"]] = modes.get(record["input_mode"], 0) + 1
            print(
                f"[{position:04d}/{len(table):04d}] {relative} | plants="
                f"{len(record['plant_features'])} | {record['input_mode']} | "
                f"created={created} skipped={skipped} failed={feature_failed}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - audit the full dataset in one run
            feature_failed += 1
            failed_keys.add(str(source.resolve()))
            failures_by_split[split] = failures_by_split.get(split, 0) + 1
            append_jsonl(
                failure_log,
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "stage": "dino_features",
                    "split": split,
                    "filename": relative,
                    "source_image_path": str(source),
                    "feature_cache_path": str(destination),
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[{position:04d}/{len(table):04d}] FAILED {relative}: {error}", flush=True)
        finally:
            if image is not None:
                image.close()
    failed = len(failed_keys)
    summary = {
        "splits": splits,
        "unique_images": len(table),
        "masks": {
            "created": mask_created,
            "reused_from_existing_grid_cache": mask_reused,
            "skipped": mask_skipped,
        },
        "features": {"created": created, "skipped": skipped},
        "failed": failed,
        "failures_by_split": failures_by_split,
        "input_modes": modes,
        "plant_instances": {
            "minimum": min(counts) if counts else None,
            "mean": sum(counts) / len(counts) if counts else None,
            "maximum": max(counts) if counts else None,
        },
        "mask_coverage": {
            "minimum": min(coverages) if coverages else None,
            "mean": sum(coverages) / len(coverages) if coverages else None,
        },
        "feature_cache_dir": str(Path(config.features.cache_dir).resolve()),
        "device": str(device),
    }
    write_json(run_dir / "feature_cache_summary.json", summary)
    strong_failures = sum(
        count for split, count in failures_by_split.items() if split != "weak_pretrain"
    )
    if strong_failures:
        raise RuntimeError(
            f"Feature extraction failed for {strong_failures} gold image(s); inspect {failure_log}"
        )
    weak_total = int((table.get("split", pd.Series(dtype=str)) == "weak_pretrain").sum())
    weak_failed = failures_by_split.get("weak_pretrain", 0)
    if weak_total and weak_failed / weak_total > config.data.maximum_weak_failure_fraction:
        raise RuntimeError(
            f"Weak feature failure rate {weak_failed / weak_total:.2%} exceeds configured "
            f"{config.data.maximum_weak_failure_fraction:.2%}; inspect {failure_log}"
        )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("pretrain", "finetune", "validation", "test"),
        help="Repeatable; defaults to all four splits.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args(argv)
    report = run(
        load_config(arguments.config),
        splits=arguments.split or ["pretrain", "finetune", "validation", "test"],
        overwrite=arguments.overwrite,
        limit=arguments.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
