"""Precompute frozen SAM masks for the clean grid crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from rapeseed_damage.artifacts import write_json
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import image_path, load_scores
from .preprocessing import load_or_create_grid_crop
from .segmentation import (
    SAM_CACHE_SCHEMA_VERSION,
    create_segmenter,
    generate_mask,
    load_cached_mask,
    log_sam_failure,
    save_mask_cache,
)


def run(config: Config, *, overwrite: bool = False) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    table = load_scores(config)
    device = resolve_device(config.segmentation.device)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_log = run_dir / config.output.sam_failure_log
    segmenter = None
    created, reused, failures = 0, 0, []
    foreground_fractions: list[float] = []
    started = perf_counter()
    total = len(table)

    for position, (_, row) in enumerate(table.iterrows(), start=1):
        filename = str(row[config.data.filename_column])
        source = image_path(config, filename)
        grid_image = None
        mask_image = None
        try:
            if not overwrite:
                try:
                    mask_image, _, metadata = load_cached_mask(source, config)
                    foreground_fractions.append(
                        float(metadata["quality"]["foreground_fraction"])
                    )
                    reused += 1
                    if position % 10 == 0 or position == total:
                        print(
                            f"SAM cache {position}/{total} | created {created} | "
                            f"reused {reused} | failed {len(failures)}",
                            flush=True,
                        )
                    continue
                except FileNotFoundError:
                    pass

            grid_image, grid_path, _ = load_or_create_grid_crop(
                source,
                config.data.grid_cache_dir,
                size=config.data.grid_crop_size,
                inner_margin_fraction=config.data.grid_inner_margin_fraction,
            )
            if segmenter is None:
                print(
                    f"Loading frozen {config.segmentation.model_name} on {device}...",
                    flush=True,
                )
                segmenter = create_segmenter(config, device)
            mask = generate_mask(segmenter, grid_image, config)
            _, _, metadata = save_mask_cache(
                mask,
                source=source,
                grid_crop_path=grid_path,
                config=config,
            )
            quality = metadata["quality"]
            if not quality["valid"]:
                raise ValueError("; ".join(quality["quality_reasons"]))
            foreground_fractions.append(float(quality["foreground_fraction"]))
            created += 1
        except Exception as error:
            failures.append(filename)
            log_sam_failure(
                failure_log,
                error=error,
                image_path=source,
                filename=filename,
                dataset_index=position - 1,
            )
        finally:
            if grid_image is not None:
                grid_image.close()
            if mask_image is not None:
                mask_image.close()

        if position % 10 == 0 or position == total:
            print(
                f"SAM cache {position}/{total} | created {created} | "
                f"reused {reused} | failed {len(failures)}",
                flush=True,
            )

    fractions = np.asarray(foreground_fractions, dtype=float)
    report = {
        "images": total,
        "created": created,
        "reused": reused,
        "failures": len(failures),
        "failed_filenames": failures,
        "valid_masks": len(foreground_fractions),
        "mask_cache_dir": str(Path(config.segmentation.mask_cache_dir).resolve()),
        "sam_cache_schema_version": SAM_CACHE_SCHEMA_VERSION,
        "sam_model": config.segmentation.model_name,
        "prompts": list(config.segmentation.prompts),
        "score_threshold": config.segmentation.score_threshold,
        "mask_threshold": config.segmentation.mask_threshold,
        "device": str(device),
        "foreground_fraction": {
            "minimum": float(fractions.min()) if fractions.size else None,
            "median": float(np.median(fractions)) if fractions.size else None,
            "mean": float(fractions.mean()) if fractions.size else None,
            "maximum": float(fractions.max()) if fractions.size else None,
        },
        "failure_log": str(failure_log.resolve()),
        "seconds": perf_counter() - started,
    }
    write_json(run_dir / "sam_cache_summary.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks even when this exact SAM configuration is cached.",
    )
    arguments = parser.parse_args(argv)
    report = run(load_config(arguments.config), overwrite=arguments.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
