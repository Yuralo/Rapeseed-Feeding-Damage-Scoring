"""Resumable, holdout-safe SAM/DINO preparation of counterfactual patch pairs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from experiments.dinov3_grid_lora_patch_attention_sam_fusion.segmentation import (
    create_segmenter,
    validate_mask,
)
from experiments.dinov3_grid_sam_adaptive_mil.crops import make_adaptive_crop_layout
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from rapeseed_damage.artifacts import append_jsonl, write_json
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .synthesis import make_pair

SCHEMA = 1
EXCLUDED_COHORTS = {"gg_insects_t1_bbch10", "gg_insects_t2_bbch13"}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _manifest(config: Config, limit: int | None = None) -> pd.DataFrame:
    path = Path(config.adaptation_manifest)
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"image_id", "absolute_path", "relative_path", "sha256", "cohort_id",
                "plot_group_id"}
    missing = required - set(table.columns)
    if table.empty or missing:
        raise ValueError(f"Empty or incomplete adaptation manifest {path}: {sorted(missing)}")
    if table["image_id"].duplicated().any() or table["sha256"].duplicated().any():
        raise ValueError("Adaptation manifest contains duplicate image identities")
    if set(table["cohort_id"]) & EXCLUDED_COHORTS:
        raise ValueError("GG insect timepoints must remain excluded from synthetic pretraining")
    gold_dir = Path(config.plant.base.data.manifest_dir)
    holdout = pd.concat([pd.read_csv(gold_dir / f"{split}.csv", dtype=str,
                                     keep_default_na=False)
                         for split in ("validation", "test")], ignore_index=True)
    for column in ("sha256", "absolute_path", "plot_group_id"):
        training = set(table[column]) - {""}
        withheld = set(holdout[column]) - {""}
        if training & withheld:
            raise ValueError(f"Adaptation manifest overlaps gold holdout by {column}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        table = table.head(limit)
    return table.reset_index(drop=True)


def _identity(config: Config, row: pd.Series) -> str:
    relevant = (
        SCHEMA, row["sha256"], row["relative_path"],
        tuple(config.severity_levels), config.minimum_leaf_pixels,
        config.minimum_removed_pixels, config.seed,
        config.plant.patches_per_side, config.plant.overlap_fraction,
        config.plant.base.features.backbone, config.plant.base.features.processor,
        config.plant.base.features.representation,
        config.plant.base.data.raw_source_folders,
        config.plant.base.data.grid_inner_margin_fraction,
        asdict(config.plant.base.segmentation),
    )
    return _sha(repr(relevant))


def _pair_path(config: Config, row: pd.Series) -> Path:
    return Path(config.pair_cache_dir) / f"{row['image_id']}_{_identity(config, row)[:16]}.npz"


def load_pair(path: Path, expected_identity: str | None = None) -> dict:
    with np.load(path, allow_pickle=False) as raw:
        record = {key: np.asarray(raw[key]) for key in raw.files}
    if int(record["schema_version"]) != SCHEMA:
        raise ValueError(f"Pair cache schema mismatch: {path}")
    if expected_identity is not None and str(record["identity"]) != expected_identity:
        raise ValueError(f"Stale counterfactual cache: {path}")
    levels = np.asarray(record["realized_fractions"], dtype=np.float32)
    original = np.asarray(record["original_feature"], dtype=np.float32)
    reference = np.asarray(record["plant_feature"], dtype=np.float32)
    bites = np.asarray(record["bite_features"], dtype=np.float32)
    shams = np.asarray(record["sham_features"], dtype=np.float32)
    if original.ndim != 1 or reference.shape != original.shape:
        raise ValueError(f"Invalid original/reference feature shape: {path}")
    if bites.shape != shams.shape or bites.shape != (len(levels), len(original)):
        raise ValueError(f"Invalid bite/sham feature shape: {path}")
    if not np.isfinite(levels).all() or not (levels > 0).all():
        raise ValueError(f"Invalid realized fractions: {path}")
    if not all(np.isfinite(x).all() for x in (original, reference, bites, shams)):
        raise ValueError(f"Nonfinite DINO features: {path}")
    return record


def _save_pair(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, schema_version=np.asarray(SCHEMA, dtype=np.int16), **record)
    os.replace(temporary, path)


def _preview(original: Image.Image, leaf_mask: np.ndarray, pairs: list,
             levels: list[float], path: Path) -> None:
    width, height = original.size
    scale = min(1, 360 / max(width, height))
    size = (round(width * scale), round(height * scale))
    columns = 4
    canvas = Image.new("RGB", (size[0] * columns, (size[1] + 25) * len(pairs)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (pair, level) in enumerate(zip(pairs, levels)):
        y = index * (size[1] + 25)
        leaf = np.asarray(leaf_mask, dtype=bool)
        overlay = np.zeros((*leaf.shape, 3), dtype=np.uint8)
        overlay[leaf] = (60, 180, 50)
        overlay[pair.removed_mask] = (245, 40, 20)
        with Image.fromarray(overlay) as mask_view:
            mask_preview = mask_view.copy()
        for column, view in enumerate((original, pair.bite, pair.sham, mask_preview)):
            thumb = view.resize(size, Image.Resampling.BILINEAR)
            canvas.paste(thumb, (column * size[0], y + 25))
            thumb.close()
        mask_preview.close()
        draw.text((4, y + 4), f"requested {level:.1%} / realized {pair.realized_fraction:.1%}",
                  fill="black")
        draw.text((size[0] + 4, y + 4), f"{pair.kind} bite", fill="black")
        draw.text((2 * size[0] + 4, y + 4), "soil sham", fill="black")
        draw.text((3 * size[0] + 4, y + 4), "leaf / removed", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)
    canvas.close()


def _choose_patch(mask: np.ndarray, config: Config) -> tuple[tuple[int, ...], tuple[int, ...]]:
    from experiments.dinov3_plant_damage_mil.features import patch_boxes

    layout = make_adaptive_crop_layout(mask, config.plant.base.adaptive_crops)
    plant = np.asarray(layout.boxes[0], dtype=np.int32)
    boxes = patch_boxes(plant, config.plant.patches_per_side,
                        config.plant.overlap_fraction)
    counts = [int(mask[y0:y1, x0:x1].sum()) for x0, y0, x1, y1 in boxes]
    selected = boxes[int(np.argmax(counts))]
    if max(counts) < config.minimum_leaf_pixels:
        raise ValueError(f"Selected patch has only {max(counts)} leaf pixels")
    return tuple(map(int, plant)), tuple(map(int, selected))


def _make_features(config: Config, row: pd.Series, extractor,
                   preview_path: Path | None) -> tuple[dict, list[dict]]:
    from experiments.dinov3_hierarchical_three_view_mil.features import (
        mask_cache_path,
        prepare_image,
    )

    source = Path(row["absolute_path"])
    relative = row["relative_path"]
    image, _, _ = prepare_image(source, relative, config.plant.base)
    try:
        mask_path = mask_cache_path(config.plant.base, relative, source)
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("L")) >= 128
        if mask.shape != (image.height, image.width):
            raise ValueError("Saved SAM mask does not match routed image")
        plant_box, patch_box = _choose_patch(mask, config)
        plant = image.crop(plant_box)
        patch = image.crop(patch_box)
        seed = int(_sha(f"{config.seed}:{row['sha256']}")[:16], 16)
        rng = np.random.default_rng(seed)
        pairs, requested = [], []
        for level in config.severity_levels:
            try:
                pair = make_pair(image, mask, patch_box, level, rng,
                                 config.minimum_removed_pixels)
            except ValueError:
                continue
            pairs.append(pair)
            requested.append(level)
        if len(pairs) < config.minimum_accepted_levels:
            raise ValueError(f"Only {len(pairs)} realistic levels were generated")
        if preview_path is not None:
            x0, y0, x1, y1 = patch_box
            _preview(patch, mask[y0:y1, x0:x1], pairs, requested, preview_path)
        views = [plant, patch]
        for pair in pairs:
            views.extend((pair.bite, pair.sham))
        try:
            values = extractor.extract(views)
        finally:
            for view in views:
                view.close()
        dtype = np.float16 if config.plant.base.features.storage_dtype == "float16" else np.float32
        record = {
            "identity": np.asarray(_identity(config, row)),
            "plant_feature": values[0].astype(dtype),
            "original_feature": values[1].astype(dtype),
            "bite_features": values[2::2].astype(dtype),
            "sham_features": values[3::2].astype(dtype),
            "requested_fractions": np.asarray(requested, dtype=np.float32),
            "realized_fractions": np.asarray([p.realized_fraction for p in pairs], dtype=np.float32),
            "kinds": np.asarray([p.kind for p in pairs]),
            "plant_box": np.asarray(plant_box, dtype=np.int32),
            "patch_box": np.asarray(patch_box, dtype=np.int32),
        }
        details = [{"level_index": index, "requested_fraction": level,
                    "realized_fraction": pair.realized_fraction, "kind": pair.kind}
                   for index, (pair, level) in enumerate(zip(pairs, requested))]
        return record, details
    finally:
        image.close()


def _index_rows(config: Config, table: pd.DataFrame) -> list[dict]:
    index = []
    for _, row in table.iterrows():
        path = _pair_path(config, row)
        if not path.is_file():
            continue
        try:
            record = load_pair(path, _identity(config, row))
        except (ValueError, OSError, KeyError):
            continue
        for level, realized in enumerate(record["realized_fractions"]):
            index.append({"image_id": row["image_id"], "sha256": row["sha256"],
                          "relative_path": row["relative_path"], "cohort_id": row["cohort_id"],
                          "plot_group_id": row["plot_group_id"], "pair_path": str(path),
                          "identity": _identity(config, row), "level_index": level,
                          "requested_fraction": float(record["requested_fractions"][level]),
                          "realized_fraction": float(realized),
                          "kind": str(record["kinds"][level])})
    return index


def _write_index(path: Path, rows: list[dict]) -> None:
    columns = ("image_id", "sha256", "relative_path", "cohort_id", "plot_group_id",
               "pair_path", "identity", "level_index", "requested_fraction",
               "realized_fraction", "kind")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(config: Config, *, limit: int | None = None, masks_only: bool = False,
        overwrite: bool = False) -> dict:
    from experiments.dinov3_hierarchical_three_view_mil.features import (
        mask_cache_path,
        prepare_image,
        save_mask,
    )
    from experiments.dinov3_hierarchical_three_view_mil.prepare_features import (
        generate_memory_bounded_mask,
    )

    table = _manifest(config, limit)
    seed_everything(config.seed, config.plant.base.runtime.deterministic)
    device = resolve_device(config.plant.base.runtime.device)
    configure_acceleration(config.plant.base, device)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_log = run_dir / "pair_preparation_failures.jsonl"
    if overwrite:
        failure_log.unlink(missing_ok=True)
    segmenter = None
    mask_failed = set()
    mask_counts = Counter()
    by_cohort = defaultdict(Counter)
    for position, (_, row) in enumerate(table.iterrows(), 1):
        source = Path(row["absolute_path"])
        relative = row["relative_path"]
        image = None
        try:
            mask_path = mask_cache_path(config.plant.base, relative, source)
            if mask_path.is_file() and not overwrite:
                with Image.open(mask_path) as stored:
                    mask = np.asarray(stored.convert("L")) >= 128
                image, _, _ = prepare_image(source, relative, config.plant.base)
                if mask.shape != (image.height, image.width):
                    raise ValueError("Cached mask shape differs from routed image")
                mask_counts["reused"] += 1
            else:
                image, _, _ = prepare_image(source, relative, config.plant.base)
                if segmenter is None:
                    segmenter = create_segmenter(config.plant.base, device)
                mask, _, _ = generate_memory_bounded_mask(segmenter, image, config.plant.base)
                mask_counts["created"] += 1
            quality = validate_mask(mask, config.plant.base)
            if not quality["valid"]:
                raise ValueError("SAM quality: " + "; ".join(quality["quality_reasons"]))
            _choose_patch(mask, config)
            if not mask_path.is_file() or overwrite:
                save_mask(mask, mask_path)
            by_cohort[row["cohort_id"]]["masked"] += 1
        except Exception as error:  # noqa: BLE001 - complete per-cohort failure audit
            mask_failed.add(row["image_id"])
            by_cohort[row["cohort_id"]]["mask_failed"] += 1
            append_jsonl(failure_log, {"stage": "mask", "image_id": row["image_id"],
                                       "relative_path": relative, "error": str(error),
                                       "traceback": traceback.format_exc()})
        finally:
            if image is not None:
                image.close()
        if position % 100 == 0 or position == len(table):
            print(f"[masks {position}/{len(table)}] usable={position-len(mask_failed)}", flush=True)
    del segmenter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pair_counts = Counter()
    preview_counts = Counter()
    if not masks_only:
        extractor = None
        # The SHA order is fixed before synthesis; take the first 25 accepted per cohort.
        pair_table = table.assign(qa_order=[_sha(f"qa:{config.seed}:{sha}")
                                            for sha in table["sha256"]])
        pair_table = pair_table.sort_values(["cohort_id", "qa_order"])
        for position, (_, row) in enumerate(pair_table.iterrows(), 1):
            if row["image_id"] in mask_failed:
                continue
            path = _pair_path(config, row)
            try:
                cohort = row["cohort_id"]
                preview_path = (run_dir / "quality_examples" / cohort /
                                f"{row['image_id']}.jpg")
                wants_preview = preview_counts[cohort] < config.quality_examples_per_cohort
                cached = path.is_file() and not overwrite
                if cached:
                    load_pair(path, _identity(config, row))
                    pair_counts["reused"] += 1
                if not cached or (wants_preview and not preview_path.is_file()):
                    if extractor is None:
                        from experiments.dinov3_grid_tiled_mil.features import FrozenDinoExtractor

                        extractor = FrozenDinoExtractor(config.plant.base, device)
                    record, _ = _make_features(config, row, extractor,
                                               preview_path if wants_preview else None)
                    if not cached:
                        _save_pair(path, record)
                        pair_counts["created"] += 1
                if wants_preview and preview_path.is_file():
                    preview_counts[cohort] += 1
                by_cohort[row["cohort_id"]]["paired"] += 1
            except Exception as error:  # noqa: BLE001 - complete per-cohort failure audit
                pair_counts["failed"] += 1
                by_cohort[row["cohort_id"]]["pair_failed"] += 1
                append_jsonl(failure_log, {"stage": "features", "image_id": row["image_id"],
                                           "relative_path": row["relative_path"],
                                           "error": str(error), "traceback": traceback.format_exc()})
            if position % 100 == 0 or position == len(table):
                print(f"[pairs {position}/{len(table)}] created={pair_counts['created']} "
                      f"reused={pair_counts['reused']} failed={pair_counts['failed']}", flush=True)
        del extractor
    rows = _index_rows(config, table) if not masks_only else []
    if not masks_only:
        _write_index(run_dir / "pair_index.csv", rows)
    paired_images = len({row["image_id"] for row in rows})
    coverage = paired_images / len(table) if not masks_only else (len(table) - len(mask_failed)) / len(table)
    report = {"requested_images": len(table), "mask_counts": dict(mask_counts),
              "pair_counts": dict(pair_counts), "usable_images": paired_images if not masks_only else
              len(table) - len(mask_failed), "pair_levels": len(rows), "coverage_fraction": coverage,
              "quality_examples_by_cohort": dict(preview_counts),
              "by_cohort": {key: dict(value) for key, value in sorted(by_cohort.items())},
              "failure_log": str(failure_log), "masks_only": masks_only,
              "preparation_config": asdict(config),
              "adaptation_manifest_sha256": hashlib.sha256(Path(config.adaptation_manifest).read_bytes()).hexdigest()}
    write_json(run_dir / "preparation_summary.json", report)
    if not masks_only and limit is None:
        shortfall = {cohort: config.quality_examples_per_cohort - preview_counts[cohort]
                     for cohort in table["cohort_id"].unique()
                     if preview_counts[cohort] < config.quality_examples_per_cohort}
        if shortfall:
            raise RuntimeError(f"Target-blind QA examples are incomplete by cohort: {shortfall}")
    if coverage < config.minimum_coverage_fraction and limit is None:
        raise RuntimeError(f"Coverage {coverage:.1%} is below configured "
                           f"{config.minimum_coverage_fraction:.1%}; inspect {failure_log}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--masks-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), limit=args.limit,
                         masks_only=args.masks_only, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
