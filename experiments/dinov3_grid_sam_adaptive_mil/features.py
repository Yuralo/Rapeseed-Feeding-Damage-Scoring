"""Versioned per-image adaptive-instance frozen DINO feature cache."""

from __future__ import annotations

import os
from dataclasses import asdict
from hashlib import sha1
from pathlib import Path

import numpy as np

from experiments.dinov3_grid_lora_patch_attention_sam_fusion.segmentation import (
    mask_cache_paths,
)
from experiments.dinov3_grid_tiled_mil.features import FrozenDinoExtractor

from .config import Config
from .crops import crop_instances, make_adaptive_crop_layout

ADAPTIVE_FEATURE_SCHEMA_VERSION = 1


def cache_identity(config: Config, filename: str, source: Path) -> str:
    stat = source.stat()
    _, _, sam_signature = mask_cache_paths(source, config)
    values = (
        ADAPTIVE_FEATURE_SCHEMA_VERSION,
        str(source.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        filename,
        sam_signature,
        tuple(sorted(asdict(config.adaptive_crops).items())),
        config.features.backbone,
        config.features.processor,
        config.features.representation,
    )
    return sha1(repr(values).encode("utf-8")).hexdigest()


def feature_cache_path(config: Config, filename: str, source: Path) -> Path:
    digest = cache_identity(config, filename, source)[:16]
    safe_stem = Path(filename).stem.replace("/", "_")
    return Path(config.features.cache_dir) / f"{safe_stem}_{digest}.npz"


def save_feature_record(
    destination: Path,
    *,
    features: np.ndarray,
    boxes: np.ndarray,
    foreground_pixels: np.ndarray,
    mask_coverage: float,
    components_before_merge: int,
    processed_image_path: str,
    mask_path: str,
    identity: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(ADAPTIVE_FEATURE_SCHEMA_VERSION, dtype=np.int16),
        identity=np.asarray(identity),
        features=features,
        boxes=np.asarray(boxes, dtype=np.int32),
        foreground_pixels=np.asarray(foreground_pixels, dtype=np.int32),
        mask_coverage=np.asarray(mask_coverage, dtype=np.float32),
        components_before_merge=np.asarray(components_before_merge, dtype=np.int16),
        processed_image_path=np.asarray(processed_image_path),
        mask_path=np.asarray(mask_path),
    )
    os.replace(temporary, destination)


def load_feature_record(path: Path, *, expected_identity: str | None = None) -> dict:
    try:
        with np.load(path, allow_pickle=False) as record:
            schema = int(record["schema_version"])
            identity = str(record["identity"])
            features = np.asarray(record["features"], dtype=np.float32)
            boxes = np.asarray(record["boxes"], dtype=np.int32)
            foreground_pixels = np.asarray(record["foreground_pixels"], dtype=np.int32)
            mask_coverage = float(record["mask_coverage"])
            components_before_merge = int(record["components_before_merge"])
            processed_image_path = str(record["processed_image_path"])
            mask_path = str(record["mask_path"])
    except Exception as error:
        raise RuntimeError(f"Could not read adaptive feature cache {path}: {error}") from error
    if schema != ADAPTIVE_FEATURE_SCHEMA_VERSION:
        raise ValueError(f"Adaptive feature-cache schema mismatch in {path}: {schema}")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(f"Stale adaptive feature-cache identity in {path}")
    if features.ndim != 2 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Invalid adaptive feature-cache shapes in {path}")
    if not len(features) or len(features) != len(boxes) or len(boxes) != len(foreground_pixels):
        raise ValueError(f"Adaptive feature/cache instance counts disagree in {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite adaptive features in {path}")
    return {
        "features": features,
        "boxes": boxes,
        "foreground_pixels": foreground_pixels,
        "mask_coverage": mask_coverage,
        "components_before_merge": components_before_merge,
        "processed_image_path": processed_image_path,
        "mask_path": mask_path,
        "identity": identity,
    }


def extract_adaptive_features(extractor, image, mask, config: Config):
    layout = make_adaptive_crop_layout(np.asarray(mask) >= 128, config.adaptive_crops)
    instances = crop_instances(image, layout.boxes)
    features = extractor.extract(instances)
    dtype = np.float16 if config.features.storage_dtype == "float16" else np.float32
    return features.astype(dtype), layout


__all__ = [
    "ADAPTIVE_FEATURE_SCHEMA_VERSION",
    "FrozenDinoExtractor",
    "cache_identity",
    "extract_adaptive_features",
    "feature_cache_path",
    "load_feature_record",
    "save_feature_record",
]
