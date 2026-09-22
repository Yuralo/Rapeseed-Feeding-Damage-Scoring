"""Routed 3x3 tile embeddings paired with existing DINO/SAM plant records."""

from __future__ import annotations

import os
from hashlib import sha1
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from experiments.dinov3_grid_tiled_mil.tiling import make_tile_layout
from experiments.dinov3_hierarchical_three_view_mil.features import cache_identity as base_identity

from .config import Config

SCHEMA = 1


def identity(config: Config, relative: str, source: Path) -> str:
    context = config.adaptive.context
    values = (
        SCHEMA, base_identity(config.weak.routed_base, relative, source),
        context.rows, context.columns, context.overlap_fraction,
        config.adaptive.features.backbone, config.adaptive.features.processor,
    )
    return sha1(repr(values).encode()).hexdigest()


def cache_path(config: Config, relative: str, source: Path) -> Path:
    return Path(config.context_cache_dir) / f"{source.stem}_{identity(config, relative, source)[:16]}.npz"


def extract(extractor, config: Config, base_record: dict) -> dict:
    """Embed only nine tiles; reuse the base record's exact global representation."""
    path = Path(base_record["processed_image_path"])
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        context = config.adaptive.context
        layout = make_tile_layout(
            image.width, image.height, context.rows, context.columns,
            context.overlap_fraction,
        )
        views = [image.crop(tuple(map(int, box))) for box in layout.boxes]
        try:
            features = extractor.extract(views)
        finally:
            for view in views:
                view.close()
            image.close()
    if features.shape != (9, len(base_record["global_feature"])):
        raise ValueError(f"3x3 tile feature shape differs from base features: {path}")
    dtype = np.float16 if config.weak.routed_base.features.storage_dtype == "float16" else np.float32
    return {
        "tile_features": features.astype(dtype),
        "tile_boxes": layout.boxes,
        "processed_image_path": str(path),
    }


def save(path: Path, record: dict, expected_identity: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(SCHEMA, dtype=np.int16),
        identity=np.asarray(expected_identity),
        **record,
    )
    os.replace(temporary, path)


def load(path: Path, expected_identity: str) -> dict:
    with np.load(path, allow_pickle=False) as raw:
        if int(raw["schema_version"]) != SCHEMA or str(raw["identity"]) != expected_identity:
            raise ValueError(f"Stale adaptive-context cache: {path}")
        features = np.asarray(raw["tile_features"], dtype=np.float32)
        boxes = np.asarray(raw["tile_boxes"], dtype=np.int32)
        processed = str(raw["processed_image_path"])
    if features.ndim != 2 or features.shape[0] != 9 or boxes.shape != (9, 4):
        raise ValueError(f"Invalid 3x3 adaptive-context cache shapes: {path}")
    if not np.isfinite(features).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError(f"Invalid adaptive-context cache contents: {path}")
    return {"tile_features": features, "tile_boxes": boxes, "processed_image_path": processed}
