"""Frozen DINOv3 embeddings for high-resolution patches inside SAM plant crops."""

from __future__ import annotations

import os
from hashlib import sha1
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.dinov3_hierarchical_three_view_mil.features import (
    cache_identity as base_identity,
)
from experiments.dinov3_hierarchical_three_view_mil.features import (
    feature_cache_path as base_cache_path,
)
from experiments.dinov3_hierarchical_three_view_mil.features import (
    load_record as load_base_record,
)

from .config import Config

SCHEMA = 1


def identity(config: Config, relative: str, source: Path) -> str:
    base = config.base
    values = (
        SCHEMA, base_identity(base, relative, source), config.patches_per_side,
        config.overlap_fraction, config.minimum_foreground_fraction,
        base.features.backbone, base.features.processor, base.features.representation,
    )
    return sha1(repr(values).encode()).hexdigest()


def cache_path(config: Config, relative: str, source: Path) -> Path:
    return Path(config.cache_dir) / f"{source.stem}_{identity(config, relative, source)[:16]}.npz"


def patch_boxes(plant_box: np.ndarray, side: int, overlap: float) -> np.ndarray:
    x0, y0, x1, y1 = map(float, plant_box)
    width, height = x1 - x0, y1 - y0
    if width < 2 or height < 2:
        raise ValueError(f"Invalid SAM plant box: {plant_box}")
    stride_fraction = 1 / side
    size_fraction = min(1.0, stride_fraction * (1 + overlap))
    boxes = []
    for row in range(side):
        for column in range(side):
            center_x = x0 + width * (column + 0.5) / side
            center_y = y0 + height * (row + 0.5) / side
            left = max(x0, center_x - width * size_fraction / 2)
            right = min(x1, center_x + width * size_fraction / 2)
            top = max(y0, center_y - height * size_fraction / 2)
            bottom = min(y1, center_y + height * size_fraction / 2)
            boxes.append([round(left), round(top), round(right), round(bottom)])
    return np.asarray(boxes, dtype=np.int32)


def extract(extractor, config: Config, base_record: dict) -> dict:
    image_path = Path(base_record["processed_image_path"])
    mask_path = Path(base_record["mask_path"])
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"Missing processed image or SAM mask: {image_path}, {mask_path}")
    with Image.open(mask_path) as handle:
        mask = np.asarray(handle.convert("L")) >= 128
    boxes = np.stack([
        patch_boxes(box, config.patches_per_side, config.overlap_fraction)
        for box in base_record["plant_boxes"]
    ])
    coverage = np.zeros(boxes.shape[:2], dtype=np.float32)
    views = []
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        if mask.shape != (image.height, image.width):
            raise ValueError(f"SAM mask size does not match processed image: {mask_path}")
        try:
            for plant_index, plant_boxes in enumerate(boxes):
                for patch_index, (left, top, right, bottom) in enumerate(plant_boxes):
                    if right <= left or bottom <= top:
                        raise ValueError(f"Empty patch in {image_path}")
                    coverage[plant_index, patch_index] = mask[top:bottom, left:right].mean()
                    views.append(image.crop((int(left), int(top), int(right), int(bottom))))
            features = extractor.extract(views).reshape(*boxes.shape[:2], -1)
        finally:
            for view in views:
                view.close()
            image.close()
    valid = coverage >= config.minimum_foreground_fraction
    for index in range(len(valid)):
        if not valid[index].any():
            valid[index, int(coverage[index].argmax())] = True
    dtype = np.float16 if config.base.features.storage_dtype == "float16" else np.float32
    return {
        "patch_features": features.astype(dtype),
        "patch_boxes": boxes,
        "patch_foreground_fraction": coverage,
        "patch_valid": valid,
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
            raise ValueError(f"Stale high-resolution patch cache: {path}")
        record = {key: np.asarray(raw[key]) for key in (
            "patch_features", "patch_boxes", "patch_foreground_fraction", "patch_valid"
        )}
    features = np.asarray(record["patch_features"], dtype=np.float32)
    boxes = record["patch_boxes"]
    coverage = record["patch_foreground_fraction"]
    valid = record["patch_valid"].astype(bool)
    if features.ndim != 3 or boxes.shape != (*features.shape[:2], 4):
        raise ValueError(f"Invalid patch array shapes: {path}")
    if coverage.shape != valid.shape or valid.shape != features.shape[:2]:
        raise ValueError(f"Invalid patch validity shape: {path}")
    if not valid.any(axis=1).all() or not np.isfinite(features).all():
        raise ValueError(f"Invalid patch contents: {path}")
    record["patch_features"] = features
    record["patch_valid"] = valid
    return record


def load_base(config: Config, relative: str, source: Path) -> dict:
    base = config.base
    return load_base_record(
        base_cache_path(base, relative, source),
        expected_identity=base_identity(base, relative, source),
    )
