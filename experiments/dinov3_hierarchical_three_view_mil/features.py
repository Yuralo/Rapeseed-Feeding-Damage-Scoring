"""Routed preprocessing and a versioned global/cell/plant feature cache."""

from __future__ import annotations

import os
from dataclasses import asdict
from hashlib import sha1
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from experiments.dinov3_grid_sam_adaptive_mil.crops import (
    crop_instances,
    make_adaptive_crop_layout,
)
from experiments.dinov3_grid_tiled_mil.features import FrozenDinoExtractor
from rapeseed_damage.grid import detect_grid, to_rgb, warp_big_square

from .config import Config

FEATURE_SCHEMA_VERSION = 2
RAW_MODE = "raw_full_image"
GRID_MODE = "grid_crop_inset075"


def source_folder(relative_path: str) -> str:
    parts = Path(relative_path).parts
    return parts[0] if len(parts) > 1 else "<dataset-root>"


def input_mode(relative_path: str, config: Config) -> str:
    return RAW_MODE if source_folder(relative_path) in set(config.data.raw_source_folders) else GRID_MODE


def _source_signature(source: Path) -> tuple:
    stat = source.stat()
    return str(source.resolve()), stat.st_size, stat.st_mtime_ns


def processed_cache_path(source: Path, relative_path: str, config: Config) -> Path:
    mode = input_mode(relative_path, config)
    signature = (
        FEATURE_SCHEMA_VERSION,
        *_source_signature(source),
        mode,
        config.data.crop_size,
        config.data.grid_inner_margin_fraction,
    )
    digest = sha1(repr(signature).encode()).hexdigest()[:12]
    suffix = "raw" if mode == RAW_MODE else "grid"
    return Path(config.data.processed_cache_dir) / f"{source.stem}_{suffix}_{digest}.jpg"


def cache_identity(config: Config, relative_path: str, source: Path) -> str:
    values = (
        FEATURE_SCHEMA_VERSION,
        *_source_signature(source),
        relative_path,
        input_mode(relative_path, config),
        config.data.crop_size,
        config.data.grid_inner_margin_fraction,
        config.data.cell_inner_margin_fraction,
        config.data.sam_inference_max_side,
        tuple(sorted(asdict(config.adaptive_crops).items())),
        tuple(sorted(asdict(config.segmentation).items())),
        config.features.backbone,
        config.features.processor,
        config.features.representation,
    )
    return sha1(repr(values).encode()).hexdigest()


def feature_cache_path(config: Config, relative_path: str, source: Path) -> Path:
    digest = cache_identity(config, relative_path, source)[:16]
    return Path(config.features.cache_dir) / f"{source.stem}_{digest}.npz"


def mask_cache_path(config: Config, relative_path: str, source: Path) -> Path:
    digest = cache_identity(config, relative_path, source)[:16]
    return Path(config.features.mask_cache_dir) / f"{source.stem}_{digest}.png"


def _oriented_rgb(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        result = ImageOps.exif_transpose(handle).convert("RGB")
        result.load()
        return result.copy()


def _atomic_jpeg(image: Image.Image, destination: Path, quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    image.save(temporary, format="JPEG", quality=quality, subsampling=0)
    os.replace(temporary, destination)


def prepare_image(source: Path, relative_path: str, config: Config) -> tuple[Image.Image, Path, str]:
    mode = input_mode(relative_path, config)
    if mode == RAW_MODE:
        return _oriented_rgb(source), source.resolve(), mode
    destination = processed_cache_path(source, relative_path, config).resolve()
    if destination.is_file():
        return _oriented_rgb(destination), destination, mode
    image = _oriented_rgb(source)
    try:
        import cv2

        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        points, _, _ = detect_grid(bgr)
        cropped = Image.fromarray(
            to_rgb(
                warp_big_square(
                    bgr,
                    points,
                    size=config.data.crop_size,
                    inner_margin_fraction=config.data.grid_inner_margin_fraction,
                )
            )
        ).convert("RGB")
    finally:
        image.close()
    _atomic_jpeg(cropped, destination, config.data.processed_jpeg_quality)
    return cropped, destination, mode


def cell_boxes(width: int, height: int, margin_fraction: float) -> np.ndarray:
    boxes = []
    for row in range(2):
        for column in range(2):
            outer_x0, outer_x1 = round(column * width / 2), round((column + 1) * width / 2)
            outer_y0, outer_y1 = round(row * height / 2), round((row + 1) * height / 2)
            margin_x = round((outer_x1 - outer_x0) * margin_fraction)
            margin_y = round((outer_y1 - outer_y0) * margin_fraction)
            boxes.append(
                [outer_x0 + margin_x, outer_y0 + margin_y, outer_x1 - margin_x, outer_y1 - margin_y]
            )
    return np.asarray(boxes, dtype=np.int32)


def assign_cells(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    return ((centers_y >= height / 2).astype(np.int16) * 2 + (centers_x >= width / 2)).astype(
        np.int16
    )


def save_mask(mask: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(
        temporary, format="PNG", optimize=False, compress_level=1
    )
    os.replace(temporary, destination)


def save_record(
    destination: Path,
    *,
    global_feature: np.ndarray,
    cell_features: np.ndarray,
    cell_boxes_array: np.ndarray,
    plant_features: np.ndarray,
    plant_boxes: np.ndarray,
    plant_cell_indices: np.ndarray,
    foreground_pixels: np.ndarray,
    mask_coverage: float,
    processed_image_path: str,
    mask_path: str,
    mode: str,
    identity: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int16),
        identity=np.asarray(identity),
        global_feature=global_feature,
        cell_features=cell_features,
        cell_boxes=cell_boxes_array.astype(np.int32),
        plant_features=plant_features,
        plant_boxes=plant_boxes.astype(np.int32),
        plant_cell_indices=plant_cell_indices.astype(np.int16),
        foreground_pixels=foreground_pixels.astype(np.int32),
        mask_coverage=np.asarray(mask_coverage, dtype=np.float32),
        processed_image_path=np.asarray(processed_image_path),
        mask_path=np.asarray(mask_path),
        input_mode=np.asarray(mode),
    )
    os.replace(temporary, destination)


def load_record(path: Path, *, expected_identity: str | None = None) -> dict:
    try:
        with np.load(path, allow_pickle=False) as raw:
            record = {name: np.asarray(raw[name]) for name in raw.files}
    except Exception as error:
        raise RuntimeError(f"Could not read three-view feature cache {path}: {error}") from error
    if int(record["schema_version"]) != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"Feature schema mismatch in {path}")
    identity = str(record["identity"])
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(f"Stale three-view feature cache identity in {path}")
    global_feature = np.asarray(record["global_feature"], dtype=np.float32)
    cells = np.asarray(record["cell_features"], dtype=np.float32)
    plants = np.asarray(record["plant_features"], dtype=np.float32)
    boxes = np.asarray(record["plant_boxes"], dtype=np.int32)
    cell_indices = np.asarray(record["plant_cell_indices"], dtype=np.int16)
    foreground = np.asarray(record["foreground_pixels"], dtype=np.int32)
    if global_feature.ndim != 1 or cells.shape != (4, global_feature.shape[0]):
        raise ValueError(f"Invalid global/cell feature shapes in {path}")
    if plants.ndim != 2 or plants.shape[1] != global_feature.shape[0]:
        raise ValueError(f"Invalid plant feature shape in {path}")
    if not len(plants) or not (len(plants) == len(boxes) == len(cell_indices) == len(foreground)):
        raise ValueError(f"Plant arrays disagree or are empty in {path}")
    if np.any((cell_indices < 0) | (cell_indices > 3)):
        raise ValueError(f"Plant cell indices must be in [0, 3] in {path}")
    if not all(np.isfinite(value).all() for value in (global_feature, cells, plants)):
        raise ValueError(f"Non-finite features in {path}")
    return {
        "global_feature": global_feature,
        "cell_features": cells,
        "cell_boxes": np.asarray(record["cell_boxes"], dtype=np.int32),
        "plant_features": plants,
        "plant_boxes": boxes,
        "plant_cell_indices": cell_indices,
        "foreground_pixels": foreground,
        "mask_coverage": float(record["mask_coverage"]),
        "processed_image_path": str(record["processed_image_path"]),
        "mask_path": str(record["mask_path"]),
        "input_mode": str(record["input_mode"]),
        "identity": identity,
    }


def extract_record(extractor, image: Image.Image, mask: np.ndarray, config: Config):
    """Embed all three views from an already prepared SAM mask."""
    layout = make_adaptive_crop_layout(mask, config.adaptive_crops)
    width, height = image.size
    cells = cell_boxes(width, height, config.data.cell_inner_margin_fraction)
    views = [image]
    views.extend(image.crop(tuple(map(int, box))) for box in cells)
    plants = crop_instances(image, layout.boxes)
    views.extend(plants)
    try:
        embeddings = extractor.extract(views)
    finally:
        for view in views[1:]:
            view.close()
    dtype = np.float16 if config.features.storage_dtype == "float16" else np.float32
    return {
        "global_feature": embeddings[0].astype(dtype),
        "cell_features": embeddings[1:5].astype(dtype),
        "cell_boxes": cells,
        "plant_features": embeddings[5:].astype(dtype),
        "plant_boxes": layout.boxes,
        "plant_cell_indices": assign_cells(layout.boxes, width, height),
        "foreground_pixels": layout.foreground_pixels,
        "mask_coverage": layout.mask_coverage,
    }


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "GRID_MODE",
    "RAW_MODE",
    "FrozenDinoExtractor",
    "assign_cells",
    "cache_identity",
    "cell_boxes",
    "extract_record",
    "feature_cache_path",
    "input_mode",
    "load_record",
    "mask_cache_path",
    "prepare_image",
    "save_mask",
    "save_record",
    "source_folder",
]
