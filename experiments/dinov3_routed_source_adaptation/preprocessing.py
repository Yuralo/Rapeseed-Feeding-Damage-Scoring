"""Source routing, cached grid crops, and shared adaptation tiling."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from experiments.dinov3_mixed_domain_adaptation.preprocessing import (
    ScoredTileCandidate,
    TileCandidate,
    TileSelection,
    choose_adaptation_tile,
    deserialize_tile_candidates,
    probable_label_mask,
    probable_vegetation_mask,
    score_tile_candidates,
    select_adaptation_tile,
    selection_masks,
    serialize_tile_candidates,
    tile_candidates,
)
from rapeseed_damage.grid import detect_grid, to_rgb, warp_big_square

from .config import Config

RAW_FULL_TILED_MODE = "raw_full_tiled"
GRID_INSET_TILED_MODE = "grid_inset075_tiled"
ALLOWED_INPUT_MODES = {RAW_FULL_TILED_MODE, GRID_INSET_TILED_MODE}
PREPARED_SCHEMA_VERSION = 1
CROP_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PreparedImage:
    image: Image.Image
    prepared_path: Path
    input_mode: str
    cache_created: bool
    source_width: int
    source_height: int
    grid_points_json: str


def source_folder(row: dict[str, str], config: Config) -> str:
    relative = Path(row[config.data.relative_path_column])
    return relative.parts[0] if len(relative.parts) > 1 else "<dataset-root>"


def route_for_source(folder: str, config: Config) -> str:
    if folder in set(config.preprocessing.raw_source_folders):
        return RAW_FULL_TILED_MODE
    return GRID_INSET_TILED_MODE


def _load_oriented_rgb(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        image.load()
        return image.copy()


def crop_cache_path(source: Path, config: Config) -> Path:
    settings = config.preprocessing
    identity = (
        f"schema={CROP_CACHE_SCHEMA_VERSION}|{source.resolve()}|"
        f"size={settings.crop_size}|"
        f"inset={settings.grid_inner_margin_fraction:.8f}"
    )
    digest = sha1(identity.encode("utf-8")).hexdigest()[:12]
    return (
        Path(settings.crop_cache_dir)
        / f"{source.stem}_gridr{CROP_CACHE_SCHEMA_VERSION}_{digest}.jpg"
    )


def _detect_and_crop(image: Image.Image, config: Config) -> tuple[Image.Image, str]:
    import cv2

    rgb = np.asarray(image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    grid_points, _, _ = detect_grid(bgr)
    cropped_bgr = warp_big_square(
        bgr,
        grid_points,
        size=config.preprocessing.crop_size,
        inner_margin_fraction=config.preprocessing.grid_inner_margin_fraction,
    )
    cropped = Image.fromarray(to_rgb(cropped_bgr)).convert("RGB")
    return cropped, json.dumps(grid_points.tolist(), separators=(",", ":"))


def _write_jpeg_atomic(image: Image.Image, destination: Path, quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    image.save(temporary, format="JPEG", quality=quality, subsampling=0)
    os.replace(temporary, destination)


def prepare_image(source: Path, folder: str, config: Config) -> PreparedImage:
    """Create the routed input once; raw sources are never copied or modified."""
    input_mode = route_for_source(folder, config)
    source_image = _load_oriented_rgb(source)
    source_width, source_height = source_image.size
    if input_mode == RAW_FULL_TILED_MODE:
        return PreparedImage(
            image=source_image,
            prepared_path=source.resolve(),
            input_mode=input_mode,
            cache_created=False,
            source_width=source_width,
            source_height=source_height,
            grid_points_json="",
        )

    destination = crop_cache_path(source, config).resolve()
    if destination.is_file() and not config.preprocessing.overwrite_cached_crops:
        source_image.close()
        return PreparedImage(
            image=_load_oriented_rgb(destination),
            prepared_path=destination,
            input_mode=input_mode,
            cache_created=False,
            source_width=source_width,
            source_height=source_height,
            grid_points_json="",
        )

    try:
        cropped, grid_points_json = _detect_and_crop(source_image, config)
    finally:
        source_image.close()
    try:
        _write_jpeg_atomic(
            cropped,
            destination,
            quality=config.preprocessing.crop_jpeg_quality,
        )
    except Exception:
        cropped.close()
        raise
    return PreparedImage(
        image=cropped,
        prepared_path=destination,
        input_mode=input_mode,
        cache_created=True,
        source_width=source_width,
        source_height=source_height,
        grid_points_json=grid_points_json,
    )


def load_prepared_image(record: dict[str, str]) -> Image.Image:
    path = Path(record["prepared_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Prepared adaptation image is missing: {path}")
    return _load_oriented_rgb(path)


__all__ = [
    "ALLOWED_INPUT_MODES",
    "GRID_INSET_TILED_MODE",
    "PREPARED_SCHEMA_VERSION",
    "RAW_FULL_TILED_MODE",
    "ScoredTileCandidate",
    "TileCandidate",
    "TileSelection",
    "choose_adaptation_tile",
    "crop_cache_path",
    "deserialize_tile_candidates",
    "load_prepared_image",
    "prepare_image",
    "probable_label_mask",
    "probable_vegetation_mask",
    "route_for_source",
    "score_tile_candidates",
    "select_adaptation_tile",
    "selection_masks",
    "serialize_tile_candidates",
    "source_folder",
    "tile_candidates",
]
