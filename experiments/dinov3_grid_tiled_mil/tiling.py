"""Deterministic, complete-coverage overlapping image tiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileLayout:
    boxes: np.ndarray
    tile_width: int
    tile_height: int


def _axis_layout(length: int, count: int, overlap_fraction: float) -> tuple[list[int], int]:
    if length < 1 or count < 1:
        raise ValueError("Axis length and tile count must be positive")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    if count == 1:
        return [0], length
    denominator = count - overlap_fraction * (count - 1)
    tile_size = min(length, max(1, round(length / denominator)))
    starts = np.linspace(0, length - tile_size, count).round().astype(int).tolist()
    return starts, tile_size


def make_tile_layout(
    width: int,
    height: int,
    rows: int,
    columns: int,
    overlap_fraction: float,
) -> TileLayout:
    x_starts, tile_width = _axis_layout(width, columns, overlap_fraction)
    y_starts, tile_height = _axis_layout(height, rows, overlap_fraction)
    boxes = np.asarray(
        [(x, y, x + tile_width, y + tile_height) for y in y_starts for x in x_starts],
        dtype=np.int32,
    )
    return TileLayout(boxes=boxes, tile_width=tile_width, tile_height=tile_height)


def global_and_tiled_views(
    image: Image.Image,
    *,
    rows: int,
    columns: int,
    overlap_fraction: float,
) -> tuple[list[Image.Image], np.ndarray]:
    image = image.convert("RGB")
    layout = make_tile_layout(image.width, image.height, rows, columns, overlap_fraction)
    tiles = [image.crop(tuple(map(int, box))) for box in layout.boxes]
    return [image, *tiles], layout.boxes
