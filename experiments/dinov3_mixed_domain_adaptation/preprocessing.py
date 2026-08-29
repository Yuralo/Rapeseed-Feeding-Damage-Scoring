"""Explicit mixed-source routing and reusable local-crop selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image, ImageOps

from .config import Config

GRID_CROP_MODE = "grid_crop"
RAW_MODE = "raw"
PREPARED_SCHEMA_VERSION = 1


def preprocessing_mode(filename: str, config: Config) -> str:
    """Route by source filename; never silently fall back between modes."""
    name = Path(filename).name
    if re.fullmatch(config.data.timestamp_pattern, name, flags=re.IGNORECASE):
        return GRID_CROP_MODE
    if re.fullmatch(config.data.raw_pattern, name, flags=re.IGNORECASE):
        return RAW_MODE
    raise ValueError(
        f"Unsupported adaptation filename {name!r}; it matches neither the timestamp "
        "grid-crop pattern nor the IMG raw-input pattern"
    )


def probable_label_mask(image: Image.Image) -> np.ndarray:
    """Find compact, bright, low-saturation collector-card regions.

    This mask is used only to reject local training crops. The image itself is
    never painted over or altered by this heuristic.
    """
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidate = ((hsv[..., 1] < 70) & (hsv[..., 2] > 150)).astype(np.uint8)
    minimum = min(candidate.shape)
    kernel_size = max(3, round(minimum * 0.012))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    result = np.zeros_like(candidate, dtype=np.uint8)
    image_area = candidate.size
    for component in range(1, count):
        x, y, width, height, area = map(int, statistics[component])
        area_fraction = area / image_area
        aspect = width / max(1, height)
        extent = area / max(1, width * height)
        if 0.002 <= area_fraction <= 0.15 and 0.25 <= aspect <= 4.0 and extent >= 0.42:
            result[labels == component] = 1
            padding = max(2, round(minimum * 0.008))
            x0, y0 = max(0, x - padding), max(0, y - padding)
            x1 = min(result.shape[1], x + width + padding)
            y1 = min(result.shape[0], y + height + padding)
            result[y0:y1, x0:x1] = 1
    return result


@dataclass(frozen=True)
class CropSelection:
    box: tuple[int, int, int, int]
    label_overlap_fraction: float


def _candidate_box(image_size: tuple[int, int], config: Config, rng: Random):
    width, height = image_size
    short_side = min(width, height)
    scale = rng.uniform(config.crops.minimum_scale, config.crops.maximum_scale)
    side = max(1, min(short_side, round(short_side * scale)))
    left = rng.randint(0, max(0, width - side))
    top = rng.randint(0, max(0, height - side))
    return left, top, left + side, top + side


def _overlap(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = box
    region = mask[top:bottom, left:right]
    return float(region.mean()) if region.size else 1.0


def _fallback_boxes(image_size: tuple[int, int], config: Config):
    """Cover corners and center when random candidates all intersect a label."""
    width, height = image_size
    side = max(1, round(min(width, height) * config.crops.minimum_scale))
    left_positions = (0, max(0, (width - side) // 2), max(0, width - side))
    top_positions = (0, max(0, (height - side) // 2), max(0, height - side))
    for top in top_positions:
        for left in left_positions:
            yield left, top, left + side, top + side


def select_local_crop(
    image: Image.Image,
    config: Config,
    rng: Random,
    *,
    label_mask: np.ndarray | None = None,
) -> CropSelection:
    mask = probable_label_mask(image) if label_mask is None else label_mask
    best: CropSelection | None = None
    for _ in range(config.crops.candidate_attempts):
        box = _candidate_box(image.size, config, rng)
        selection = CropSelection(box, _overlap(mask, box))
        if best is None or selection.label_overlap_fraction < best.label_overlap_fraction:
            best = selection
        if selection.label_overlap_fraction <= config.crops.label_overlap_limit:
            return selection
    for box in _fallback_boxes(image.size, config):
        selection = CropSelection(box, _overlap(mask, box))
        if best is None or selection.label_overlap_fraction < best.label_overlap_fraction:
            best = selection
        if selection.label_overlap_fraction <= config.crops.label_overlap_limit:
            return selection
    if best is None:  # pragma: no cover - candidate_attempts is validated positive.
        raise RuntimeError("No local crop candidate was generated")
    return best


def load_prepared_image(record: dict[str, str]) -> Image.Image:
    path = Path(record["processed_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Prepared image is missing: {path}")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()
