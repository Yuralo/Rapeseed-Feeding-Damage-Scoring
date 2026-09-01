"""Raw-image tile selection for DINOv3 domain adaptation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image, ImageOps

from .config import Config

RAW_TILED_MODE = "raw_tiled"
PREPARED_SCHEMA_VERSION = 2


def probable_label_mask(image: Image.Image) -> np.ndarray:
    """Find compact, bright collector-card regions without altering the image."""
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


def probable_vegetation_mask(image: Image.Image) -> np.ndarray:
    """Return a permissive green-vegetation mask used only for tile sampling."""
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (
        (hsv[..., 0] >= 25)
        & (hsv[..., 0] <= 95)
        & (hsv[..., 1] >= 40)
        & (hsv[..., 2] >= 35)
    ).astype(np.uint8)


def selection_masks(image: Image.Image, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Compute inexpensive label/vegetation masks on a bounded-size thumbnail."""
    maximum = config.tiles.mask_analysis_max_side
    scale = min(1.0, maximum / max(image.size))
    size = tuple(max(1, round(value * scale)) for value in image.size)
    analysis_image = image if size == image.size else image.resize(size, Image.Resampling.BILINEAR)
    try:
        return probable_label_mask(analysis_image), probable_vegetation_mask(analysis_image)
    finally:
        if analysis_image is not image:
            analysis_image.close()


@dataclass(frozen=True)
class TileCandidate:
    grid_size: int
    row: int
    column: int
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class TileSelection:
    grid_size: int
    row: int
    column: int
    box: tuple[int, int, int, int]
    label_overlap_fraction: float
    vegetation_fraction: float
    sampling_strategy: str


@dataclass(frozen=True)
class ScoredTileCandidate:
    grid_size: int
    row: int
    column: int
    box: tuple[int, int, int, int]
    label_overlap_fraction: float
    vegetation_fraction: float


def _positions(length: int, side: int, count: int) -> list[int]:
    available = max(0, length - side)
    return [round(index * available / (count - 1)) for index in range(count)]


def tile_candidates(image_size: tuple[int, int], config: Config) -> list[TileCandidate]:
    """Build square, overlapping 3x3/4x4 grids that cover the complete raw image."""
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image size: {image_size}")
    maximum = max(width, height)
    candidates = []
    for grid_size in config.tiles.grid_sizes:
        effective_spans = grid_size - (grid_size - 1) * config.tiles.overlap_fraction
        side = min(min(width, height), math.ceil(maximum / effective_spans))
        left_positions = _positions(width, side, grid_size)
        top_positions = _positions(height, side, grid_size)
        for row, top in enumerate(top_positions):
            for column, left in enumerate(left_positions):
                candidates.append(
                    TileCandidate(
                        grid_size=grid_size,
                        row=row,
                        column=column,
                        box=(left, top, left + side, top + side),
                    )
                )
    return candidates


def _mask_fraction(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> float:
    image_width, image_height = image_size
    mask_height, mask_width = mask.shape
    left, top, right, bottom = box
    x0 = max(0, min(mask_width, math.floor(left * mask_width / image_width)))
    x1 = max(0, min(mask_width, math.ceil(right * mask_width / image_width)))
    y0 = max(0, min(mask_height, math.floor(top * mask_height / image_height)))
    y1 = max(0, min(mask_height, math.ceil(bottom * mask_height / image_height)))
    region = mask[y0:y1, x0:x1]
    return float(region.mean()) if region.size else 0.0


def score_tile_candidates(
    image: Image.Image,
    config: Config,
    *,
    label_mask: np.ndarray | None = None,
    vegetation_mask: np.ndarray | None = None,
) -> list[ScoredTileCandidate]:
    """Analyze the fixed candidate boxes once without modifying source pixels."""
    if label_mask is None or vegetation_mask is None:
        detected_label, detected_vegetation = selection_masks(image, config)
        label_mask = detected_label if label_mask is None else label_mask
        vegetation_mask = detected_vegetation if vegetation_mask is None else vegetation_mask
    scored: list[ScoredTileCandidate] = []
    for candidate in tile_candidates(image.size, config):
        scored.append(
            ScoredTileCandidate(
                grid_size=candidate.grid_size,
                row=candidate.row,
                column=candidate.column,
                box=candidate.box,
                label_overlap_fraction=_mask_fraction(
                    label_mask, candidate.box, image.size
                ),
                vegetation_fraction=_mask_fraction(
                    vegetation_mask, candidate.box, image.size
                ),
            )
        )
    return scored


def serialize_tile_candidates(candidates: list[ScoredTileCandidate]) -> str:
    return json.dumps([asdict(candidate) for candidate in candidates], separators=(",", ":"))


def deserialize_tile_candidates(value: str) -> list[ScoredTileCandidate]:
    try:
        raw = json.loads(value)
        candidates = [
            ScoredTileCandidate(
                grid_size=int(item["grid_size"]),
                row=int(item["row"]),
                column=int(item["column"]),
                box=tuple(map(int, item["box"])),
                label_overlap_fraction=float(item["label_overlap_fraction"]),
                vegetation_fraction=float(item["vegetation_fraction"]),
            )
            for item in raw
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Invalid serialized tile-candidate metadata") from error
    if not candidates or any(len(candidate.box) != 4 for candidate in candidates):
        raise ValueError("Serialized tile-candidate metadata is empty or malformed")
    return candidates


def choose_adaptation_tile(
    scored: list[ScoredTileCandidate], config: Config, rng: Random
) -> TileSelection:
    """Choose one label-safe scale, then use mixed plant-biased/uniform sampling."""
    if not scored:
        raise ValueError("No scored tile candidates were provided")
    eligible = [
        item
        for item in scored
        if item.label_overlap_fraction <= config.tiles.label_overlap_limit
    ]
    if not eligible:
        minimum_overlap = min(item.label_overlap_fraction for item in scored)
        eligible = [item for item in scored if item.label_overlap_fraction == minimum_overlap]
    available_scales = sorted({item.grid_size for item in eligible})
    selected_scale = rng.choice(available_scales)
    pool = [item for item in eligible if item.grid_size == selected_scale]
    use_plant_bias = (
        rng.random() < config.tiles.plant_biased_probability
        and any(item.vegetation_fraction > 0 for item in pool)
    )
    if use_plant_bias:
        weights = [
            item.vegetation_fraction**config.tiles.vegetation_score_power for item in pool
        ]
        candidate = rng.choices(pool, weights=weights, k=1)[0]
        strategy = "plant_biased"
    else:
        candidate = rng.choice(pool)
        strategy = "uniform"
    return TileSelection(
        grid_size=candidate.grid_size,
        row=candidate.row,
        column=candidate.column,
        box=candidate.box,
        label_overlap_fraction=candidate.label_overlap_fraction,
        vegetation_fraction=candidate.vegetation_fraction,
        sampling_strategy=strategy,
    )


def select_adaptation_tile(
    image: Image.Image,
    config: Config,
    rng: Random,
    *,
    label_mask: np.ndarray | None = None,
    vegetation_mask: np.ndarray | None = None,
) -> TileSelection:
    """Convenience path for audits/tests; training uses precomputed candidate scores."""
    return choose_adaptation_tile(
        score_tile_candidates(
            image,
            config,
            label_mask=label_mask,
            vegetation_mask=vegetation_mask,
        ),
        config,
        rng,
    )


def load_prepared_image(record: dict[str, str]) -> Image.Image:
    """Load the original validated image; no geometric preprocessing is applied."""
    path = Path(record["source_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Raw adaptation image is missing: {path}")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()
