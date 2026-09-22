"""Measured soil-revealing holes and edge bites with a compositing sham."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BitePair:
    bite: Image.Image
    sham: Image.Image
    removed_mask: np.ndarray
    sham_mask: np.ndarray
    realized_fraction: float
    kind: str


def _irregular_disk(height: int, width: int, cx: int, cy: int, radius: float,
                    radii: np.ndarray, angle: float) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, len(radii), endpoint=False) + angle
    vertices = np.stack((cx + radius * radii * np.cos(theta),
                         cy + radius * radii * np.sin(theta)), axis=1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [vertices.round().astype(np.int32)], 1)
    return mask.astype(bool)


def _sample_cut(leaf: np.ndarray, target: float, rng: np.random.Generator,
                minimum_removed: int) -> tuple[np.ndarray, str]:
    if leaf.ndim != 2 or leaf.sum() < minimum_removed * 3:
        raise ValueError("Too little leaf foreground for a measured bite")
    height, width = leaf.shape
    eroded = cv2.erode(leaf.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    kind = "interior" if rng.random() < 0.5 else "edge"
    candidate = eroded if kind == "interior" else leaf & ~eroded
    if candidate.sum() < 3:
        candidate = leaf
    points = np.column_stack(np.nonzero(candidate))
    desired = max(minimum_removed, round(float(leaf.sum()) * target))
    best, best_error = None, float("inf")
    for _ in range(24):
        y, x = map(int, points[int(rng.integers(len(points)))])
        radii = rng.uniform(0.7, 1.3, size=13)
        angle = float(rng.uniform(0, 2 * np.pi))
        lo, hi = 1.0, float(max(height, width))
        for _ in range(13):
            radius = (lo + hi) / 2
            cut = _irregular_disk(height, width, x, y, radius, radii, angle) & leaf
            count = int(cut.sum())
            if abs(count - desired) < best_error:
                best, best_error = cut, abs(count - desired)
            if count < desired:
                lo = radius
            else:
                hi = radius
        if best_error <= max(3, round(desired * 0.15)):
            break
    if best is None or best.sum() < minimum_removed or best_error > max(5, desired * 0.35):
        raise ValueError("Could not realize the requested leaf loss")
    return best, kind


def _soil_donor(image: np.ndarray, mask: np.ndarray, shape: tuple[int, int],
                rng: np.random.Generator) -> np.ndarray:
    height, width = shape
    full_height, full_width = mask.shape
    if height > full_height or width > full_width:
        raise ValueError("Donor region larger than image")
    # Source and target use the same camera, soil, lighting, and image compression.
    for _ in range(150):
        y = int(rng.integers(0, full_height - height + 1))
        x = int(rng.integers(0, full_width - width + 1))
        candidate = mask[y:y + height, x:x + width]
        if candidate.mean() < 0.02:
            return image[y:y + height, x:x + width].copy()
    raise ValueError("No sufficiently clean same-image soil donor")


def make_pair(image: Image.Image, plant_mask: np.ndarray, patch_box: tuple[int, int, int, int],
              requested_fraction: float, rng: np.random.Generator,
              minimum_removed_pixels: int = 8) -> BitePair:
    """Return a bite and sham crop; realized_fraction is relative to original leaf pixels."""
    rgb = np.asarray(image.convert("RGB"))
    mask = np.asarray(plant_mask, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("Plant mask must match the EXIF-oriented processed image")
    x0, y0, x1, y1 = patch_box
    if x0 < 0 or y0 < 0 or x1 > rgb.shape[1] or y1 > rgb.shape[0] or x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid patch box")
    crop = rgb[y0:y1, x0:x1]
    leaf = mask[y0:y1, x0:x1]
    cut, kind = _sample_cut(leaf, requested_fraction, rng, minimum_removed_pixels)
    rows, cols = np.nonzero(cut)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    donor = _soil_donor(rgb, mask, (bottom - top, right - left), rng)
    bite = crop.copy()
    piece = bite[top:bottom, left:right]
    piece[cut[top:bottom, left:right]] = donor[cut[top:bottom, left:right]]

    # Reuse the same cut shape and donor on soil only. The sham tests whether a model
    # exploits paste seams instead of lost leaf tissue.
    sham = crop.copy()
    sham_mask = np.zeros_like(leaf)
    soil = ~leaf
    ys, xs = np.nonzero(soil)
    if not len(ys):
        raise ValueError("No soil foreground for a sham edit")
    for _ in range(100):
        index = int(rng.integers(len(ys)))
        dy, dx = int(ys[index] - rows.mean()), int(xs[index] - cols.mean())
        shifted = np.zeros_like(cut)
        src_y0, src_y1 = max(0, -dy), min(cut.shape[0], cut.shape[0] - dy)
        src_x0, src_x1 = max(0, -dx), min(cut.shape[1], cut.shape[1] - dx)
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            continue
        shifted[src_y0 + dy:src_y1 + dy, src_x0 + dx:src_x1 + dx] = (
            cut[src_y0:src_y1, src_x0:src_x1]
        )
        shifted &= soil
        if shifted.sum() >= max(3, round(cut.sum() * 0.8)):
            sham_mask = shifted
            break
    if not sham_mask.any():
        raise ValueError("Could not place a comparable soil-only sham")
    sy, sx = np.nonzero(sham_mask)
    st, sb, sl, sr = int(sy.min()), int(sy.max()) + 1, int(sx.min()), int(sx.max()) + 1
    sham_donor = _soil_donor(rgb, mask, (sb - st, sr - sl), rng)
    sham_piece = sham[st:sb, sl:sr]
    sham_piece[sham_mask[st:sb, sl:sr]] = sham_donor[sham_mask[st:sb, sl:sr]]
    return BitePair(Image.fromarray(bite), Image.fromarray(sham), cut, sham_mask,
                    float(cut.sum() / leaf.sum()), kind)
