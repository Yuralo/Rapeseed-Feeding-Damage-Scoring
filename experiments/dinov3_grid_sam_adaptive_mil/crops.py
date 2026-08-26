"""Deterministic SAM-foreground grouping and plant-centred crop generation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .config import AdaptiveCropSettings


@dataclass(frozen=True)
class AdaptiveCropLayout:
    boxes: np.ndarray
    foreground_pixels: np.ndarray
    mask_coverage: float
    component_count_before_merge: int


def _component_boxes(mask: np.ndarray, dilation_px: int) -> list[list[int]]:
    binary = np.asarray(mask, dtype=np.uint8)
    if dilation_px:
        size = 2 * dilation_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        grouped = cv2.dilate(binary, kernel)
    else:
        grouped = binary
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    boxes = []
    for label in range(1, count):
        x, y, width, height, _ = statistics[label]
        original_pixels = int(binary[labels == label].sum())
        if original_pixels:
            boxes.append([int(x), int(y), int(x + width), int(y + height)])
    return boxes


def _box_distance(left: list[int], right: list[int]) -> float:
    left_x = 0.5 * (left[0] + left[2])
    left_y = 0.5 * (left[1] + left[3])
    right_x = 0.5 * (right[0] + right[2])
    right_y = 0.5 * (right[1] + right[3])
    return (left_x - right_x) ** 2 + (left_y - right_y) ** 2


def _merge_to_limit(boxes: list[list[int]], limit: int) -> list[list[int]]:
    boxes = [list(box) for box in boxes]
    while len(boxes) > limit:
        pair = min(
            ((i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))),
            key=lambda indices: _box_distance(boxes[indices[0]], boxes[indices[1]]),
        )
        first, second = pair
        merged = [
            min(boxes[first][0], boxes[second][0]),
            min(boxes[first][1], boxes[second][1]),
            max(boxes[first][2], boxes[second][2]),
            max(boxes[first][3], boxes[second][3]),
        ]
        boxes[first] = merged
        boxes.pop(second)
    return boxes


def _square_context_box(
    box: list[int], width: int, height: int, settings: AdaptiveCropSettings
) -> list[int]:
    x0, y0, x1, y1 = box
    span = max(x1 - x0, y1 - y0)
    desired = max(settings.minimum_crop_size, round(span * settings.context_scale))
    desired = min(desired, settings.maximum_crop_size)
    desired = min(max(desired, span), width, height)
    center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    start_x = round(center_x - desired / 2)
    start_y = round(center_y - desired / 2)
    start_x = min(max(0, start_x), width - desired)
    start_y = min(max(0, start_y), height - desired)
    return [start_x, start_y, start_x + desired, start_y + desired]


def make_adaptive_crop_layout(
    mask: np.ndarray, settings: AdaptiveCropSettings
) -> AdaptiveCropLayout:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("Adaptive crops require a nonempty two-dimensional SAM mask")
    height, width = binary.shape
    groups = _component_boxes(binary, settings.grouping_dilation_px)
    if not groups:
        rows, columns = np.nonzero(binary)
        groups = [[columns.min(), rows.min(), columns.max() + 1, rows.max() + 1]]
    before_merge = len(groups)
    groups = _merge_to_limit(groups, settings.maximum_instances)
    boxes = np.asarray(
        [_square_context_box(group, width, height, settings) for group in groups],
        dtype=np.int32,
    )
    covered = np.zeros_like(binary)
    foreground_pixels = []
    for x0, y0, x1, y1 in boxes:
        pixels = int(binary[y0:y1, x0:x1].sum())
        foreground_pixels.append(pixels)
        covered[y0:y1, x0:x1] = True
    coverage = float(binary[covered].sum() / binary.sum())
    if coverage < settings.minimum_mask_coverage:
        raise ValueError(
            f"Adaptive crops cover {coverage:.4f} of SAM foreground; "
            f"required {settings.minimum_mask_coverage:.4f}"
        )
    order = np.argsort(np.asarray(foreground_pixels))[::-1]
    return AdaptiveCropLayout(
        boxes=boxes[order],
        foreground_pixels=np.asarray(foreground_pixels, dtype=np.int32)[order],
        mask_coverage=coverage,
        component_count_before_merge=before_merge,
    )


def crop_instances(image: Image.Image, boxes: np.ndarray) -> list[Image.Image]:
    rgb = image.convert("RGB")
    return [rgb.crop(tuple(map(int, box))) for box in boxes]
