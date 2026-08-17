"""Public access to the grid/quadrat extraction code from the notebook.

The implementation remains in ``utils.image`` so existing notebook and viewer
imports continue to work during the migration.
"""

from utils.image import (  # noqa: F401
    CELL_SIZE,
    INNER_MARGIN_FRACTION,
    MAX_DETECTION_DIMENSION,
    detect_grid,
    image_to_ndarray,
    resize_for_detection,
    to_rgb,
    warp_big_square,
    warp_cell,
)

__all__ = [
    "CELL_SIZE",
    "INNER_MARGIN_FRACTION",
    "MAX_DETECTION_DIMENSION",
    "detect_grid",
    "image_to_ndarray",
    "resize_for_detection",
    "to_rgb",
    "warp_big_square",
    "warp_cell",
]

