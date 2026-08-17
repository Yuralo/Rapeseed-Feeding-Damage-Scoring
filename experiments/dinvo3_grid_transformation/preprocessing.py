from pathlib import Path

from PIL import Image

from rapeseed_damage.grid import (
    detect_grid,
    image_to_ndarray,
    to_rgb,
    warp_big_square,
)


def load_grid_crop(path: str | Path, size: int = 1400) -> Image.Image:
    image_bgr = image_to_ndarray(str(path))

    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    grid_points, _, _ = detect_grid(image_bgr)

    cropped_bgr = warp_big_square(
        image_bgr,
        grid_points,
        size=size,
    )

    cropped_rgb = to_rgb(cropped_bgr)
    return Image.fromarray(cropped_rgb)