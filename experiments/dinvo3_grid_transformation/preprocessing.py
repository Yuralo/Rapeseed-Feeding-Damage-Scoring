from pathlib import Path
from datetime import datetime, timezone
import os
import traceback

from PIL import Image
from torch.utils.data import get_worker_info

from rapeseed_damage.artifacts import append_jsonl

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


def log_grid_failure(
    log_path: str | Path,
    *,
    error: Exception,
    image_path: str | Path,
    filename: str,
    dataset_index: int,
) -> None:
    worker = get_worker_info()
    append_jsonl(
        log_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "detect_and_warp_grid",
            "filename": filename,
            "image_path": str(image_path),
            "dataset_index": dataset_index,
            "worker_id": worker.id if worker is not None else None,
            "process_id": os.getpid(),
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
    )
