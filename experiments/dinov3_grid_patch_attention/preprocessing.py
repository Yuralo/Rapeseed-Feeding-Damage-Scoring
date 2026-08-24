"""Grid crop caching and structured failure logging for this experiment."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
import os
import traceback

from PIL import Image
from torch.utils.data import get_worker_info

from rapeseed_damage.artifacts import append_jsonl
from rapeseed_damage.grid import detect_grid, image_to_ndarray, to_rgb, warp_big_square


def load_grid_crop(path: str | Path, size: int = 1400) -> Image.Image:
    image_bgr = image_to_ndarray(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    grid_points, _, _ = detect_grid(image_bgr)
    cropped_bgr = warp_big_square(image_bgr, grid_points, size=size)
    return Image.fromarray(to_rgb(cropped_bgr))


def cache_path_for(source: str | Path, cache_dir: str | Path) -> Path:
    source = Path(source)
    digest = sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / f"{source.stem}_{digest}.jpg"


def load_or_create_grid_crop(
    source: str | Path,
    cache_dir: str | Path,
    *,
    size: int,
    overwrite: bool = False,
) -> tuple[Image.Image, Path, bool]:
    destination = cache_path_for(source, cache_dir).resolve()
    if destination.is_file() and not overwrite:
        with Image.open(destination) as cached:
            return cached.convert("RGB").copy(), destination, False
    image = load_grid_crop(source, size=size).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    image.save(temporary, format="JPEG", quality=95, subsampling=0)
    os.replace(temporary, destination)
    return image, destination, True


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
            "stage": "detect_warp_and_cache_grid",
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
