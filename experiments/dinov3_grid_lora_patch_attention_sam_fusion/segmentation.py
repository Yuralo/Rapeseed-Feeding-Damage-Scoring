"""Versioned SAM mask caching, validation, and failure reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image

from rapeseed_damage.artifacts import append_jsonl, write_json
from .config import Config
from .preprocessing import CACHE_SCHEMA_VERSION as GRID_CACHE_SCHEMA_VERSION

SAM_CACHE_SCHEMA_VERSION = 1


def _cache_signature(source: str | Path, config: Config) -> str:
    settings = config.segmentation
    components = (
        f"sam_schema={SAM_CACHE_SCHEMA_VERSION}",
        f"grid_schema={GRID_CACHE_SCHEMA_VERSION}",
        f"source={Path(source).resolve()}",
        f"grid_size={config.data.grid_crop_size}",
        f"grid_margin={config.data.grid_inner_margin_fraction:.8f}",
        f"model={settings.model_name}",
        "prompts=" + "|".join(map(str, settings.prompts)),
        f"score={settings.score_threshold:.8f}",
        f"mask={settings.mask_threshold:.8f}",
        f"minimum_foreground={settings.minimum_foreground_fraction:.8f}",
        f"maximum_foreground={settings.maximum_foreground_fraction:.8f}",
    )
    return sha1("\n".join(components).encode("utf-8")).hexdigest()


def mask_cache_paths(source: str | Path, config: Config) -> tuple[Path, Path, str]:
    source = Path(source)
    signature = _cache_signature(source, config)
    stem = f"{source.stem}_samv{SAM_CACHE_SCHEMA_VERSION}_{signature[:12]}"
    directory = Path(config.segmentation.mask_cache_dir).resolve()
    return directory / f"{stem}.png", directory / f"{stem}.json", signature


def mask_statistics(mask: np.ndarray) -> dict[str, object]:
    boolean = np.asarray(mask, dtype=bool)
    if boolean.ndim != 2:
        raise ValueError(f"SAM mask must be two-dimensional, got shape {boolean.shape}")
    height, width = boolean.shape
    pixels = int(boolean.sum())
    fraction = float(pixels / boolean.size) if boolean.size else 0.0
    if pixels:
        rows, columns = np.nonzero(boolean)
        bounding_box = [
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ]
    else:
        bounding_box = None
    edge_pixels = 0
    if height and width:
        edge_pixels = int(
            boolean[0].sum()
            + boolean[-1].sum()
            + boolean[1:-1, 0].sum()
            + boolean[1:-1, -1].sum()
        )
    return {
        "height": height,
        "width": width,
        "foreground_pixels": pixels,
        "foreground_fraction": fraction,
        "bounding_box_xyxy": bounding_box,
        "foreground_edge_pixels": edge_pixels,
    }


def validate_mask(mask: np.ndarray, config: Config) -> dict[str, object]:
    statistics = mask_statistics(mask)
    fraction = float(statistics["foreground_fraction"])
    minimum = config.segmentation.minimum_foreground_fraction
    maximum = config.segmentation.maximum_foreground_fraction
    reasons = []
    if fraction < minimum:
        reasons.append(f"foreground fraction {fraction:.6f} is below {minimum:.6f}")
    if fraction > maximum:
        reasons.append(f"foreground fraction {fraction:.6f} is above {maximum:.6f}")
    return {**statistics, "valid": not reasons, "quality_reasons": reasons}


def save_mask_cache(
    mask: np.ndarray,
    *,
    source: str | Path,
    grid_crop_path: str | Path,
    config: Config,
) -> tuple[Path, Path, dict[str, object]]:
    mask_path, metadata_path, signature = mask_cache_paths(source, config)
    boolean = np.asarray(mask, dtype=bool)
    quality = validate_mask(boolean, config)
    expected_shape = (config.data.grid_crop_size, config.data.grid_crop_size)
    if boolean.shape != expected_shape:
        raise ValueError(
            f"SAM mask shape {boolean.shape} does not match grid crop {expected_shape}"
        )
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mask_path.with_name(f".{mask_path.name}.{os.getpid()}.tmp")
    Image.fromarray(boolean.astype(np.uint8) * 255, mode="L").save(
        temporary,
        format="PNG",
        optimize=True,
    )
    os.replace(temporary, mask_path)
    metadata = {
        "sam_cache_schema_version": SAM_CACHE_SCHEMA_VERSION,
        "grid_cache_schema_version": GRID_CACHE_SCHEMA_VERSION,
        "signature": signature,
        "source_image_path": str(Path(source).resolve()),
        "grid_crop_path": str(Path(grid_crop_path).resolve()),
        "mask_path": str(mask_path),
        "model_name": config.segmentation.model_name,
        "prompts": list(config.segmentation.prompts),
        "score_threshold": config.segmentation.score_threshold,
        "mask_threshold": config.segmentation.mask_threshold,
        "minimum_foreground_fraction": (
            config.segmentation.minimum_foreground_fraction
        ),
        "maximum_foreground_fraction": (
            config.segmentation.maximum_foreground_fraction
        ),
        "quality": quality,
    }
    write_json(metadata_path, metadata)
    return mask_path, metadata_path, metadata


def load_cached_mask(
    source: str | Path,
    config: Config,
    *,
    require_valid: bool = True,
) -> tuple[Image.Image, Path, dict[str, object]]:
    mask_path, metadata_path, signature = mask_cache_paths(source, config)
    if not mask_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"SAM cache is missing for {Path(source).name}. Run prepare_sam_cache first. "
            f"Expected {mask_path} and {metadata_path}."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("signature") != signature:
        raise ValueError(f"Stale SAM metadata signature for {Path(source).name}")
    if metadata.get("sam_cache_schema_version") != SAM_CACHE_SCHEMA_VERSION:
        raise ValueError(f"SAM cache schema mismatch for {Path(source).name}")
    if require_valid and not metadata.get("quality", {}).get("valid", False):
        reasons = metadata.get("quality", {}).get("quality_reasons", [])
        raise ValueError(
            f"SAM mask failed quality checks for {Path(source).name}: " + "; ".join(reasons)
        )
    with Image.open(mask_path) as stored:
        mask = stored.convert("L").copy()
    expected_size = (config.data.grid_crop_size, config.data.grid_crop_size)
    if mask.size != expected_size:
        mask.close()
        raise ValueError(
            f"Cached SAM mask for {Path(source).name} has size {mask.size}, "
            f"expected {expected_size}"
        )
    return mask, mask_path, metadata


def make_masked_image(
    image: Image.Image,
    mask: Image.Image,
    *,
    background_value: int,
) -> Image.Image:
    rgb = image.convert("RGB")
    binary = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
    background = Image.new("RGB", rgb.size, color=(background_value,) * 3)
    return Image.composite(rgb, background, binary)


def create_segmenter(config: Config, device):
    from rapeseed_damage.segmentation import SamPlantSegmenter

    return SamPlantSegmenter(config.segmentation.model_name, device=device)


def generate_mask(segmenter, image: Image.Image, config: Config) -> np.ndarray:
    mask, _ = segmenter(
        image,
        prompts=config.segmentation.prompts,
        score_threshold=config.segmentation.score_threshold,
        mask_threshold=config.segmentation.mask_threshold,
    )
    return np.asarray(mask, dtype=bool)


def log_sam_failure(
    log_path: str | Path,
    *,
    error: Exception,
    image_path: str | Path,
    filename: str,
    dataset_index: int,
) -> None:
    append_jsonl(
        log_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "load_grid_segment_validate_and_cache_sam",
            "filename": filename,
            "image_path": str(image_path),
            "dataset_index": dataset_index,
            "process_id": os.getpid(),
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
    )
