"""Resumable per-image frozen DINOv3 feature cache."""

from __future__ import annotations

import os
from contextlib import nullcontext
from hashlib import sha1
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from experiments.dinov3_grid_lora_patch_attention.preprocessing import (
    CACHE_SCHEMA_VERSION as GRID_CACHE_SCHEMA_VERSION,
)

from .config import Config
from .tiling import global_and_tiled_views

FEATURE_CACHE_SCHEMA_VERSION = 1


def cache_identity(config: Config, filename: str, source: Path) -> str:
    stat = source.stat()
    values = (
        FEATURE_CACHE_SCHEMA_VERSION,
        GRID_CACHE_SCHEMA_VERSION,
        str(source.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        filename,
        config.data.grid_crop_size,
        config.data.grid_inner_margin_fraction,
        config.features.backbone,
        config.features.processor,
        config.features.representation,
        config.tiles.rows,
        config.tiles.columns,
        config.tiles.overlap_fraction,
        config.tiles.include_global_view,
    )
    return sha1(repr(values).encode("utf-8")).hexdigest()


def feature_cache_path(config: Config, filename: str, source: Path) -> Path:
    digest = cache_identity(config, filename, source)[:16]
    safe_stem = Path(filename).stem.replace("/", "_")
    return Path(config.features.cache_dir) / f"{safe_stem}_{digest}.npz"


def save_feature_record(
    destination: Path,
    *,
    features: np.ndarray,
    tile_boxes: np.ndarray,
    processed_image_path: str,
    identity: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(FEATURE_CACHE_SCHEMA_VERSION, dtype=np.int16),
        identity=np.asarray(identity),
        features=features,
        tile_boxes=np.asarray(tile_boxes, dtype=np.int32),
        processed_image_path=np.asarray(processed_image_path),
    )
    os.replace(temporary, destination)


def load_feature_record(path: Path, *, expected_identity: str | None = None) -> dict:
    try:
        with np.load(path, allow_pickle=False) as record:
            schema = int(record["schema_version"])
            identity = str(record["identity"])
            features = np.asarray(record["features"], dtype=np.float32)
            tile_boxes = np.asarray(record["tile_boxes"], dtype=np.int32)
            processed_image_path = str(record["processed_image_path"])
    except Exception as error:
        raise RuntimeError(f"Could not read feature cache {path}: {error}") from error
    if schema != FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Feature cache schema mismatch in {path}: {schema}")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(f"Stale feature cache identity in {path}")
    if features.ndim != 2 or tile_boxes.ndim != 2 or tile_boxes.shape[1] != 4:
        raise ValueError(f"Invalid feature cache shapes in {path}")
    if features.shape[0] != tile_boxes.shape[0] + 1:
        raise ValueError(f"Expected one global feature plus one feature per tile in {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite frozen features in {path}")
    return {
        "features": features,
        "tile_boxes": tile_boxes,
        "processed_image_path": processed_image_path,
        "identity": identity,
    }


class FrozenDinoExtractor:
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(config.features.processor)
        self.backbone = AutoModel.from_pretrained(config.features.backbone).to(device).eval()
        self.backbone.requires_grad_(False)

    def _autocast(self):
        mode = self.config.runtime.mixed_precision
        if self.device.type != "cuda" or mode == "none":
            return nullcontext()
        dtype = torch.float16 if mode == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def extract(self, views) -> np.ndarray:
        chunks = []
        batch_size = self.config.features.extraction_batch_size
        with torch.inference_mode():
            for start in range(0, len(views), batch_size):
                pixels = self.processor(
                    images=views[start : start + batch_size], return_tensors="pt"
                )["pixel_values"].to(self.device)
                with self._autocast():
                    tokens = self.backbone(pixel_values=pixels).last_hidden_state
                    registers = int(getattr(self.backbone.config, "num_register_tokens", 0))
                    cls_token = tokens[:, 0]
                    patch_mean = tokens[:, 1 + registers :].mean(dim=1)
                    representation = torch.cat([cls_token, patch_mean], dim=-1)
                chunks.append(representation.float().cpu())
        return torch.cat(chunks).numpy()

    def extract_image(self, image) -> tuple[np.ndarray, np.ndarray]:
        views, boxes = global_and_tiled_views(
            image,
            rows=self.config.tiles.rows,
            columns=self.config.tiles.columns,
            overlap_fraction=self.config.tiles.overlap_fraction,
        )
        features = self.extract(views)
        dtype = np.float16 if self.config.features.storage_dtype == "float16" else np.float32
        return features.astype(dtype), boxes
