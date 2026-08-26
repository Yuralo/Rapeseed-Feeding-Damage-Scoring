"""Regression metrics and regional/local attention diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.metrics import (
    attention_entropy,
    mean_baseline,
    regression_metrics,
)


def scale_attention_metrics(weights: np.ndarray) -> dict[str, float | int]:
    entropy = attention_entropy(weights)
    top = weights.max(axis=1)
    return {
        "mean_normalized_entropy": float(entropy.mean()),
        "minimum_normalized_entropy": float(entropy.min()),
        "maximum_normalized_entropy": float(entropy.max()),
        "mean_top_tile_mass": float(top.mean()),
        "maximum_top_tile_mass": float(top.max()),
        "tiles_per_image": int(weights.shape[1]),
    }


@dataclass(frozen=True)
class Predictions:
    targets: np.ndarray
    predictions: np.ndarray
    filenames: list[str]
    source_image_paths: list[str]
    processed_image_paths: list[str]
    context_feature_cache_paths: list[str]
    regional_feature_cache_paths: list[str]
    local_feature_cache_paths: list[str]
    objective_mse: float
    regional_weights: np.ndarray
    local_weights: np.ndarray
    context_boxes: np.ndarray
    regional_boxes: np.ndarray
    local_boxes: np.ndarray

    def metrics(self) -> dict[str, float]:
        result = regression_metrics(self.targets, self.predictions, self.objective_mse)
        result["normalized_mse"] = self.objective_mse
        return result

    def attention_metrics(self) -> dict[str, dict[str, float | int]]:
        return {
            "regional_4x4": scale_attention_metrics(self.regional_weights),
            "local_5x5": scale_attention_metrics(self.local_weights),
        }


def predict(model, loader, device, scaler: TargetScaler) -> Predictions:
    model.eval()
    predicted, actual = [], []
    regional_weights, local_weights = [], []
    context_boxes, regional_boxes, local_boxes = [], [], []
    filenames, source_paths, processed_paths = [], [], []
    context_cache_paths, regional_cache_paths, local_cache_paths = [], [], []
    squared_error, samples = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            context = batch["context_features"].to(device, non_blocking=True)
            regional = batch["regional_features"].to(device, non_blocking=True)
            local = batch["local_features"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True).reshape(-1)
            predictions, regional_attention, local_attention = model(
                context, regional, local, return_attention=True
            )
            predictions = predictions.reshape(-1)
            squared_error += torch.nn.functional.mse_loss(
                predictions.float(), targets, reduction="sum"
            ).item()
            samples += targets.numel()
            predicted.append(predictions.float().cpu())
            actual.append(targets.cpu())
            regional_weights.append(regional_attention.float().cpu())
            local_weights.append(local_attention.float().cpu())
            context_boxes.append(batch["context_tile_boxes"].cpu())
            regional_boxes.append(batch["regional_tile_boxes"].cpu())
            local_boxes.append(batch["local_tile_boxes"].cpu())
            filenames.extend(map(str, batch["filename"]))
            source_paths.extend(map(str, batch["source_image_path"]))
            processed_paths.extend(map(str, batch["processed_image_path"]))
            context_cache_paths.extend(map(str, batch["context_feature_cache_path"]))
            regional_cache_paths.extend(map(str, batch["regional_feature_cache_path"]))
            local_cache_paths.extend(map(str, batch["local_feature_cache_path"]))
    if not samples:
        raise ValueError("Cannot evaluate an empty loader")
    regional_attention = torch.cat(regional_weights).numpy()
    local_attention = torch.cat(local_weights).numpy()
    for name, weights in (("regional", regional_attention), ("local", local_attention)):
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError(f"{name} tile attention weights do not sum to one")
    return Predictions(
        targets=scaler.inverse(torch.cat(actual).numpy()),
        predictions=scaler.inverse(torch.cat(predicted).numpy()),
        filenames=filenames,
        source_image_paths=source_paths,
        processed_image_paths=processed_paths,
        context_feature_cache_paths=context_cache_paths,
        regional_feature_cache_paths=regional_cache_paths,
        local_feature_cache_paths=local_cache_paths,
        objective_mse=squared_error / samples,
        regional_weights=regional_attention,
        local_weights=local_attention,
        context_boxes=torch.cat(context_boxes).numpy(),
        regional_boxes=torch.cat(regional_boxes).numpy(),
        local_boxes=torch.cat(local_boxes).numpy(),
    )


__all__ = [
    "Predictions",
    "attention_entropy",
    "mean_baseline",
    "predict",
    "regression_metrics",
    "scale_attention_metrics",
]
