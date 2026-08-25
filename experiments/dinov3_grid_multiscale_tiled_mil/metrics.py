"""Regression metrics and per-scale attention diagnostics."""

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
    coarse_feature_cache_paths: list[str]
    fine_feature_cache_paths: list[str]
    objective_mse: float
    coarse_weights: np.ndarray
    fine_weights: np.ndarray
    coarse_boxes: np.ndarray
    fine_boxes: np.ndarray

    def metrics(self) -> dict[str, float]:
        result = regression_metrics(self.targets, self.predictions, self.objective_mse)
        result["normalized_mse"] = self.objective_mse
        return result

    def attention_metrics(self) -> dict[str, dict[str, float | int]]:
        return {
            "coarse": scale_attention_metrics(self.coarse_weights),
            "fine": scale_attention_metrics(self.fine_weights),
        }


def predict(model, loader, device, scaler: TargetScaler) -> Predictions:
    model.eval()
    predicted, actual = [], []
    coarse_weights, fine_weights, coarse_boxes, fine_boxes = [], [], [], []
    filenames, source_paths, processed_paths = [], [], []
    coarse_cache_paths, fine_cache_paths = [], []
    squared_error, samples = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            coarse = batch["coarse_features"].to(device, non_blocking=True)
            fine = batch["fine_features"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True).reshape(-1)
            predictions, coarse_attention, fine_attention = model(
                coarse, fine, return_attention=True
            )
            predictions = predictions.reshape(-1)
            squared_error += torch.nn.functional.mse_loss(
                predictions.float(), targets, reduction="sum"
            ).item()
            samples += targets.numel()
            predicted.append(predictions.float().cpu())
            actual.append(targets.cpu())
            coarse_weights.append(coarse_attention.float().cpu())
            fine_weights.append(fine_attention.float().cpu())
            coarse_boxes.append(batch["coarse_tile_boxes"].cpu())
            fine_boxes.append(batch["fine_tile_boxes"].cpu())
            filenames.extend(map(str, batch["filename"]))
            source_paths.extend(map(str, batch["source_image_path"]))
            processed_paths.extend(map(str, batch["processed_image_path"]))
            coarse_cache_paths.extend(map(str, batch["coarse_feature_cache_path"]))
            fine_cache_paths.extend(map(str, batch["fine_feature_cache_path"]))
    if not samples:
        raise ValueError("Cannot evaluate an empty loader")
    coarse_attention = torch.cat(coarse_weights).numpy()
    fine_attention = torch.cat(fine_weights).numpy()
    for name, weights in (("coarse", coarse_attention), ("fine", fine_attention)):
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError(f"{name} tile attention weights do not sum to one")
    return Predictions(
        targets=scaler.inverse(torch.cat(actual).numpy()),
        predictions=scaler.inverse(torch.cat(predicted).numpy()),
        filenames=filenames,
        source_image_paths=source_paths,
        processed_image_paths=processed_paths,
        coarse_feature_cache_paths=coarse_cache_paths,
        fine_feature_cache_paths=fine_cache_paths,
        objective_mse=squared_error / samples,
        coarse_weights=coarse_attention,
        fine_weights=fine_attention,
        coarse_boxes=torch.cat(coarse_boxes).numpy(),
        fine_boxes=torch.cat(fine_boxes).numpy(),
    )


__all__ = [
    "Predictions",
    "attention_entropy",
    "mean_baseline",
    "predict",
    "regression_metrics",
    "scale_attention_metrics",
]
