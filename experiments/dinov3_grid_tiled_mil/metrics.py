"""Regression metrics and tile-attention diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import TargetScaler


def attention_entropy(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    clipped = np.clip(weights, 1e-12, 1.0)
    denominator = np.log(weights.shape[1]) if weights.shape[1] > 1 else 1.0
    return -(clipped * np.log(clipped)).sum(axis=1) / denominator


def regression_metrics(targets, predictions, objective_mse=None) -> dict[str, float]:
    targets = np.asarray(targets, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    result = {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "r2": float(r2_score(targets, predictions)) if len(targets) > 1 else float("nan"),
    }
    if objective_mse is not None:
        result["objective_mse"] = float(objective_mse)
    return result


@dataclass(frozen=True)
class Predictions:
    targets: np.ndarray
    predictions: np.ndarray
    filenames: list[str]
    source_image_paths: list[str]
    processed_image_paths: list[str]
    feature_cache_paths: list[str]
    objective_mse: float
    tile_weights: np.ndarray
    tile_boxes: np.ndarray

    def metrics(self) -> dict[str, float]:
        result = regression_metrics(self.targets, self.predictions, self.objective_mse)
        result["normalized_mse"] = self.objective_mse
        return result

    def attention_metrics(self) -> dict[str, float | int]:
        entropy = attention_entropy(self.tile_weights)
        top = self.tile_weights.max(axis=1)
        return {
            "mean_normalized_entropy": float(entropy.mean()),
            "minimum_normalized_entropy": float(entropy.min()),
            "maximum_normalized_entropy": float(entropy.max()),
            "mean_top_tile_mass": float(top.mean()),
            "maximum_top_tile_mass": float(top.max()),
            "tiles_per_image": int(self.tile_weights.shape[1]),
        }


def predict(model, loader, device, scaler: TargetScaler) -> Predictions:
    model.eval()
    predicted, actual, all_weights, all_boxes = [], [], [], []
    filenames, source_paths, processed_paths, cache_paths = [], [], [], []
    squared_error, samples = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True).reshape(-1)
            predictions, weights = model(features, return_attention=True)
            predictions = predictions.reshape(-1)
            squared_error += torch.nn.functional.mse_loss(
                predictions.float(), targets, reduction="sum"
            ).item()
            samples += targets.numel()
            predicted.append(predictions.float().cpu())
            actual.append(targets.cpu())
            all_weights.append(weights.float().cpu())
            all_boxes.append(batch["tile_boxes"].cpu())
            filenames.extend(map(str, batch["filename"]))
            source_paths.extend(map(str, batch["source_image_path"]))
            processed_paths.extend(map(str, batch["processed_image_path"]))
            cache_paths.extend(map(str, batch["feature_cache_path"]))
    if not samples:
        raise ValueError("Cannot evaluate an empty loader")
    weights = torch.cat(all_weights).numpy()
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Tile attention weights do not sum to one")
    return Predictions(
        targets=scaler.inverse(torch.cat(actual).numpy()),
        predictions=scaler.inverse(torch.cat(predicted).numpy()),
        filenames=filenames,
        source_image_paths=source_paths,
        processed_image_paths=processed_paths,
        feature_cache_paths=cache_paths,
        objective_mse=squared_error / samples,
        tile_weights=weights,
        tile_boxes=torch.cat(all_boxes).numpy(),
    )


def mean_baseline(targets, training_mean: float) -> dict[str, float]:
    targets = np.asarray(targets)
    return regression_metrics(targets, np.full(targets.shape, training_mean))
