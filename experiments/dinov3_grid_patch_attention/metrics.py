"""Regression metrics plus patch-attention diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import Config
from .data import TargetScaler
from .runtime import autocast_context


def attention_entropy(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    clipped = np.clip(weights, 1e-12, 1.0)
    denominator = np.log(weights.shape[1]) if weights.shape[1] > 1 else 1.0
    return -(clipped * np.log(clipped)).sum(axis=1) / denominator


def top_fraction_mass(weights: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    count = max(1, int(np.ceil(weights.shape[1] * fraction)))
    return np.sort(weights, axis=1)[:, -count:].sum(axis=1)


@dataclass(frozen=True)
class Predictions:
    targets: np.ndarray
    predictions: np.ndarray
    filenames: list[str]
    source_image_paths: list[str]
    processed_image_paths: list[str]
    objective_mse: float
    attention_weights: np.ndarray
    attention_grid_rows: np.ndarray
    attention_grid_columns: np.ndarray

    def metrics(self) -> dict[str, float]:
        result = regression_metrics(self.targets, self.predictions, self.objective_mse)
        result["normalized_mse"] = self.objective_mse
        return result

    def attention_metrics(self) -> dict[str, float]:
        entropies = attention_entropy(self.attention_weights)
        top_mass = top_fraction_mass(self.attention_weights)
        return {
            "mean_normalized_entropy": float(entropies.mean()),
            "minimum_normalized_entropy": float(entropies.min()),
            "maximum_normalized_entropy": float(entropies.max()),
            "mean_top_10_percent_mass": float(top_mass.mean()),
            "patches_per_image": int(self.attention_weights.shape[1]),
        }


def regression_metrics(targets, predictions, objective_mse=None) -> dict[str, float]:
    targets = np.asarray(targets, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(targets) == 0 or targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must have equal, non-empty shapes")
    result = {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "r2": float(r2_score(targets, predictions)) if len(targets) > 1 else float("nan"),
    }
    if objective_mse is not None:
        result["objective_mse"] = float(objective_mse)
    return result


def predict(model, loader, device, scaler: TargetScaler, config: Config) -> Predictions:
    model.eval()
    predicted, actual, all_attention = [], [], []
    filenames, source_paths, processed_paths = [], [], []
    grid_rows, grid_columns = [], []
    squared_error, sample_count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True).reshape(-1)
            with autocast_context(config, device):
                predictions, attention = model(pixels, return_attention=True)
                predictions = predictions.reshape(-1)
                batch_error = torch.nn.functional.mse_loss(
                    predictions.float(), targets, reduction="sum"
                )
            rows, columns = model.attention_grid(pixels, attention.shape[1])
            squared_error += batch_error.item()
            sample_count += targets.numel()
            predicted.append(predictions.float().cpu())
            actual.append(targets.cpu())
            all_attention.append(attention.float().cpu())
            filenames.extend(map(str, batch["filename"]))
            source_paths.extend(map(str, batch["source_image_path"]))
            processed_paths.extend(map(str, batch["processed_image_path"]))
            grid_rows.extend([rows] * targets.numel())
            grid_columns.extend([columns] * targets.numel())
    if not sample_count:
        raise ValueError("Cannot evaluate an empty loader")
    weights = torch.cat(all_attention).numpy()
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Patch attention weights do not sum to one")
    return Predictions(
        targets=scaler.inverse(torch.cat(actual).numpy()),
        predictions=scaler.inverse(torch.cat(predicted).numpy()),
        filenames=filenames,
        source_image_paths=source_paths,
        processed_image_paths=processed_paths,
        objective_mse=squared_error / sample_count,
        attention_weights=weights,
        attention_grid_rows=np.asarray(grid_rows, dtype=np.int16),
        attention_grid_columns=np.asarray(grid_columns, dtype=np.int16),
    )


def mean_baseline(targets, training_mean: float) -> dict[str, float]:
    targets = np.asarray(targets)
    return regression_metrics(targets, np.full(targets.shape, training_mean))
