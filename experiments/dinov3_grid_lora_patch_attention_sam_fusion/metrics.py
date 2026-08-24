"""Regression, dual-attention, mask, and fusion diagnostics."""

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


def _attention_summary(weights: np.ndarray) -> dict[str, float]:
    entropies = attention_entropy(weights)
    top_mass = top_fraction_mass(weights)
    return {
        "mean_normalized_entropy": float(entropies.mean()),
        "minimum_normalized_entropy": float(entropies.min()),
        "maximum_normalized_entropy": float(entropies.max()),
        "mean_top_10_percent_mass": float(top_mass.mean()),
        "patches_per_image": int(weights.shape[1]),
    }


@dataclass(frozen=True)
class Predictions:
    targets: np.ndarray
    predictions: np.ndarray
    base_predictions: np.ndarray
    fusion_deltas: np.ndarray
    filenames: list[str]
    source_image_paths: list[str]
    processed_image_paths: list[str]
    mask_paths: list[str]
    mask_foreground_fractions: np.ndarray
    objective_mse: float
    original_attention_weights: np.ndarray
    masked_attention_weights: np.ndarray
    attention_grid_rows: np.ndarray
    attention_grid_columns: np.ndarray
    fusion_weights: np.ndarray

    def metrics(self) -> dict[str, float]:
        result = regression_metrics(self.targets, self.predictions, self.objective_mse)
        result["normalized_mse"] = self.objective_mse
        return result

    def diagnostics(self) -> dict[str, object]:
        return {
            "original_attention": _attention_summary(self.original_attention_weights),
            "masked_attention": _attention_summary(self.masked_attention_weights),
            "fusion": {
                "mean_original_weight": float(self.fusion_weights[:, 0].mean()),
                "mean_masked_weight": float(self.fusion_weights[:, 1].mean()),
                "mean_binary_mask_weight": float(self.fusion_weights[:, 2].mean()),
                "mean_absolute_delta": float(np.abs(self.fusion_deltas).mean()),
                "maximum_absolute_delta": float(np.abs(self.fusion_deltas).max()),
            },
            "masks": {
                "mean_foreground_fraction": float(
                    self.mask_foreground_fractions.mean()
                ),
                "minimum_foreground_fraction": float(
                    self.mask_foreground_fractions.min()
                ),
                "maximum_foreground_fraction": float(
                    self.mask_foreground_fractions.max()
                ),
            },
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
    predicted, base_predicted, fusion_deltas, actual = [], [], [], []
    original_attention, masked_attention, fusion_weights = [], [], []
    foreground_fractions = []
    filenames, source_paths, processed_paths, mask_paths = [], [], [], []
    grid_rows, grid_columns = [], []
    squared_error, sample_count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            original_pixels = batch["original_pixel_values"].to(
                device, non_blocking=True
            )
            masked_pixels = batch["masked_pixel_values"].to(device, non_blocking=True)
            masks = batch["mask_values"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True).reshape(-1)
            with autocast_context(config, device):
                predictions, diagnostics = model(
                    original_pixels,
                    masked_pixels,
                    masks,
                    return_diagnostics=True,
                )
                predictions = predictions.reshape(-1)
                batch_error = torch.nn.functional.mse_loss(
                    predictions.float(), targets, reduction="sum"
                )
            attention = diagnostics["original_attention"]
            rows, columns = model.attention_grid(original_pixels, attention.shape[1])
            squared_error += batch_error.item()
            sample_count += targets.numel()
            predicted.append(predictions.float().cpu())
            base_predicted.append(diagnostics["base_predictions"].float().cpu())
            fusion_deltas.append(diagnostics["fusion_delta"].float().cpu())
            actual.append(targets.cpu())
            original_attention.append(attention.float().cpu())
            masked_attention.append(diagnostics["masked_attention"].float().cpu())
            fusion_weights.append(diagnostics["fusion_weights"].float().cpu())
            foreground_fractions.append(batch["mask_foreground_fraction"].float())
            filenames.extend(map(str, batch["filename"]))
            source_paths.extend(map(str, batch["source_image_path"]))
            processed_paths.extend(map(str, batch["processed_image_path"]))
            mask_paths.extend(map(str, batch["mask_path"]))
            grid_rows.extend([rows] * targets.numel())
            grid_columns.extend([columns] * targets.numel())
    if not sample_count:
        raise ValueError("Cannot evaluate an empty loader")
    original_weights = torch.cat(original_attention).numpy()
    masked_weights = torch.cat(masked_attention).numpy()
    for name, weights in (
        ("original", original_weights),
        ("masked", masked_weights),
    ):
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-4):
            raise RuntimeError(f"{name} patch-attention weights do not sum to one")
    gates = torch.cat(fusion_weights).numpy()
    if not np.allclose(gates.sum(axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Fusion weights do not sum to one")
    normalized_delta = torch.cat(fusion_deltas).numpy()
    return Predictions(
        targets=scaler.inverse(torch.cat(actual).numpy()),
        predictions=scaler.inverse(torch.cat(predicted).numpy()),
        base_predictions=scaler.inverse(torch.cat(base_predicted).numpy()),
        fusion_deltas=normalized_delta * scaler.std,
        filenames=filenames,
        source_image_paths=source_paths,
        processed_image_paths=processed_paths,
        mask_paths=mask_paths,
        mask_foreground_fractions=torch.cat(foreground_fractions).numpy(),
        objective_mse=squared_error / sample_count,
        original_attention_weights=original_weights,
        masked_attention_weights=masked_weights,
        attention_grid_rows=np.asarray(grid_rows, dtype=np.int16),
        attention_grid_columns=np.asarray(grid_columns, dtype=np.int16),
        fusion_weights=gates,
    )


def mean_baseline(targets, training_mean: float) -> dict[str, float]:
    targets = np.asarray(targets)
    return regression_metrics(targets, np.full(targets.shape, training_mean))
