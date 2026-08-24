"""Inference and metrics for the configured mean-score regression objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import Config
from .data import TargetScaler
from .runtime import autocast_context


@dataclass(frozen=True)
class Predictions:
    targets: np.ndarray
    predictions: np.ndarray
    filenames: list[str]
    source_image_paths: list[str]
    processed_image_paths: list[str]
    objective_mse: float
    targets_normalized: bool

    def metrics(self) -> dict[str, float]:
        result = regression_metrics(self.targets, self.predictions, self.objective_mse)
        if self.targets_normalized:
            result["normalized_mse"] = self.objective_mse
        return result


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
    predicted, actual = [], []
    filenames, source_paths, processed_paths = [], [], []
    squared_error, sample_count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True).reshape(-1)
            with autocast_context(config, device):
                predictions = model(pixels).reshape(-1)
                batch_error = torch.nn.functional.mse_loss(
                    predictions.float(), targets, reduction="sum"
                )
            squared_error += batch_error.item()
            sample_count += targets.numel()
            predicted.append(predictions.float().cpu())
            actual.append(targets.cpu())
            filenames.extend(map(str, batch["filename"]))
            source_paths.extend(map(str, batch["source_image_path"]))
            processed_paths.extend(map(str, batch["processed_image_path"]))
    if not sample_count:
        raise ValueError("Cannot evaluate an empty loader")
    return Predictions(
        targets=scaler.inverse(torch.cat(actual).numpy()),
        predictions=scaler.inverse(torch.cat(predicted).numpy()),
        filenames=filenames,
        source_image_paths=source_paths,
        processed_image_paths=processed_paths,
        objective_mse=squared_error / sample_count,
        targets_normalized=config.data.normalize_targets,
    )


def mean_baseline(targets, training_mean: float) -> dict[str, float]:
    targets = np.asarray(targets)
    return regression_metrics(targets, np.full(targets.shape, training_mean))
