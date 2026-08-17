"""Loss, inference, and metrics for the mean-score regression objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import TargetScaler


@dataclass(frozen=True)
class Predictions:
    targets: np.ndarray
    predictions: np.ndarray
    filenames: list[str]
    image_paths: list[str]
    normalized_mse: float

    def metrics(self) -> dict[str, float]:
        return regression_metrics(self.targets, self.predictions, self.normalized_mse)


def regression_metrics(targets, predictions, normalized_mse=None) -> dict[str, float]:
    targets = np.asarray(targets, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(targets) == 0 or targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must have equal, non-empty shapes")
    result = {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "r2": float(r2_score(targets, predictions)) if len(targets) > 1 else float("nan"),
    }
    if normalized_mse is not None:
        result["normalized_mse"] = float(normalized_mse)
    return result


def predict(model, loader, device, scaler: TargetScaler) -> Predictions:
    model.eval()
    predicted, actual, filenames, paths = [], [], [], []
    squared_error, sample_count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True).reshape(-1)
            predictions = model(pixels).reshape(-1)
            squared_error += torch.nn.functional.mse_loss(
                predictions, targets, reduction="sum"
            ).item()
            sample_count += targets.numel()
            predicted.append(predictions.cpu())
            actual.append(targets.cpu())
            filenames.extend(map(str, batch["filename"]))
            paths.extend(map(str, batch["image_path"]))
    if not sample_count:
        raise ValueError("Cannot evaluate an empty loader")
    return Predictions(
        targets=scaler.inverse(torch.cat(actual).numpy()),
        predictions=scaler.inverse(torch.cat(predicted).numpy()),
        filenames=filenames,
        image_paths=paths,
        normalized_mse=squared_error / sample_count,
    )


def mean_baseline(targets, training_mean: float) -> dict[str, float]:
    targets = np.asarray(targets)
    return regression_metrics(targets, np.full(targets.shape, training_mean))
