"""CSV and plot artifacts meaningful to this regression experiment."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rapeseed_damage.artifacts import write_json

from .config import Config
from .data import TargetScaler
from .metrics import Predictions, mean_baseline
from .preprocessing import load_grid_crop

def _save_figure(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_history_plot(history, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(range(1, len(history["train_loss"]) + 1), history["train_loss"])
    axes[0].plot(history["val_epochs"], history["val_loss"], marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Normalized MSE", title="Loss")
    axes[1].plot(history["val_epochs"], history["val_mae"], marker="o")
    axes[1].set(xlabel="Epoch", ylabel="MAE", title="Validation MAE")
    axes[2].plot(history["val_epochs"], history["val_r2"], marker="o")
    axes[2].set(xlabel="Epoch", ylabel="R²", title="Validation R²")
    for axis in axes:
        axis.grid(alpha=0.25)
    _save_figure(figure, path)


def save_label_plot(targets, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(range(len(targets)), targets, alpha=0.7, edgecolors="none")
    axis.set(xlabel="Sample", ylabel="Mean damage score", title="Target distribution")
    axis.grid(alpha=0.25)
    _save_figure(figure, path)


def save_regression_plot(result: Predictions, path: Path) -> None:
    targets, predictions = result.targets, result.predictions
    residuals = predictions - targets
    metrics = result.metrics()
    lower, upper = min(targets.min(), predictions.min()), max(
        targets.max(), predictions.max()
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(targets, predictions, alpha=0.65, edgecolors="none")
    axes[0].plot([lower, upper], [lower, upper], "--", color="red")
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="DINOv3 regression")
    axes[0].text(
        0.05,
        0.95,
        f"MAE={metrics['mae']:.3f}\nRMSE={metrics['rmse']:.3f}\nR²={metrics['r2']:.3f}",
        transform=axes[0].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axes[1].scatter(predictions, residuals, alpha=0.65, edgecolors="none")
    axes[1].axhline(0, linestyle="--", color="red")
    axes[1].set(xlabel="Predicted", ylabel="Prediction − target", title="Residuals")
    for axis in axes:
        axis.grid(alpha=0.25)
    _save_figure(figure, path)


def save_examples(result: Predictions, path: Path, count: int, columns: int) -> None:
    count = min(count, len(result.targets))
    if count == 0:
        return
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for index in range(count):
        image = load_grid_crop(result.image_paths[index])
        error = result.predictions[index] - result.targets[index]
        axes[index].imshow(image)
        axes[index].axis("off")
        axes[index].set_title(
            f"Target: {result.targets[index]:.2f}\n"
            f"Prediction: {result.predictions[index]:.2f}\nError: {error:+.2f}"
        )
    for axis in axes[count:]:
        axis.axis("off")
    _save_figure(figure, path)


def save_evaluation(
    result: Predictions,
    scaler: TargetScaler,
    destination: Path,
    config: Config,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "model": result.metrics(),
        "mean_baseline": mean_baseline(result.targets, scaler.mean),
        "samples": len(result.targets),
    }
    write_json(destination / "metrics.json", report)
    pd.DataFrame(
        {
            "filename": result.filenames,
            "target": result.targets,
            "prediction": result.predictions,
            "residual": result.predictions - result.targets,
            "image_path": result.image_paths,
        }
    ).to_csv(destination / "predictions.csv", index=False)
    if config.output.save_plots:
        save_regression_plot(result, destination / "regression.png")
        save_examples(
            result,
            destination / "prediction_examples.png",
            config.output.example_images,
            config.output.example_columns,
        )
    return report

