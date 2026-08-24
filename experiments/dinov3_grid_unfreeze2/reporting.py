"""CSV and plot artifacts for the final-two-block regression experiment."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from rapeseed_damage.artifacts import write_json

from .config import Config
from .data import TargetScaler
from .metrics import Predictions, mean_baseline


def _save_figure(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_history_plot(history, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.reshape(-1)
    axes[0].plot(range(1, len(history["train_loss"]) + 1), history["train_loss"])
    axes[0].plot(history["val_epochs"], history["val_loss"], marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Objective MSE", title="Loss")
    axes[1].plot(history["val_epochs"], history["val_mae"], marker="o", label="MAE")
    axes[1].plot(history["val_epochs"], history["val_r2"], marker="o", label="R²")
    axes[1].set(xlabel="Epoch", title="Validation metrics")
    axes[1].legend()
    for group in sorted({name for values in history["learning_rates"] for name in values}):
        axes[2].plot(
            range(1, len(history["learning_rates"]) + 1),
            [values.get(group, np.nan) for values in history["learning_rates"]],
            label=group,
        )
    axes[2].set(xlabel="Epoch", ylabel="Learning rate", title="End-of-epoch LR", yscale="log")
    axes[2].legend(fontsize=8)
    axes[3].plot(
        range(1, len(history["epoch_seconds"]) + 1),
        history["epoch_seconds"],
        marker="o",
        label="seconds",
    )
    axes[3].set(xlabel="Epoch", ylabel="Seconds", title="Epoch duration")
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
    lower, upper = min(targets.min(), predictions.min()), max(targets.max(), predictions.max())
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
        with Image.open(result.processed_image_paths[index]) as image:
            axes[index].imshow(image.convert("RGB"))
        error = result.predictions[index] - result.targets[index]
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
        "mean_baseline": mean_baseline(result.targets, scaler.baseline_mean),
        "samples": len(result.targets),
        "target_processing": {
            "normalized": scaler.enabled,
            "training_mean": scaler.baseline_mean,
            "transform_mean": scaler.mean,
            "transform_std": scaler.std,
        },
    }
    write_json(destination / "metrics.json", report)
    pd.DataFrame(
        {
            "filename": result.filenames,
            "target": result.targets,
            "prediction": result.predictions,
            "residual": result.predictions - result.targets,
            "source_image_path": result.source_image_paths,
            "processed_image_path": result.processed_image_paths,
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
