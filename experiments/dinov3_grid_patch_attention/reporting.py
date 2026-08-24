"""Regression artifacts and interpretable patch-attention overlays."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from rapeseed_damage.artifacts import write_json

from .config import Config
from .data import TargetScaler
from .metrics import Predictions, attention_entropy, mean_baseline, top_fraction_mass


def _save_figure(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_history_plot(history, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(19, 10))
    axes = axes.reshape(-1)
    axes[0].plot(range(1, len(history["train_loss"]) + 1), history["train_loss"])
    axes[0].plot(history["val_epochs"], history["val_loss"], marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Normalized MSE", title="Loss")
    axes[1].plot(history["val_epochs"], history["val_mae"], marker="o", label="MAE")
    axes[1].plot(history["val_epochs"], history["val_r2"], marker="o", label="R²")
    axes[1].set(xlabel="Epoch", title="Validation metrics")
    axes[1].legend()
    axes[2].plot(
        history["val_epochs"],
        history["val_attention_entropy"],
        marker="o",
        label="normalized entropy",
    )
    axes[2].plot(
        history["val_epochs"],
        history["val_attention_top_10_mass"],
        marker="o",
        label="top-10% mass",
    )
    axes[2].set(xlabel="Epoch", ylabel="Fraction", title="Attention concentration")
    axes[2].legend()
    for group in sorted({name for values in history["learning_rates"] for name in values}):
        axes[3].plot(
            range(1, len(history["learning_rates"]) + 1),
            [values.get(group, np.nan) for values in history["learning_rates"]],
            label=group,
        )
    axes[3].set(xlabel="Epoch", ylabel="Learning rate", title="End-of-epoch LR", yscale="log")
    axes[3].legend(fontsize=8)
    axes[4].plot(range(1, len(history["epoch_seconds"]) + 1), history["epoch_seconds"], marker="o")
    axes[4].set(xlabel="Epoch", ylabel="Seconds", title="Epoch duration")
    axes[5].plot(
        range(1, len(history["peak_cuda_memory_gb"]) + 1),
        history["peak_cuda_memory_gb"],
        marker="o",
    )
    axes[5].set(xlabel="Epoch", ylabel="GB", title="Peak CUDA memory")
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
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="Patch-attention regression")
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


def save_attention_examples(
    result: Predictions,
    path: Path,
    count: int,
    columns: int,
    *,
    ratio_min: float,
    ratio_max: float,
) -> None:
    count = min(count, len(result.targets))
    if count == 0:
        return
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    entropies = attention_entropy(result.attention_weights)
    for index in range(count):
        grid = result.attention_weights[index].reshape(
            int(result.attention_grid_rows[index]),
            int(result.attention_grid_columns[index]),
        )
        relative_to_uniform = grid * grid.size
        with Image.open(result.processed_image_paths[index]) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            axes[index].imshow(rgb)
        axes[index].imshow(
            relative_to_uniform,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=ratio_min, vcenter=1.0, vmax=ratio_max),
            alpha=0.52,
            interpolation="bilinear",
            extent=(0, width, height, 0),
        )
        error = result.predictions[index] - result.targets[index]
        axes[index].axis("off")
        axes[index].set_title(
            f"Target {result.targets[index]:.2f} | Pred {result.predictions[index]:.2f}\n"
            f"Error {error:+.2f} | H {entropies[index]:.4f} "
            f"| max {relative_to_uniform.max():.2f}× uniform"
        )
    for axis in axes[count:]:
        axis.axis("off")
    _save_figure(figure, path)


def _representative_indices(targets: np.ndarray, count: int) -> list[int]:
    count = min(count, len(targets))
    ordered = np.argsort(targets)
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return ordered[positions].tolist()


def save_attention_inspection(
    result: Predictions,
    path: Path,
    *,
    count: int,
    top_fraction: float,
    ratio_min: float,
    ratio_max: float,
) -> None:
    indices = _representative_indices(result.targets, count)
    if not indices:
        return
    entropies = attention_entropy(result.attention_weights)
    top_mass = top_fraction_mass(result.attention_weights, top_fraction)
    figure, axes = plt.subplots(len(indices), 3, figsize=(15, 4.7 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        grid_rows = int(result.attention_grid_rows[index])
        grid_columns = int(result.attention_grid_columns[index])
        flat_weights = result.attention_weights[index]
        grid = flat_weights.reshape(grid_rows, grid_columns)
        relative_to_uniform = grid * grid.size
        with Image.open(result.processed_image_paths[index]) as image:
            rgb = image.convert("RGB").copy()
        width, height = rgb.size
        error = result.predictions[index] - result.targets[index]

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(
            f"Original | {result.filenames[index]}\n"
            f"Target {result.targets[index]:.2f} | Prediction {result.predictions[index]:.2f} "
            f"| Error {error:+.2f}"
        )

        axes[row, 1].imshow(rgb)
        axes[row, 1].imshow(
            relative_to_uniform,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=ratio_min, vcenter=1.0, vmax=ratio_max),
            alpha=0.5,
            interpolation="bilinear",
            extent=(0, width, height, 0),
        )
        axes[row, 1].set_title(
            f"Attention relative to uniform | max {relative_to_uniform.max():.2f}×"
        )

        axes[row, 2].imshow(rgb)
        top_count = max(1, int(np.ceil(flat_weights.size * top_fraction)))
        top_indices = np.argsort(flat_weights)[-top_count:]
        patch_width, patch_height = width / grid_columns, height / grid_rows
        for patch_index in top_indices:
            patch_row, patch_column = divmod(int(patch_index), grid_columns)
            axes[row, 2].add_patch(
                Rectangle(
                    (patch_column * patch_width, patch_row * patch_height),
                    patch_width,
                    patch_height,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.7,
                )
            )
        axes[row, 2].set_title(
            f"Top {100 * top_fraction:.0f}% patches | mass {top_mass[index]:.3f}\n"
            f"Normalized entropy {entropies[index]:.3f}"
        )
        for column in range(3):
            axes[row, column].axis("off")
    _save_figure(figure, path)


def save_evaluation(
    result: Predictions,
    scaler: TargetScaler,
    destination: Path,
    config: Config,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    entropies = attention_entropy(result.attention_weights)
    top_mass = top_fraction_mass(result.attention_weights)
    report = {
        "model": result.metrics(),
        "attention": result.attention_metrics(),
        "mean_baseline": mean_baseline(result.targets, scaler.baseline_mean),
        "samples": len(result.targets),
        "target_processing": {
            "normalized": True,
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
            "attention_normalized_entropy": entropies,
            "attention_top_10_percent_mass": top_mass,
            "source_image_path": result.source_image_paths,
            "processed_image_path": result.processed_image_paths,
        }
    ).to_csv(destination / "predictions.csv", index=False)
    np.savez_compressed(
        destination / config.output.attention_arrays_name,
        filenames=np.asarray(result.filenames),
        weights=result.attention_weights.astype(np.float16),
        grid_rows=result.attention_grid_rows,
        grid_columns=result.attention_grid_columns,
    )
    if config.output.save_plots:
        save_regression_plot(result, destination / "regression.png")
        save_attention_examples(
            result,
            destination / "attention_examples.png",
            config.output.example_images,
            config.output.example_columns,
            ratio_min=config.output.attention_ratio_min,
            ratio_max=config.output.attention_ratio_max,
        )
        save_attention_inspection(
            result,
            destination / "attention_inspection.png",
            count=config.output.attention_inspection_images,
            top_fraction=config.output.attention_top_fraction,
            ratio_min=config.output.attention_ratio_min,
            ratio_max=config.output.attention_ratio_max,
        )
    return report
