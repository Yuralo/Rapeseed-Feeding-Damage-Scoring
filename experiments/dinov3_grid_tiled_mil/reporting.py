"""Evaluation artifacts and interpretable tile-attention plots."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image

from rapeseed_damage.artifacts import write_json

from .config import Config
from .data import TargetScaler
from .metrics import Predictions, attention_entropy, mean_baseline


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_label_plot(targets, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(range(len(targets)), targets, alpha=0.7, edgecolors="none")
    axis.set(xlabel="Sample", ylabel="Mean damage score", title="Target distribution")
    axis.grid(alpha=0.25)
    _save(figure, path)


def save_history_plot(history, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.reshape(-1)
    axes[0].plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="train")
    axes[0].plot(history["val_epochs"], history["val_loss"], marker="o", label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Normalized MSE", title="Loss")
    axes[0].legend()
    axes[1].plot(history["val_epochs"], history["val_mae"], marker="o", label="MAE")
    axes[1].plot(history["val_epochs"], history["val_r2"], marker="o", label="R²")
    axes[1].set(xlabel="Epoch", title="Validation metrics")
    axes[1].legend()
    axes[2].plot(
        history["val_epochs"], history["val_attention_entropy"], marker="o", label="entropy"
    )
    axes[2].plot(history["val_epochs"], history["val_top_tile_mass"], marker="o", label="top tile")
    axes[2].set(xlabel="Epoch", ylabel="Fraction", title="Tile-attention diagnostics")
    axes[2].legend()
    axes[3].plot(range(1, len(history["epoch_seconds"]) + 1), history["epoch_seconds"], marker="o")
    axes[3].set(xlabel="Epoch", ylabel="Seconds", title="Epoch duration")
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(figure, path)


def save_regression_plot(result: Predictions, path: Path) -> None:
    residuals = result.predictions - result.targets
    metrics = result.metrics()
    lower = min(result.targets.min(), result.predictions.min())
    upper = max(result.targets.max(), result.predictions.max())
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(result.targets, result.predictions, alpha=0.65, edgecolors="none")
    axes[0].plot([lower, upper], [lower, upper], "--", color="red")
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="Global + tiled MIL")
    axes[0].text(
        0.05,
        0.95,
        f"MAE={metrics['mae']:.3f}\nRMSE={metrics['rmse']:.3f}\nR²={metrics['r2']:.3f}",
        transform=axes[0].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axes[1].scatter(result.predictions, residuals, alpha=0.65, edgecolors="none")
    axes[1].axhline(0, linestyle="--", color="red")
    axes[1].set(xlabel="Predicted", ylabel="Prediction − target", title="Residuals")
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(figure, path)


def _attention_heatmap(width: int, height: int, boxes: np.ndarray, weights: np.ndarray):
    weighted = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    relative = weights * len(weights)
    for box, value in zip(boxes, relative, strict=True):
        x0, y0, x1, y1 = map(int, box)
        weighted[y0:y1, x0:x1] += float(value)
        coverage[y0:y1, x0:x1] += 1
    return weighted / np.maximum(coverage, 1)


def save_prediction_examples(result: Predictions, path: Path, count: int, columns: int) -> None:
    count = min(count, len(result.targets))
    if not count:
        return
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for index in range(count):
        with Image.open(result.processed_image_paths[index]) as image:
            axes[index].imshow(image.convert("RGB"))
        error = result.predictions[index] - result.targets[index]
        axes[index].set_title(
            f"{Path(result.filenames[index]).name}\nTarget {result.targets[index]:.2f} | "
            f"Pred {result.predictions[index]:.2f}\nError {error:+.2f}"
        )
        axes[index].axis("off")
    for axis in axes[count:]:
        axis.axis("off")
    _save(figure, path)


def save_attention_examples(result: Predictions, path: Path, count: int, columns: int) -> None:
    count = min(count, len(result.targets))
    if not count:
        return
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    entropy = attention_entropy(result.tile_weights)
    for index in range(count):
        with Image.open(result.processed_image_paths[index]) as image:
            rgb = image.convert("RGB")
        width, height = rgb.size
        heatmap = _attention_heatmap(
            width, height, result.tile_boxes[index], result.tile_weights[index]
        )
        axes[index].imshow(rgb)
        axes[index].imshow(
            heatmap,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=max(2.0, float(heatmap.max()))),
            alpha=0.5,
            extent=(0, width, height, 0),
        )
        top = int(result.tile_weights[index].argmax())
        x0, y0, x1, y1 = result.tile_boxes[index, top]
        axes[index].add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="cyan", linewidth=2)
        )
        axes[index].set_title(
            f"{Path(result.filenames[index]).name}\nTarget {result.targets[index]:.2f} | "
            f"Pred {result.predictions[index]:.2f}\nTop tile {top}="
            f"{result.tile_weights[index, top]:.3f} | H={entropy[index]:.3f}"
        )
        axes[index].axis("off")
    for axis in axes[count:]:
        axis.axis("off")
    _save(figure, path)


def _representative_indices(targets: np.ndarray, count: int) -> list[int]:
    count = min(count, len(targets))
    if not count:
        return []
    ordered = np.argsort(targets)
    return ordered[np.linspace(0, len(ordered) - 1, count).round().astype(int)].tolist()


def save_attention_inspection(result: Predictions, path: Path, count: int) -> None:
    indices = _representative_indices(result.targets, count)
    if not indices:
        return
    figure, axes = plt.subplots(len(indices), 3, figsize=(15, 4.8 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        with Image.open(result.processed_image_paths[index]) as image:
            rgb = image.convert("RGB").copy()
        width, height = rgb.size
        boxes, weights = result.tile_boxes[index], result.tile_weights[index]
        top = int(weights.argmax())
        axes[row, 0].imshow(rgb)
        for tile_index, (x0, y0, x1, y1) in enumerate(boxes):
            axes[row, 0].add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="white", linewidth=1)
            )
            axes[row, 0].text(x0 + 8, y0 + 28, str(tile_index), color="cyan", fontsize=9)
        axes[row, 0].set_title(
            f"{result.filenames[index]} | target {result.targets[index]:.2f} | "
            f"prediction {result.predictions[index]:.2f}"
        )
        heatmap = _attention_heatmap(width, height, boxes, weights)
        axes[row, 1].imshow(rgb)
        axes[row, 1].imshow(
            heatmap,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=max(2.0, float(heatmap.max()))),
            alpha=0.55,
            extent=(0, width, height, 0),
        )
        axes[row, 1].set_title("Tile attention relative to uniform")
        x0, y0, x1, y1 = boxes[top]
        axes[row, 2].imshow(rgb.crop((int(x0), int(y0), int(x1), int(y1))))
        axes[row, 2].set_title(f"Top tile {top} | attention mass {weights[top]:.3f}")
        for column in range(3):
            axes[row, column].axis("off")
    _save(figure, path)


def save_evaluation(
    result: Predictions,
    scaler: TargetScaler,
    destination: Path,
    config: Config,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    entropy = attention_entropy(result.tile_weights)
    top_indices = result.tile_weights.argmax(axis=1)
    top_weights = result.tile_weights.max(axis=1)
    report = {
        "model": result.metrics(),
        "tile_attention": result.attention_metrics(),
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
    columns = {
        "filename": result.filenames,
        "target": result.targets,
        "prediction": result.predictions,
        "residual": result.predictions - result.targets,
        "tile_attention_normalized_entropy": entropy,
        "top_tile_index": top_indices,
        "top_tile_weight": top_weights,
        "source_image_path": result.source_image_paths,
        "processed_image_path": result.processed_image_paths,
        "feature_cache_path": result.feature_cache_paths,
    }
    for tile_index in range(result.tile_weights.shape[1]):
        columns[f"tile_{tile_index:02d}_weight"] = result.tile_weights[:, tile_index]
    pd.DataFrame(columns).to_csv(destination / "predictions.csv", index=False)
    np.savez_compressed(
        destination / config.output.attention_arrays_name,
        filenames=np.asarray(result.filenames),
        weights=result.tile_weights.astype(np.float16),
        boxes=result.tile_boxes.astype(np.int16),
    )
    if config.output.save_plots:
        save_regression_plot(result, destination / "regression.png")
        save_prediction_examples(
            result,
            destination / "prediction_examples.png",
            config.output.example_images,
            config.output.example_columns,
        )
        save_attention_examples(
            result,
            destination / "tile_attention_examples.png",
            config.output.example_images,
            config.output.example_columns,
        )
        save_attention_inspection(
            result,
            destination / "tile_attention_inspection.png",
            config.output.attention_inspection_images,
        )
    return report
