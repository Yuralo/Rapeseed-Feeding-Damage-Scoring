"""Tri-scale regression reports and scale-specific attention inspection."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image

from experiments.dinov3_grid_tiled_mil.reporting import save_label_plot
from rapeseed_damage.artifacts import write_json

from .config import Config
from .data import TargetScaler
from .metrics import Predictions, attention_entropy, mean_baseline


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_history_plot(history, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.reshape(-1)
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(history["val_epochs"], history["val_loss"], marker="o", label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Normalized MSE", title="Loss")
    axes[0].legend()
    axes[1].plot(history["val_epochs"], history["val_mae"], marker="o", label="MAE")
    axes[1].plot(history["val_epochs"], history["val_r2"], marker="o", label="R²")
    axes[1].set(xlabel="Epoch", title="Validation metrics")
    axes[1].legend()
    axes[2].plot(
        history["val_epochs"], history["val_regional_attention_entropy"], label="4x4 entropy"
    )
    axes[2].plot(history["val_epochs"], history["val_local_attention_entropy"], label="5x5 entropy")
    axes[2].plot(history["val_epochs"], history["val_regional_top_tile_mass"], label="4x4 top")
    axes[2].plot(history["val_epochs"], history["val_local_top_tile_mass"], label="5x5 top")
    axes[2].set(xlabel="Epoch", ylabel="Fraction", title="Learned attention by scale")
    axes[2].legend()
    axes[3].plot(epochs, history["epoch_seconds"], marker="o")
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
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="3x3 + 4x4 + 5x5 MIL")
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


def _attention_heatmap(width: int, height: int, boxes: np.ndarray, weights: np.ndarray):
    weighted = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    relative = weights * len(weights)
    for box, value in zip(boxes, relative, strict=True):
        x0, y0, x1, y1 = map(int, box)
        weighted[y0:y1, x0:x1] += float(value)
        coverage[y0:y1, x0:x1] += 1
    return weighted / np.maximum(coverage, 1)


def _representative_indices(targets: np.ndarray, count: int) -> list[int]:
    count = min(count, len(targets))
    if not count:
        return []
    ordered = np.argsort(targets)
    return ordered[np.linspace(0, len(ordered) - 1, count).round().astype(int)].tolist()


def _draw_scale(axis_heatmap, axis_crop, rgb, boxes, weights, label: str) -> None:
    width, height = rgb.size
    heatmap = _attention_heatmap(width, height, boxes, weights)
    axis_heatmap.imshow(rgb)
    axis_heatmap.imshow(
        heatmap,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=max(2.0, float(heatmap.max()))),
        alpha=0.55,
        extent=(0, width, height, 0),
    )
    top = int(weights.argmax())
    x0, y0, x1, y1 = boxes[top]
    axis_crop.imshow(rgb.crop((int(x0), int(y0), int(x1), int(y1))))
    axis_heatmap.set_title(f"{label} attention relative to uniform")
    axis_crop.set_title(f"{label} top tile {top} | mass {weights[top]:.3f}")


def _draw_context_layout(axis, rgb, boxes) -> None:
    axis.imshow(rgb)
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        axis.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="white", linewidth=1)
        )
        axis.text(x0 + 8, y0 + 28, str(index), color="cyan", fontsize=9)
    axis.set_title("3x3 context tiles | mean pooled, no attention")


def save_attention_inspection(result: Predictions, path: Path, count: int) -> None:
    indices = _representative_indices(result.targets, count)
    if not indices:
        return
    figure, axes = plt.subplots(len(indices), 6, figsize=(27, 4.6 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        with Image.open(result.processed_image_paths[index]) as image:
            rgb = image.convert("RGB").copy()
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(
            f"{result.filenames[index]}\ntarget {result.targets[index]:.2f} | "
            f"prediction {result.predictions[index]:.2f}"
        )
        _draw_context_layout(axes[row, 1], rgb, result.context_boxes[index])
        _draw_scale(
            axes[row, 2],
            axes[row, 3],
            rgb,
            result.regional_boxes[index],
            result.regional_weights[index],
            "4x4",
        )
        _draw_scale(
            axes[row, 4],
            axes[row, 5],
            rgb,
            result.local_boxes[index],
            result.local_weights[index],
            "5x5",
        )
        for column in range(6):
            axes[row, column].axis("off")
    _save(figure, path)


def save_evaluation(
    result: Predictions,
    scaler: TargetScaler,
    destination: Path,
    config: Config,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    attention = result.attention_metrics()
    report = {
        "model": result.metrics(),
        "triscale_attention": attention,
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
        "regional_attention_normalized_entropy": attention_entropy(result.regional_weights),
        "regional_top_tile_index": result.regional_weights.argmax(axis=1),
        "regional_top_tile_weight": result.regional_weights.max(axis=1),
        "local_attention_normalized_entropy": attention_entropy(result.local_weights),
        "local_top_tile_index": result.local_weights.argmax(axis=1),
        "local_top_tile_weight": result.local_weights.max(axis=1),
        "source_image_path": result.source_image_paths,
        "processed_image_path": result.processed_image_paths,
        "context_feature_cache_path": result.context_feature_cache_paths,
        "regional_feature_cache_path": result.regional_feature_cache_paths,
        "local_feature_cache_path": result.local_feature_cache_paths,
    }
    for index in range(result.regional_weights.shape[1]):
        columns[f"regional_tile_{index:02d}_weight"] = result.regional_weights[:, index]
    for index in range(result.local_weights.shape[1]):
        columns[f"local_tile_{index:02d}_weight"] = result.local_weights[:, index]
    pd.DataFrame(columns).to_csv(destination / "predictions.csv", index=False)
    np.savez_compressed(
        destination / config.output.attention_arrays_name,
        filenames=np.asarray(result.filenames),
        context_boxes=result.context_boxes.astype(np.int16),
        regional_weights=result.regional_weights.astype(np.float16),
        regional_boxes=result.regional_boxes.astype(np.int16),
        local_weights=result.local_weights.astype(np.float16),
        local_boxes=result.local_boxes.astype(np.int16),
    )
    if config.output.save_plots:
        save_regression_plot(result, destination / "regression.png")
        save_prediction_examples(
            result,
            destination / "prediction_examples.png",
            config.output.example_images,
            config.output.example_columns,
        )
        save_attention_inspection(
            result,
            destination / "triscale_attention_inspection.png",
            config.output.attention_inspection_images,
        )
    return report


__all__ = [
    "save_attention_inspection",
    "save_evaluation",
    "save_history_plot",
    "save_label_plot",
    "save_prediction_examples",
    "save_regression_plot",
]
