"""Full evaluation artifacts for fixed-tile plus SAM-adaptive hybrid MIL."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from experiments.dinov3_grid_multiscale_tiled_mil.reporting import _draw_scale
from experiments.dinov3_grid_sam_adaptive_mil.reporting import (
    error_analysis,
    save_adaptive_diagnostics,
    save_prediction_examples,
    save_worst_error_examples,
    target_range_metrics,
)
from experiments.dinov3_grid_tiled_mil.metrics import attention_entropy
from experiments.dinov3_grid_tiled_mil.reporting import save_label_plot
from rapeseed_damage.artifacts import write_json

from .metrics import Predictions, mean_baseline


def _save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_history_plot(history, path):
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
        history["val_epochs"], history["val_fine_attention_entropy"], label="4x4 entropy"
    )
    axes[2].plot(
        history["val_epochs"], history["val_plant_attention_entropy"], label="plant entropy"
    )
    axes[2].plot(history["val_epochs"], history["val_fine_top_mass"], label="4x4 top")
    axes[2].plot(history["val_epochs"], history["val_plant_top_mass"], label="plant top")
    axes[2].set(xlabel="Epoch", ylabel="Fraction", title="Hybrid attention diagnostics")
    axes[2].legend()
    axes[3].plot(epochs, history["epoch_seconds"], marker="o")
    axes[3].set(xlabel="Epoch", ylabel="Seconds", title="Epoch duration")
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(figure, path)


def save_regression_plot(result: Predictions, path: Path):
    residuals = result.predictions - result.targets
    metrics = result.metrics()
    lower = min(result.targets.min(), result.predictions.min())
    upper = max(result.targets.max(), result.predictions.max())
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    points = axes[0].scatter(
        result.targets,
        result.predictions,
        c=result.instance_counts,
        cmap="viridis",
        alpha=0.72,
        edgecolors="none",
    )
    axes[0].plot([lower, upper], [lower, upper], "--", color="red")
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="4x4 + SAM adaptive hybrid MIL")
    axes[0].text(
        0.05,
        0.95,
        f"MAE={metrics['mae']:.3f}\nRMSE={metrics['rmse']:.3f}\nR²={metrics['r2']:.3f}",
        transform=axes[0].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    figure.colorbar(points, ax=axes[0], label="SAM instances")
    axes[1].scatter(result.predictions, residuals, alpha=0.68, edgecolors="none")
    axes[1].axhline(0, linestyle="--", color="red")
    axes[1].set(xlabel="Predicted", ylabel="Prediction − target", title="Residuals")
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(figure, path)


def _representative_indices(targets, count):
    count = min(count, len(targets))
    if not count:
        return []
    ordered = np.argsort(targets)
    return ordered[np.linspace(0, len(ordered) - 1, count).round().astype(int)].tolist()


def save_hybrid_attention_inspection(result: Predictions, path: Path, count: int):
    indices = _representative_indices(result.targets, count)
    if not indices:
        return
    statistics = result.attention_statistics()
    figure, axes = plt.subplots(len(indices), 5, figsize=(23, 4.8 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        with Image.open(result.processed_image_paths[index]) as image:
            rgb = image.convert("RGB").copy()
        with Image.open(result.mask_paths[index]) as mask:
            mask_array = np.asarray(mask.convert("L"))
        axes[row, 0].imshow(rgb)
        error = result.predictions[index] - result.targets[index]
        axes[row, 0].set_title(
            f"{Path(result.filenames[index]).name}\ntarget {result.targets[index]:.2f} | "
            f"pred {result.predictions[index]:.2f} | error {error:+.2f}"
        )
        _draw_scale(
            axes[row, 1],
            axes[row, 2],
            rgb,
            result.fine_boxes[index],
            result.fine_weights[index],
            "4x4",
        )
        axes[row, 3].imshow(rgb)
        axes[row, 3].imshow(mask_array, cmap="Greens", alpha=0.35)
        valid = np.flatnonzero(result.valid[index])
        top = int(statistics["top_index"][index])
        for instance in valid:
            x0, y0, x1, y1 = result.boxes[index, instance]
            axes[row, 3].add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    color="cyan" if instance == top else "white",
                    linewidth=2 if instance == top else 1,
                )
            )
        axes[row, 3].set_title(
            f"SAM instances {len(valid)} | H={statistics['normalized_entropy'][index]:.3f}"
        )
        x0, y0, x1, y1 = result.boxes[index, top]
        axes[row, 4].imshow(rgb.crop((x0, y0, x1, y1)))
        axes[row, 4].set_title(
            f"Top plant {top} | mass {result.weights[index, top]:.3f}"
        )
        for axis in axes[row]:
            axis.axis("off")
    _save(figure, path)


def save_evaluation(result: Predictions, scaler, destination: Path, config):
    destination.mkdir(parents=True, exist_ok=True)
    residuals = result.predictions - result.targets
    plant = result.attention_statistics()
    attention = result.attention_metrics()
    report = {
        "model": result.metrics(),
        "hybrid_attention": attention,
        "error_analysis": error_analysis(result),
        "target_ranges": target_range_metrics(result),
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
        "residual": residuals,
        "absolute_error": np.abs(residuals),
        "instance_count": result.instance_counts,
        "mask_coverage": result.mask_coverages,
        "fine_attention_normalized_entropy": attention_entropy(result.fine_weights),
        "fine_top_tile_index": result.fine_weights.argmax(1),
        "fine_top_tile_weight": result.fine_weights.max(1),
        "plant_attention_normalized_entropy": plant["normalized_entropy"],
        "plant_attention_effective_instance_count": plant["effective_instance_count"],
        "top_plant_index": plant["top_index"],
        "top_plant_weight": plant["top_weight"],
        "source_image_path": result.source_image_paths,
        "processed_image_path": result.processed_image_paths,
        "mask_path": result.mask_paths,
        "context_feature_cache_path": result.context_feature_cache_paths,
        "fine_feature_cache_path": result.fine_feature_cache_paths,
        "adaptive_feature_cache_path": result.adaptive_feature_cache_paths,
    }
    for index in range(result.fine_weights.shape[1]):
        columns[f"fine_tile_{index:02d}_weight"] = result.fine_weights[:, index]
    for index in range(result.weights.shape[1]):
        columns[f"plant_{index:02d}_weight"] = result.weights[:, index]
        columns[f"plant_{index:02d}_foreground_pixels"] = result.foreground_pixels[:, index]
    pd.DataFrame(columns).to_csv(destination / "predictions.csv", index=False)
    np.savez_compressed(
        destination / config.output.attention_arrays_name,
        filenames=np.asarray(result.filenames),
        fine_weights=result.fine_weights.astype(np.float16),
        fine_boxes=result.fine_boxes.astype(np.int16),
        plant_weights=result.weights.astype(np.float16),
        plant_valid=result.valid,
        plant_boxes=result.boxes.astype(np.int16),
        plant_foreground_pixels=result.foreground_pixels.astype(np.float32),
    )
    if config.output.save_plots:
        save_regression_plot(result, destination / "regression.png")
        save_adaptive_diagnostics(result, destination / "hybrid_diagnostics.png")
        save_prediction_examples(
            result,
            destination / "prediction_examples.png",
            config.output.example_images,
            config.output.example_columns,
        )
        save_worst_error_examples(
            result,
            destination / "worst_error_examples.png",
            config.output.example_images,
            config.output.example_columns,
        )
        save_hybrid_attention_inspection(
            result,
            destination / "hybrid_attention_inspection.png",
            config.output.attention_inspection_images,
        )
    return report


__all__ = ["save_evaluation", "save_history_plot", "save_label_plot"]
