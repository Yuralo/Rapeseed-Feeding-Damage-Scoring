"""Complete evaluation reports for SAM-guided adaptive-instance MIL."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from experiments.dinov3_grid_tiled_mil.metrics import regression_metrics
from experiments.dinov3_grid_tiled_mil.reporting import save_label_plot
from rapeseed_damage.artifacts import write_json

from .metrics import Predictions, mean_baseline


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 2 or np.isclose(left.std(), 0) or np.isclose(right.std(), 0):
        return None
    return float(np.corrcoef(left, right)[0, 1])


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
        history["val_epochs"],
        history["val_attention_entropy"],
        marker="o",
        label="normalized entropy",
    )
    axes[2].plot(
        history["val_epochs"], history["val_top_mass"], marker="o", label="top instance"
    )
    axes[2].set(xlabel="Epoch", ylabel="Fraction", title="Plant-attention diagnostics")
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
    points = axes[0].scatter(
        result.targets,
        result.predictions,
        c=result.instance_counts,
        cmap="viridis",
        alpha=0.72,
        edgecolors="none",
    )
    axes[0].plot([lower, upper], [lower, upper], "--", color="red")
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="SAM adaptive MIL")
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


def _panel(
    result: Predictions,
    indices: np.ndarray | list[int],
    path: Path,
    columns: int,
    title: str,
) -> None:
    indices = list(map(int, indices))
    if not indices:
        return
    statistics = result.attention_statistics()
    rows = math.ceil(len(indices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.4 * rows))
    figure.suptitle(title, fontsize=14)
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, index in zip(axes, indices, strict=False):
        with Image.open(result.processed_image_paths[index]) as image:
            axis.imshow(image.convert("RGB"))
        error = result.predictions[index] - result.targets[index]
        axis.set_title(
            f"{Path(result.filenames[index]).name}\n"
            f"Target {result.targets[index]:.2f} | Pred {result.predictions[index]:.2f}\n"
            f"Error {error:+.2f} | n={result.instance_counts[index]} | "
            f"H={statistics['normalized_entropy'][index]:.2f}"
        )
        axis.axis("off")
    for axis in axes[len(indices) :]:
        axis.axis("off")
    _save(figure, path)


def _representative_indices(targets: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(targets))
    if not count:
        return np.asarray([], dtype=int)
    ordered = np.argsort(targets)
    return ordered[np.linspace(0, len(ordered) - 1, count).round().astype(int)]


def save_prediction_examples(result: Predictions, path: Path, count: int, columns: int) -> None:
    _panel(
        result,
        _representative_indices(result.targets, count),
        path,
        columns,
        "Representative predictions across the target range",
    )


def save_worst_error_examples(result: Predictions, path: Path, count: int, columns: int) -> None:
    residuals = result.predictions - result.targets
    indices = np.argsort(np.abs(residuals))[::-1][: min(count, len(residuals))]
    _panel(result, indices, path, columns, "Largest absolute validation errors")


def save_attention_inspection(result: Predictions, path: Path, count: int) -> None:
    indices = _representative_indices(result.targets, count)
    if not len(indices):
        return
    statistics = result.attention_statistics()
    figure, axes = plt.subplots(len(indices), 3, figsize=(15, 4.8 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        with Image.open(result.processed_image_paths[index]) as image:
            rgb = image.convert("RGB").copy()
        with Image.open(result.mask_paths[index]) as mask:
            mask_array = np.asarray(mask.convert("L"))
        valid = np.flatnonzero(result.valid[index])
        top = int(statistics["top_index"][index])
        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(rgb)
        axes[row, 1].imshow(mask_array, cmap="Greens", alpha=0.4)
        for instance in valid:
            x0, y0, x1, y1 = result.boxes[index, instance]
            axes[row, 0].add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    color="cyan" if instance == top else "white",
                    linewidth=2 if instance == top else 1,
                )
            )
            axes[row, 0].text(x0 + 5, y0 + 20, str(instance), color="yellow", fontsize=8)
        x0, y0, x1, y1 = result.boxes[index, top]
        axes[row, 2].imshow(rgb.crop((x0, y0, x1, y1)))
        error = result.predictions[index] - result.targets[index]
        axes[row, 0].set_title(
            f"{Path(result.filenames[index]).name} | target {result.targets[index]:.2f} | "
            f"pred {result.predictions[index]:.2f} | error {error:+.2f}"
        )
        axes[row, 1].set_title(
            f"SAM mask | {len(valid)} instances | coverage {result.mask_coverages[index]:.3f}"
        )
        axes[row, 2].set_title(
            f"Top crop {top} | mass {result.weights[index, top]:.3f} | "
            f"H={statistics['normalized_entropy'][index]:.3f}"
        )
        for axis in axes[row]:
            axis.axis("off")
    _save(figure, path)


def save_adaptive_diagnostics(result: Predictions, path: Path) -> None:
    residuals = result.predictions - result.targets
    absolute = np.abs(residuals)
    statistics = result.attention_statistics()
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0, 0].scatter(result.instance_counts, absolute, alpha=0.65, edgecolors="none")
    axes[0, 0].set(xlabel="SAM instance count", ylabel="Absolute error", title="Instances vs error")
    axes[0, 1].scatter(
        statistics["normalized_entropy"], absolute, alpha=0.65, edgecolors="none"
    )
    axes[0, 1].set(
        xlabel="Normalized attention entropy", ylabel="Absolute error", title="Entropy vs error"
    )
    axes[1, 0].scatter(statistics["top_weight"], absolute, alpha=0.65, edgecolors="none")
    axes[1, 0].set(
        xlabel="Top-instance attention mass", ylabel="Absolute error", title="Focus vs error"
    )
    points = axes[1, 1].scatter(
        result.targets,
        residuals,
        c=result.instance_counts,
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
    )
    axes[1, 1].axhline(0, color="red", linestyle="--")
    axes[1, 1].set(xlabel="Target", ylabel="Prediction − target", title="Error by target")
    figure.colorbar(points, ax=axes[1, 1], label="SAM instances")
    for axis in axes.reshape(-1):
        axis.grid(alpha=0.25)
    _save(figure, path)


def target_range_metrics(result: Predictions) -> dict[str, dict]:
    ranges = (
        ("0_to_2_5", result.targets <= 2.5),
        ("over_2_5_to_7_5", (result.targets > 2.5) & (result.targets <= 7.5)),
        ("over_7_5_to_15", (result.targets > 7.5) & (result.targets <= 15)),
        ("over_15", result.targets > 15),
    )
    report = {}
    for name, selected in ranges:
        if not selected.any():
            report[name] = {"samples": 0}
            continue
        report[name] = {
            "samples": int(selected.sum()),
            **regression_metrics(result.targets[selected], result.predictions[selected]),
            "mean_residual": float((result.predictions[selected] - result.targets[selected]).mean()),
        }
    return report


def error_analysis(result: Predictions) -> dict:
    residuals = result.predictions - result.targets
    absolute = np.abs(residuals)
    statistics = result.attention_statistics()
    worst = int(absolute.argmax())
    slope, intercept = np.polyfit(result.targets, result.predictions, 1)
    return {
        "mean_residual": float(residuals.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "within_1_point_fraction": float(np.mean(absolute <= 1)),
        "within_2_5_points_fraction": float(np.mean(absolute <= 2.5)),
        "within_5_points_fraction": float(np.mean(absolute <= 5)),
        "prediction_vs_target_slope": float(slope),
        "prediction_vs_target_intercept": float(intercept),
        "absolute_error_correlations": {
            "instance_count": _correlation(result.instance_counts, absolute),
            "normalized_attention_entropy": _correlation(
                statistics["normalized_entropy"], absolute
            ),
            "top_instance_mass": _correlation(statistics["top_weight"], absolute),
            "target": _correlation(result.targets, absolute),
        },
        "worst_sample": {
            "filename": result.filenames[worst],
            "target": float(result.targets[worst]),
            "prediction": float(result.predictions[worst]),
            "residual": float(residuals[worst]),
            "absolute_error": float(absolute[worst]),
        },
    }


def save_evaluation(result: Predictions, scaler, destination: Path, config) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    residuals = result.predictions - result.targets
    statistics = result.attention_statistics()
    report = {
        "model": result.metrics(),
        "adaptive_attention": result.attention_metrics(),
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
        "attention_normalized_entropy": statistics["normalized_entropy"],
        "attention_effective_instance_count": statistics["effective_instance_count"],
        "top_instance_index": statistics["top_index"],
        "top_instance_weight": statistics["top_weight"],
        "source_image_path": result.source_image_paths,
        "processed_image_path": result.processed_image_paths,
        "mask_path": result.mask_paths,
        "context_feature_cache_path": result.context_feature_cache_paths,
        "adaptive_feature_cache_path": result.adaptive_feature_cache_paths,
    }
    for index in range(result.weights.shape[1]):
        columns[f"instance_{index:02d}_weight"] = result.weights[:, index]
        columns[f"instance_{index:02d}_foreground_pixels"] = result.foreground_pixels[:, index]
    pd.DataFrame(columns).to_csv(destination / "predictions.csv", index=False)
    np.savez_compressed(
        destination / config.output.attention_arrays_name,
        filenames=np.asarray(result.filenames),
        weights=result.weights.astype(np.float16),
        valid=result.valid,
        boxes=result.boxes.astype(np.int16),
        foreground_pixels=result.foreground_pixels.astype(np.float32),
        normalized_entropy=statistics["normalized_entropy"].astype(np.float32),
        top_indices=statistics["top_index"].astype(np.int16),
    )
    if config.output.save_plots:
        save_regression_plot(result, destination / "regression.png")
        save_adaptive_diagnostics(result, destination / "adaptive_diagnostics.png")
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
        save_attention_inspection(
            result,
            destination / "adaptive_attention_inspection.png",
            config.output.attention_inspection_images,
        )
    return report


__all__ = [
    "error_analysis",
    "save_evaluation",
    "save_history_plot",
    "save_label_plot",
    "target_range_metrics",
]
