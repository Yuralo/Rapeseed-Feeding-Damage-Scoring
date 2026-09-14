"""Complete evaluation tables and plots for the three-view experiment."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from experiments.dinov3_grid_tiled_mil.reporting import save_label_plot
from rapeseed_damage.artifacts import write_json

from .metrics import Predictions, mean_baseline, normalized_entropy


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_history_plot(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.reshape(-1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(history["val_epochs"], history["val_loss"], marker="o", label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Normalized loss", title="Loss")
    axes[0].legend()
    axes[1].plot(history["val_epochs"], history["val_mae"], marker="o", label="MAE")
    axes[1].plot(history["val_epochs"], history["val_r2"], marker="o", label="R²")
    axes[1].set(xlabel="Epoch", title="Gold validation metrics")
    axes[1].legend()
    axes[2].plot(history["val_epochs"], history["val_cell_entropy"], label="cell entropy")
    axes[2].plot(history["val_epochs"], history["val_plant_entropy"], label="plant entropy")
    axes[2].plot(history["val_epochs"], history["val_top_cell_mass"], label="top cell mass")
    axes[2].plot(history["val_epochs"], history["val_top_plant_mass"], label="top plant mass")
    axes[2].set(xlabel="Epoch", ylabel="Fraction", title="Hierarchical attention")
    axes[2].legend()
    axes[3].plot(epochs, history["epoch_seconds"], marker="o")
    axes[3].set(xlabel="Epoch", ylabel="Seconds", title="Epoch duration")
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(figure, path)


def _correlation(left, right) -> float | None:
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def error_analysis(result: Predictions) -> dict:
    residual = result.predictions - result.targets
    absolute = np.abs(residual)
    slope, intercept = np.polyfit(result.targets, result.predictions, 1)
    worst = int(absolute.argmax())
    return {
        "mean_residual": float(residual.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "within_1_point_fraction": float(np.mean(absolute <= 1)),
        "within_2_5_points_fraction": float(np.mean(absolute <= 2.5)),
        "within_5_points_fraction": float(np.mean(absolute <= 5)),
        "prediction_vs_target_slope": float(slope),
        "prediction_vs_target_intercept": float(intercept),
        "absolute_error_correlations": {
            "target": _correlation(result.targets, absolute),
            "plant_count": _correlation(result.plant_counts, absolute),
            "cell_entropy": _correlation(normalized_entropy(result.cell_weights), absolute),
            "plant_entropy": _correlation(
                normalized_entropy(result.plant_weights, result.plant_valid), absolute
            ),
        },
        "worst_sample": {
            "filename": result.filenames[worst],
            "target": float(result.targets[worst]),
            "prediction": float(result.predictions[worst]),
            "residual": float(residual[worst]),
        },
    }


def target_range_metrics(result: Predictions) -> dict:
    from experiments.dinov3_grid_tiled_mil.metrics import regression_metrics

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
        else:
            residual = result.predictions[selected] - result.targets[selected]
            report[name] = {
                "samples": int(selected.sum()),
                **regression_metrics(result.targets[selected], result.predictions[selected]),
                "mean_residual": float(residual.mean()),
            }
    return report


def save_regression_plot(result: Predictions, path: Path) -> None:
    residual = result.predictions - result.targets
    metrics = result.metrics()
    lower = min(result.targets.min(), result.predictions.min())
    upper = max(result.targets.max(), result.predictions.max())
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    points = axes[0].scatter(
        result.targets, result.predictions, c=result.plant_counts, cmap="viridis", alpha=0.72
    )
    axes[0].plot([lower, upper], [lower, upper], "--", color="red")
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="Hierarchical three-view MIL")
    axes[0].text(
        0.05, 0.95,
        f"MAE={metrics['mae']:.3f}\nRMSE={metrics['rmse']:.3f}\nR²={metrics['r2']:.3f}",
        transform=axes[0].transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    figure.colorbar(points, ax=axes[0], label="SAM plants")
    axes[1].scatter(result.predictions, residual, alpha=0.68)
    axes[1].axhline(0, linestyle="--", color="red")
    axes[1].set(xlabel="Predicted", ylabel="Prediction − target", title="Residuals")
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(figure, path)


def _representative(targets: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(targets))
    order = np.argsort(targets)
    return order[np.linspace(0, len(order) - 1, count).round().astype(int)] if count else np.array([])


def _prediction_panel(result, indices, path, columns, title) -> None:
    indices = list(map(int, indices))
    if not indices:
        return
    rows = math.ceil(len(indices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.4 * rows))
    figure.suptitle(title)
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, index in zip(axes, indices, strict=False):
        with Image.open(result.processed_image_paths[index]) as image:
            axis.imshow(image.convert("RGB"))
        error = result.predictions[index] - result.targets[index]
        axis.set_title(
            f"{Path(result.filenames[index]).name}\nTarget {result.targets[index]:.2f} | "
            f"Pred {result.predictions[index]:.2f}\nError {error:+.2f} | "
            f"plants={result.plant_counts[index]}"
        )
        axis.axis("off")
    for axis in axes[len(indices):]:
        axis.axis("off")
    _save(figure, path)


def save_attention_inspection(result: Predictions, path: Path, count: int) -> None:
    indices = _representative(result.targets, count)
    if not len(indices):
        return
    figure, axes = plt.subplots(len(indices), 3, figsize=(15, 4.8 * len(indices)), squeeze=False)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for row, index in enumerate(indices):
        with Image.open(result.processed_image_paths[index]) as handle:
            image = handle.convert("RGB").copy()
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(image)
        top_cell = int(result.cell_weights[index].argmax())
        top_plant = int(result.plant_weights[index].argmax())
        for cell, box in enumerate(result.cell_boxes[index]):
            x0, y0, x1, y1 = box
            axes[row, 0].add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=colors[cell],
                          linewidth=4 if cell == top_cell else 2)
            )
            axes[row, 0].text(x0 + 5, y0 + 25, f"C{cell} {result.cell_weights[index, cell]:.2f}",
                              color=colors[cell], fontsize=9)
        for plant in np.flatnonzero(result.plant_valid[index]):
            x0, y0, x1, y1 = result.plant_boxes[index, plant]
            cell = int(result.plant_cell_indices[index, plant])
            axes[row, 1].add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=colors[cell],
                          linewidth=4 if plant == top_plant else 1)
            )
            axes[row, 1].text(x0 + 4, y0 + 18, f"{result.plant_weights[index, plant]:.2f}",
                              color="yellow", fontsize=8)
        x0, y0, x1, y1 = result.plant_boxes[index, top_plant]
        axes[row, 2].imshow(image.crop((int(x0), int(y0), int(x1), int(y1))))
        axes[row, 0].set_title(
            f"{Path(result.filenames[index]).name}\ntarget {result.targets[index]:.2f} | "
            f"pred {result.predictions[index]:.2f}\ntop cell C{top_cell}"
        )
        axes[row, 1].set_title(f"Plant attention by owning cell | n={result.plant_counts[index]}")
        axes[row, 2].set_title(
            f"Top plant {top_plant} | mass {result.plant_weights[index, top_plant]:.3f}"
        )
        for axis in axes[row]:
            axis.axis("off")
    _save(figure, path)


def save_hierarchical_diagnostics(result: Predictions, path: Path) -> None:
    residual = result.predictions - result.targets
    absolute = np.abs(residual)
    cell_entropy = normalized_entropy(result.cell_weights)
    plant_entropy = normalized_entropy(result.plant_weights, result.plant_valid)
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0, 0].scatter(result.plant_counts, absolute, alpha=0.65)
    axes[0, 0].set(
        xlabel="SAM plant count", ylabel="Absolute error", title="Plant count vs error"
    )
    axes[0, 1].scatter(cell_entropy, absolute, alpha=0.65)
    axes[0, 1].set(
        xlabel="Cell-attention entropy", ylabel="Absolute error", title="Cell focus vs error"
    )
    axes[1, 0].scatter(plant_entropy, absolute, alpha=0.65)
    axes[1, 0].set(
        xlabel="Plant-attention entropy", ylabel="Absolute error", title="Plant focus vs error"
    )
    points = axes[1, 1].scatter(
        result.targets, residual, c=result.plant_counts, cmap="viridis", alpha=0.7
    )
    axes[1, 1].axhline(0, color="red", linestyle="--")
    axes[1, 1].set(xlabel="Target", ylabel="Prediction − target", title="Error by target")
    figure.colorbar(points, ax=axes[1, 1], label="SAM plants")
    for axis in axes.reshape(-1):
        axis.grid(alpha=0.25)
    _save(figure, path)


def save_evaluation(result: Predictions, scaler, destination: Path, config) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    cell_entropy = normalized_entropy(result.cell_weights)
    plant_entropy = normalized_entropy(result.plant_weights, result.plant_valid)
    report = {
        "model": result.metrics(),
        "hierarchical_attention": result.attention_metrics(),
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
        "residual": result.predictions - result.targets,
        "absolute_error": np.abs(result.predictions - result.targets),
        "plant_count": result.plant_counts,
        "mask_coverage": result.mask_coverages,
        "cell_attention_normalized_entropy": cell_entropy,
        "plant_attention_normalized_entropy": plant_entropy,
        "top_cell_index": result.cell_weights.argmax(axis=1),
        "top_cell_weight": result.cell_weights.max(axis=1),
        "top_plant_index": result.plant_weights.argmax(axis=1),
        "top_plant_weight": result.plant_weights.max(axis=1),
        "input_mode": result.input_modes,
        "source_image_path": result.source_image_paths,
        "processed_image_path": result.processed_image_paths,
        "mask_path": result.mask_paths,
        "feature_cache_path": result.feature_cache_paths,
    }
    for index in range(4):
        columns[f"cell_{index}_weight"] = result.cell_weights[:, index]
    for index in range(result.plant_weights.shape[1]):
        columns[f"plant_{index:02d}_weight"] = result.plant_weights[:, index]
        columns[f"plant_{index:02d}_cell"] = result.plant_cell_indices[:, index]
    pd.DataFrame(columns).to_csv(destination / "predictions.csv", index=False)
    np.savez_compressed(
        destination / config.output.attention_arrays_name,
        filenames=np.asarray(result.filenames),
        cell_weights=result.cell_weights.astype(np.float16),
        plant_weights=result.plant_weights.astype(np.float16),
        plant_valid=result.plant_valid,
        cell_boxes=result.cell_boxes.astype(np.int16),
        plant_boxes=result.plant_boxes.astype(np.int16),
        plant_cell_indices=result.plant_cell_indices.astype(np.int8),
    )
    if config.output.save_plots:
        save_regression_plot(result, destination / "regression.png")
        save_hierarchical_diagnostics(result, destination / "hierarchical_diagnostics.png")
        _prediction_panel(
            result, _representative(result.targets, config.output.example_images),
            destination / "prediction_examples.png", config.output.example_columns,
            "Representative predictions across the target range",
        )
        worst = np.argsort(np.abs(result.predictions - result.targets))[::-1][
            : config.output.example_images
        ]
        _prediction_panel(
            result, worst, destination / "worst_error_examples.png",
            config.output.example_columns, "Largest absolute validation errors",
        )
        save_attention_inspection(
            result, destination / "hierarchical_attention_inspection.png",
            config.output.attention_inspection_images,
        )
    return report


__all__ = ["save_evaluation", "save_history_plot", "save_label_plot"]
