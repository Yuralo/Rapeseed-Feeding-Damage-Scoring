"""Metrics, filename-aware predictions, and per-image SAM-fusion diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image

from rapeseed_damage.artifacts import write_json

from .config import Config
from .data import TargetScaler
from .metrics import Predictions, attention_entropy, mean_baseline, top_fraction_mass
from .segmentation import make_masked_image


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
        history["val_original_attention_entropy"],
        marker="o",
        label="original",
    )
    axes[2].plot(
        history["val_epochs"],
        history["val_masked_attention_entropy"],
        marker="o",
        label="masked",
    )
    axes[2].set(xlabel="Epoch", ylabel="Normalized entropy", title="Patch attention")
    axes[2].legend()
    for index, label in enumerate(("original", "masked", "binary mask")):
        axes[3].plot(
            history["val_epochs"],
            [weights[index] for weights in history["val_fusion_weights"]],
            marker="o",
            label=label,
        )
    axes[3].set(xlabel="Epoch", ylabel="Mean weight", title="Fusion gates")
    axes[3].legend()
    for group in sorted({name for values in history["learning_rates"] for name in values}):
        axes[4].plot(
            range(1, len(history["learning_rates"]) + 1),
            [values.get(group, np.nan) for values in history["learning_rates"]],
            label=group,
        )
    axes[4].set(xlabel="Epoch", ylabel="Learning rate", title="End-of-epoch LR", yscale="log")
    axes[4].legend(fontsize=8)
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
    lower = min(targets.min(), predictions.min())
    upper = max(targets.max(), predictions.max())
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(targets, predictions, alpha=0.65, edgecolors="none")
    axes[0].plot([lower, upper], [lower, upper], "--", color="red")
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="SAM-fusion regression")
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


def save_prediction_examples(
    result: Predictions,
    path: Path,
    count: int,
    columns: int,
) -> None:
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
        axes[index].axis("off")
        axes[index].set_title(
            f"{Path(result.filenames[index]).name}\n"
            f"Target {result.targets[index]:.2f} | Pred {result.predictions[index]:.2f}\n"
            f"Error {error:+.2f} | SAM delta {result.fusion_deltas[index]:+.2f}"
        )
    for axis in axes[count:]:
        axis.axis("off")
    _save_figure(figure, path)


def _attention_grid(result: Predictions, index: int, *, masked: bool) -> np.ndarray:
    weights = (
        result.masked_attention_weights[index]
        if masked
        else result.original_attention_weights[index]
    )
    rows = int(result.attention_grid_rows[index])
    columns = int(result.attention_grid_columns[index])
    return weights.reshape(rows, columns) * weights.size


def save_sam_fusion_inspections(
    result: Predictions,
    destination: Path,
    config: Config,
) -> None:
    count = min(config.output.attention_inspection_images, len(result.targets))
    if not count:
        return
    destination.mkdir(parents=True, exist_ok=True)
    ordered = np.argsort(result.targets)
    indices = ordered[np.linspace(0, len(ordered) - 1, count).round().astype(int)]
    norm = TwoSlopeNorm(
        vmin=config.output.attention_ratio_min,
        vcenter=1.0,
        vmax=config.output.attention_ratio_max,
    )
    for position, index_value in enumerate(indices, start=1):
        index = int(index_value)
        with Image.open(result.processed_image_paths[index]) as opened:
            original = opened.convert("RGB").copy()
        with Image.open(result.mask_paths[index]) as opened:
            mask = opened.convert("L").copy()
        masked = make_masked_image(
            original,
            mask,
            background_value=config.segmentation.background_value,
        )
        width, height = original.size
        figure, axes = plt.subplots(1, 5, figsize=(19, 4), squeeze=False)
        axes = axes[0]
        axes[0].imshow(original)
        axes[0].set_title(
            f"{result.filenames[index]}\nTarget {result.targets[index]:.2f} | "
            f"Pred {result.predictions[index]:.2f}"
        )
        axes[1].imshow(original)
        mask_array = np.asarray(mask)
        axes[1].imshow(
            np.ma.masked_where(mask_array < 128, mask_array),
            cmap="spring",
            alpha=0.48,
            vmin=0,
            vmax=255,
        )
        axes[1].set_title(
            f"Binary mask\n{100 * result.mask_foreground_fractions[index]:.2f}% foreground"
        )
        axes[2].imshow(masked)
        gates = result.fusion_weights[index]
        axes[2].set_title(
            "Masked input\n"
            f"gates O/M/B={gates[0]:.2f}/{gates[1]:.2f}/{gates[2]:.2f}"
        )
        axes[3].imshow(original)
        axes[3].imshow(
            _attention_grid(result, index, masked=False),
            cmap="coolwarm",
            norm=norm,
            alpha=0.52,
            interpolation="bilinear",
            extent=(0, width, height, 0),
        )
        axes[3].set_title("Original-branch patch attention")
        axes[4].imshow(masked)
        axes[4].imshow(
            _attention_grid(result, index, masked=True),
            cmap="coolwarm",
            norm=norm,
            alpha=0.52,
            interpolation="bilinear",
            extent=(0, width, height, 0),
        )
        axes[4].set_title(
            f"Masked-branch attention\nSAM delta {result.fusion_deltas[index]:+.2f}"
        )
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in Path(result.filenames[index]).stem
        )
        figure.savefig(
            destination / f"{position:02d}_{safe_name}.jpg",
            format="jpeg",
            dpi=120,
            bbox_inches="tight",
            pil_kwargs={"quality": 88, "optimize": True},
        )
        plt.close(figure)
        original.close()
        mask.close()
        masked.close()


def save_evaluation(
    result: Predictions,
    scaler: TargetScaler,
    destination: Path,
    config: Config,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    original_entropy = attention_entropy(result.original_attention_weights)
    masked_entropy = attention_entropy(result.masked_attention_weights)
    report = {
        "model": result.metrics(),
        "diagnostics": result.diagnostics(),
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
            "base_prediction": result.base_predictions,
            "fusion_delta": result.fusion_deltas,
            "residual": result.predictions - result.targets,
            "mask_foreground_fraction": result.mask_foreground_fractions,
            "fusion_original_weight": result.fusion_weights[:, 0],
            "fusion_masked_weight": result.fusion_weights[:, 1],
            "fusion_binary_mask_weight": result.fusion_weights[:, 2],
            "original_attention_entropy": original_entropy,
            "masked_attention_entropy": masked_entropy,
            "original_attention_top_10_mass": top_fraction_mass(
                result.original_attention_weights
            ),
            "masked_attention_top_10_mass": top_fraction_mass(
                result.masked_attention_weights
            ),
            "source_image_path": result.source_image_paths,
            "processed_image_path": result.processed_image_paths,
            "mask_path": result.mask_paths,
        }
    ).to_csv(destination / "predictions.csv", index=False)
    np.savez_compressed(
        destination / config.output.attention_arrays_name,
        filenames=np.asarray(result.filenames),
        original_attention=result.original_attention_weights.astype(np.float16),
        masked_attention=result.masked_attention_weights.astype(np.float16),
        fusion_weights=result.fusion_weights.astype(np.float16),
        mask_foreground_fraction=result.mask_foreground_fractions.astype(np.float32),
        grid_rows=result.attention_grid_rows,
        grid_columns=result.attention_grid_columns,
    )
    if config.output.save_plots:
        save_regression_plot(result, destination / "regression.png")
        save_prediction_examples(
            result,
            destination / "prediction_examples.png",
            config.output.example_images,
            config.output.example_columns,
        )
        save_sam_fusion_inspections(
            result,
            destination / "sam_fusion_inspection",
            config,
        )
    return report
