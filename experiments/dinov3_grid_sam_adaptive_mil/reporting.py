"""Adaptive crop attention reports."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from experiments.dinov3_grid_tiled_mil.reporting import save_label_plot
from rapeseed_damage.artifacts import write_json

from .metrics import mean_baseline


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_history_plot(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(history["train_loss"])
    axes[0].plot(history["val_epochs"], history["val_loss"])
    axes[1].plot(history["val_epochs"], history["val_mae"], label="MAE")
    axes[1].plot(history["val_epochs"], history["val_r2"], label="R²")
    axes[1].legend()
    axes[2].plot(history["val_epochs"], history["val_attention_entropy"], label="entropy")
    axes[2].plot(history["val_epochs"], history["val_top_mass"], label="top mass")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(fig, path)


def save_evaluation(result, scaler, destination: Path, config):
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "model": result.metrics(),
        "adaptive_attention": result.attention_metrics(),
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
        "instance_count": result.instance_counts,
        "mask_coverage": result.mask_coverages,
        "processed_image_path": result.processed_image_paths,
        "mask_path": result.mask_paths,
    }
    for index in range(result.weights.shape[1]):
        columns[f"instance_{index:02d}_weight"] = result.weights[:, index]
    pd.DataFrame(columns).to_csv(destination / "predictions.csv", index=False)
    np.savez_compressed(
        destination / config.output.attention_arrays_name,
        filenames=np.asarray(result.filenames),
        weights=result.weights.astype(np.float16),
        valid=result.valid,
        boxes=result.boxes.astype(np.int16),
    )
    if config.output.save_plots:
        count = min(config.output.example_images, len(result.targets))
        rows = math.ceil(count / config.output.example_columns)
        fig, axes = plt.subplots(
            rows,
            config.output.example_columns,
            figsize=(4 * config.output.example_columns, 4 * rows),
        )
        axes = np.atleast_1d(axes).reshape(-1)
        for i in range(count):
            with Image.open(result.processed_image_paths[i]) as image:
                axes[i].imshow(image.convert("RGB"))
            axes[i].set_title(
                f"{result.filenames[i]}\nTarget {result.targets[i]:.2f} | Pred {result.predictions[i]:.2f}"
            )
            axes[i].axis("off")
        for axis in axes[count:]:
            axis.axis("off")
        _save(fig, destination / "prediction_examples.png")
        indices = np.argsort(result.targets)[
            np.linspace(
                0,
                len(result.targets) - 1,
                min(config.output.attention_inspection_images, len(result.targets)),
            )
            .round()
            .astype(int)
        ]
        fig, axes = plt.subplots(len(indices), 3, figsize=(14, 4.5 * len(indices)), squeeze=False)
        for row, i in enumerate(indices):
            with Image.open(result.processed_image_paths[i]) as image:
                rgb = image.convert("RGB").copy()
            with Image.open(result.mask_paths[i]) as mask:
                mask_array = np.asarray(mask.convert("L"))
            axes[row, 0].imshow(rgb)
            axes[row, 1].imshow(rgb)
            axes[row, 1].imshow(mask_array, cmap="Greens", alpha=0.4)
            valid = np.flatnonzero(result.valid[i])
            top = valid[np.argmax(result.weights[i, valid])]
            for j in valid:
                x0, y0, x1, y1 = result.boxes[i, j]
                axes[row, 0].add_patch(
                    Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        color="cyan" if j == top else "white",
                        linewidth=2 if j == top else 1,
                    )
                )
            x0, y0, x1, y1 = result.boxes[i, top]
            axes[row, 2].imshow(rgb.crop((x0, y0, x1, y1)))
            axes[row, 0].set_title(
                f"{result.filenames[i]} | target {result.targets[i]:.2f} | pred {result.predictions[i]:.2f}"
            )
            axes[row, 1].set_title(f"SAM mask | {len(valid)} instances")
            axes[row, 2].set_title(f"Top plant crop {top} | mass {result.weights[i, top]:.3f}")
            for axis in axes[row]:
                axis.axis("off")
        _save(fig, destination / "adaptive_attention_inspection.png")
    return report


__all__ = ["save_evaluation", "save_history_plot", "save_label_plot"]
