"""Paired baseline/new-model metrics and inspectable plant-patch evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from experiments.dinov3_hierarchical_three_view_mil.metrics import predict as predict_base
from experiments.dinov3_hierarchical_three_view_mil.reporting import save_evaluation
from rapeseed_damage.artifacts import write_json


@dataclass
class DamageResult:
    result: object
    base_predictions: np.ndarray
    residuals: np.ndarray
    patch_boxes: np.ndarray
    patch_valid: np.ndarray
    patch_weights: np.ndarray
    patch_evidence: np.ndarray
    patch_coverage: np.ndarray


def predict(model, loader, device, scaler) -> DamageResult:
    model.eval()
    base = predict_base(model.base, loader, device, scaler)
    outputs, targets, names = [], [], []
    boxes, valid, weights, evidence, coverage = [], [], [], [], []
    normalized_squared = samples = 0
    with torch.inference_mode():
        for batch in loader:
            moved = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output, attention = model(moved, return_attention=True)
            target = moved["target"].float()
            normalized_squared += torch.square(output.float() - target).sum().item()
            samples += target.numel()
            outputs.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            names.extend(map(str, batch["filename"]))
            boxes.append(batch["patch_boxes"].numpy())
            valid.append(batch["patch_valid"].numpy())
            coverage.append(batch["patch_foreground_fraction"].numpy())
            weights.append(attention["patch_weights"].cpu().numpy())
            evidence.append(attention["patch_evidence"].cpu().numpy())
    if not samples or names != base.filenames:
        raise ValueError("Patch and base predictions are empty or out of order")
    final = scaler.inverse(np.concatenate(outputs))
    if not np.allclose(scaler.inverse(np.concatenate(targets)), base.targets, atol=1e-4):
        raise ValueError("Baseline and damage targets differ")
    result = replace(base, predictions=final, objective_mse=normalized_squared / samples)
    return DamageResult(
        result=result,
        base_predictions=base.predictions,
        residuals=final - base.predictions,
        patch_boxes=np.concatenate(boxes),
        patch_valid=np.concatenate(valid),
        patch_weights=np.concatenate(weights),
        patch_evidence=np.concatenate(evidence),
        patch_coverage=np.concatenate(coverage),
    )


def _save_patch_csv(value: DamageResult, destination: Path) -> None:
    rows = []
    for index, name in enumerate(value.result.filenames):
        for plant, patch in np.argwhere(value.patch_valid[index]):
            x0, y0, x1, y1 = value.patch_boxes[index, plant, patch]
            rows.append({
                "filename": name, "target": float(value.result.targets[index]),
                "base_prediction": float(value.base_predictions[index]),
                "prediction": float(value.result.predictions[index]),
                "plant_index": int(plant), "patch_index": int(patch),
                "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
                "patch_weight": float(value.patch_weights[index, plant, patch]),
                "candidate_local_evidence": float(value.patch_evidence[index, plant, patch]),
                "sam_foreground_fraction": float(value.patch_coverage[index, plant, patch]),
            })
    pd.DataFrame(rows).to_csv(destination / "patch_evidence.csv", index=False)


def _patch_inspection(value: DamageResult, destination: Path, count: int) -> None:
    if count <= 0:
        return
    targets = value.result.targets
    selected = np.argsort(targets)[np.linspace(0, len(targets) - 1, min(count, len(targets))).round().astype(int)]
    fig, axes = plt.subplots(len(selected), 2, figsize=(11, 4.5 * len(selected)), squeeze=False)
    for row, index in enumerate(selected):
        with Image.open(value.result.processed_image_paths[index]) as handle:
            image = handle.convert("RGB")
            axes[row, 0].imshow(image)
            axes[row, 1].imshow(image)
        magnitudes = np.abs(value.patch_evidence[index] * value.patch_weights[index])
        peak = max(float(magnitudes[value.patch_valid[index]].max(initial=0)), 1e-6)
        for plant, patch in np.argwhere(value.patch_valid[index]):
            x0, y0, x1, y1 = value.patch_boxes[index, plant, patch]
            strength = float(magnitudes[plant, patch] / peak)
            axes[row, 1].add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor=(1, 0.15, 0, 0.38 * strength),
                edgecolor=(1, 0.1, 0, 0.35 + 0.65 * strength), linewidth=1.5,
            ))
        name = Path(value.result.filenames[index]).name
        axes[row, 0].set_title(f"{name} | target {targets[index]:.2f}\n"
                               f"base {value.base_predictions[index]:.2f}")
        axes[row, 1].set_title(f"Candidate patch evidence | final {value.result.predictions[index]:.2f}\n"
                               f"correction {value.residuals[index]:+.2f}")
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle("Weakly supervised local evidence — not verified lesion segmentation")
    fig.tight_layout()
    fig.savefig(destination / "patch_evidence_inspection.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def save(value: DamageResult, scaler, destination: Path, config) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    report = save_evaluation(value.result, scaler, destination, config.base)
    base_error = np.abs(value.base_predictions - value.result.targets)
    new_error = np.abs(value.result.predictions - value.result.targets)
    report["paired_comparison"] = {
        "base_mae": float(base_error.mean()),
        "final_mae": float(new_error.mean()),
        "mae_improvement": float(base_error.mean() - new_error.mean()),
        "base_rmse": float(np.sqrt(np.mean((value.base_predictions - value.result.targets) ** 2))),
        "final_rmse": float(np.sqrt(np.mean((value.result.predictions - value.result.targets) ** 2))),
        "improved_images": int(np.sum(new_error < base_error)),
        "worsened_images": int(np.sum(new_error > base_error)),
        "mean_absolute_correction": float(np.abs(value.residuals).mean()),
    }
    ranges = {
        "0_to_2_5": value.result.targets <= 2.5,
        "over_2_5_to_7_5": (value.result.targets > 2.5) & (value.result.targets <= 7.5),
        "over_7_5_to_15": (value.result.targets > 7.5) & (value.result.targets <= 15),
        "over_15": value.result.targets > 15,
    }
    report["paired_target_ranges"] = {
        name: {
            "samples": int(selected.sum()),
            "base_mae": float(base_error[selected].mean()) if selected.any() else None,
            "final_mae": float(new_error[selected].mean()) if selected.any() else None,
            "mean_correction": float(value.residuals[selected].mean()) if selected.any() else None,
        }
        for name, selected in ranges.items()
    }
    report["interpretation_warning"] = (
        "Patch evidence is weakly supervised by image scores and must not be called a lesion mask "
        "without pixel-level validation."
    )
    predictions = pd.read_csv(destination / "predictions.csv")
    predictions.insert(3, "base_prediction", value.base_predictions)
    predictions.insert(4, "local_correction", value.residuals)
    predictions.insert(5, "base_absolute_error", base_error)
    predictions.to_csv(destination / "predictions.csv", index=False)
    _save_patch_csv(value, destination)
    if config.base.output.save_plots:
        _patch_inspection(value, destination, config.inspection_images)
    write_json(destination / "metrics.json", report)
    return report
