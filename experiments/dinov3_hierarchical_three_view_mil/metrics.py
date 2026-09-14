"""Predictions and attention diagnostics for hierarchical three-view MIL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.dinov3_grid_tiled_mil.metrics import mean_baseline, regression_metrics


def normalized_entropy(weights: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if valid is None:
        valid = np.ones_like(values, dtype=bool)
    result = []
    for row, mask in zip(values, valid, strict=True):
        selected = row[np.asarray(mask, dtype=bool)]
        selected = selected / selected.sum()
        entropy = float(-(selected * np.log(np.clip(selected, 1e-12, 1))).sum())
        result.append(entropy / np.log(len(selected)) if len(selected) > 1 else 0.0)
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class Predictions:
    targets: np.ndarray
    predictions: np.ndarray
    objective_mse: float
    filenames: list[str]
    source_image_paths: list[str]
    processed_image_paths: list[str]
    mask_paths: list[str]
    feature_cache_paths: list[str]
    input_modes: list[str]
    cell_weights: np.ndarray
    plant_weights: np.ndarray
    plant_valid: np.ndarray
    cell_boxes: np.ndarray
    plant_boxes: np.ndarray
    plant_cell_indices: np.ndarray
    plant_counts: np.ndarray
    mask_coverages: np.ndarray

    def metrics(self) -> dict:
        result = regression_metrics(self.targets, self.predictions, self.objective_mse)
        result["normalized_mse"] = self.objective_mse
        return result

    def attention_metrics(self) -> dict:
        cell_entropy = normalized_entropy(self.cell_weights)
        plant_entropy = normalized_entropy(self.plant_weights, self.plant_valid)
        return {
            "cells": {
                "mean_normalized_entropy": float(cell_entropy.mean()),
                "minimum_normalized_entropy": float(cell_entropy.min()),
                "maximum_normalized_entropy": float(cell_entropy.max()),
                "mean_top_mass": float(self.cell_weights.max(axis=1).mean()),
            },
            "plants": {
                "mean_normalized_entropy": float(plant_entropy.mean()),
                "minimum_normalized_entropy": float(plant_entropy.min()),
                "maximum_normalized_entropy": float(plant_entropy.max()),
                "mean_top_mass": float(self.plant_weights.max(axis=1).mean()),
                "mean_instances_per_image": float(self.plant_counts.mean()),
                "minimum_instances_per_image": int(self.plant_counts.min()),
                "maximum_instances_per_image": int(self.plant_counts.max()),
                "mean_mask_coverage": float(self.mask_coverages.mean()),
            },
        }


def predict(model, loader, device, scaler) -> Predictions:
    model.eval()
    predictions, targets, cell_weights, plant_weights = [], [], [], []
    valid, cell_boxes, plant_boxes, plant_cells, counts, coverages = [], [], [], [], [], []
    filenames, sources, processed, masks, caches, modes = [], [], [], [], [], []
    squared_error = samples = 0
    with torch.inference_mode():
        for batch in loader:
            global_feature = batch["global_feature"].to(device, non_blocking=True)
            cells = batch["cell_features"].to(device, non_blocking=True)
            plants = batch["plant_features"].to(device, non_blocking=True)
            plant_valid = batch["plant_valid"].to(device, non_blocking=True)
            plant_cell_indices = batch["plant_cell_indices"].to(device, non_blocking=True)
            target = batch["target"].float().to(device)
            output, attention = model(
                global_feature,
                cells,
                plants,
                plant_valid,
                plant_cell_indices,
                return_attention=True,
            )
            squared_error += torch.nn.functional.mse_loss(
                output.float(), target, reduction="sum"
            ).item()
            samples += target.numel()
            predictions.append(output.float().cpu())
            targets.append(target.cpu())
            cell_weights.append(attention["cell_weights"].cpu())
            plant_weights.append(attention["plant_weights"].cpu())
            valid.append(plant_valid.cpu())
            cell_boxes.append(batch["cell_boxes"])
            plant_boxes.append(batch["plant_boxes"])
            plant_cells.append(batch["plant_cell_indices"])
            counts.append(batch["plant_count"])
            coverages.append(batch["mask_coverage"])
            filenames.extend(map(str, batch["filename"]))
            sources.extend(map(str, batch["source_image_path"]))
            processed.extend(map(str, batch["processed_image_path"]))
            masks.extend(map(str, batch["mask_path"]))
            caches.extend(map(str, batch["feature_cache_path"]))
            modes.extend(map(str, batch["input_mode"]))
    if not samples:
        raise ValueError("Cannot evaluate an empty loader")
    combined_plants = torch.cat(plant_weights).numpy()
    combined_valid = torch.cat(valid).numpy()
    if not np.allclose((combined_plants * combined_valid).sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Valid plant attention weights do not sum to one")
    return Predictions(
        targets=scaler.inverse(torch.cat(targets).numpy()),
        predictions=scaler.inverse(torch.cat(predictions).numpy()),
        objective_mse=squared_error / samples,
        filenames=filenames,
        source_image_paths=sources,
        processed_image_paths=processed,
        mask_paths=masks,
        feature_cache_paths=caches,
        input_modes=modes,
        cell_weights=torch.cat(cell_weights).numpy(),
        plant_weights=combined_plants,
        plant_valid=combined_valid,
        cell_boxes=torch.cat(cell_boxes).numpy(),
        plant_boxes=torch.cat(plant_boxes).numpy(),
        plant_cell_indices=torch.cat(plant_cells).numpy(),
        plant_counts=torch.cat(counts).numpy(),
        mask_coverages=torch.cat(coverages).numpy(),
    )


__all__ = ["Predictions", "mean_baseline", "normalized_entropy", "predict"]
