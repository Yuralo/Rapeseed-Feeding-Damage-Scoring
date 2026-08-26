"""Predictions and diagnostics for fixed-tile plus adaptive hybrid MIL."""

from dataclasses import dataclass

import numpy as np
import torch

from experiments.dinov3_grid_multiscale_tiled_mil.metrics import scale_attention_metrics
from experiments.dinov3_grid_sam_adaptive_mil.metrics import Predictions as AdaptivePredictions
from experiments.dinov3_grid_tiled_mil.metrics import mean_baseline


@dataclass(frozen=True)
class Predictions(AdaptivePredictions):
    fine_feature_cache_paths: list[str]
    fine_weights: np.ndarray
    fine_boxes: np.ndarray

    def attention_metrics(self):
        adaptive = super().attention_metrics()
        return {
            "fine_4x4": scale_attention_metrics(self.fine_weights),
            "adaptive": adaptive,
        }


def predict(model, loader, device, scaler):
    model.eval()
    predicted, actual = [], []
    fine_weights, fine_boxes = [], []
    plant_weights, valid, plant_boxes, pixels = [], [], [], []
    counts, coverages = [], []
    filenames, source_paths, processed_paths, mask_paths = [], [], [], []
    context_cache_paths, fine_cache_paths, adaptive_cache_paths = [], [], []
    squared_error, samples = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            context = batch["context_features"].to(device, non_blocking=True)
            fine = batch["fine_features"].to(device, non_blocking=True)
            instances = batch["instance_features"].to(device, non_blocking=True)
            mask = batch["instance_valid"].to(device, non_blocking=True)
            target = batch["target"].float().to(device, non_blocking=True).reshape(-1)
            output, fine_attention, plant_attention = model(
                context, fine, instances, mask, return_attention=True
            )
            output = output.reshape(-1)
            squared_error += torch.nn.functional.mse_loss(
                output.float(), target, reduction="sum"
            ).item()
            samples += target.numel()
            predicted.append(output.float().cpu())
            actual.append(target.cpu())
            fine_weights.append(fine_attention.float().cpu())
            fine_boxes.append(batch["fine_tile_boxes"].cpu())
            plant_weights.append(plant_attention.float().cpu())
            valid.append(mask.cpu())
            plant_boxes.append(batch["instance_boxes"].cpu())
            pixels.append(batch["instance_foreground_pixels"].cpu())
            counts.append(batch["instance_count"].cpu())
            coverages.append(batch["mask_coverage"].cpu())
            filenames.extend(map(str, batch["filename"]))
            source_paths.extend(map(str, batch["source_image_path"]))
            processed_paths.extend(map(str, batch["processed_image_path"]))
            mask_paths.extend(map(str, batch["mask_path"]))
            context_cache_paths.extend(map(str, batch["context_feature_cache_path"]))
            fine_cache_paths.extend(map(str, batch["fine_feature_cache_path"]))
            adaptive_cache_paths.extend(map(str, batch["adaptive_feature_cache_path"]))
    if not samples:
        raise ValueError("Cannot evaluate an empty loader")
    fine_attention = torch.cat(fine_weights).numpy()
    plant_attention = torch.cat(plant_weights).numpy()
    combined_valid = torch.cat(valid).numpy()
    if not np.allclose(fine_attention.sum(1), 1.0, atol=1e-5):
        raise RuntimeError("4x4 attention weights do not sum to one")
    if not np.allclose((plant_attention * combined_valid).sum(1), 1.0, atol=1e-5):
        raise RuntimeError("Valid plant attention weights do not sum to one")
    return Predictions(
        targets=scaler.inverse(torch.cat(actual).numpy()),
        predictions=scaler.inverse(torch.cat(predicted).numpy()),
        filenames=filenames,
        source_image_paths=source_paths,
        processed_image_paths=processed_paths,
        mask_paths=mask_paths,
        context_feature_cache_paths=context_cache_paths,
        adaptive_feature_cache_paths=adaptive_cache_paths,
        objective_mse=squared_error / samples,
        weights=plant_attention,
        valid=combined_valid,
        boxes=torch.cat(plant_boxes).numpy(),
        foreground_pixels=torch.cat(pixels).numpy(),
        instance_counts=torch.cat(counts).numpy(),
        mask_coverages=torch.cat(coverages).numpy(),
        fine_feature_cache_paths=fine_cache_paths,
        fine_weights=fine_attention,
        fine_boxes=torch.cat(fine_boxes).numpy(),
    )


__all__ = ["Predictions", "mean_baseline", "predict"]

