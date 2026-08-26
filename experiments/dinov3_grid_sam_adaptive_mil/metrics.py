"""Adaptive-instance predictions and diagnostics."""

from dataclasses import dataclass

import numpy as np
import torch

from experiments.dinov3_grid_tiled_mil.metrics import mean_baseline, regression_metrics


@dataclass(frozen=True)
class Predictions:
    targets: np.ndarray
    predictions: np.ndarray
    filenames: list[str]
    processed_image_paths: list[str]
    mask_paths: list[str]
    objective_mse: float
    weights: np.ndarray
    valid: np.ndarray
    boxes: np.ndarray
    instance_counts: np.ndarray
    mask_coverages: np.ndarray

    def metrics(self):
        result = regression_metrics(self.targets, self.predictions, self.objective_mse)
        result["normalized_mse"] = self.objective_mse
        return result

    def attention_metrics(self):
        entropy, tops = [], []
        for weights, valid in zip(self.weights, self.valid, strict=True):
            values = weights[valid]
            denominator = np.log(len(values)) if len(values) > 1 else 1.0
            entropy.append(float(-(values * np.log(np.clip(values, 1e-12, 1))).sum() / denominator))
            tops.append(float(values.max()))
        return {
            "mean_normalized_entropy": float(np.mean(entropy)),
            "mean_top_instance_mass": float(np.mean(tops)),
            "mean_instances_per_image": float(self.instance_counts.mean()),
            "minimum_instances_per_image": int(self.instance_counts.min()),
            "maximum_instances_per_image": int(self.instance_counts.max()),
            "mean_mask_coverage": float(self.mask_coverages.mean()),
        }


def predict(model, loader, device, scaler):
    model.eval()
    predicted, actual, weights, valid, boxes, counts, coverages = [], [], [], [], [], [], []
    filenames, image_paths, mask_paths = [], [], []
    squared_error = samples = 0
    with torch.inference_mode():
        for batch in loader:
            context = batch["context_features"].to(device, non_blocking=True)
            instances = batch["instance_features"].to(device, non_blocking=True)
            mask = batch["instance_valid"].to(device, non_blocking=True)
            target = batch["target"].float().to(device)
            output, attention = model(context, instances, mask, return_attention=True)
            squared_error += torch.nn.functional.mse_loss(
                output.float(), target, reduction="sum"
            ).item()
            samples += target.numel()
            predicted.append(output.float().cpu())
            actual.append(target.cpu())
            weights.append(attention.cpu())
            valid.append(mask.cpu())
            boxes.append(batch["instance_boxes"])
            counts.append(batch["instance_count"])
            coverages.append(batch["mask_coverage"])
            filenames.extend(map(str, batch["filename"]))
            image_paths.extend(map(str, batch["processed_image_path"]))
            mask_paths.extend(map(str, batch["mask_path"]))
    return Predictions(
        scaler.inverse(torch.cat(actual).numpy()),
        scaler.inverse(torch.cat(predicted).numpy()),
        filenames,
        image_paths,
        mask_paths,
        squared_error / samples,
        torch.cat(weights).numpy(),
        torch.cat(valid).numpy(),
        torch.cat(boxes).numpy(),
        torch.cat(counts).numpy(),
        torch.cat(coverages).numpy(),
    )


__all__ = ["Predictions", "mean_baseline", "predict"]
