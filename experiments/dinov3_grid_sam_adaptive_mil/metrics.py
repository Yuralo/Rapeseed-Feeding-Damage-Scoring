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
    source_image_paths: list[str]
    processed_image_paths: list[str]
    mask_paths: list[str]
    context_feature_cache_paths: list[str]
    adaptive_feature_cache_paths: list[str]
    objective_mse: float
    weights: np.ndarray
    valid: np.ndarray
    boxes: np.ndarray
    foreground_pixels: np.ndarray
    instance_counts: np.ndarray
    mask_coverages: np.ndarray

    def metrics(self):
        result = regression_metrics(self.targets, self.predictions, self.objective_mse)
        result["normalized_mse"] = self.objective_mse
        return result

    def attention_metrics(self):
        statistics = self.attention_statistics()
        entropy = statistics["normalized_entropy"]
        tops = statistics["top_weight"]
        return {
            "mean_normalized_entropy": float(np.mean(entropy)),
            "minimum_normalized_entropy": float(np.min(entropy)),
            "maximum_normalized_entropy": float(np.max(entropy)),
            "mean_top_instance_mass": float(np.mean(tops)),
            "minimum_top_instance_mass": float(np.min(tops)),
            "maximum_top_instance_mass": float(np.max(tops)),
            "mean_instances_per_image": float(self.instance_counts.mean()),
            "median_instances_per_image": float(np.median(self.instance_counts)),
            "minimum_instances_per_image": int(self.instance_counts.min()),
            "maximum_instances_per_image": int(self.instance_counts.max()),
            "mean_mask_coverage": float(self.mask_coverages.mean()),
            "minimum_mask_coverage": float(self.mask_coverages.min()),
            "maximum_mask_coverage": float(self.mask_coverages.max()),
        }

    def attention_statistics(self):
        entropy, raw_entropy, tops, top_indices, effective = [], [], [], [], []
        for weights, valid in zip(self.weights, self.valid, strict=True):
            indices = np.flatnonzero(valid)
            if not len(indices):
                raise ValueError("Every image must have at least one valid adaptive instance")
            values = np.asarray(weights[valid], dtype=np.float64)
            if values.sum() <= 0:
                raise ValueError("Valid adaptive-instance weights must have positive mass")
            values = values / values.sum()
            sample_entropy = float(-(values * np.log(np.clip(values, 1e-12, 1))).sum())
            normalized = sample_entropy / np.log(len(values)) if len(values) > 1 else 0.0
            local_top = int(values.argmax())
            entropy.append(normalized)
            raw_entropy.append(sample_entropy)
            tops.append(float(values[local_top]))
            top_indices.append(int(indices[local_top]))
            effective.append(float(np.exp(sample_entropy)))
        return {
            "normalized_entropy": np.asarray(entropy, dtype=np.float64),
            "raw_entropy": np.asarray(raw_entropy, dtype=np.float64),
            "top_weight": np.asarray(tops, dtype=np.float64),
            "top_index": np.asarray(top_indices, dtype=np.int64),
            "effective_instance_count": np.asarray(effective, dtype=np.float64),
        }


def predict(model, loader, device, scaler):
    model.eval()
    predicted, actual, weights, valid, boxes, pixels, counts, coverages = (
        [], [], [], [], [], [], [], []
    )
    filenames, source_paths, image_paths, mask_paths = [], [], [], []
    context_cache_paths, adaptive_cache_paths = [], []
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
            pixels.append(batch["instance_foreground_pixels"])
            counts.append(batch["instance_count"])
            coverages.append(batch["mask_coverage"])
            filenames.extend(map(str, batch["filename"]))
            source_paths.extend(map(str, batch["source_image_path"]))
            image_paths.extend(map(str, batch["processed_image_path"]))
            mask_paths.extend(map(str, batch["mask_path"]))
            context_cache_paths.extend(map(str, batch["context_feature_cache_path"]))
            adaptive_cache_paths.extend(map(str, batch["adaptive_feature_cache_path"]))
    if not samples:
        raise ValueError("Cannot evaluate an empty loader")
    combined_weights = torch.cat(weights).numpy()
    combined_valid = torch.cat(valid).numpy()
    if not np.allclose((combined_weights * combined_valid).sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Valid adaptive-instance attention weights do not sum to one")
    return Predictions(
        targets=scaler.inverse(torch.cat(actual).numpy()),
        predictions=scaler.inverse(torch.cat(predicted).numpy()),
        filenames=filenames,
        source_image_paths=source_paths,
        processed_image_paths=image_paths,
        mask_paths=mask_paths,
        context_feature_cache_paths=context_cache_paths,
        adaptive_feature_cache_paths=adaptive_cache_paths,
        objective_mse=squared_error / samples,
        weights=combined_weights,
        valid=combined_valid,
        boxes=torch.cat(boxes).numpy(),
        foreground_pixels=torch.cat(pixels).numpy(),
        instance_counts=torch.cat(counts).numpy(),
        mask_coverages=torch.cat(coverages).numpy(),
    )


__all__ = ["Predictions", "mean_baseline", "predict"]
