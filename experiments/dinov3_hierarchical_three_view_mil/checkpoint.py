"""Checkpoint schema for the hierarchical weak-to-gold experiment."""

from __future__ import annotations

from typing import Any

from .config import Config
from .features import FEATURE_SCHEMA_VERSION

EXPERIMENT_ID = "dinov3_hierarchical_three_view_mil"
CHECKPOINT_VERSION = 1


def payload(
    *, model, optimizer, scheduler, stage: str, stage_epoch: int, global_epoch: int,
    metrics: dict, scaler, config: Config, feature_dim: int, history: dict,
    best_validation_loss: float, best_validation_mae: float,
    evaluations_without_improvement: int, manifests: dict[str, list[str]], environment: dict,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_format": "hierarchical_three_view_head_only",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": int(feature_dim),
        "stage": stage,
        "stage_epoch": int(stage_epoch),
        "epoch": int(global_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "target_mean": scaler.mean,
        "target_std": scaler.std,
        "target_training_mean": scaler.baseline_mean,
        "targets_normalized": True,
        "history": history,
        "best_validation_loss": float(best_validation_loss),
        "best_validation_mae": float(best_validation_mae),
        "evaluations_without_improvement": int(evaluations_without_improvement),
        "manifests": manifests,
        "config": config.to_dict(),
        "environment": environment,
    }


def validate_for(state: dict, config: Config, feature_dim: int | None = None) -> None:
    if state.get("experiment") != EXPERIMENT_ID:
        raise ValueError(f"Expected {EXPERIMENT_ID!r}, got {state.get('experiment')!r}")
    if state.get("model_state_format") != "hierarchical_three_view_head_only":
        raise ValueError("Checkpoint is not a hierarchical three-view head checkpoint")
    if state.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Checkpoint feature schema is incompatible")
    if not state.get("targets_normalized") or "model_state_dict" not in state:
        raise ValueError("Checkpoint is incomplete or targets were not normalized")
    if state.get("stage") not in {"pretrain", "finetune"}:
        raise ValueError(f"Unknown checkpoint stage: {state.get('stage')!r}")
    if feature_dim is not None and int(state.get("feature_dim", -1)) != int(feature_dim):
        raise ValueError("Checkpoint/cache feature dimensions differ")
    saved = state.get("config", {})
    checks = {
        "data": ("raw_source_folders", "crop_size", "grid_inner_margin_fraction", "cell_inner_margin_fraction"),
        "segmentation": ("model_name", "prompts", "score_threshold", "mask_threshold"),
        "adaptive_crops": (
            "grouping_dilation_px", "context_scale", "minimum_crop_size", "maximum_crop_size",
            "maximum_instances", "minimum_mask_coverage",
        ),
        "features": ("backbone", "processor", "representation"),
        "model": (
            "projection_dim", "attention_hidden_dim", "attention_dropout",
            "attention_temperature", "head_hidden_dim", "dropout",
        ),
    }
    for section, keys in checks.items():
        configured = getattr(config, section)
        for key in keys:
            if key in saved.get(section, {}) and saved[section][key] != getattr(configured, key):
                raise ValueError(f"Checkpoint mismatch for {section}.{key}")


__all__ = ["EXPERIMENT_ID", "payload", "validate_for"]
