"""Head-only checkpoint schema for tri-scale cached-feature MIL."""

from __future__ import annotations

from typing import Any

from experiments.dinov3_grid_tiled_mil.features import FEATURE_CACHE_SCHEMA_VERSION

from .config import Config

EXPERIMENT_ID = "dinov3_grid_triscale_tiled_mil"


def payload(
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    scaler,
    config: Config,
    feature_dim: int,
    training_filenames: list[str],
    validation_filenames: list[str],
    history: dict[str, Any],
    best_validation_loss: float,
    best_validation_mae: float,
    training_state: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "checkpoint_version": 1,
        "model_state_format": "triscale_mil_head_only",
        "feature_cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "feature_dim": int(feature_dim),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "val_loss": metrics["objective_mse"],
        "target_mean": scaler.mean,
        "target_std": scaler.std,
        "target_training_mean": scaler.baseline_mean,
        "targets_normalized": True,
        "training_filenames": training_filenames,
        "validation_filenames": validation_filenames,
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_validation_mae": best_validation_mae,
        "training_state": training_state,
        "config": config.to_dict(),
        "environment": environment,
    }


def validate_for(state: dict[str, Any], config: Config, feature_dim: int | None = None) -> None:
    if state.get("experiment") != EXPERIMENT_ID:
        raise ValueError(
            f"Expected a {EXPERIMENT_ID!r} checkpoint, got {state.get('experiment')!r}"
        )
    if state.get("model_state_format") != "triscale_mil_head_only":
        raise ValueError("Checkpoint is not a tri-scale MIL head-only checkpoint")
    if "model_state_dict" not in state or not state.get("targets_normalized", True):
        raise ValueError("Checkpoint is incomplete or does not use normalized targets")
    if state.get("feature_cache_schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError("Checkpoint feature-cache schema is incompatible")
    if feature_dim is not None and int(state.get("feature_dim", -1)) != int(feature_dim):
        raise ValueError(
            f"Checkpoint feature dimension {state.get('feature_dim')} != cache {feature_dim}"
        )
    saved = state.get("config", {})
    checks = {
        "data": ("grid_crop_size", "grid_inner_margin_fraction"),
        "features": ("backbone", "processor", "representation"),
        "context": ("rows", "columns", "overlap_fraction", "include_global_view"),
        "regional": ("rows", "columns", "overlap_fraction", "include_global_view"),
        "local": ("rows", "columns", "overlap_fraction", "include_global_view"),
        "model": (
            "projection_dim",
            "attention_hidden_dim",
            "attention_dropout",
            "attention_temperature",
            "head_hidden_dim",
            "dropout",
        ),
    }
    for section, keys in checks.items():
        configured = getattr(config, section)
        for key in keys:
            if key in saved.get(section, {}) and saved[section][key] != getattr(configured, key):
                raise ValueError(
                    f"Checkpoint mismatch for {section}.{key}: "
                    f"{saved[section][key]!r} != {getattr(configured, key)!r}"
                )
