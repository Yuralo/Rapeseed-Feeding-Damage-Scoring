"""Checkpoint schema for the fixed-tile plus adaptive hybrid head."""

from experiments.dinov3_grid_sam_adaptive_mil.features import (
    ADAPTIVE_FEATURE_SCHEMA_VERSION,
)
from experiments.dinov3_grid_tiled_mil.features import FEATURE_CACHE_SCHEMA_VERSION

EXPERIMENT_ID = "dinov3_grid_4x4_sam_adaptive_hybrid_mil"


def validate_for(state, config, feature_dim=None):
    if (
        state.get("experiment") != EXPERIMENT_ID
        or state.get("model_state_format") != "hybrid_mil_head_only"
    ):
        raise ValueError("Incompatible fixed-tile plus adaptive hybrid checkpoint")
    if state.get("adaptive_feature_schema_version") != ADAPTIVE_FEATURE_SCHEMA_VERSION:
        raise ValueError("Checkpoint adaptive feature-cache schema mismatch")
    if state.get("tiled_feature_schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError("Checkpoint tiled feature-cache schema mismatch")
    if feature_dim is not None and int(state.get("feature_dim", -1)) != int(feature_dim):
        raise ValueError("Checkpoint feature dimension mismatch")
    if not state.get("targets_normalized", False):
        raise ValueError("Hybrid checkpoint must use normalized targets")
    saved = state.get("config", {})
    for section in ("context", "fine", "adaptive_crops", "features", "model"):
        configured = getattr(config, section)
        for key, value in saved.get(section, {}).items():
            if hasattr(configured, key) and getattr(configured, key) != value:
                raise ValueError(f"Checkpoint mismatch for {section}.{key}")


def payload(
    *,
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
    scaler,
    config,
    feature_dim,
    training_filenames,
    validation_filenames,
    history,
    best_validation_loss,
    best_validation_mae,
    training_state,
    environment,
):
    return {
        "experiment": EXPERIMENT_ID,
        "checkpoint_version": 1,
        "model_state_format": "hybrid_mil_head_only",
        "adaptive_feature_schema_version": ADAPTIVE_FEATURE_SCHEMA_VERSION,
        "tiled_feature_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
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


__all__ = ["payload", "validate_for"]
