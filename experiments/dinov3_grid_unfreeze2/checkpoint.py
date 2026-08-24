"""Checkpoint contents and compatibility rules for this experiment."""

from __future__ import annotations

from typing import Any

from .config import Config
from .data import TargetScaler
from .preprocessing import CACHE_SCHEMA_VERSION

EXPERIMENT_ID = "dinov3_grid_unfreeze2"


def payload(
    *,
    model,
    optimizer,
    scheduler,
    grad_scaler,
    epoch: int,
    metrics: dict[str, float],
    scaler: TargetScaler,
    config: Config,
    training_filenames: list[str],
    validation_filenames: list[str],
    history: dict[str, Any],
    best_validation_loss: float,
    training_state: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "checkpoint_version": 3,
        "grid_cache_schema_version": CACHE_SCHEMA_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "grad_scaler_state_dict": grad_scaler.state_dict() if grad_scaler else None,
        "metrics": metrics,
        "val_loss": metrics["objective_mse"],
        "val_mae": metrics["mae"],
        "val_r2": metrics["r2"],
        "target_mean": scaler.mean,
        "target_std": scaler.std,
        "target_training_mean": scaler.baseline_mean,
        "targets_normalized": scaler.enabled,
        "training_filenames": training_filenames,
        "validation_filenames": validation_filenames,
        "history": history,
        "best_validation_loss": best_validation_loss,
        "training_state": training_state,
        "config": config.to_dict(),
        "environment": environment,
    }


def scaler_from(state: dict[str, Any]) -> TargetScaler:
    if state.get("target_mean") is None or state.get("target_std") is None:
        raise ValueError("Checkpoint has no target normalization statistics")
    saved_data = state.get("config", {}).get("data", {})
    enabled = bool(
        state.get("targets_normalized", saved_data.get("normalize_targets", True))
    )
    training_mean = float(state.get("target_training_mean", state["target_mean"]))
    return TargetScaler(
        float(state["target_mean"]),
        float(state["target_std"]),
        enabled=enabled,
        training_mean=training_mean,
    )


def validate_for(state: dict[str, Any], config: Config) -> None:
    if state.get("experiment") != EXPERIMENT_ID:
        raise ValueError(
            f"Expected a {EXPERIMENT_ID!r} checkpoint, got {state.get('experiment')!r}"
        )
    if "model_state_dict" not in state:
        raise ValueError("Checkpoint has no model_state_dict")
    saved_data = state.get("config", {}).get("data", {})
    saved_normalization = bool(
        state.get("targets_normalized", saved_data.get("normalize_targets", True))
    )
    if saved_normalization != config.data.normalize_targets:
        raise ValueError(
            "Checkpoint target-normalization mismatch: "
            f"checkpoint={saved_normalization}, config={config.data.normalize_targets}"
        )
    saved_cache_schema = state.get("grid_cache_schema_version", CACHE_SCHEMA_VERSION)
    if saved_cache_schema != CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Checkpoint grid-detector/cache schema mismatch: "
            f"checkpoint={saved_cache_schema}, current={CACHE_SCHEMA_VERSION}"
        )
    for key, default in (
        ("grid_crop_size", 1400),
        ("grid_inner_margin_fraction", 0.0),
    ):
        saved_value = saved_data.get(key, default)
        configured_value = getattr(config.data, key)
        if saved_value != configured_value:
            raise ValueError(
                f"Checkpoint preprocessing mismatch for {key}: "
                f"checkpoint={saved_value!r}, config={configured_value!r}"
            )
    saved = state.get("config", {}).get("model", {})
    keys = (
        "backbone",
        "processor",
        "unfreeze_last_n_blocks",
        "unfreeze_final_norm",
        "head_hidden_dim",
        "dropout",
    )
    for key in keys:
        if key in saved and saved[key] != getattr(config.model, key):
            raise ValueError(
                f"Checkpoint model mismatch for {key}: "
                f"checkpoint={saved[key]!r}, config={getattr(config.model, key)!r}"
            )
