"""Checkpoint contents and compatibility rules for this experiment."""

from __future__ import annotations

from typing import Any

from .config import Config
from .data import TargetScaler

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
        "checkpoint_version": 2,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "grad_scaler_state_dict": grad_scaler.state_dict() if grad_scaler else None,
        "metrics": metrics,
        "val_loss": metrics["normalized_mse"],
        "val_mae": metrics["mae"],
        "val_r2": metrics["r2"],
        "target_mean": scaler.mean,
        "target_std": scaler.std,
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
    return TargetScaler(float(state["target_mean"]), float(state["target_std"]))


def validate_for(state: dict[str, Any], config: Config) -> None:
    if state.get("experiment") != EXPERIMENT_ID:
        raise ValueError(
            f"Expected a {EXPERIMENT_ID!r} checkpoint, got {state.get('experiment')!r}"
        )
    if "model_state_dict" not in state:
        raise ValueError("Checkpoint has no model_state_dict")
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
