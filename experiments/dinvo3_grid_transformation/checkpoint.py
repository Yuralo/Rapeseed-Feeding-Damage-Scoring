"""Checkpoint contents and compatibility rules for this experiment only."""

from __future__ import annotations

from typing import Any

from .config import Config
from .data import TargetScaler


def payload(
    *,
    model,
    optimizer,
    epoch: int,
    metrics: dict[str, float],
    scaler: TargetScaler,
    config: Config,
    training_filenames: list[str],
    validation_filenames: list[str],
    history: dict[str, list[float]],
    best_validation_loss: float,
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": "dinov3_regression",
        "checkpoint_version": 1,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
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
        "config": config.to_dict(),
        "environment": environment,
    }


def scaler_from(state: dict[str, Any]) -> TargetScaler:
    if state.get("target_mean") is None or state.get("target_std") is None:
        raise ValueError("Checkpoint has no target normalization statistics")
    return TargetScaler(float(state["target_mean"]), float(state["target_std"]))


def validate_for(state: dict[str, Any], config: Config) -> None:
    if "model_state_dict" not in state:
        raise ValueError("Checkpoint has no model_state_dict")
    saved = state.get("config", {}).get("model", {})
    for key in ("backbone", "processor", "freeze_backbone", "head_hidden_dim", "dropout"):
        if key in saved and saved[key] != getattr(config.model, key):
            raise ValueError(
                f"Checkpoint model mismatch for {key}: "
                f"checkpoint={saved[key]!r}, config={getattr(config.model, key)!r}"
            )
