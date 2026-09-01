"""Compact checkpoint payload and compatibility validation."""

from __future__ import annotations

from typing import Any

from .config import Config
from .preprocessing import PREPARED_SCHEMA_VERSION

EXPERIMENT_ID = "dinov3_mixed_domain_adaptation"
CHECKPOINT_VERSION = 2


def payload(
    *,
    model,
    optimizer,
    scheduler,
    grad_scaler,
    epoch: int,
    metrics: dict[str, float],
    config: Config,
    training_ids: list[str],
    validation_ids: list[str],
    history: dict[str, Any],
    best_validation_loss: float,
    training_state: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_format": "trainable_only",
        "prepared_schema_version": PREPARED_SCHEMA_VERSION,
        "epoch": epoch,
        "model_state_dict": model.adaptation_state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "grad_scaler_state_dict": grad_scaler.state_dict() if grad_scaler else None,
        "metrics": metrics,
        "training_ids": training_ids,
        "validation_ids": validation_ids,
        "history": history,
        "best_validation_loss": best_validation_loss,
        "training_state": training_state,
        "config": config.to_dict(),
        "environment": environment,
    }


def validate_for(state: dict[str, Any], config: Config) -> None:
    if state.get("experiment") != EXPERIMENT_ID:
        raise ValueError(
            f"Expected a {EXPERIMENT_ID!r} checkpoint, got {state.get('experiment')!r}"
        )
    if state.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported adaptation checkpoint version")
    if state.get("model_state_format") != "trainable_only":
        raise ValueError("Adaptation checkpoint must contain a trainable-only state")
    if "model_state_dict" not in state:
        raise ValueError("Checkpoint has no model_state_dict")
    if state.get("prepared_schema_version") != PREPARED_SCHEMA_VERSION:
        raise ValueError("Prepared-manifest schema changed; rebuild inputs and restart training")
    saved = state.get("config", {})
    comparisons = {
        "tiles": (
            "grid_sizes",
            "overlap_fraction",
            "plant_biased_probability",
            "vegetation_score_power",
            "label_overlap_limit",
            "mask_analysis_max_side",
        ),
        "model": (
            "backbone",
            "processor",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "train_final_norm",
        ),
        "objective": ("cross_view_weight", "same_view_anchor_weight"),
    }
    current = config.to_dict()
    for section, keys in comparisons.items():
        for key in keys:
            saved_value = saved.get(section, {}).get(key)
            current_value = current[section][key]
            if saved_value is not None and saved_value != current_value:
                raise ValueError(
                    f"Checkpoint mismatch for {section}.{key}: "
                    f"checkpoint={saved_value!r}, config={current_value!r}"
                )
