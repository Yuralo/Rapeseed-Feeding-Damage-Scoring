"""Checkpoint schema for weak-pretraining plus gold-finetuning MIL."""

from __future__ import annotations

from typing import Any

from experiments.dinov3_grid_tiled_mil.features import FEATURE_CACHE_SCHEMA_VERSION

from .config import Config

EXPERIMENT_ID = "dinov3_grid_multicohort_multiscale_mil"
CHECKPOINT_VERSION = 1


def payload(
    *,
    model,
    optimizer,
    scheduler,
    stage: str,
    stage_epoch: int,
    global_epoch: int,
    metrics: dict,
    scaler,
    config: Config,
    feature_dim: int,
    history: dict[str, Any],
    best_validation_loss: float,
    best_validation_mae: float,
    evaluations_without_improvement: int,
    manifests: dict[str, list[str]],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_format": "multicohort_multiscale_mil_head_only",
        "feature_cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "feature_dim": int(feature_dim),
        "stage": stage,
        "stage_epoch": int(stage_epoch),
        "epoch": int(global_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
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


def validate_for(state: dict[str, Any], config: Config, feature_dim: int | None = None) -> None:
    if state.get("experiment") != EXPERIMENT_ID:
        raise ValueError(
            f"Expected a {EXPERIMENT_ID!r} checkpoint, got {state.get('experiment')!r}"
        )
    if state.get("model_state_format") != "multicohort_multiscale_mil_head_only":
        raise ValueError("Checkpoint is not a multicohort multiscale head checkpoint")
    if state.get("feature_cache_schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError("Checkpoint feature-cache schema is incompatible")
    if "model_state_dict" not in state or not state.get("targets_normalized"):
        raise ValueError("Checkpoint is incomplete or targets were not normalized")
    if state.get("stage") not in {"pretrain", "finetune"}:
        raise ValueError(f"Unknown checkpoint stage: {state.get('stage')!r}")
    if feature_dim is not None and int(state.get("feature_dim", -1)) != int(feature_dim):
        raise ValueError(
            f"Checkpoint feature dimension {state.get('feature_dim')} != cache {feature_dim}"
        )
    saved = state.get("config", {})
    checks = {
        "data": ("grid_crop_size", "grid_inner_margin_fraction"),
        "features": ("backbone", "processor", "representation"),
        "coarse": ("rows", "columns", "overlap_fraction", "include_global_view"),
        "fine": ("rows", "columns", "overlap_fraction", "include_global_view"),
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
