from experiments.dinov3_grid_tiled_mil.features import FEATURE_CACHE_SCHEMA_VERSION

from .features import ADAPTIVE_FEATURE_SCHEMA_VERSION

EXPERIMENT_ID = "dinov3_grid_sam_adaptive_mil"


def validate_for(state, config, feature_dim=None):
    if (
        state.get("experiment") != EXPERIMENT_ID
        or state.get("model_state_format") != "adaptive_mil_head_only"
    ):
        raise ValueError("Incompatible adaptive-MIL checkpoint")
    if (
        state.get("adaptive_feature_schema_version") != ADAPTIVE_FEATURE_SCHEMA_VERSION
        or state.get("context_feature_schema_version") != FEATURE_CACHE_SCHEMA_VERSION
    ):
        raise ValueError("Checkpoint feature-cache schema mismatch")
    if feature_dim is not None and state.get("feature_dim") != feature_dim:
        raise ValueError("Checkpoint feature dimension mismatch")
    saved = state.get("config", {})
    for section in ("context", "adaptive_crops", "features", "model"):
        for key, value in saved.get(section, {}).items():
            if (
                hasattr(getattr(config, section), key)
                and getattr(getattr(config, section), key) != value
            ):
                raise ValueError(f"Checkpoint mismatch for {section}.{key}")


def payload(
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
    scaler,
    config,
    feature_dim,
    train_names,
    val_names,
    history,
    best_loss,
    best_mae,
    training_state,
    environment,
):
    return {
        "experiment": EXPERIMENT_ID,
        "model_state_format": "adaptive_mil_head_only",
        "adaptive_feature_schema_version": ADAPTIVE_FEATURE_SCHEMA_VERSION,
        "context_feature_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "feature_dim": feature_dim,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "target_mean": scaler.mean,
        "target_std": scaler.std,
        "target_training_mean": scaler.baseline_mean,
        "targets_normalized": True,
        "training_filenames": train_names,
        "validation_filenames": val_names,
        "history": history,
        "best_validation_loss": best_loss,
        "best_validation_mae": best_mae,
        "training_state": training_state,
        "config": config.to_dict(),
        "environment": environment,
    }
