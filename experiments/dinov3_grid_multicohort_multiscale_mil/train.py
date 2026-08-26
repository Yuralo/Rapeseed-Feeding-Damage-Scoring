"""Weak multicohort pretraining followed by gold calibration-set fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from experiments.dinov3_grid_multiscale_tiled_mil.metrics import predict
from experiments.dinov3_grid_multiscale_tiled_mil.reporting import (
    save_evaluation,
    save_history_plot,
    save_label_plot,
)
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration, learning_rates
from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .checkpoint import payload, validate_for
from .config import Config, StageSettings, load_config
from .data import make_loader, prepare_data
from .model import MultiScaleTiledMILRegressor

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def empty_history() -> dict[str, list]:
    return {
        "train_loss": [],
        "train_stage": [],
        "val_epochs": [],
        "val_loss": [],
        "val_mae": [],
        "val_r2": [],
        "val_coarse_attention_entropy": [],
        "val_coarse_top_tile_mass": [],
        "val_fine_attention_entropy": [],
        "val_fine_top_tile_mass": [],
        "learning_rates": [],
        "epoch_seconds": [],
    }


def _optimizer(model, settings: StageSettings):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        destination = decay if parameter.ndim > 1 and not name.endswith(".bias") else no_decay
        destination.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": settings.weight_decay, "group_name": "head_decay"},
            {"params": no_decay, "weight_decay": 0.0, "group_name": "head_no_decay"},
        ],
        lr=settings.learning_rate,
    )


def _scheduler(optimizer, settings: StageSettings, loader_length: int):
    total_steps = max(1, settings.epochs * loader_length)
    warmup_steps = int(total_steps * settings.warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        minimum = settings.minimum_learning_rate_ratio
        return minimum + (1 - minimum) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), total_steps, warmup_steps


def _manifest_names(config: Config, tables: dict[str, object]) -> dict[str, list[str]]:
    column = config.data.filename_column
    return {name: table[column].astype(str).tolist() for name, table in tables.items()}


def _validate_checkpoint_manifests(state: dict, manifests: dict[str, list[str]]) -> None:
    saved = state.get("manifests") or {}
    for split, names in manifests.items():
        if list(map(str, saved.get(split, []))) != list(map(str, names)):
            raise ValueError(
                f"Checkpoint {split} manifest differs from the configured manifest. "
                "Use the original manifests or start a new run."
            )


def _train_stage(
    *,
    stage_name: str,
    model,
    train_loader,
    validation_loader,
    scaler,
    config: Config,
    settings: StageSettings,
    device,
    feature_dim: int,
    history: dict[str, list],
    manifests: dict[str, list[str]],
    environment: dict,
    best_path: Path,
    best_mae_path: Path | None,
    last_path: Path,
    resume_state: dict | None = None,
) -> tuple[dict, Path, Path | None]:
    optimizer = _optimizer(model, settings)
    scheduler, total_steps, warmup_steps = _scheduler(optimizer, settings, len(train_loader))
    start_epoch = 1
    best_loss = best_mae = float("inf")
    patience = 0
    if resume_state is not None:
        if resume_state["stage"] != stage_name:
            raise ValueError(
                f"Cannot resume {stage_name} from a {resume_state['stage']} checkpoint"
            )
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = int(resume_state["stage_epoch"]) + 1
        best_loss = float(resume_state["best_validation_loss"])
        best_mae = float(resume_state["best_validation_mae"])
        patience = int(resume_state.get("evaluations_without_improvement", 0))
    print(
        f"{stage_name}: epochs={settings.epochs}, start={start_epoch}, "
        f"steps={total_steps}, warmup={warmup_steps}, samples={len(train_loader.dataset)}",
        flush=True,
    )
    if start_epoch > settings.epochs or patience >= settings.early_stopping_patience:
        if not best_path.is_file():
            raise RuntimeError(
                f"{stage_name} is already complete but its best checkpoint is missing: {best_path}"
            )
        return (
            {
                "best_validation_loss": best_loss,
                "best_validation_mae": best_mae,
                "stopped_early": patience >= settings.early_stopping_patience,
                "completed_stage_epochs": min(start_epoch - 1, settings.epochs),
            },
            best_path,
            best_mae_path,
        )
    stopped_early = False
    for stage_epoch in range(start_epoch, settings.epochs + 1):
        train_loader.sampler.set_epoch(stage_epoch)
        model.train()
        weighted_loss_sum = weight_sum = 0.0
        started = perf_counter()
        for batch in train_loader:
            coarse = batch["coarse_features"].to(device, non_blocking=True)
            fine = batch["fine_features"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True)
            weights = batch["sample_weight"].float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(coarse, fine)
            per_sample = torch.nn.functional.mse_loss(
                predictions.float(), targets, reduction="none"
            )
            loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            weighted_loss_sum += float((per_sample.detach() * weights).sum().item())
            weight_sum += float(weights.sum().item())
        train_loss = weighted_loss_sum / weight_sum
        seconds = perf_counter() - started
        history["train_loss"].append(train_loss)
        history["train_stage"].append(stage_name)
        history["learning_rates"].append(learning_rates(optimizer))
        history["epoch_seconds"].append(seconds)
        global_epoch = len(history["train_loss"])
        result = predict(model, validation_loader, device, scaler)
        metrics = result.metrics()
        attention = result.attention_metrics()
        history["val_epochs"].append(global_epoch)
        history["val_loss"].append(metrics["objective_mse"])
        history["val_mae"].append(metrics["mae"])
        history["val_r2"].append(metrics["r2"])
        history["val_coarse_attention_entropy"].append(
            attention["coarse"]["mean_normalized_entropy"]
        )
        history["val_coarse_top_tile_mass"].append(attention["coarse"]["mean_top_tile_mass"])
        history["val_fine_attention_entropy"].append(attention["fine"]["mean_normalized_entropy"])
        history["val_fine_top_tile_mass"].append(attention["fine"]["mean_top_tile_mass"])
        improved_mse = metrics["objective_mse"] < best_loss - settings.early_stopping_min_delta
        improved_mae = metrics["mae"] < best_mae
        if improved_mse:
            best_loss, patience = metrics["objective_mse"], 0
        else:
            patience += 1
        if improved_mae:
            best_mae = metrics["mae"]
        stopped_early = patience >= settings.early_stopping_patience
        state = payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            stage=stage_name,
            stage_epoch=stage_epoch,
            global_epoch=global_epoch,
            metrics=metrics,
            scaler=scaler,
            config=config,
            feature_dim=feature_dim,
            history=history,
            best_validation_loss=best_loss,
            best_validation_mae=best_mae,
            evaluations_without_improvement=patience,
            manifests=manifests,
            environment=environment,
        )
        save_checkpoint(last_path, state)
        if improved_mse:
            save_checkpoint(best_path, state)
        if improved_mae and best_mae_path is not None:
            save_checkpoint(best_mae_path, state)
        print(
            f"{stage_name} {stage_epoch:03d}/{settings.epochs} | train {train_loss:.5f} | "
            f"val {metrics['objective_mse']:.5f} | MAE {metrics['mae']:.3f} | "
            f"R² {metrics['r2']:.3f} | patience {patience}/{settings.early_stopping_patience} "
            f"| {seconds:.2f}s",
            flush=True,
        )
        write_json(Path(config.output.run_dir) / "history.json", history)
        if config.output.save_plots:
            save_history_plot(history, Path(config.output.run_dir) / "training_history.png")
        if stopped_early:
            print(f"Early stopping {stage_name} after epoch {stage_epoch}", flush=True)
            break
    if not best_path.is_file():
        raise RuntimeError(f"{stage_name} did not produce {best_path}")
    return (
        {
            "best_validation_loss": best_loss,
            "best_validation_mae": best_mae,
            "stopped_early": stopped_early,
            "completed_stage_epochs": stage_epoch,
        },
        best_path,
        best_mae_path,
    )


def run(config: Config, resume: str | Path | None = None) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config.single_scale_config(config.coarse), device)
    pretrain, finetune, validation, test, scaler, feature_dim = prepare_data(config)
    tables = {
        "pretrain": pretrain,
        "finetune": finetune,
        "validation": validation,
        "test": test,
    }
    manifests = _manifest_names(config, tables)
    pretrain_loader = make_loader(
        pretrain, scaler, config, config.training.pretraining, training=True
    )
    finetune_loader = make_loader(
        finetune, scaler, config, config.training.finetuning, training=True, seed_offset=1000
    )
    validation_loader = make_loader(
        validation,
        scaler,
        config,
        config.training.finetuning,
        training=False,
        seed_offset=2000,
    )
    model = MultiScaleTiledMILRegressor(feature_dim, config).to(device)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_info(device, REPOSITORY_ROOT)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "model_parameters.json", model.parameter_summary())
    if config.output.save_plots:
        save_label_plot(finetune[config.data.target_column].to_numpy(), run_dir / "targets.png")
    history = empty_history()
    state = load_checkpoint(resume, device) if resume else None
    if state:
        validate_for(state, config, feature_dim)
        _validate_checkpoint_manifests(state, manifests)
        model.load_state_dict(state["model_state_dict"])
        history.update(state.get("history") or {})
        print(f"Resuming {state['stage']} from {resume}", flush=True)

    pretrain_path = run_dir / config.output.pretrain_checkpoint_name
    best_path = run_dir / config.output.best_checkpoint_name
    best_mae_path = run_dir / config.output.best_mae_checkpoint_name
    last_path = run_dir / config.output.last_checkpoint_name
    stage_reports = {}
    if state is None or state["stage"] == "pretrain":
        report, _, _ = _train_stage(
            stage_name="pretrain",
            model=model,
            train_loader=pretrain_loader,
            validation_loader=validation_loader,
            scaler=scaler,
            config=config,
            settings=config.training.pretraining,
            device=device,
            feature_dim=feature_dim,
            history=history,
            manifests=manifests,
            environment=environment,
            best_path=pretrain_path,
            best_mae_path=None,
            last_path=last_path,
            resume_state=state,
        )
        stage_reports["pretrain"] = report
        pretrain_state = load_checkpoint(pretrain_path, device)
        validate_for(pretrain_state, config, feature_dim)
        model.load_state_dict(pretrain_state["model_state_dict"])
        state = None
    elif state["stage"] != "finetune":
        raise ValueError(f"Cannot resume unknown stage {state['stage']!r}")

    report, _, _ = _train_stage(
        stage_name="finetune",
        model=model,
        train_loader=finetune_loader,
        validation_loader=validation_loader,
        scaler=scaler,
        config=config,
        settings=config.training.finetuning,
        device=device,
        feature_dim=feature_dim,
        history=history,
        manifests=manifests,
        environment=environment,
        best_path=best_path,
        best_mae_path=best_mae_path,
        last_path=last_path,
        resume_state=state,
    )
    stage_reports["finetune"] = report

    best_state = load_checkpoint(best_path, device)
    validate_for(best_state, config, feature_dim)
    model.load_state_dict(best_state["model_state_dict"])
    evaluation = save_evaluation(
        predict(model, validation_loader, device, scaler), scaler, run_dir, config
    )
    mae_state = load_checkpoint(best_mae_path, device)
    validate_for(mae_state, config, feature_dim)
    model.load_state_dict(mae_state["model_state_dict"])
    mae_dir = run_dir / "best_mae_evaluation"
    mae_evaluation = save_evaluation(
        predict(model, validation_loader, device, scaler), scaler, mae_dir, config
    )
    evaluation.update(
        {
            "selection_metric": "objective_mse",
            "best_checkpoint": str(best_path),
            "best_mae_checkpoint": str(best_mae_path),
            "pretrain_checkpoint": str(pretrain_path),
            "last_checkpoint": str(last_path),
            "stage_reports": stage_reports,
            "pretrain_samples": len(pretrain),
            "finetune_gold_samples": len(finetune),
            "validation_gold_samples": len(validation),
            "reserved_test_gold_samples": len(test),
            "test_evaluated_during_training": False,
            "feature_dim": feature_dim,
            "device": str(device),
            "best_mae_evaluation": mae_evaluation,
        }
    )
    write_json(run_dir / "summary.json", evaluation)
    return evaluation


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume")
    mode.add_argument("--from-scratch", action="store_true")
    arguments = parser.parse_args(argv)
    report = run(load_config(arguments.config), arguments.resume)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
