"""Reliability-weighted weak pretraining followed by gold-only calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration, learning_rates
from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .checkpoint import payload, validate_for
from .config import Config, StageSettings, load_config
from .data import make_loader, prepare_data
from .metrics import predict
from .model import HierarchicalThreeViewRegressor
from .reporting import save_evaluation, save_history_plot, save_label_plot

ROOT = Path(__file__).resolve().parents[2]


def empty_history() -> dict[str, list]:
    return {
        "train_loss": [], "train_stage": [], "train_objective": [], "val_epochs": [],
        "val_loss": [], "val_mae": [], "val_r2": [], "val_cell_entropy": [],
        "val_plant_entropy": [], "val_top_cell_mass": [], "val_top_plant_mass": [],
        "learning_rates": [], "epoch_seconds": [],
    }


def _optimizer(model, settings: StageSettings):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim > 1 and not name.endswith(".bias") else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": settings.weight_decay, "group_name": "head_decay"},
            {"params": no_decay, "weight_decay": 0.0, "group_name": "head_no_decay"},
        ],
        lr=settings.learning_rate,
    )


def _scheduler(optimizer, settings: StageSettings, loader_length: int):
    total = max(1, settings.epochs * loader_length)
    warmup = int(total * settings.warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return max(1, step + 1) / warmup
        progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return settings.minimum_learning_rate_ratio + (1 - settings.minimum_learning_rate_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), total, warmup


def _forward(model, batch, device):
    return model(
        batch["global_feature"].to(device, non_blocking=True),
        batch["cell_features"].to(device, non_blocking=True),
        batch["plant_features"].to(device, non_blocking=True),
        batch["plant_valid"].to(device, non_blocking=True),
        batch["plant_cell_indices"].to(device, non_blocking=True),
    )


def _validate_manifests(state: dict, manifests: dict[str, list[str]]) -> None:
    saved = state.get("manifests") or {}
    for split, names in manifests.items():
        if list(map(str, saved.get(split, []))) != list(map(str, names)):
            raise ValueError(f"Checkpoint {split} manifest differs from this run")


def _train_stage(
    *, stage_name: str, model, train_loader, validation_loader, scaler, config: Config,
    settings: StageSettings, device, feature_dim: int, history: dict, manifests: dict,
    environment: dict, best_path: Path, best_mae_path: Path | None, last_path: Path,
    resume_state: dict | None,
):
    optimizer = _optimizer(model, settings)
    scheduler, total_steps, warmup = _scheduler(optimizer, settings, len(train_loader))
    start_epoch, best_loss, best_mae, patience = 1, float("inf"), float("inf"), 0
    if resume_state is not None:
        if resume_state["stage"] != stage_name:
            raise ValueError(f"Cannot resume {stage_name} from {resume_state['stage']}")
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = int(resume_state["stage_epoch"]) + 1
        best_loss = float(resume_state["best_validation_loss"])
        best_mae = float(resume_state["best_validation_mae"])
        patience = int(resume_state.get("evaluations_without_improvement", 0))
    objective = "weighted_huber" if stage_name == "pretrain" else "gold_normalized_mse"
    print(
        f"{stage_name}: objective={objective} epochs={settings.epochs} start={start_epoch} "
        f"steps={total_steps} warmup={warmup} samples={len(train_loader.dataset)}",
        flush=True,
    )
    if start_epoch > settings.epochs or patience >= settings.early_stopping_patience:
        if not best_path.is_file():
            raise RuntimeError(f"Completed stage is missing its best checkpoint: {best_path}")
        return {"completed_stage_epochs": start_epoch - 1, "stopped_early": True}
    stopped_early = False
    for stage_epoch in range(start_epoch, settings.epochs + 1):
        train_loader.sampler.set_epoch(stage_epoch)
        model.train()
        loss_sum = weight_sum = 0.0
        started = perf_counter()
        for batch in train_loader:
            target = batch["target"].float().to(device, non_blocking=True)
            weights = batch["sample_weight"].float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch, device)
            if stage_name == "pretrain":
                per_sample = torch.nn.functional.smooth_l1_loss(
                    output.float(), target, reduction="none", beta=settings.huber_delta
                )
                loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
                loss_sum += float((per_sample.detach() * weights).sum().item())
                weight_sum += float(weights.sum().item())
            else:
                per_sample = torch.nn.functional.mse_loss(output.float(), target, reduction="none")
                loss = per_sample.mean()
                loss_sum += float(per_sample.detach().sum().item())
                weight_sum += float(target.numel())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
        train_loss = loss_sum / max(weight_sum, 1e-8)
        seconds = perf_counter() - started
        result = predict(model, validation_loader, device, scaler)
        metrics = result.metrics()
        attention = result.attention_metrics()
        history["train_loss"].append(train_loss)
        history["train_stage"].append(stage_name)
        history["train_objective"].append(objective)
        history["val_epochs"].append(len(history["train_loss"]))
        history["val_loss"].append(metrics["objective_mse"])
        history["val_mae"].append(metrics["mae"])
        history["val_r2"].append(metrics["r2"])
        history["val_cell_entropy"].append(attention["cells"]["mean_normalized_entropy"])
        history["val_plant_entropy"].append(attention["plants"]["mean_normalized_entropy"])
        history["val_top_cell_mass"].append(attention["cells"]["mean_top_mass"])
        history["val_top_plant_mass"].append(attention["plants"]["mean_top_mass"])
        history["learning_rates"].append(learning_rates(optimizer))
        history["epoch_seconds"].append(seconds)
        improved_mse = metrics["objective_mse"] < best_loss - settings.early_stopping_min_delta
        improved_mae = metrics["mae"] < best_mae
        patience = 0 if improved_mse else patience + 1
        if improved_mse:
            best_loss = metrics["objective_mse"]
        if improved_mae:
            best_mae = metrics["mae"]
        state = payload(
            model=model, optimizer=optimizer, scheduler=scheduler, stage=stage_name,
            stage_epoch=stage_epoch, global_epoch=len(history["train_loss"]), metrics=metrics,
            scaler=scaler, config=config, feature_dim=feature_dim, history=history,
            best_validation_loss=best_loss, best_validation_mae=best_mae,
            evaluations_without_improvement=patience, manifests=manifests, environment=environment,
        )
        save_checkpoint(last_path, state)
        if improved_mse:
            save_checkpoint(best_path, state)
        if improved_mae and best_mae_path is not None:
            save_checkpoint(best_mae_path, state)
        write_json(Path(config.output.run_dir) / "history.json", history)
        if config.output.save_plots:
            save_history_plot(history, Path(config.output.run_dir) / "training_history.png")
        print(
            f"{stage_name} {stage_epoch:03d}/{settings.epochs} | train {train_loss:.5f} | "
            f"val MSE {metrics['objective_mse']:.5f} | MAE {metrics['mae']:.3f} | "
            f"R² {metrics['r2']:.3f} | patience {patience}/{settings.early_stopping_patience} "
            f"| {seconds:.2f}s",
            flush=True,
        )
        if patience >= settings.early_stopping_patience:
            stopped_early = True
            break
    return {"completed_stage_epochs": stage_epoch, "stopped_early": stopped_early,
            "best_validation_loss": best_loss, "best_validation_mae": best_mae}


def run(config: Config, resume: str | Path | None = None) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config, device)
    tables, scaler, feature_dim, skipped_weak = prepare_data(config)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tables["pretrain"].to_csv(run_dir / config.output.usable_pretrain_manifest, index=False)
    loaders = {
        "finetune": make_loader(
            tables["finetune"], scaler, config, config.training.finetuning,
            training=True, seed_offset=1000,
        ),
        "validation": make_loader(
            tables["validation"], scaler, config, config.training.finetuning,
            training=False, seed_offset=2000,
        ),
    }
    if config.training.use_weak_pretraining:
        loaders["pretrain"] = make_loader(
            tables["pretrain"], scaler, config, config.training.pretraining, training=True,
            cohort_balanced=config.training.cohort_balanced_pretraining,
        )
    model = HierarchicalThreeViewRegressor(feature_dim, config).to(device)
    environment = environment_info(device, ROOT)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "model_parameters.json", model.parameter_summary())
    if config.output.save_plots:
        save_label_plot(tables["finetune"][config.data.target_column].to_numpy(), run_dir / "targets.png")
    manifests = {
        name: table[config.data.filename_column].astype(str).tolist() for name, table in tables.items()
    }
    state = load_checkpoint(resume, device) if resume else None
    history = empty_history()
    if state:
        validate_for(state, config, feature_dim)
        _validate_manifests(state, manifests)
        model.load_state_dict(state["model_state_dict"])
        history.update(state.get("history") or {})
    pretrain_path = run_dir / config.output.pretrain_checkpoint_name
    best_path = run_dir / config.output.best_checkpoint_name
    mae_path = run_dir / config.output.best_mae_checkpoint_name
    last_path = run_dir / config.output.last_checkpoint_name
    reports = {}
    if config.training.use_weak_pretraining and (state is None or state["stage"] == "pretrain"):
        reports["pretrain"] = _train_stage(
            stage_name="pretrain", model=model, train_loader=loaders["pretrain"],
            validation_loader=loaders["validation"], scaler=scaler, config=config,
            settings=config.training.pretraining, device=device, feature_dim=feature_dim,
            history=history, manifests=manifests, environment=environment,
            best_path=pretrain_path, best_mae_path=None, last_path=last_path, resume_state=state,
        )
        selected = load_checkpoint(pretrain_path, device)
        validate_for(selected, config, feature_dim)
        model.load_state_dict(selected["model_state_dict"])
        state = None
    elif state is not None and state["stage"] != "finetune":
        raise ValueError(f"Cannot resume stage {state['stage']!r} with this training mode")
    reports["finetune"] = _train_stage(
        stage_name="finetune", model=model, train_loader=loaders["finetune"],
        validation_loader=loaders["validation"], scaler=scaler, config=config,
        settings=config.training.finetuning, device=device, feature_dim=feature_dim,
        history=history, manifests=manifests, environment=environment, best_path=best_path,
        best_mae_path=mae_path, last_path=last_path, resume_state=state,
    )
    evaluations = []
    for checkpoint, destination, selection in (
        (best_path, run_dir, "objective_mse"),
        (mae_path, run_dir / "best_mae_evaluation", "mae"),
    ):
        selected = load_checkpoint(checkpoint, device)
        validate_for(selected, config, feature_dim)
        model.load_state_dict(selected["model_state_dict"])
        report = save_evaluation(
            predict(model, loaders["validation"], device, scaler), scaler, destination, config
        )
        report.update({"checkpoint": str(checkpoint), "selection_metric": selection})
        write_json(destination / "summary.json", report)
        evaluations.append(report)
    result = evaluations[0]
    result.update({
        "training_mode": "weak_then_gold" if config.training.use_weak_pretraining else "gold_only",
        "stage_reports": reports,
        "weak_pretrain_samples": len(tables["pretrain"]) if config.training.use_weak_pretraining else 0,
        "weak_records_skipped": skipped_weak,
        "gold_finetune_samples": len(tables["finetune"]),
        "gold_validation_samples": len(tables["validation"]),
        "reserved_gold_test_samples": len(tables["test"]),
        "test_evaluated_during_training": False,
        "best_checkpoint": str(best_path), "best_mae_checkpoint": str(mae_path),
        "pretrain_checkpoint": str(pretrain_path) if config.training.use_weak_pretraining else None,
        "last_checkpoint": str(last_path), "best_mae_evaluation": evaluations[1],
        "feature_dim": feature_dim, "device": str(device),
    })
    write_json(run_dir / "summary.json", result)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume")
    mode.add_argument("--from-scratch", action="store_true")
    arguments = parser.parse_args(argv)
    print(json.dumps(run(load_config(arguments.config), arguments.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
