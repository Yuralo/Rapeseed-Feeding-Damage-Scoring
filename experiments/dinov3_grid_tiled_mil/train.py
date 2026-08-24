"""Train a small MIL head on cached global and tiled DINOv3 features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .checkpoint import payload, validate_for
from .config import Config, load_config
from .data import prepare_data
from .metrics import predict
from .model import GlobalTiledMILRegressor
from .reporting import save_evaluation, save_history_plot, save_label_plot
from .runtime import configure_acceleration, learning_rates, make_optimizer, make_scheduler

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def empty_history() -> dict[str, list]:
    return {
        "train_loss": [],
        "val_epochs": [],
        "val_loss": [],
        "val_mae": [],
        "val_r2": [],
        "val_attention_entropy": [],
        "val_top_tile_mass": [],
        "learning_rates": [],
        "epoch_seconds": [],
    }


def restore_history(state) -> dict[str, list]:
    history = empty_history()
    for key, values in (state.get("history") or {}).items():
        if key in history:
            history[key] = values
    return history


def run(config: Config, resume: str | Path | None = None) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config, device)
    state = load_checkpoint(resume, device) if resume else None
    if state:
        validate_for(state, config)
    table, train, validation, scaler, feature_dim, train_loader, validation_loader = prepare_data(
        config, state
    )
    if state:
        validate_for(state, config, feature_dim)
    model = GlobalTiledMILRegressor(feature_dim, config).to(device)
    optimizer = make_optimizer(model, config)
    scheduler, total_steps, warmup_steps = make_scheduler(optimizer, config, len(train_loader))

    start_epoch, history, best_loss = 1, empty_history(), float("inf")
    global_step, evaluations_without_improvement = 0, 0
    if state:
        model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict"):
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        history = restore_history(state)
        best_loss = float(state.get("best_validation_loss", state.get("val_loss", best_loss)))
        training_state = state.get("training_state") or {}
        global_step = int(training_state.get("global_step", 0))
        evaluations_without_improvement = int(
            training_state.get("evaluations_without_improvement", 0)
        )
        print(f"Resuming at epoch {start_epoch} from {resume}", flush=True)

    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_info(device, REPOSITORY_ROOT)
    parameter_summary = model.parameter_summary()
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "model_parameters.json", parameter_summary)
    if config.output.save_plots:
        save_label_plot(table[config.data.target_column].to_numpy(), run_dir / "targets.png")
    print(
        f"Cached DINO features: dim={feature_dim}, views/image="
        f"{1 + config.tiles.rows * config.tiles.columns}, tiles/image="
        f"{config.tiles.rows * config.tiles.columns}",
        flush=True,
    )
    print(
        f"Trainable MIL head: {parameter_summary['trainable_parameters']:,} parameters | "
        f"batch={config.training.batch_size} | steps={total_steps} | warmup={warmup_steps}",
        flush=True,
    )
    print(
        f"Targets: z-score mean={scaler.mean:.4f}, std={scaler.std:.4f} | "
        f"train={len(train)}, validation={len(validation)}",
        flush=True,
    )

    train_names = train[config.data.filename_column].astype(str).tolist()
    validation_names = validation[config.data.filename_column].astype(str).tolist()
    best_path = run_dir / config.output.best_checkpoint_name
    last_path = run_dir / config.output.last_checkpoint_name
    stopped_early = False

    for epoch in range(start_epoch, config.training.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        total_loss, samples = 0.0, 0
        epoch_started = perf_counter()
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = torch.nn.functional.mse_loss(predictions.float(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            total_loss += loss.detach().item() * targets.numel()
            samples += targets.numel()
        epoch_seconds = perf_counter() - epoch_started
        train_loss = total_loss / samples
        history["train_loss"].append(float(train_loss))
        history["learning_rates"].append(learning_rates(optimizer))
        history["epoch_seconds"].append(float(epoch_seconds))
        message = (
            f"Epoch {epoch:03d}/{config.training.epochs} | train {train_loss:.5f} | "
            f"{epoch_seconds:.2f}s"
        )
        should_evaluate = epoch % config.training.eval_every == 0 or epoch == config.training.epochs
        if should_evaluate:
            result = predict(model, validation_loader, device, scaler)
            metrics = result.metrics()
            attention = result.attention_metrics()
            history["val_epochs"].append(epoch)
            history["val_loss"].append(metrics["objective_mse"])
            history["val_mae"].append(metrics["mae"])
            history["val_r2"].append(metrics["r2"])
            history["val_attention_entropy"].append(attention["mean_normalized_entropy"])
            history["val_top_tile_mass"].append(attention["mean_top_tile_mass"])
            improved = (
                metrics["objective_mse"] < best_loss - config.training.early_stopping_min_delta
            )
            if improved:
                best_loss = metrics["objective_mse"]
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            stopped_early = (
                evaluations_without_improvement >= config.training.early_stopping_patience
            )
            checkpoint = payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                scaler=scaler,
                config=config,
                feature_dim=feature_dim,
                training_filenames=train_names,
                validation_filenames=validation_names,
                history=history,
                best_validation_loss=best_loss,
                training_state={
                    "global_step": global_step,
                    "evaluations_without_improvement": evaluations_without_improvement,
                    "stopped_early": stopped_early,
                },
                environment=environment,
            )
            save_checkpoint(last_path, checkpoint)
            if improved:
                save_checkpoint(best_path, checkpoint)
            message += (
                f" | val {metrics['objective_mse']:.5f} | MAE {metrics['mae']:.3f} | "
                f"R² {metrics['r2']:.3f} | tile H {attention['mean_normalized_entropy']:.3f} | "
                f"top {attention['mean_top_tile_mass']:.3f} | patience "
                f"{evaluations_without_improvement}/{config.training.early_stopping_patience}"
            )
        print(message, flush=True)
        write_json(run_dir / "history.json", history)
        if config.output.save_plots:
            save_history_plot(history, run_dir / "training_history.png")
        if stopped_early:
            print(f"Early stopping after epoch {epoch}; best val loss {best_loss:.5f}", flush=True)
            break

    previous_best = Path(resume).parent / config.output.best_checkpoint_name if resume else None
    if best_path.is_file():
        evaluation_checkpoint = best_path
    elif previous_best is not None and previous_best.is_file():
        evaluation_checkpoint = previous_best
    else:
        raise RuntimeError("Training did not produce a best checkpoint")
    best_state = load_checkpoint(evaluation_checkpoint, device)
    validate_for(best_state, config, feature_dim)
    model.load_state_dict(best_state["model_state_dict"])
    report = save_evaluation(
        predict(model, validation_loader, device, scaler), scaler, run_dir, config
    )
    report.update(
        {
            "best_checkpoint": str(evaluation_checkpoint),
            "last_checkpoint": str(last_path if last_path.is_file() else resume),
            "train_samples": len(train),
            "validation_samples": len(validation),
            "feature_dim": feature_dim,
            "device": str(device),
            "stopped_early": stopped_early,
            "completed_epochs": len(history["train_loss"]),
            "global_optimizer_steps": global_step,
            "model_parameters": parameter_summary,
        }
    )
    write_json(run_dir / "summary.json", report)
    return report


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
