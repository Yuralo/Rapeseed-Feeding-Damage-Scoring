"""Train the SAM-guided adaptive-instance MIL head."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.dinov3_grid_tiled_mil.runtime import (
    configure_acceleration,
    learning_rates,
    make_optimizer,
    make_scheduler,
)
from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .checkpoint import payload, validate_for
from .config import load_config
from .data import prepare_data
from .metrics import predict
from .model import SamAdaptiveMILRegressor
from .reporting import save_evaluation, save_history_plot, save_label_plot

ROOT = Path(__file__).resolve().parents[2]


def empty_history():
    return {
        "train_loss": [],
        "val_epochs": [],
        "val_loss": [],
        "val_mae": [],
        "val_r2": [],
        "val_attention_entropy": [],
        "val_top_mass": [],
        "learning_rates": [],
        "epoch_seconds": [],
    }


def run(config, resume=None):
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config, device)
    state = load_checkpoint(resume, device) if resume else None
    if state:
        validate_for(state, config)
    table, train, val, scaler, dim, train_loader, val_loader = prepare_data(config, state)
    model = SamAdaptiveMILRegressor(dim, config).to(device)
    optimizer = make_optimizer(model, config)
    scheduler, _, _ = make_scheduler(optimizer, config, len(train_loader))
    history = empty_history()
    start = 1
    best_loss = best_mae = float("inf")
    stale = steps = 0
    if state:
        validate_for(state, config, dim)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        history.update(state.get("history", {}))
        start = state["epoch"] + 1
        best_loss = state.get("best_validation_loss", best_loss)
        best_mae = state.get("best_validation_mae", best_mae)
        stale = state.get("training_state", {}).get("stale", 0)
        steps = state.get("training_state", {}).get("steps", 0)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_info(device, ROOT)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "model_parameters.json", model.parameter_summary())
    if config.output.save_plots:
        save_label_plot(table[config.data.target_column].to_numpy(), run_dir / "targets.png")
    train_names = train[config.data.filename_column].astype(str).tolist()
    val_names = val[config.data.filename_column].astype(str).tolist()
    mse_path = run_dir / config.output.best_checkpoint_name
    mae_path = run_dir / config.output.best_mae_checkpoint_name
    last_path = run_dir / config.output.last_checkpoint_name
    stopped_early = False
    for epoch in range(start, config.training.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        total = samples = 0
        began = perf_counter()
        for batch in train_loader:
            context = batch["context_features"].to(device)
            instances = batch["instance_features"].to(device)
            valid = batch["instance_valid"].to(device)
            target = batch["target"].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(context, instances, valid)
            loss = torch.nn.functional.mse_loss(output.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            steps += 1
            total += loss.item() * target.numel()
            samples += target.numel()
        history["train_loss"].append(total / samples)
        history["learning_rates"].append(learning_rates(optimizer))
        history["epoch_seconds"].append(perf_counter() - began)
        result = predict(model, val_loader, device, scaler)
        metrics = result.metrics()
        attention = result.attention_metrics()
        history["val_epochs"].append(epoch)
        history["val_loss"].append(metrics["objective_mse"])
        history["val_mae"].append(metrics["mae"])
        history["val_r2"].append(metrics["r2"])
        history["val_attention_entropy"].append(attention["mean_normalized_entropy"])
        history["val_top_mass"].append(attention["mean_top_instance_mass"])
        improved_mse = (
            metrics["objective_mse"] < best_loss - config.training.early_stopping_min_delta
        )
        improved_mae = metrics["mae"] < best_mae
        if improved_mse:
            best_loss = metrics["objective_mse"]
            stale = 0
        else:
            stale += 1
        if improved_mae:
            best_mae = metrics["mae"]
        state = payload(
            model,
            optimizer,
            scheduler,
            epoch,
            metrics,
            scaler,
            config,
            dim,
            train_names,
            val_names,
            history,
            best_loss,
            best_mae,
            {"stale": stale, "steps": steps},
            environment,
        )
        save_checkpoint(last_path, state)
        if improved_mse:
            save_checkpoint(mse_path, state)
        if improved_mae:
            save_checkpoint(mae_path, state)
        print(
            f"Epoch {epoch:03d} | train {history['train_loss'][-1]:.4f} | val {metrics['objective_mse']:.4f} | MAE {metrics['mae']:.3f} | R² {metrics['r2']:.3f} | instances {attention['mean_instances_per_image']:.1f} | H {attention['mean_normalized_entropy']:.3f} | patience {stale}/{config.training.early_stopping_patience}",
            flush=True,
        )
        write_json(run_dir / "history.json", history)
        if config.output.save_plots:
            save_history_plot(history, run_dir / "training_history.png")
        if stale >= config.training.early_stopping_patience:
            stopped_early = True
            break
    reports = []
    for checkpoint, destination, selection in (
        (mse_path, run_dir, "objective_mse"),
        (mae_path, run_dir / "best_mae_evaluation", "mae"),
    ):
        selected = load_checkpoint(checkpoint, device)
        model.load_state_dict(selected["model_state_dict"])
        report = save_evaluation(
            predict(model, val_loader, device, scaler), scaler, destination, config
        )
        report.update({"checkpoint": str(checkpoint), "selection_metric": selection})
        write_json(destination / "summary.json", report)
        reports.append(report)
    root = reports[0]
    root.update(
        {
            "best_mse_checkpoint": str(mse_path),
            "best_mae_checkpoint": str(mae_path),
            "last_checkpoint": str(last_path),
            "best_mae_evaluation": reports[1],
            "train_samples": len(train),
            "validation_samples": len(val),
            "completed_epochs": len(history["train_loss"]),
            "global_optimizer_steps": steps,
            "stopped_early": stopped_early,
            "feature_dim": dim,
            "device": str(device),
            "model_parameters": model.parameter_summary(),
        }
    )
    write_json(run_dir / "summary.json", root)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
