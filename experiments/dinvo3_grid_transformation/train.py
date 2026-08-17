"""Train the DINOv3 regression experiment.

This loop deliberately lives beside the model and dataset. A future experiment
can replace it completely instead of conforming to a universal trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .checkpoint import payload, validate_for
from .config import Config, load_config
from .metrics import predict
from .model import DinoV3Regressor, make_optimizer
from .reporting import save_evaluation, save_history_plot, save_label_plot
from .setup import prepare_data

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def empty_history() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "val_epochs": [],
        "val_loss": [],
        "val_mae": [],
        "val_r2": [],
    }


def run(config: Config, resume: str | Path | None = None) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    state = load_checkpoint(resume, device) if resume else None
    if state:
        validate_for(state, config)

    table, train, validation, scaler, train_loader, validation_loader = prepare_data(
        config, state
    )
    model = DinoV3Regressor(config).to(device)
    optimizer = make_optimizer(model, config)
    start_epoch, history, best_loss = 1, empty_history(), float("inf")
    if state:
        model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict"):
            optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        history = state.get("history") or history
        best_loss = float(state.get("best_validation_loss", state.get("val_loss", best_loss)))
        print(f"Resuming at epoch {start_epoch} from {resume}", flush=True)

    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_environment = environment_info(device, REPOSITORY_ROOT)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "environment.json", run_environment)
    print(
        f"Grid preprocessing failures will be logged to "
        f"{run_dir / config.output.grid_failure_log}",
        flush=True,
    )
    if config.output.save_plots:
        save_label_plot(table[config.data.target_column].to_numpy(), run_dir / "targets.png")

    train_names = train[config.data.filename_column].astype(str).tolist()
    validation_names = validation[config.data.filename_column].astype(str).tolist()
    best_path = run_dir / config.output.best_checkpoint_name
    last_path = run_dir / config.output.last_checkpoint_name
    best_saved = False

    for epoch in range(start_epoch, config.training.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        total_loss, samples = 0.0, 0
        for batch in train_loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(pixels), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.numel()
            samples += targets.numel()
        train_loss = total_loss / samples
        history["train_loss"].append(float(train_loss))
        message = f"Epoch {epoch:03d}/{config.training.epochs} | train {train_loss:.5f}"

        if epoch % config.training.eval_every == 0 or epoch == config.training.epochs:
            result = predict(model, validation_loader, device, scaler)
            metrics = result.metrics()
            history["val_epochs"].append(epoch)
            history["val_loss"].append(metrics["normalized_mse"])
            history["val_mae"].append(metrics["mae"])
            history["val_r2"].append(metrics["r2"])
            improved = metrics["normalized_mse"] < best_loss
            best_loss = min(best_loss, metrics["normalized_mse"])
            checkpoint = payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                scaler=scaler,
                config=config,
                training_filenames=train_names,
                validation_filenames=validation_names,
                history=history,
                best_validation_loss=best_loss,
                environment=run_environment,
            )
            save_checkpoint(last_path, checkpoint)
            if improved:
                save_checkpoint(best_path, checkpoint)
                best_saved = True
            message += (
                f" | val {metrics['normalized_mse']:.5f}"
                f" | MAE {metrics['mae']:.3f} | R² {metrics['r2']:.3f}"
            )
        print(message, flush=True)

    write_json(run_dir / "history.json", history)
    if config.output.save_plots:
        save_history_plot(history, run_dir / "training_history.png")

    if best_saved:
        evaluation_checkpoint = best_path
    elif resume:
        previous_best = Path(resume).parent / config.output.best_checkpoint_name
        evaluation_checkpoint = previous_best if previous_best.is_file() else Path(resume)
    else:
        evaluation_checkpoint = best_path
    best_state = load_checkpoint(evaluation_checkpoint, device)
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
            "device": str(device),
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
