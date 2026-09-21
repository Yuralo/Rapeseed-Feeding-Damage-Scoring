"""Gold-only residual training on high-resolution SAM-plant patches."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.checkpoint import validate_for as validate_base
from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import make_loader, prepare_tables
from .model import PlantDamageRegressor
from .reporting import predict, save

ROOT = Path(__file__).resolve().parents[2]


def _scaler(state: dict) -> TargetScaler:
    return TargetScaler(
        mean=float(state["target_mean"]), std=float(state["target_std"]),
        training_mean=float(state.get("target_training_mean", state["target_mean"])),
    )


def _state(model, optimizer, epoch: int, config: Config, base_state: dict,
           dimension: int, validation: dict, history: list[dict]):
    return {
        "experiment": "dinov3_plant_damage_mil",
        "version": 1,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "feature_dim": dimension,
        "base_checkpoint": str(Path(config.base_checkpoint).resolve()),
        "base_checkpoint_epoch": base_state.get("epoch"),
        "base_checkpoint_manifests": base_state.get("manifests"),
        "target_mean": float(base_state["target_mean"]),
        "target_std": float(base_state["target_std"]),
        "target_training_mean": float(base_state.get("target_training_mean", base_state["target_mean"])),
        "config": asdict(config),
        "base_config": config.base.to_dict(),
        "validation": validation,
        "history": history,
    }


def _train_epoch(model, loader, optimizer, device, penalty: float):
    model.train()
    total = samples = 0
    for batch in loader:
        moved = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        target = moved["target"].float()
        optimizer.zero_grad(set_to_none=True)
        output, attention = model(moved, return_attention=True)
        error = torch.square(output.float() - target)
        regularizer = penalty * torch.square(attention["residual"]).mean()
        loss = error.mean() + regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        total += float(error.detach().sum())
        samples += target.numel()
    return total / samples


def _history_plot(history: list[dict], destination: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [entry["epoch"] for entry in history]
    axes[0].plot(epochs, [entry["train_mse"] for entry in history], label="train")
    axes[0].plot(epochs, [entry["validation_mse"] for entry in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Normalized MSE", title="Residual training")
    axes[0].legend()
    axes[1].plot(epochs, [entry["validation_mae"] for entry in history], label="new")
    axes[1].axhline(history[0]["base_mae"], linestyle="--", color="gray", label="frozen base")
    axes[1].set(xlabel="Epoch", ylabel="MAE", title="Paired gold validation")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run(config: Config) -> dict:
    seed_everything(config.seed, config.base.runtime.deterministic)
    device = resolve_device(config.base.runtime.device)
    configure_acceleration(config.base, device)
    base_state = load_checkpoint(config.base_checkpoint, device)
    validate_base(base_state, config.base)
    if base_state["stage"] != "finetune":
        raise ValueError("The reference three-view checkpoint must be gold-finetuned")
    tables, dimension = prepare_tables(config, base_state)
    validate_base(base_state, config.base, dimension)
    scaler = _scaler(base_state)
    gold_mean = float(tables["finetune"][config.base.data.target_column].mean())
    if not np.isclose(gold_mean, scaler.mean, atol=1e-4):
        raise ValueError("Base checkpoint target scaler does not match gold training split")
    train_loader = make_loader(tables["finetune"], scaler, config, training=True, offset=1000)
    val_loader = make_loader(tables["validation"], scaler, config, training=False, offset=2000)
    model = PlantDamageRegressor(dimension, config, base_state["model_state_dict"]).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    output = Path(config.run_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", asdict(config))
    write_json(output / "model_parameters.json", model.parameter_summary())
    write_json(output / "environment.json", environment_info(device, ROOT))
    initial = predict(model, val_loader, device, scaler)
    initial_metrics = initial.result.metrics()
    base_mae = float(np.mean(np.abs(initial.base_predictions - initial.result.targets)))
    if not np.allclose(initial.base_predictions, initial.result.predictions, atol=1e-5):
        raise RuntimeError("Residual branch must reproduce the base model before training")
    best_mse = float(initial_metrics["objective_mse"])
    best_mae = float(initial_metrics["mae"])
    history = [{"epoch": 0, "train_mse": None, "validation_mse": best_mse,
                "validation_mae": best_mae, "base_mae": base_mae}]
    state = _state(model, optimizer, 0, config, base_state, dimension, initial_metrics, history)
    save_checkpoint(output / "best.pt", state)
    save_checkpoint(output / "best_mae.pt", state)
    patience = 0
    for epoch in range(1, config.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        train_mse = _train_epoch(model, train_loader, optimizer, device, config.residual_penalty)
        validation = predict(model, val_loader, device, scaler)
        metrics = validation.result.metrics()
        history.append({"epoch": epoch, "train_mse": train_mse,
                        "validation_mse": float(metrics["objective_mse"]),
                        "validation_mae": float(metrics["mae"]), "base_mae": base_mae})
        state = _state(model, optimizer, epoch, config, base_state, dimension, metrics, history)
        save_checkpoint(output / "last.pt", state)
        if metrics["objective_mse"] < best_mse - 1e-4:
            best_mse = float(metrics["objective_mse"])
            patience = 0
            save_checkpoint(output / "best.pt", state)
        else:
            patience += 1
        if metrics["mae"] < best_mae - 1e-4:
            best_mae = float(metrics["mae"])
            save_checkpoint(output / "best_mae.pt", state)
        write_json(output / "history.json", history)
        if config.base.output.save_plots:
            _history_plot(history, output / "training_history.png")
        print(f"Epoch {epoch:03d}/{config.epochs} | train MSE {train_mse:.4f} | "
              f"val MSE {metrics['objective_mse']:.4f} | val MAE {metrics['mae']:.3f} | "
              f"base MAE {base_mae:.3f} | patience {patience}/{config.patience}", flush=True)
        if patience >= config.patience:
            break
    selected_reports = {}
    for name, checkpoint, destination in (
        ("best_mse", output / "best.pt", output),
        ("best_mae", output / "best_mae.pt", output / "best_mae_evaluation"),
    ):
        selected = load_checkpoint(checkpoint, device)
        model.load_state_dict(selected["model_state_dict"])
        detail = predict(model, val_loader, device, scaler)
        report = save(detail, scaler, destination, config)
        report.update({"selection": name, "checkpoint": str(checkpoint),
                       "checkpoint_epoch": int(selected["epoch"]), "split": "validation"})
        write_json(destination / "summary.json", report)
        selected_reports[name] = report
    return {"best_mse": selected_reports["best_mse"]["paired_comparison"],
            "best_mae": selected_reports["best_mae"]["paired_comparison"],
            "best_mse_epoch": selected_reports["best_mse"]["checkpoint_epoch"],
            "best_mae_epoch": selected_reports["best_mae"]["checkpoint_epoch"],
            "gold_train_images": len(tables["finetune"]),
            "validation_images": len(tables["validation"]),
            "reserved_test_images": len(tables["test"]),
            "test_evaluated_during_training": False,
            "run_dir": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
