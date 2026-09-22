"""Retrain the strongest plant-damage model on plot-isolated paired means."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.dinov3_grid_tiled_mil.metrics import regression_metrics
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_plant_damage_mil.model import PlantDamageRegressor
from experiments.dinov3_plant_damage_mil.reporting import predict, save
from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import make_loader, prepare_data
from .manifests import make_manifests

ROOT = Path(__file__).resolve().parents[2]


def validate_checkpoint(state, config: Config, manifests: dict, dimension: int | None = None):
    if state.get("experiment") != "dinov3_plant_damage_paired_mean" or state.get("version") != 1:
        raise ValueError("Not a paired-mean plant-damage checkpoint")
    if state.get("config") != asdict(config):
        raise ValueError("Checkpoint configuration differs")
    if state.get("manifest_sha256") != manifests["manifest_sha256"]:
        raise ValueError("Paired-score manifests changed after checkpoint creation")
    if dimension is not None and int(state.get("feature_dim", -1)) != dimension:
        raise ValueError("Checkpoint feature dimension differs")


def _train_epoch(model, loader, optimizer, device, config):
    model.train()
    squared_sum = samples = 0
    for batch in loader:
        moved = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        prediction, attention = model(moved, return_attention=True)
        squared = torch.square(prediction.float() - moved["target"].float())
        loss = squared.mean() + config.residual_penalty * torch.square(attention["residual"]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config.gradient_clip_norm,
        )
        optimizer.step()
        squared_sum += float(squared.detach().sum())
        samples += moved["target"].numel()
    return squared_sum / samples


def _history_plot(history, destination):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [entry["epoch"] for entry in history]
    axes[0].plot(epochs, [entry["train_mse"] for entry in history], label="paired train")
    axes[0].plot(epochs, [entry["validation_mse"] for entry in history], label="gold validation")
    axes[0].set(xlabel="Epoch", ylabel="Normalized MSE", title="Paired-mean training")
    axes[0].legend()
    axes[1].plot(epochs, [entry["validation_mae"] for entry in history], label="final")
    axes[1].plot(epochs, [entry["base_mae"] for entry in history], "--", label="frozen base")
    axes[1].set(xlabel="Epoch", ylabel="MAE", title="Gold validation")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _state(
    model,
    optimizer,
    config,
    manifests,
    base_state,
    dimension,
    epoch,
    history,
    best_mse,
    patience,
    train_names,
    val_names,
    test_names,
    metrics,
):
    return {
        "experiment": "dinov3_plant_damage_paired_mean",
        "version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "manifest_sha256": manifests["manifest_sha256"],
        "feature_dim": dimension,
        "base_checkpoint": str(Path(config.base_checkpoint).resolve()),
        "base_checkpoint_epoch": base_state.get("epoch"),
        "base_checkpoint_manifests": base_state.get("manifests"),
        "target_mean": float(base_state["target_mean"]),
        "target_std": float(base_state["target_std"]),
        "target_training_mean": float(
            base_state.get("target_training_mean", base_state["target_mean"])
        ),
        "train_filenames": train_names,
        "validation_filenames": val_names,
        "test_filenames": test_names,
        "epoch": epoch,
        "history": history,
        "validation": metrics,
        "best_validation_mse": best_mse,
        "patience": patience,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "test_used_for_selection": False,
    }


def _metric_values(target, prediction):
    result = regression_metrics(target, prediction)
    result["mse"] = result["rmse"] ** 2
    return result


def _save_split(
    model,
    loader,
    scaler,
    destination,
    config,
    device,
    checkpoint,
    split,
    reference_predictions=None,
):
    details = predict(model, loader, device, scaler)
    report = save(details, scaler, destination, config)
    report["model"]["mse"] = report["model"]["rmse"] ** 2
    report.update(
        {
            "split": split,
            "checkpoint": str(checkpoint),
            "target_definition": "arithmetic mean of score_jlu and score_gau",
            "selection": "lowest normalized MSE on gold validation only",
        }
    )
    if reference_predictions is not None:
        old = pd.read_csv(reference_predictions)
        new = pd.read_csv(destination / "predictions.csv")
        comparison = new[["filename", "target", "prediction"]].merge(
            old[["filename", "target", "prediction"]],
            on="filename",
            how="inner",
            validate="one_to_one",
            suffixes=("_new", "_previous"),
        )
        if len(comparison) != len(new) or len(comparison) != len(old):
            raise ValueError(
                "Previous-best validation rows do not exactly match this validation set"
            )
        if not np.allclose(comparison["target_new"], comparison["target_previous"], atol=1e-5):
            raise ValueError("Previous-best validation targets changed")
        comparison["new_absolute_error"] = np.abs(
            comparison["prediction_new"] - comparison["target_new"]
        )
        comparison["previous_absolute_error"] = np.abs(
            comparison["prediction_previous"] - comparison["target_new"]
        )
        comparison["absolute_error_reduction"] = (
            comparison["previous_absolute_error"] - comparison["new_absolute_error"]
        )
        comparison.to_csv(destination / "paired_previous_best_comparison.csv", index=False)
        previous = _metric_values(comparison["target_new"], comparison["prediction_previous"])
        current = _metric_values(comparison["target_new"], comparison["prediction_new"])
        report["previous_best_comparison"] = {
            "reference_predictions": str(reference_predictions),
            "samples": len(comparison),
            "previous_model": previous,
            "paired_mean_model": current,
            "mae_reduction": previous["mae"] - current["mae"],
            "mse_reduction": previous["mse"] - current["mse"],
            "r2_increase": current["r2"] - previous["r2"],
            "improved_images": int((comparison["absolute_error_reduction"] > 0).sum()),
            "worsened_images": int((comparison["absolute_error_reduction"] < 0).sum()),
        }
    write_json(destination / "summary.json", report)
    return report


def run(config: Config, resume: str | Path | None = None):
    manifests = make_manifests(config)
    seed_everything(config.seed, config.base.runtime.deterministic)
    device = resolve_device(config.base.runtime.device)
    configure_acceleration(config.base, device)
    base_state = load_checkpoint(config.base_checkpoint, device)
    train, validation, test, scaler, dimension, missing_base, missing_patches = prepare_data(
        config, base_state
    )
    train_loader = make_loader(train, scaler, config, training=True, offset=1000)
    val_loader = make_loader(validation, scaler, config, training=False, offset=2000)
    test_loader = make_loader(test, scaler, config, training=False, offset=3000)
    model = PlantDamageRegressor(dimension, config, base_state["model_state_dict"]).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    output = Path(config.run_dir)
    output.mkdir(parents=True, exist_ok=True)
    name = config.base.data.filename_column
    train_names = train[name].astype(str).tolist()
    val_names = validation[name].astype(str).tolist()
    test_names = test[name].astype(str).tolist()
    write_json(output / "config.json", asdict(config))
    write_json(output / "plant_reference_config.json", asdict(config.reference))
    write_json(output / "model_parameters.json", model.parameter_summary())
    write_json(output / "environment.json", environment_info(device, ROOT))
    write_json(
        output / "data_summary.json",
        {
            "paired_training": len(train),
            "gold_validation": len(validation),
            "gold_test": len(test),
            "discordant_missing_base_features": missing_base,
            "discordant_missing_patch_features": len(missing_patches),
            "manifest_counts": manifests,
        },
    )
    initial = predict(model, val_loader, device, scaler)
    initial_metrics = initial.result.metrics()
    base_mae = float(np.abs(initial.base_predictions - initial.result.targets).mean())
    if not np.allclose(initial.base_predictions, initial.result.predictions, atol=1e-5):
        raise RuntimeError("Fresh residual branch must reproduce the frozen best-model base")
    history = [
        {
            "epoch": 0,
            "train_mse": None,
            "validation_mse": float(initial_metrics["objective_mse"]),
            "validation_mae": float(initial_metrics["mae"]),
            "validation_r2": float(initial_metrics["r2"]),
            "base_mae": base_mae,
        }
    ]
    start, best_mse, patience = 1, float(initial_metrics["objective_mse"]), 0
    initial_state = _state(
        model,
        optimizer,
        config,
        manifests,
        base_state,
        dimension,
        0,
        history,
        best_mse,
        patience,
        train_names,
        val_names,
        test_names,
        initial_metrics,
    )
    save_checkpoint(output / "best_mse.pt", initial_state)
    if resume:
        state = load_checkpoint(resume, device)
        validate_checkpoint(state, config, manifests, dimension)
        if (
            state["train_filenames"] != train_names
            or state["validation_filenames"] != val_names
            or state["test_filenames"] != test_names
        ):
            raise ValueError("Selected rows differ from checkpoint")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        start = int(state["epoch"]) + 1
        best_mse = float(state["best_validation_mse"])
        patience = int(state["patience"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])
    for epoch in range(start, config.epochs + 1):
        if patience >= config.early_stopping_patience:
            break
        train_loader.sampler.set_epoch(epoch)
        train_mse = _train_epoch(model, train_loader, optimizer, device, config)
        validation_details = predict(model, val_loader, device, scaler)
        metrics = validation_details.result.metrics()
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": float(metrics["objective_mse"]),
                "validation_mae": float(metrics["mae"]),
                "validation_r2": float(metrics["r2"]),
                "base_mae": base_mae,
            }
        )
        improved = metrics["objective_mse"] < best_mse - 1e-4
        if improved:
            best_mse, patience = float(metrics["objective_mse"]), 0
        else:
            patience += 1
        state = _state(
            model,
            optimizer,
            config,
            manifests,
            base_state,
            dimension,
            epoch,
            history,
            best_mse,
            patience,
            train_names,
            val_names,
            test_names,
            metrics,
        )
        save_checkpoint(output / "last.pt", state)
        if improved:
            save_checkpoint(output / "best_mse.pt", state)
        write_json(output / "history.json", history)
        if config.base.output.save_plots:
            _history_plot(history, output / "training_history.png")
        print(
            f"Epoch {epoch:03d}/{config.epochs} | paired train MSE {train_mse:.4f} | "
            f"gold val MSE {metrics['objective_mse']:.4f} | MAE {metrics['mae']:.3f} | "
            f"R² {metrics['r2']:.3f} | patience "
            f"{patience}/{config.early_stopping_patience}",
            flush=True,
        )
    checkpoint = output / "best_mse.pt"
    selected = load_checkpoint(checkpoint, device)
    validate_checkpoint(selected, config, manifests, dimension)
    model.load_state_dict(selected["model_state_dict"])
    validation_report = _save_split(
        model,
        val_loader,
        scaler,
        output / "gold_validation",
        config,
        device,
        checkpoint,
        "gold_validation",
        config.reference_validation_predictions,
    )
    test_report = _save_split(
        model, test_loader, scaler, output / "gold_test", config, device, checkpoint, "gold_test"
    )
    summary = {
        "checkpoint": str(checkpoint),
        "best_epoch": int(selected["epoch"]),
        "paired_training_images": len(train),
        "gold_validation": validation_report["model"],
        "gold_validation_vs_previous_best": validation_report["previous_best_comparison"],
        "gold_test": test_report["model"],
        "gold_test_used_for_selection": False,
    }
    write_json(output / "run_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
