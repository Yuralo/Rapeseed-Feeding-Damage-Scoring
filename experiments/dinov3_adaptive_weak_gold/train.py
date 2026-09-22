"""Fresh adaptive MIL head: weak-only optimizer, all-gold checkpoint selection."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from experiments.dinov3_grid_sam_adaptive_mil.metrics import predict
from experiments.dinov3_grid_sam_adaptive_mil.model import SamAdaptiveMILRegressor
from experiments.dinov3_grid_sam_adaptive_mil.reporting import (
    save_evaluation,
    save_history_plot,
    save_label_plot,
)
from experiments.dinov3_grid_tiled_mil.runtime import (
    configure_acceleration,
    learning_rates,
    make_optimizer,
    make_scheduler,
)
from experiments.dinov3_weak_only_gold_validation.data import manifest_hashes
from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import make_loader, prepare_data

ROOT = Path(__file__).resolve().parents[2]


def empty_history():
    return {
        "train_loss": [], "val_epochs": [], "val_loss": [], "val_mae": [],
        "val_r2": [], "val_attention_entropy": [], "val_top_mass": [],
        "learning_rates": [], "epoch_seconds": [],
    }


def _checkpoint(model, optimizer, scheduler, config, scaler, dimension, epoch,
                history, metrics, best_mse, best_mae, patience, train_names, gold_names):
    return {
        "experiment": "dinov3_adaptive_weak_gold",
        "version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(config),
        "adaptive_config": config.adaptive.to_dict(),
        "manifest_hashes": manifest_hashes(config.weak),
        "weak_training_filenames": train_names,
        "gold_validation_filenames": gold_names,
        "feature_dim": dimension,
        "target_mean": scaler.mean,
        "target_std": scaler.std,
        "target_training_mean": scaler.baseline_mean,
        "epoch": epoch,
        "metrics": metrics,
        "history": history,
        "best_validation_mse": best_mse,
        "best_validation_mae": best_mae,
        "patience": patience,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "gold_used_for_training": False,
        "gold_used_for_checkpoint_selection": True,
    }


def validate_checkpoint(state, config: Config, dimension: int | None = None):
    if state.get("experiment") != "dinov3_adaptive_weak_gold" or state.get("version") != 1:
        raise ValueError("Not an adaptive weak/gold checkpoint")
    if state.get("config") != asdict(config) or state.get("adaptive_config") != config.adaptive.to_dict():
        raise ValueError("Checkpoint configuration differs")
    if state.get("manifest_hashes") != manifest_hashes(config.weak):
        raise ValueError("Weak/gold manifests changed after checkpoint creation")
    if dimension is not None and state.get("feature_dim") != dimension:
        raise ValueError("Checkpoint feature dimension differs")
    if state.get("gold_used_for_training") or not state.get("gold_used_for_checkpoint_selection"):
        raise ValueError("Checkpoint does not satisfy gold-validation-only policy")


def _save_selected(model, loader, scaler, device, checkpoint, destination, config, selection):
    state = load_checkpoint(checkpoint, device)
    model.load_state_dict(state["model_state_dict"])
    result = predict(model, loader, device, scaler)
    report = save_evaluation(result, scaler, destination, config.adaptive)
    report.update({
        "selection_metric": selection,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(state["epoch"]),
        "weak_training_images": len(state["weak_training_filenames"]),
        "gold_training_images": 0,
        "gold_validation_images": len(state["gold_validation_filenames"]),
        "independent_gold_test_exists": False,
        "caution": "The same 470 gold images select checkpoints; this is validation, not a final test.",
    })
    write_json(destination / "summary.json", report)
    return report


def _weak_diagnostics(model, table, scaler, config, device, destination):
    result = predict(model, make_loader(table, scaler, config, training=False,
                                        seed_offset=3000), device, scaler)
    rows = pd.DataFrame({
        "filename": result.filenames,
        "cohort_id": table[config.weak.routed_base.data.cohort_column].astype(str),
        "target": result.targets,
        "prediction": result.predictions,
        "residual": result.predictions - result.targets,
    })
    rows.to_csv(destination / "weak_train_predictions.csv", index=False)
    groups = {}
    for cohort, group in rows.groupby("cohort_id"):
        error = group["residual"].to_numpy(dtype=np.float64)
        groups[cohort] = {
            "samples": len(group), "mae_vs_weak_labels": float(np.abs(error).mean()),
            "mean_residual": float(error.mean()),
        }
    write_json(destination / "weak_train_cohort_diagnostics.json", {
        "cohorts": groups,
        "warning": "In-sample fit to weak labels, not external validation.",
    })
    return groups


def run(config: Config, resume: str | Path | None = None):
    seed_everything(config.seed, config.adaptive.runtime.deterministic)
    device = resolve_device(config.adaptive.runtime.device)
    configure_acceleration(config.adaptive, device)
    train_table, gold_table, scaler, dimension, omitted = prepare_data(config)
    train_loader = make_loader(train_table, scaler, config, training=True, seed_offset=1000)
    gold_loader = make_loader(gold_table, scaler, config, training=False, seed_offset=2000)
    model = SamAdaptiveMILRegressor(dimension, config.adaptive).to(device)
    optimizer = make_optimizer(model, config.adaptive)
    scheduler, _, _ = make_scheduler(optimizer, config.adaptive, len(train_loader))
    output = Path(config.run_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_names = train_table[config.weak.routed_base.data.filename_column].astype(str).tolist()
    gold_names = gold_table[config.weak.routed_base.data.filename_column].astype(str).tolist()
    write_json(output / "config.json", asdict(config))
    write_json(output / "adaptive_reference_config.json", config.adaptive.to_dict())
    write_json(output / "model_parameters.json", model.parameter_summary())
    write_json(output / "environment.json", environment_info(device, ROOT))
    data_report = {
        "eligible_weak_images": len(train_table) + len(omitted),
        "actual_weak_training_images": len(train_table),
        "omitted_weak_images": len(omitted),
        "gold_validation_images": len(gold_table),
        "gold_training_images": 0,
    }
    write_json(output / "data_summary.json", data_report)
    omitted.to_csv(output / "omitted_weak_feature_rows.csv", index=False)
    if config.adaptive.output.save_plots:
        save_label_plot(train_table[config.weak.routed_base.data.target_column].to_numpy(),
                        output / "weak_train_targets.png")
        save_label_plot(gold_table[config.weak.routed_base.data.target_column].to_numpy(),
                        output / "gold_validation_targets.png")
    history = empty_history()
    start, best_mse, best_mae, patience = 1, float("inf"), float("inf"), 0
    if resume:
        state = load_checkpoint(resume, device)
        validate_checkpoint(state, config, dimension)
        if state["weak_training_filenames"] != train_names or state["gold_validation_filenames"] != gold_names:
            raise ValueError("Selected weak/gold samples differ from checkpoint")
        if not math.isclose(float(state["target_mean"]), scaler.mean, abs_tol=1e-6):
            raise ValueError("Weak target scaler differs from checkpoint")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_states"]])
        history = state["history"]
        start = int(state["epoch"]) + 1
        best_mse = float(state["best_validation_mse"])
        best_mae = float(state["best_validation_mae"])
        patience = int(state["patience"])
    for epoch in range(start, config.epochs + 1):
        if patience >= config.early_stopping_patience:
            break
        train_loader.sampler.set_epoch(epoch)
        model.train()
        total = samples = 0
        started = perf_counter()
        for batch in train_loader:
            context = batch["context_features"].to(device, non_blocking=True)
            plants = batch["instance_features"].to(device, non_blocking=True)
            valid = batch["instance_valid"].to(device, non_blocking=True)
            target = batch["target"].float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(context, plants, valid)
            squared = torch.square(prediction.float() - target)
            loss = squared.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            total += float(squared.detach().sum())
            samples += target.numel()
        result = predict(model, gold_loader, device, scaler)
        metrics, attention = result.metrics(), result.attention_metrics()
        history["train_loss"].append(total / samples)
        history["val_epochs"].append(epoch)
        history["val_loss"].append(metrics["objective_mse"])
        history["val_mae"].append(metrics["mae"])
        history["val_r2"].append(metrics["r2"])
        history["val_attention_entropy"].append(attention["mean_normalized_entropy"])
        history["val_top_mass"].append(attention["mean_top_instance_mass"])
        history["learning_rates"].append(learning_rates(optimizer))
        history["epoch_seconds"].append(perf_counter() - started)
        improved_mse = metrics["objective_mse"] < best_mse - config.adaptive.training.early_stopping_min_delta
        improved_mae = metrics["mae"] < best_mae - 1e-4
        if improved_mse:
            best_mse, patience = float(metrics["objective_mse"]), 0
        else:
            patience += 1
        if improved_mae:
            best_mae = float(metrics["mae"])
        state = _checkpoint(model, optimizer, scheduler, config, scaler, dimension, epoch,
                            history, metrics, best_mse, best_mae, patience, train_names, gold_names)
        save_checkpoint(output / "last.pt", state)
        if improved_mse:
            save_checkpoint(output / "best_mse.pt", state)
        if improved_mae:
            save_checkpoint(output / "best_mae.pt", state)
        write_json(output / "history.json", history)
        if config.adaptive.output.save_plots:
            save_history_plot(history, output / "training_history.png")
        print(
            f"Epoch {epoch:03d}/{config.epochs} | weak train MSE {total / samples:.5f} | "
            f"gold val MSE {metrics['objective_mse']:.5f} | MAE {metrics['mae']:.3f} | "
            f"R² {metrics['r2']:.3f} | patience {patience}/{config.early_stopping_patience} | "
            f"{history['epoch_seconds'][-1]:.1f}s", flush=True,
        )
    if not (output / "best_mse.pt").is_file() or not (output / "best_mae.pt").is_file():
        raise RuntimeError("Training did not produce selected checkpoints")
    mse_report = _save_selected(model, gold_loader, scaler, device, output / "best_mse.pt",
                                output, config, "mse")
    mae_report = _save_selected(model, gold_loader, scaler, device, output / "best_mae.pt",
                                output / "best_mae_evaluation", config, "mae")
    cohorts = _weak_diagnostics(model, train_table, scaler, config, device, output)
    summary = {
        **data_report, "best_mse": mse_report["model"], "best_mae": mae_report["model"],
        "best_mse_epoch": mse_report["checkpoint_epoch"],
        "best_mae_epoch": mae_report["checkpoint_epoch"],
        "weak_training_cohorts": cohorts,
        "independent_gold_test_exists": False,
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
