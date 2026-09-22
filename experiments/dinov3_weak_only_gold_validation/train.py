"""Train a fresh three-view head on weak scores; evaluate only on all gold images."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.metrics import predict
from experiments.dinov3_hierarchical_three_view_mil.model import HierarchicalThreeViewRegressor
from experiments.dinov3_hierarchical_three_view_mil.reporting import (
    save_evaluation,
    save_history_plot,
    save_label_plot,
)
from experiments.dinov3_hierarchical_three_view_mil.train import empty_history
from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import make_loader, manifest_hashes, prepare_data

ROOT = Path(__file__).resolve().parents[2]


def _forward(model, batch, device):
    return model(
        batch["global_feature"].to(device, non_blocking=True),
        batch["cell_features"].to(device, non_blocking=True),
        batch["plant_features"].to(device, non_blocking=True),
        batch["plant_valid"].to(device, non_blocking=True),
        batch["plant_cell_indices"].to(device, non_blocking=True),
    )


def _scheduler(optimizer, config: Config, steps_per_epoch: int):
    total = max(1, config.epochs * steps_per_epoch)
    warmup = int(total * config.warmup_fraction)

    def multiplier(step: int):
        if warmup and step < warmup:
            return max(1, step + 1) / warmup
        progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return config.minimum_learning_rate_ratio + (1 - config.minimum_learning_rate_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _checkpoint(model, optimizer, scheduler, config: Config, scaler, dimension: int,
                epoch: int, metrics: dict, history: dict, best_mse: float, best_mae: float,
                patience: int):
    state = {
        "experiment": "dinov3_weak_only_gold_validation",
        "checkpoint_version": 1,
        "mode": config.mode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "feature_dim": dimension,
        "epoch": epoch,
        "metrics": metrics,
        "history": history,
        "best_validation_mse": best_mse,
        "best_validation_mae": best_mae,
        "patience": patience,
        "target_mean": scaler.mean,
        "target_std": scaler.std,
        "target_training_mean": scaler.baseline_mean,
        "config": asdict(config),
        "base_config": config.base.to_dict(),
        "manifest_hashes": manifest_hashes(config),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "gold_used_for_training": False,
        "gold_used_for_checkpoint_selection": True,
    }
    return state


def _restore(state: dict, model, optimizer, scheduler, config: Config, dimension: int):
    if state.get("experiment") != "dinov3_weak_only_gold_validation":
        raise ValueError("Cannot resume from an unrelated checkpoint")
    if state.get("checkpoint_version") != 1 or state.get("mode") != config.mode:
        raise ValueError("Checkpoint version or weak-label mode differs")
    if state.get("config") != asdict(config) or state.get("base_config") != config.base.to_dict():
        raise ValueError("Checkpoint configuration differs from this run")
    if state.get("manifest_hashes") != manifest_hashes(config):
        raise ValueError("Training or gold validation manifest changed")
    if int(state.get("feature_dim", -1)) != dimension:
        raise ValueError("Checkpoint feature dimension differs")
    if not state.get("gold_used_for_checkpoint_selection") or state.get("gold_used_for_training"):
        raise ValueError("Checkpoint does not satisfy gold validation-only policy")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng_states"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_states"]])


def _save_selected(model, loader, scaler, device, checkpoint: Path, destination: Path,
                   config: Config, selection: str):
    state = load_checkpoint(checkpoint, device)
    model.load_state_dict(state["model_state_dict"])
    result = predict(model, loader, device, scaler)
    report = save_evaluation(result, scaler, destination, config.routed_base)
    report.update({
        "selection_metric": selection,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(state["epoch"]),
        "training_supervision": config.mode,
        "gold_training_images": 0,
        "gold_validation_images": len(loader.dataset),
        "independent_gold_test_exists": False,
        "caution": (
            "All 470 gold images select the checkpoint. These validation metrics are not an "
            "independent final test estimate."
        ),
    })
    write_json(destination / "summary.json", report)
    return report


def _save_weak_training_diagnostics(model, train_table, scaler, config: Config, device,
                                    destination: Path):
    loader = make_loader(train_table, scaler, config, training=False, seed_offset=3000)
    result = predict(model, loader, device, scaler)
    base = config.routed_base
    expected_names = train_table[base.data.filename_column].astype(str).tolist()
    if result.filenames != expected_names:
        raise ValueError("Weak training predictions are out of manifest order")
    rows = []
    for index, (_, record) in enumerate(train_table.iterrows()):
        rows.append({
            "filename": result.filenames[index],
            "cohort_id": str(record[base.data.cohort_column]),
            "supervision_tier": str(record[base.data.supervision_tier_column]),
            "target": float(result.targets[index]),
            "prediction": float(result.predictions[index]),
            "residual": float(result.predictions[index] - result.targets[index]),
        })
    with (destination / "weak_train_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {}
    for cohort in sorted({row["cohort_id"] for row in rows}):
        subset = [row for row in rows if row["cohort_id"] == cohort]
        residuals = np.asarray([row["residual"] for row in subset], dtype=np.float64)
        report[cohort] = {
            "samples": len(subset),
            "mae_vs_weak_labels": float(np.abs(residuals).mean()),
            "rmse_vs_weak_labels": float(np.sqrt(np.square(residuals).mean())),
            "mean_residual": float(residuals.mean()),
        }
    write_json(destination / "weak_train_cohort_diagnostics.json", {
        "cohorts": report,
        "warning": "These are in-sample fits to weak labels, not external validation metrics.",
    })
    return report


def run(config: Config, resume: str | Path | None = None):
    seed_everything(config.seed, config.base.runtime.deterministic)
    device = resolve_device(config.base.runtime.device)
    configure_acceleration(config.base, device)
    train_table, gold_table, scaler, dimension = prepare_data(config)
    train_loader = make_loader(train_table, scaler, config, training=True, seed_offset=1000)
    validation_loader = make_loader(gold_table, scaler, config, training=False, seed_offset=2000)
    model = HierarchicalThreeViewRegressor(dimension, config.routed_base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = _scheduler(optimizer, config, len(train_loader))
    destination = Path(config.run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "config.json", asdict(config))
    write_json(destination / "model_parameters.json", model.parameter_summary())
    write_json(destination / "environment.json", environment_info(device, ROOT))
    write_json(destination / "data_summary.json", json.loads(
        (Path(config.manifest_dir) / "summary.json").read_text(encoding="utf-8")
    ))
    if config.base.output.save_plots:
        save_label_plot(train_table[config.base.data.target_column].to_numpy(),
                        destination / "weak_train_targets.png")
        save_label_plot(gold_table[config.base.data.target_column].to_numpy(),
                        destination / "gold_validation_targets.png")
    history = empty_history()
    start_epoch, best_mse, best_mae, patience = 1, float("inf"), float("inf"), 0
    if resume:
        state = load_checkpoint(resume, device)
        _restore(state, model, optimizer, scheduler, config, dimension)
        if not math.isclose(float(state["target_mean"]), scaler.mean, abs_tol=1e-6):
            raise ValueError("Weak target scaler differs from checkpoint")
        start_epoch = int(state["epoch"]) + 1
        best_mse = float(state["best_validation_mse"])
        best_mae = float(state["best_validation_mae"])
        patience = int(state["patience"])
        history = state["history"]
    for epoch in range(start_epoch, config.epochs + 1):
        if patience >= config.early_stopping_patience:
            break
        train_loader.sampler.set_epoch(epoch)
        model.train()
        started = perf_counter()
        total_squared = samples = 0
        for batch in train_loader:
            target = batch["target"].float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch, device)
            squared = torch.square(output.float() - target)
            loss = squared.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            total_squared += float(squared.detach().sum())
            samples += target.numel()
        train_loss = total_squared / samples
        seconds = perf_counter() - started
        validation = predict(model, validation_loader, device, scaler)
        metrics = validation.metrics()
        attention = validation.attention_metrics()
        history["train_loss"].append(train_loss)
        history["train_stage"].append("weak_only")
        history["train_objective"].append("normalized_mse")
        history["val_epochs"].append(epoch)
        history["val_loss"].append(metrics["objective_mse"])
        history["val_mae"].append(metrics["mae"])
        history["val_r2"].append(metrics["r2"])
        history["val_cell_entropy"].append(attention["cells"]["mean_normalized_entropy"])
        history["val_plant_entropy"].append(attention["plants"]["mean_normalized_entropy"])
        history["val_top_cell_mass"].append(attention["cells"]["mean_top_mass"])
        history["val_top_plant_mass"].append(attention["plants"]["mean_top_mass"])
        history["learning_rates"].append([float(group["lr"]) for group in optimizer.param_groups])
        history["epoch_seconds"].append(seconds)
        improved_mse = metrics["objective_mse"] < best_mse - 1e-4
        improved_mae = metrics["mae"] < best_mae - 1e-4
        if improved_mse:
            best_mse = float(metrics["objective_mse"])
            patience = 0
        else:
            patience += 1
        if improved_mae:
            best_mae = float(metrics["mae"])
        state = _checkpoint(model, optimizer, scheduler, config, scaler, dimension,
                            epoch, metrics, history, best_mse, best_mae, patience)
        save_checkpoint(destination / "last.pt", state)
        if improved_mse:
            save_checkpoint(destination / "best_mse.pt", state)
        if improved_mae:
            save_checkpoint(destination / "best_mae.pt", state)
        write_json(destination / "history.json", history)
        if config.base.output.save_plots:
            save_history_plot(history, destination / "training_history.png")
        print(
            f"Epoch {epoch:03d}/{config.epochs} | weak train MSE {train_loss:.5f} | "
            f"gold val MSE {metrics['objective_mse']:.5f} | MAE {metrics['mae']:.3f} | "
            f"R² {metrics['r2']:.3f} | patience {patience}/{config.early_stopping_patience} "
            f"| {seconds:.1f}s",
            flush=True,
        )
    best_path = destination / "best_mse.pt"
    best_mae_path = destination / "best_mae.pt"
    if not best_path.is_file() or not best_mae_path.is_file():
        raise RuntimeError("Training did not produce selected checkpoints")
    mse_report = _save_selected(model, validation_loader, scaler, device, best_path,
                                destination, config, "mse")
    mae_report = _save_selected(model, validation_loader, scaler, device, best_mae_path,
                                destination / "best_mae_evaluation", config, "mae")
    cohort_diagnostics = _save_weak_training_diagnostics(
        model, train_table, scaler, config, device, destination
    )
    summary = {
        "training_mode": config.mode,
        "weak_train_images": len(train_table),
        "gold_training_images": 0,
        "gold_validation_images": len(gold_table),
        "best_mse": mse_report["model"],
        "best_mae": mae_report["model"],
        "best_mse_epoch": mse_report["checkpoint_epoch"],
        "best_mae_epoch": mae_report["checkpoint_epoch"],
        "weak_training_cohorts": cohort_diagnostics,
        "all_gold_used_for_validation": True,
        "independent_gold_test_exists": False,
        "run_dir": str(destination),
    }
    write_json(destination / "run_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
