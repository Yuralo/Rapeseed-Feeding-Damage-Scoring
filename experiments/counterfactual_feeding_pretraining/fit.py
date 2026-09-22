"""Matched gold-only residual fits for random, bite, and sham initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.checkpoint import validate_for as validate_base
from experiments.dinov3_plant_damage_mil.data import make_loader, prepare_tables
from experiments.dinov3_plant_damage_mil.model import PlantDamageRegressor
from experiments.dinov3_plant_damage_mil.reporting import predict, save
from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .pretrain import load_local_state

ROOT = Path(__file__).resolve().parents[2]


def run_dir(config: Config, arm: str, seed: int) -> Path:
    return Path(config.run_dir) / "gold_fit" / arm / f"seed_{seed}"


def _plant_config(config: Config, arm: str, seed: int):
    base = config.plant
    return replace(base, run_dir=str(run_dir(config, arm, seed)), seed=seed)


def _plot_mae(details, table: pd.DataFrame, plant_config) -> float:
    names = table[plant_config.base.data.filename_column].astype(str).tolist()
    if details.result.filenames != names:
        raise ValueError("Prediction filenames do not match the gold manifest")
    errors = np.abs(details.result.predictions - details.result.targets)
    groups = table[plant_config.base.data.group_column].astype(str).tolist()
    if any(not group or group == "nan" for group in groups):
        raise ValueError("Gold manifest has missing plot groups")
    frame = pd.DataFrame({"group": groups, "absolute_error": errors})
    return float(frame.groupby("group")["absolute_error"].mean().mean())


def _scaler(state):
    return TargetScaler(mean=float(state["target_mean"]), std=float(state["target_std"]),
                        training_mean=float(state.get("target_training_mean", state["target_mean"])))


def _initial_state(config: Config, arm: str, dimension: int, device, model):
    if arm == "random":
        return None
    path = Path(config.run_dir) / "pretraining" / arm / "best.pt"
    state = load_checkpoint(path, device)
    if (state.get("experiment"), state.get("version"), state.get("mode")) != (
        "counterfactual_feeding_pretraining", 1, arm
    ):
        raise ValueError(f"Wrong pretraining checkpoint: {path}")
    if (state.get("config") != asdict(config) or
            state.get("plant_config") != asdict(config.plant)):
        raise ValueError("Pretraining config differs from the gold fit")
    if state.get("feature_dim") != dimension:
        raise ValueError("Pretraining and gold feature dimensions differ")
    manifest_hash = hashlib.sha256(Path(config.adaptation_manifest).read_bytes()).hexdigest()
    if state.get("manifest_sha256") != manifest_hash:
        raise ValueError("Adaptation manifest changed since pretraining")
    if arm == "synthetic":
        summary = json.loads((path.parent / "summary.json").read_text())
        if not summary["diagnostics"]["passes_synthetic_gate"]:
            raise ValueError("Synthetic pretraining failed its held-out bite/sham diagnostic gate")
    load_local_state(model, state["local_state_dict"])
    return str(path)


def _train_epoch(model, loader, optimizer, device, penalty):
    model.train()
    total = samples = 0
    for batch in loader:
        moved = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor)
                 else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output, attention = model(moved, return_attention=True)
        squared = torch.square(output.float() - moved["target"].float())
        loss = squared.mean() + penalty * torch.square(attention["residual"]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        total += float(squared.detach().sum())
        samples += moved["target"].numel()
    return total / samples


def _save_report(details, scaler, table, plant_config, destination, arm, seed):
    report = save(details, scaler, destination, plant_config)
    predictions = pd.read_csv(destination / "predictions.csv")
    if predictions["filename"].astype(str).tolist() != details.result.filenames:
        raise ValueError("Saved prediction order differs from evaluated order")
    predictions["plot_group_id"] = table[plant_config.base.data.group_column].astype(str).tolist()
    predictions["arm"] = arm
    predictions["seed"] = seed
    predictions.to_csv(destination / "predictions.csv", index=False)
    report["plot_weighted_mae"] = _plot_mae(details, table, plant_config)
    report["arm"] = arm
    report["seed"] = seed
    write_json(destination / "summary.json", report)
    return report


def run(config: Config, arm: str, seed: int) -> dict:
    if arm not in {"random", "synthetic", "sham"}:
        raise ValueError("arm must be random, synthetic, or sham")
    if seed not in config.fitting_seeds:
        raise ValueError(f"Seed {seed} is not in the predeclared fitting_seeds")
    plant_config = _plant_config(config, arm, seed)
    seed_everything(seed, plant_config.base.runtime.deterministic)
    device = resolve_device(plant_config.base.runtime.device)
    configure_acceleration(plant_config.base, device)
    base_state = load_checkpoint(plant_config.base_checkpoint, device)
    validate_base(base_state, plant_config.base)
    if base_state.get("stage") != "finetune":
        raise ValueError("Gold fit requires a gold-finetuned frozen hierarchical base")
    tables, dimension = prepare_tables(plant_config, base_state)
    validate_base(base_state, plant_config.base, dimension)
    scaler = _scaler(base_state)
    gold_mean = float(tables["finetune"][plant_config.base.data.target_column].mean())
    if not np.isclose(gold_mean, scaler.mean, atol=1e-4):
        raise ValueError("Base target scaler differs from gold training mean")
    train_loader = make_loader(tables["finetune"], scaler, plant_config, training=True,
                               offset=1000)
    val_loader = make_loader(tables["validation"], scaler, plant_config, training=False,
                             offset=2000)
    model = PlantDamageRegressor(dimension, plant_config, base_state["model_state_dict"]).to(device)
    source = _initial_state(config, arm, dimension, device, model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=plant_config.learning_rate, weight_decay=plant_config.weight_decay,
    )
    destination = run_dir(config, arm, seed)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "config.json", {"experiment": asdict(config),
                                              "plant": asdict(plant_config)})
    write_json(destination / "environment.json", environment_info(device, ROOT))
    initial = predict(model, val_loader, device, scaler)
    if not np.allclose(initial.base_predictions, initial.result.predictions, atol=1e-5):
        raise RuntimeError("Epoch 0 must reproduce the frozen base exactly")
    base_plot_mae = _plot_mae(initial, tables["validation"], plant_config)
    best = base_plot_mae
    history = [{"epoch": 0, "plot_mae": best,
                "image_mae": float(initial.result.metrics()["mae"]), "train_mse": None}]
    patience = 0

    def state(epoch, validation):
        return {"experiment": "counterfactual_feeding_gold_fit", "version": 1,
                "arm": arm, "seed": seed, "epoch": epoch,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "feature_dim": dimension, "config": asdict(config),
                "plant_config": asdict(plant_config),
                "base_checkpoint_manifests": base_state.get("manifests"),
                "base_checkpoint": str(Path(plant_config.base_checkpoint).resolve()),
                "pretraining_checkpoint": source,
                "target_mean": scaler.mean, "target_std": scaler.std,
                "target_training_mean": scaler.training_mean,
                "validation": validation, "history": history,
                "test_used_for_selection": False}

    save_checkpoint(destination / "best.pt", state(0, {"plot_mae": best}))
    for epoch in range(1, plant_config.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        train_mse = _train_epoch(model, train_loader, optimizer, device,
                                 plant_config.residual_penalty)
        details = predict(model, val_loader, device, scaler)
        plot_mae = _plot_mae(details, tables["validation"], plant_config)
        image_mae = float(details.result.metrics()["mae"])
        history.append({"epoch": epoch, "plot_mae": plot_mae,
                        "image_mae": image_mae, "train_mse": train_mse})
        checkpoint = state(epoch, {"plot_mae": plot_mae, "image_mae": image_mae})
        save_checkpoint(destination / "last.pt", checkpoint)
        if plot_mae < best - 1e-4:
            best = plot_mae
            patience = 0
            save_checkpoint(destination / "best.pt", checkpoint)
        else:
            patience += 1
        write_json(destination / "history.json", history)
        print(f"{arm} seed {seed} epoch {epoch:03d}: train MSE {train_mse:.4f}, "
              f"gold plot MAE {plot_mae:.3f}, image MAE {image_mae:.3f}", flush=True)
        if patience >= plant_config.patience:
            break
    selected = load_checkpoint(destination / "best.pt", device)
    model.load_state_dict(selected["model_state_dict"])
    final = predict(model, val_loader, device, scaler)
    report = _save_report(final, scaler, tables["validation"], plant_config,
                          destination / "validation", arm, seed)
    summary = {"arm": arm, "seed": seed, "best_epoch": selected["epoch"],
               "plot_weighted_validation_mae": report["plot_weighted_mae"],
               "image_validation_mae": report["model"]["mae"],
               "frozen_base_plot_mae": base_plot_mae, "train_images": len(tables["finetune"]),
               "validation_images": len(tables["validation"]),
               "reserved_test_images": len(tables["test"]), "pretraining_checkpoint": source,
               "test_evaluated": False}
    write_json(destination / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=("random", "synthetic", "sham"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.arm, args.seed), indent=2))


if __name__ == "__main__":
    main()
