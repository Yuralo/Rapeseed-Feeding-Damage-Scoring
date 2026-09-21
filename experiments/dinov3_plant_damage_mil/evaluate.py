"""Evaluate a plant-damage checkpoint on validation or the reserved gold test split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.data import load_manifest, verify_features
from rapeseed_damage.artifacts import write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import make_loader, verify_patch_features
from .model import PlantDamageRegressor
from .reporting import predict, save


def run(config: Config, checkpoint: str | Path, split: str, output_dir: str | Path | None = None):
    if split not in {"validation", "test"}:
        raise ValueError("Only validation or reserved gold test can be evaluated")
    seed_everything(config.seed, config.base.runtime.deterministic)
    device = resolve_device(config.base.runtime.device)
    configure_acceleration(config.base, device)
    state = load_checkpoint(checkpoint, device)
    if state.get("experiment") != "dinov3_plant_damage_mil" or state.get("version") != 1:
        raise ValueError("Not a compatible plant-damage checkpoint")
    if state.get("config") != asdict(config):
        raise ValueError("Checkpoint and experiment configuration differ")
    if state.get("base_config") != config.base.to_dict():
        raise ValueError("The underlying three-view configuration changed since training")
    table = load_manifest(config.base, split)
    names = table[config.base.data.filename_column].astype(str).tolist()
    if names != list(map(str, (state.get("base_checkpoint_manifests") or {}).get(split, []))):
        raise ValueError(f"Checkpoint and {split} manifest differ")
    dimension = verify_features(table, config.base)
    verify_patch_features(config, table)
    if dimension != int(state["feature_dim"]):
        raise ValueError("Three-view feature dimension differs from checkpoint")
    base_weights = {
        key.removeprefix("base."): value
        for key, value in state["model_state_dict"].items() if key.startswith("base.")
    }
    model = PlantDamageRegressor(dimension, config, base_weights).to(device)
    model.load_state_dict(state["model_state_dict"])
    scaler = TargetScaler(
        mean=float(state["target_mean"]), std=float(state["target_std"]),
        training_mean=float(state["target_training_mean"]),
    )
    loader = make_loader(table, scaler, config, training=False, offset=3000 if split == "test" else 2000)
    destination = Path(output_dir) if output_dir else Path(config.run_dir) / f"{split}_evaluation"
    report = save(predict(model, loader, device, scaler), scaler, destination, config)
    report.update({"split": split, "checkpoint": str(Path(checkpoint).resolve()),
                   "checkpoint_epoch": int(state["epoch"]),
                   "manifest": str(config.base.manifest_path(split).resolve())})
    write_json(destination / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.checkpoint, args.split,
                         args.output_dir), indent=2))


if __name__ == "__main__":
    main()
