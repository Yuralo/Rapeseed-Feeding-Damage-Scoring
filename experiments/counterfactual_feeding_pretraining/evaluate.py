"""Explicit validation or historical gold-test evaluation for one fitted arm."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.data import load_manifest, verify_features
from experiments.dinov3_plant_damage_mil.data import make_loader, verify_patch_features
from experiments.dinov3_plant_damage_mil.model import PlantDamageRegressor
from experiments.dinov3_plant_damage_mil.reporting import predict
from rapeseed_damage.artifacts import write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .fit import _plant_config, _save_report, run_dir


def run(config: Config, arm: str, seed: int, split: str = "validation") -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("Only gold validation and historical test are supported")
    plant = _plant_config(config, arm, seed)
    seed_everything(seed, plant.base.runtime.deterministic)
    device = resolve_device(plant.base.runtime.device)
    configure_acceleration(plant.base, device)
    checkpoint = run_dir(config, arm, seed) / "best.pt"
    state = load_checkpoint(checkpoint, device)
    if (state.get("experiment"), state.get("version"), state.get("arm"), state.get("seed")) != (
        "counterfactual_feeding_gold_fit", 1, arm, seed
    ):
        raise ValueError("Wrong counterfactual gold-fit checkpoint")
    if state.get("config") != asdict(config) or state.get("plant_config") != asdict(plant):
        raise ValueError("Gold-fit checkpoint configuration differs")
    table = load_manifest(plant.base, split)
    names = table[plant.base.data.filename_column].astype(str).tolist()
    expected = (state.get("base_checkpoint_manifests") or {}).get(split, [])
    if names != list(map(str, expected)):
        raise ValueError(f"Gold {split} rows differ from the frozen base checkpoint")
    dimension = verify_features(table, plant.base)
    verify_patch_features(plant, table)
    if dimension != state["feature_dim"]:
        raise ValueError("Feature dimension differs from saved fit")
    base_weights = {key.removeprefix("base."): value for key, value in
                    state["model_state_dict"].items() if key.startswith("base.")}
    model = PlantDamageRegressor(dimension, plant, base_weights).to(device)
    model.load_state_dict(state["model_state_dict"])
    scaler = TargetScaler(mean=float(state["target_mean"]), std=float(state["target_std"]),
                          training_mean=float(state["target_training_mean"]))
    loader = make_loader(table, scaler, plant, training=False,
                         offset=3000 if split == "test" else 2000)
    destination = run_dir(config, arm, seed) / split
    report = _save_report(predict(model, loader, device, scaler), scaler, table, plant,
                          destination, arm, seed)
    report["split"] = split
    report["checkpoint"] = str(checkpoint.resolve())
    report["checkpoint_epoch"] = state["epoch"]
    report["historical_test_warning"] = ("This test split was visible in earlier experiments; "
        "it is retrospective, not an independent confirmation.") if split == "test" else None
    write_json(destination / "summary.json", report)
    return {"arm": arm, "seed": seed, "split": split, "epoch": state["epoch"],
            "plot_weighted_mae": report["plot_weighted_mae"],
            "image_mae": report["model"]["mae"]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=("random", "synthetic", "sham"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.arm, args.seed, args.split), indent=2))


if __name__ == "__main__":
    main()
