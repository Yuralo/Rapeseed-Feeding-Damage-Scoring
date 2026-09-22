"""Re-evaluate a weak-trained checkpoint on the all-gold validation manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.metrics import predict
from experiments.dinov3_hierarchical_three_view_mil.model import HierarchicalThreeViewRegressor
from experiments.dinov3_hierarchical_three_view_mil.reporting import save_evaluation
from rapeseed_damage.artifacts import write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import make_loader, manifest_hashes, prepare_gold_data


def run(config: Config, checkpoint: str | Path, output_dir: str | Path | None = None):
    seed_everything(config.seed, config.base.runtime.deterministic)
    device = resolve_device(config.base.runtime.device)
    configure_acceleration(config.base, device)
    state = load_checkpoint(checkpoint, device)
    if state.get("experiment") != "dinov3_weak_only_gold_validation":
        raise ValueError("Not a weak-only/gold-validation checkpoint")
    if state.get("checkpoint_version") != 1 or state.get("mode") != config.mode:
        raise ValueError("Checkpoint version or training mode differs")
    if state.get("config") != asdict(config) or state.get("base_config") != config.base.to_dict():
        raise ValueError("Checkpoint and current configuration differ")
    if state.get("manifest_hashes") != manifest_hashes(config):
        raise ValueError("Checkpoint manifests differ")
    if state.get("gold_used_for_training") or not state.get("gold_used_for_checkpoint_selection"):
        raise ValueError("Checkpoint violates gold-validation-only policy")
    gold, dimension = prepare_gold_data(config)
    if dimension != int(state["feature_dim"]):
        raise ValueError("Feature dimension differs from checkpoint")
    scaler = TargetScaler(
        mean=float(state["target_mean"]), std=float(state["target_std"]),
        training_mean=float(state["target_training_mean"]),
    )
    loader = make_loader(gold, scaler, config, training=False, seed_offset=2000)
    model = HierarchicalThreeViewRegressor(dimension, config.routed_base).to(device)
    model.load_state_dict(state["model_state_dict"])
    destination = Path(output_dir) if output_dir else Path(config.run_dir) / "reevaluation"
    report = save_evaluation(predict(model, loader, device, scaler), scaler,
                             destination, config.routed_base)
    report.update({
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_epoch": int(state["epoch"]),
        "weak_training_mode": config.mode,
        "gold_training_images": 0,
        "gold_validation_images": len(gold),
        "independent_gold_test_exists": False,
    })
    write_json(destination / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.checkpoint, args.output_dir),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
