"""Re-evaluate a saved adaptive weak-only checkpoint on all 470 gold images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.dinov3_grid_sam_adaptive_mil.metrics import predict
from experiments.dinov3_grid_sam_adaptive_mil.model import SamAdaptiveMILRegressor
from experiments.dinov3_grid_sam_adaptive_mil.reporting import save_evaluation
from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from rapeseed_damage.artifacts import write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import make_loader, prepare_gold_data
from .train import validate_checkpoint


def run(config: Config, checkpoint: str | Path, output_dir: str | Path | None = None):
    seed_everything(config.seed, config.adaptive.runtime.deterministic)
    device = resolve_device(config.adaptive.runtime.device)
    configure_acceleration(config.adaptive, device)
    state = load_checkpoint(checkpoint, device)
    gold, dimension = prepare_gold_data(config)
    validate_checkpoint(state, config, dimension)
    names = gold[config.weak.routed_base.data.filename_column].astype(str).tolist()
    if names != state["gold_validation_filenames"]:
        raise ValueError("Gold validation order differs from checkpoint")
    scaler = TargetScaler(
        mean=float(state["target_mean"]), std=float(state["target_std"]),
        training_mean=float(state["target_training_mean"]),
    )
    model = SamAdaptiveMILRegressor(dimension, config.adaptive).to(device)
    model.load_state_dict(state["model_state_dict"])
    loader = make_loader(gold, scaler, config, training=False, seed_offset=2000)
    destination = Path(output_dir) if output_dir else Path(config.run_dir) / "reevaluation"
    report = save_evaluation(predict(model, loader, device, scaler), scaler,
                             destination, config.adaptive)
    report.update({
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_epoch": int(state["epoch"]),
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
