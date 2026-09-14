"""Evaluate a checkpoint on an explicit gold validation or reserved test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .checkpoint import validate_for
from .config import load_config
from .data import load_manifest, make_loader, verify_features
from .metrics import predict
from .model import HierarchicalThreeViewRegressor
from .reporting import save_evaluation

ROOT = Path(__file__).resolve().parents[2]


def run(config, checkpoint: str | Path, split: str, output_dir: str | Path | None = None):
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config, device)
    state = load_checkpoint(checkpoint, device)
    validate_for(state, config)
    table = load_manifest(config, split)
    names = table[config.data.filename_column].astype(str).tolist()
    if names != list(map(str, (state.get("manifests") or {}).get(split, []))):
        raise ValueError(f"Checkpoint {split} manifest differs from {config.manifest_path(split)}")
    dimension = verify_features(table, config)
    validate_for(state, config, dimension)
    scaler = TargetScaler(
        mean=float(state["target_mean"]),
        std=float(state["target_std"]),
        training_mean=float(state.get("target_training_mean", state["target_mean"])),
    )
    loader = make_loader(
        table, scaler, config, config.training.finetuning, training=False,
        seed_offset=3000 if split == "test" else 2000,
    )
    model = HierarchicalThreeViewRegressor(dimension, config).to(device)
    model.load_state_dict(state["model_state_dict"])
    destination = Path(output_dir) if output_dir else Path(config.output.run_dir) / f"{split}_evaluation"
    report = save_evaluation(predict(model, loader, device, scaler), scaler, destination, config)
    report.update({
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_stage": state["stage"],
        "checkpoint_stage_epoch": state["stage_epoch"],
        "split": split,
        "manifest": str(config.manifest_path(split).resolve()),
        "gold_only_evaluation": True,
        "device": str(device),
    })
    write_json(destination / "summary.json", report)
    write_json(destination / "config.json", config.to_dict())
    write_json(destination / "environment.json", environment_info(device, ROOT))
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir")
    arguments = parser.parse_args(argv)
    print(json.dumps(run(
        load_config(arguments.config), arguments.checkpoint, arguments.split, arguments.output_dir
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
