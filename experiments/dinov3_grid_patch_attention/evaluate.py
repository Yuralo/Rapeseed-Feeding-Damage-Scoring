"""Evaluate a patch-attention checkpoint and save attention overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .checkpoint import validate_for
from .config import Config, load_config
from .metrics import predict
from .model import DinoV3PatchAttentionRegressor
from .reporting import save_evaluation
from .runtime import configure_acceleration
from .setup import prepare_data

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def run(config: Config, checkpoint_path: str | Path, output_dir=None) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config, device)
    state = load_checkpoint(checkpoint_path, device)
    validate_for(state, config)
    _, _, validation, scaler, _, validation_loader = prepare_data(config, state)
    model = DinoV3PatchAttentionRegressor(config).to(device)
    model.load_state_dict(state["model_state_dict"])
    destination = Path(output_dir or Path(config.output.run_dir) / "evaluation")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "config.json", config.to_dict())
    write_json(destination / "environment.json", environment_info(device, REPOSITORY_ROOT))
    report = save_evaluation(
        predict(model, validation_loader, device, scaler, config),
        scaler,
        destination,
        config,
    )
    report.update(
        {
            "checkpoint": str(checkpoint_path),
            "validation_samples": len(validation),
            "device": str(device),
        }
    )
    write_json(destination / "summary.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    arguments = parser.parse_args(argv)
    report = run(load_config(arguments.config), arguments.checkpoint, arguments.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
