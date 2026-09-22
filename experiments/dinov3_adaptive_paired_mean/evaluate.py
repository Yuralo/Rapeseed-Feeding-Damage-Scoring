"""Re-evaluate the selected paired-mean plant-damage checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_plant_damage_mil.model import PlantDamageRegressor
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import make_loader, prepare_data
from .manifests import make_manifests
from .train import _save_split, validate_checkpoint


def load_model(config: Config, checkpoint: str | Path):
    manifests = make_manifests(config)
    seed_everything(config.seed, config.base.runtime.deterministic)
    device = resolve_device(config.base.runtime.device)
    configure_acceleration(config.base, device)
    state = load_checkpoint(checkpoint, device)
    validate_checkpoint(state, config, manifests)
    base_state = load_checkpoint(config.base_checkpoint, device)
    train, validation, test, _, dimension, _, _ = prepare_data(config, base_state)
    validate_checkpoint(state, config, manifests, dimension)
    if (
        state["train_filenames"] != train[config.base.data.filename_column].astype(str).tolist()
        or state["validation_filenames"]
        != validation[config.base.data.filename_column].astype(str).tolist()
        or state["test_filenames"] != test[config.base.data.filename_column].astype(str).tolist()
    ):
        raise ValueError("Current split rows differ from checkpoint")
    scaler = TargetScaler(
        mean=float(state["target_mean"]),
        std=float(state["target_std"]),
        training_mean=float(state["target_training_mean"]),
    )
    model = PlantDamageRegressor(dimension, config, base_state["model_state_dict"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model, scaler, device, state, validation, test


def run(config: Config, checkpoint: str | Path, *, split="test", output_dir=None):
    if split not in {"validation", "test"}:
        raise ValueError("Only gold validation/test may be evaluated")
    model, scaler, device, _, validation, test = load_model(config, checkpoint)
    table = test if split == "test" else validation
    loader = make_loader(table, scaler, config, training=False, offset=3000)
    destination = Path(output_dir) if output_dir else Path(config.run_dir) / f"gold_{split}"
    return _save_split(
        model,
        loader,
        scaler,
        destination,
        config,
        device,
        checkpoint,
        f"gold_{split}",
        config.reference_validation_predictions if split == "validation" else None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                load_config(args.config),
                args.checkpoint,
                split=args.split,
                output_dir=args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
