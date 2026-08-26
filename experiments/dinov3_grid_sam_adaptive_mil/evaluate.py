import argparse
import json
from pathlib import Path

from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from rapeseed_damage.artifacts import write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .checkpoint import validate_for
from .config import load_config
from .data import prepare_data
from .metrics import predict
from .model import SamAdaptiveMILRegressor
from .reporting import save_evaluation


def run(config, checkpoint_path, output_dir=None):
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config, device)
    state = load_checkpoint(checkpoint_path, device)
    validate_for(state, config)
    _, _, validation, scaler, dim, _, loader = prepare_data(config, state)
    validate_for(state, config, dim)
    model = SamAdaptiveMILRegressor(dim, config).to(device)
    model.load_state_dict(state["model_state_dict"])
    destination = Path(output_dir or Path(config.output.run_dir) / "evaluation")
    report = save_evaluation(predict(model, loader, device, scaler), scaler, destination, config)
    report.update(
        {
            "checkpoint": str(checkpoint_path),
            "validation_samples": len(validation),
            "feature_dim": dim,
            "device": str(device),
        }
    )
    write_json(destination / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(load_config(args.config), args.checkpoint, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
