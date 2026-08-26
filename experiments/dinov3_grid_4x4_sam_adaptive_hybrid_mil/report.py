"""Regenerate full hybrid reports for both selected checkpoints."""

import argparse
import json
from pathlib import Path

from rapeseed_damage.artifacts import write_json

from .config import load_config
from .evaluate import run as evaluate
from .reporting import save_history_plot


def run(config, mse_checkpoint=None, mae_checkpoint=None, output_dir=None):
    run_dir = Path(config.output.run_dir)
    root = Path(output_dir or run_dir / "posthoc_reports")
    checkpoints = {
        "best_mse": Path(mse_checkpoint or run_dir / config.output.best_checkpoint_name),
        "best_mae": Path(mae_checkpoint or run_dir / config.output.best_mae_checkpoint_name),
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))
    reports = {
        name: evaluate(config, checkpoint, root / name)
        for name, checkpoint in checkpoints.items()
    }
    history_path = run_dir / "history.json"
    if config.output.save_plots and history_path.is_file():
        with history_path.open() as handle:
            save_history_plot(json.load(handle), root / "training_history.png")
    reports["comparison"] = {
        metric: reports["best_mae"]["model"][metric] - reports["best_mse"]["model"][metric]
        for metric in ("mae", "rmse", "r2")
    }
    write_json(root / "checkpoint_comparison.json", reports)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mse-checkpoint")
    parser.add_argument("--mae-checkpoint")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                load_config(args.config),
                args.mse_checkpoint,
                args.mae_checkpoint,
                args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
