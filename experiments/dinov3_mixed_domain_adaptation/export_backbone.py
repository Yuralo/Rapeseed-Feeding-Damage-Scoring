"""Merge the best LoRA checkpoint into a standard Hugging Face DINOv3 directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoImageProcessor

from rapeseed_damage.artifacts import write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device

from .checkpoint import validate_for
from .config import load_config
from .model import DinoV3DomainAdapter


def run(config, *, checkpoint: str | Path | None = None, destination: str | Path | None = None):
    device = resolve_device(config.runtime.device)
    checkpoint_path = (
        Path(checkpoint)
        if checkpoint
        else (Path(config.output.run_dir) / config.output.best_checkpoint_name)
    )
    state = load_checkpoint(checkpoint_path, device)
    validate_for(state, config)
    model = DinoV3DomainAdapter(config, include_teacher=False).to(device)
    model.load_adaptation_state_dict(state["model_state_dict"])
    model.eval()
    merged = model.student.merge_and_unload()
    export_dir = (
        Path(destination)
        if destination
        else (Path(config.output.run_dir) / config.output.export_dir)
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(export_dir)
    processor = AutoImageProcessor.from_pretrained(config.model.processor)
    processor.save_pretrained(export_dir)
    metadata = {
        "source_checkpoint": str(checkpoint_path.resolve()),
        "export_directory": str(export_dir.resolve()),
        "base_backbone": config.model.backbone,
        "processor": config.model.processor,
        "checkpoint_epoch": int(state["epoch"]),
        "validation_metrics": state.get("metrics", {}),
        "representation_used_during_adaptation": "cls+mean_patches",
        "downstream_usage": {
            "backbone": str(export_dir.resolve()),
            "processor": str(export_dir.resolve()),
        },
    }
    write_json(export_dir / "adaptation_metadata.json", metadata)
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--destination")
    arguments = parser.parse_args(argv)
    report = run(
        load_config(arguments.config),
        checkpoint=arguments.checkpoint,
        destination=arguments.destination,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
