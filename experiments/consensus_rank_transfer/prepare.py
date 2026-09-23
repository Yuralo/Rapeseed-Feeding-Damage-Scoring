"""Extract frozen base predictions and pooled three-view embeddings once per split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.checkpoint import validate_for
from experiments.dinov3_hierarchical_three_view_mil.config import load_config as load_base_config
from experiments.dinov3_hierarchical_three_view_mil.data import (
    filter_usable_pretrain,
    load_manifest,
    make_loader,
    verify_features,
)
from experiments.dinov3_hierarchical_three_view_mil.model import HierarchicalThreeViewRegressor
from rapeseed_damage.artifacts import environment_info
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config: Config, splits: list[str]) -> dict:
    base_config = load_base_config(config.base_config)
    seed_everything(config.cv_seed, base_config.runtime.deterministic)
    device = resolve_device(base_config.runtime.device)
    configure_acceleration(base_config, device)
    state = load_checkpoint(config.base_checkpoint, device)
    validate_for(state, base_config)
    if state.get("stage") != "finetune":
        raise ValueError("Frozen base must be a gold-finetuned hierarchical checkpoint")
    scaler = TargetScaler(float(state["target_mean"]), float(state["target_std"]),
                          float(state.get("target_training_mean", state["target_mean"])))
    tables = {split: load_manifest(base_config, split) for split in splits}
    if "pretrain" in tables:
        tables["pretrain"], weak_missing = filter_usable_pretrain(tables["pretrain"], base_config)
    else:
        weak_missing = 0
    for split in ("finetune", "validation", "test"):
        if split in tables:
            expected = (state.get("manifests") or {}).get(split)
            observed = tables[split][base_config.data.filename_column].astype(str).tolist()
            if expected is None or list(map(str, expected)) != observed:
                raise ValueError(f"Base checkpoint {split} manifest differs from current manifest")
    if "pretrain" in tables:
        weak_groups = set(tables["pretrain"][base_config.data.group_column].astype(str))
        for split in ("validation", "test"):
            other = load_manifest(base_config, split)
            if weak_groups & set(other[base_config.data.group_column].astype(str)):
                raise ValueError(f"Weak {split} plot overlap")
    dimension = verify_features(pd.concat(tables.values(), ignore_index=True), base_config)
    validate_for(state, base_config, dimension)
    model = HierarchicalThreeViewRegressor(dimension, base_config).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    pooled_batches: list[np.ndarray] = []

    def capture(_module, inputs):
        pooled_batches.append(inputs[0].detach().float().cpu().numpy().copy())

    handle = model.head[0].register_forward_pre_hook(capture)
    destination = Path(config.run_dir) / "frozen"
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"base_checkpoint": str(Path(config.base_checkpoint).resolve()),
               "base_checkpoint_sha256": _sha256(Path(config.base_checkpoint)),
               "base_config": str(Path(config.base_config).resolve()),
               "environment": environment_info(device, Path(__file__).resolve().parents[2]),
               "feature_dimension": dimension, "pooled_dimension": 3 * base_config.model.projection_dim,
               "weak_missing_features": weak_missing, "splits": {}}
    try:
        for split, table in tables.items():
            pooled_batches.clear()
            loader = make_loader(table, scaler, base_config, base_config.training.finetuning,
                                 training=False, seed_offset=7000)
            normalized: list[np.ndarray] = []
            names: list[str] = []
            with torch.inference_mode():
                for batch in loader:
                    output = model(
                        batch["global_feature"].to(device),
                        batch["cell_features"].to(device),
                        batch["plant_features"].to(device),
                        batch["plant_valid"].to(device),
                        batch["plant_cell_indices"].to(device),
                    )
                    normalized.append(output.detach().float().cpu().numpy())
                    names.extend(batch["filename"])
            expected_names = table[base_config.data.filename_column].astype(str).tolist()
            if names != expected_names or len(pooled_batches) != len(normalized):
                raise RuntimeError(f"Frozen {split} extraction order or hook count differs")
            pooled = np.concatenate(pooled_batches).astype(np.float32)
            base_prediction = scaler.inverse(np.concatenate(normalized)).astype(np.float32)
            if pooled.shape != (len(table), 3 * base_config.model.projection_dim):
                raise RuntimeError(f"Unexpected {split} pooled-feature shape: {pooled.shape}")
            if not np.isfinite(pooled).all() or not np.isfinite(base_prediction).all():
                raise RuntimeError(f"Nonfinite frozen {split} features or predictions")
            if split == "validation":
                reference_path = (Path(config.base_checkpoint).parent /
                                  "best_mae_evaluation" / "predictions.csv")
                if reference_path.is_file():
                    reference = pd.read_csv(reference_path)
                    if reference["filename"].astype(str).tolist() != names:
                        raise ValueError("Stored frozen-base validation filenames differ")
                    delta = float(np.max(np.abs(reference["prediction"].to_numpy(dtype=np.float32)
                                                - base_prediction)))
                    if delta > 1e-3:
                        raise ValueError(f"Frozen-base validation predictions changed (max {delta:.4g})")
                    summary["base_validation_reference_max_abs_difference"] = delta
            def strings(column: str, source: pd.DataFrame = table) -> np.ndarray:
                return source[column].astype(str).to_numpy(dtype=str)
            data = {
                "features": pooled,
                "base": base_prediction,
                "target": table[base_config.data.target_column].to_numpy(dtype=np.float32),
                "filename": strings(base_config.data.filename_column),
                "group": strings(base_config.data.group_column),
                "cohort": strings(base_config.data.cohort_column),
                "source_path": strings(base_config.data.absolute_path_column),
                "score_jlu": pd.to_numeric(table["score_jlu"], errors="coerce").to_numpy(dtype=np.float32),
                "score_gau": pd.to_numeric(table["score_gau"], errors="coerce").to_numpy(dtype=np.float32),
            }
            path = destination / f"{split}.npz"
            np.savez_compressed(path, **data)
            summary["splits"][split] = {"images": len(table), "file": str(path),
                                        "npz_sha256": _sha256(path),
                                        "manifest_sha256": _sha256(base_config.manifest_path(split)),
                                        "cohorts": table[base_config.data.cohort_column].value_counts().to_dict()}
            print(f"{split}: {len(table)} images -> {path}", flush=True)
    finally:
        handle.remove()
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/consensus_rank_transfer/config.toml")
    parser.add_argument("--split", action="append", choices=("pretrain", "finetune", "validation", "test"))
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.split or
                         ["pretrain", "finetune", "validation", "test"]), indent=2))


if __name__ == "__main__":
    main()
