"""External-cohort diagnostics against scored but non-gold source datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.data import (
    load_manifest,
    validate_split_isolation,
    verify_features,
)
from rapeseed_damage.artifacts import write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .data import make_loader, select_cohorts, verify_patch_features
from .model import PlantDamageRegressor
from .reporting import predict, save

DEFAULT_COHORTS = ("wg_insects_t1_bbch10", "dsv_asendorf_t1_bbch11")


def run(config: Config, checkpoint: str | Path, *, cohorts: list[str],
        limit_per_cohort: int | None, output_dir: str | Path | None = None):
    seed_everything(config.seed, config.base.runtime.deterministic)
    device = resolve_device(config.base.runtime.device)
    configure_acceleration(config.base, device)
    state = load_checkpoint(checkpoint, device)
    if state.get("experiment") != "dinov3_plant_damage_mil" or state.get("version") != 1:
        raise ValueError("Not a compatible plant-damage checkpoint")
    if state.get("config") != asdict(config) or state.get("base_config") != config.base.to_dict():
        raise ValueError("Checkpoint and current experiment configuration differ")
    tables = {
        split: load_manifest(config.base, split)
        for split in ("pretrain", "finetune", "validation", "test")
    }
    validate_split_isolation(tables, config.base)
    saved = state.get("base_checkpoint_manifests") or {}
    for split, table in tables.items():
        names = table[config.base.data.filename_column].astype(str).tolist()
        if names != list(map(str, saved.get(split, []))):
            raise ValueError(f"{split} manifest differs from the frozen base checkpoint")
    table = select_cohorts(tables["pretrain"], config, cohorts, limit_per_cohort)
    train_cohorts = set(tables["finetune"][config.base.data.cohort_column].astype(str))
    if set(cohorts) & train_cohorts:
        raise ValueError("OOD cohorts must not include the gold training cohort")
    source_tiers = set(table[config.base.data.supervision_tier_column].astype(str))
    if "gold" in source_tiers:
        raise ValueError("OOD diagnostic expected non-gold labels")
    dimension = verify_features(table, config.base)
    verify_patch_features(config, table)
    if dimension != int(state["feature_dim"]):
        raise ValueError("DINO feature dimension differs from checkpoint")
    weights = {
        key.removeprefix("base."): value
        for key, value in state["model_state_dict"].items() if key.startswith("base.")
    }
    model = PlantDamageRegressor(dimension, config, weights).to(device)
    model.load_state_dict(state["model_state_dict"])
    scaler = TargetScaler(
        mean=float(state["target_mean"]), std=float(state["target_std"]),
        training_mean=float(state["target_training_mean"]),
    )
    destination = Path(output_dir) if output_dir else Path(config.run_dir) / (
        f"ood_weak_probe_{limit_per_cohort}" if limit_per_cohort else "ood_weak_full"
    )
    destination.mkdir(parents=True, exist_ok=True)
    reports = {}
    for cohort in cohorts:
        subset = table.loc[table[config.base.data.cohort_column].astype(str) == cohort].reset_index(drop=True)
        loader = make_loader(subset, scaler, config, training=False, offset=4000)
        details = predict(model, loader, device, scaler)
        report = save(details, scaler, destination / cohort, config)
        report.update({
            "cohort_id": cohort,
            "split": "external_cohort_weak_label_diagnostic",
            "gold_standard": False,
            "supervision_tiers": sorted(set(subset[config.base.data.supervision_tier_column].astype(str))),
            "mean_weak_target": float(np.mean(details.result.targets)),
            "mean_base_prediction": float(np.mean(details.base_predictions)),
            "mean_final_prediction": float(np.mean(details.result.predictions)),
            "checkpoint": str(Path(checkpoint).resolve()),
            "warning": (
                "These are single-/weak-rater targets from a different cohort. MAE is agreement "
                "with those labels, not gold-standard accuracy; source, stage and rater can all shift."
            ),
        })
        write_json(destination / cohort / "summary.json", report)
        reports[cohort] = {
            "samples": report["samples"], "labels": report["supervision_tiers"],
            "base_mae_vs_weak": report["paired_comparison"]["base_mae"],
            "final_mae_vs_weak": report["paired_comparison"]["final_mae"],
            "mean_weak_target": report["mean_weak_target"],
            "mean_base_prediction": report["mean_base_prediction"],
            "mean_final_prediction": report["mean_final_prediction"],
        }
    summary = {
        "cohorts": reports,
        "cohort_sampling": "target-blind random subset with the experiment seed",
        "limit_per_cohort": limit_per_cohort,
        "gold_test_used": False,
        "backbone_exposure": (
            "The adapted backbone may have seen these source cohorts without labels during "
            "domain adaptation. This probes shift for the gold-trained scoring head, not a "
            "guaranteed never-seen domain for the full system."
        ),
        "interpretation": (
            "Compare base and new model within each cohort. Do not rank these weak-label "
            "MAEs against the gold validation MAE or use the cohorts to tune the gold model."
        ),
    }
    write_json(destination / "ood_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cohorts", nargs="+", default=list(DEFAULT_COHORTS))
    parser.add_argument("--limit-per-cohort", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.checkpoint,
                         cohorts=args.cohorts, limit_per_cohort=args.limit_per_cohort,
                         output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
