"""Matched weak-label external-cohort diagnostic for every arm and fitting seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.dinov3_grid_tiled_mil.data import TargetScaler
from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.data import (
    load_manifest,
    validate_split_isolation,
    verify_features,
)
from experiments.dinov3_plant_damage_mil.data import (
    make_loader,
    select_cohorts,
    verify_patch_features,
)
from experiments.dinov3_plant_damage_mil.model import PlantDamageRegressor
from experiments.dinov3_plant_damage_mil.reporting import predict
from rapeseed_damage.artifacts import write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import Config, load_config
from .fit import _plant_config, _save_report, run_dir

DEFAULT_COHORTS = ("wg_insects_t1_bbch10", "dsv_asendorf_t1_bbch11")
ARMS = ("random", "synthetic", "sham")


def _selected_table(config: Config, cohorts: list[str], limit_per_cohort: int | None):
    base = config.plant.base
    tables = {split: load_manifest(base, split)
              for split in ("pretrain", "finetune", "validation", "test")}
    validate_split_isolation(tables, base)
    selected = select_cohorts(tables["pretrain"], config.plant, cohorts, limit_per_cohort)
    if set(cohorts) & set(tables["finetune"][base.data.cohort_column].astype(str)):
        raise ValueError("External cohorts include a gold training cohort")
    if "gold" in set(selected[base.data.supervision_tier_column].astype(str)):
        raise ValueError("External diagnostic contains gold labels")
    if selected[base.data.group_column].isna().any():
        raise ValueError("External diagnostic contains missing plot groups")
    return tables, selected


def _checkpoint(config: Config, arm: str, seed: int, device, tables: dict):
    plant = _plant_config(config, arm, seed)
    path = run_dir(config, arm, seed) / "best.pt"
    state = load_checkpoint(path, device)
    if (state.get("experiment"), state.get("version"), state.get("arm"), state.get("seed")) != (
        "counterfactual_feeding_gold_fit", 1, arm, seed
    ):
        raise ValueError(f"Wrong gold-fit checkpoint: {path}")
    if state.get("config") != asdict(config) or state.get("plant_config") != asdict(plant):
        raise ValueError(f"Gold-fit checkpoint config changed: {path}")
    saved = state.get("base_checkpoint_manifests") or {}
    for split, table in tables.items():
        names = table[plant.base.data.filename_column].astype(str).tolist()
        if names != list(map(str, saved.get(split, []))):
            raise ValueError(f"Frozen base {split} manifest changed: {path}")
    return path, state, plant


def _plot(summary: dict, path: Path) -> None:
    cohorts = list(summary["cohorts"])
    figure, axes = plt.subplots(len(cohorts), 2, figsize=(11, 4 * len(cohorts)), squeeze=False)
    colors = {"base": "#555555", "random": "#4c78a8", "synthetic": "#54a24b",
              "sham": "#e4a64e"}
    for row, cohort in enumerate(cohorts):
        values = summary["cohorts"][cohort]
        labels = ["base", *ARMS]
        mae = [values["base_mae_vs_weak"]] + [values["arms"][arm]["mean_mae_vs_weak"]
                                                for arm in ARMS]
        bias = [values["base_bias_vs_weak"]] + [values["arms"][arm]["mean_bias_vs_weak"]
                                                  for arm in ARMS]
        for column, numbers in enumerate((mae, bias)):
            axis = axes[row, column]
            axis.bar(labels, numbers, color=[colors[label] for label in labels])
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{cohort}: {'MAE' if column == 0 else 'bias'} vs weak labels")
            axis.set_ylabel("Score points")
            axis.grid(axis="y", alpha=0.2)
        for x, arm in enumerate(ARMS, start=1):
            samples = values["arms"][arm]["seed_mae_vs_weak"]
            axes[row, 0].scatter([x] * len(samples), samples, c="black", s=13, zorder=3)
    figure.suptitle("External-cohort diagnostic: weak-label agreement, not gold accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(config: Config, *, cohorts: list[str] | None = None,
        limit_per_cohort: int | None = None) -> dict:
    cohorts = list(DEFAULT_COHORTS if cohorts is None else cohorts)
    if limit_per_cohort is not None and limit_per_cohort < 1:
        raise ValueError("limit_per_cohort must be positive")
    seed_everything(config.seed, config.plant.base.runtime.deterministic)
    device = resolve_device(config.plant.base.runtime.device)
    configure_acceleration(config.plant.base, device)
    tables, selected = _selected_table(config, cohorts, limit_per_cohort)
    dimension = verify_features(selected, config.plant.base)
    verify_patch_features(config.plant, selected)
    destination = Path(config.run_dir) / (f"ood_weak_probe_{limit_per_cohort}"
                                               if limit_per_cohort else "ood_weak_full")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "matched_sample_manifest.csv"
    selected.to_csv(manifest_path, index=False)
    selection_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows = []
    report_rows = []
    reference = {}
    for arm in ARMS:
        for seed in config.fitting_seeds:
            path, state, plant = _checkpoint(config, arm, seed, device, tables)
            if dimension != int(state["feature_dim"]):
                raise ValueError(f"DINO feature dimension differs from checkpoint: {path}")
            base_weights = {key.removeprefix("base."): value for key, value in
                            state["model_state_dict"].items() if key.startswith("base.")}
            model = PlantDamageRegressor(dimension, plant, base_weights).to(device)
            model.load_state_dict(state["model_state_dict"])
            scaler = TargetScaler(mean=float(state["target_mean"]),
                                  std=float(state["target_std"]),
                                  training_mean=float(state["target_training_mean"]))
            for cohort in cohorts:
                subset = selected.loc[
                    selected[plant.base.data.cohort_column].astype(str) == cohort
                ].reset_index(drop=True)
                loader = make_loader(subset, scaler, plant, training=False, offset=4000)
                details = predict(model, loader, device, scaler)
                output = destination / cohort / arm / f"seed_{seed}"
                report = _save_report(details, scaler, subset, plant, output, arm, seed)
                report.update({"cohort_id": cohort, "weak_label_diagnostic": True,
                               "gold_standard": False, "checkpoint": str(path.resolve()),
                               "sample_manifest_sha256": selection_hash})
                write_json(output / "summary.json", report)
                predictions = pd.read_csv(output / "predictions.csv")
                checked = predictions[["filename", "target", "base_prediction"]].copy()
                checked = checked.sort_values("filename").reset_index(drop=True)
                if cohort not in reference:
                    reference[cohort] = checked
                else:
                    expected = reference[cohort]
                    if (checked["filename"].tolist() != expected["filename"].tolist() or
                            not np.allclose(checked["target"], expected["target"], atol=1e-4) or
                            not np.allclose(checked["base_prediction"],
                                            expected["base_prediction"], atol=1e-4)):
                        raise ValueError(f"OOD samples, targets, or base predictions differ: {cohort}")
                predictions["cohort_id"] = cohort
                predictions["arm"] = arm
                predictions["seed"] = seed
                rows.append(predictions[["cohort_id", "arm", "seed", "filename", "plot_group_id",
                                         "target", "prediction", "base_prediction"]])
                report_rows.append({"cohort_id": cohort, "arm": arm, "seed": seed,
                                    "mae_vs_weak": float(report["model"]["mae"]),
                                    "bias_vs_weak": float((details.result.predictions -
                                                           details.result.targets).mean()),
                                    "plot_weighted_mae_vs_weak": report["plot_weighted_mae"]})
            del model
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(destination / "matched_predictions.csv", index=False)
    metrics = pd.DataFrame(report_rows)
    metrics.to_csv(destination / "seed_metrics.csv", index=False)
    summarized = {}
    for cohort in cohorts:
        frame = predictions.loc[predictions["cohort_id"] == cohort]
        baseline = frame.loc[(frame["arm"] == ARMS[0]) &
                             (frame["seed"] == config.fitting_seeds[0])]
        target = baseline["target"].to_numpy(dtype=float)
        base = baseline["base_prediction"].to_numpy(dtype=float)
        arm_metrics = {}
        for arm in ARMS:
            subset = metrics.loc[(metrics["cohort_id"] == cohort) & (metrics["arm"] == arm)]
            arm_metrics[arm] = {
                "mean_mae_vs_weak": float(subset["mae_vs_weak"].mean()),
                "mean_bias_vs_weak": float(subset["bias_vs_weak"].mean()),
                "seed_mae_vs_weak": subset["mae_vs_weak"].tolist(),
                "seed_plot_weighted_mae_vs_weak": subset["plot_weighted_mae_vs_weak"].tolist(),
            }
        plot_errors = frame.assign(
            absolute_error=(frame["prediction"] - frame["target"]).abs()
        ).groupby(["plot_group_id", "arm"])["absolute_error"].mean().unstack()
        summarized[cohort] = {"images": len(baseline),
                              "base_mae_vs_weak": float(np.abs(base - target).mean()),
                              "base_bias_vs_weak": float((base - target).mean()),
                              "synthetic_vs_random_plot_mae_reduction_vs_weak": float(
                                  (plot_errors["random"] - plot_errors["synthetic"]).mean()),
                              "arms": arm_metrics}
    summary = {
        "cohorts": summarized, "arms": list(ARMS), "seeds": list(config.fitting_seeds),
        "sample_manifest": str(manifest_path), "sample_manifest_sha256": selection_hash,
        "limit_per_cohort": limit_per_cohort, "gold_test_used": False,
        "selection_use": "Diagnostic only; do not select an arm, seed, or checkpoint on these labels.",
        "interpretation": (
            "These source cohorts have weak or single-rater scores. MAE measures agreement with "
            "those raters, not gold-standard accuracy. The adapted backbone and synthetic pretext "
            "may have seen these images without scores; this is a scoring-domain shift probe, "
            "not a fully unseen-domain claim."
        ),
    }
    _plot(summary, destination / "ood_diagnostic.png")
    write_json(destination / "ood_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohorts", nargs="+", default=list(DEFAULT_COHORTS))
    parser.add_argument("--limit-per-cohort", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), cohorts=args.cohorts,
                         limit_per_cohort=args.limit_per_cohort), indent=2))


if __name__ == "__main__":
    main()
