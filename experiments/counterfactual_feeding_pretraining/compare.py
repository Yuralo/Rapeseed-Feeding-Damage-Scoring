"""Paired, plot-clustered analysis across all predeclared gold-fitting seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rapeseed_damage.artifacts import write_json

from .config import Config, load_config
from .fit import run_dir


def _predictions(config: Config, arm: str, seed: int) -> pd.DataFrame:
    path = run_dir(config, arm, seed) / "validation" / "predictions.csv"
    table = pd.read_csv(path)
    required = {"filename", "target", "prediction", "base_prediction", "plot_group_id"}
    if required - set(table.columns):
        raise ValueError(f"Missing prediction fields in {path}")
    if table["filename"].duplicated().any() or table["plot_group_id"].isna().any():
        raise ValueError(f"Duplicate filenames or missing plot groups in {path}")
    if not np.isfinite(table[["target", "prediction", "base_prediction"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite prediction values in {path}")
    return table.sort_values("filename").reset_index(drop=True)


def _cluster_interval(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run(config: Config) -> dict:
    arms = ("random", "synthetic", "sham")
    rows = []
    reference = None
    reference_base = None
    for seed in config.fitting_seeds:
        per_seed = {arm: _predictions(config, arm, seed) for arm in arms}
        current = per_seed["random"]
        if reference is None:
            reference = current[["filename", "target", "plot_group_id"]].copy()
            reference_base = current["base_prediction"].to_numpy(dtype=float)
        for arm, table in per_seed.items():
            if table["filename"].tolist() != reference["filename"].tolist():
                raise ValueError(f"{arm} seed {seed} uses different validation filenames")
            if not np.allclose(table["target"], reference["target"], atol=1e-5):
                raise ValueError(f"{arm} seed {seed} uses different gold targets")
            if table["plot_group_id"].astype(str).tolist() != reference["plot_group_id"].astype(str).tolist():
                raise ValueError(f"{arm} seed {seed} uses different plot groups")
            if not np.allclose(table["base_prediction"], reference_base, atol=1e-4):
                raise ValueError("Frozen base predictions differ across arms or seeds")
            for index, value in table.iterrows():
                rows.append({"seed": seed, "arm": arm, "filename": value["filename"],
                             "group": str(value["plot_group_id"]),
                             "target": float(value["target"]),
                             "absolute_error": abs(float(value["prediction"] - value["target"])),
                             "base_absolute_error": abs(float(value["base_prediction"] - value["target"]))})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(config.run_dir) / "validation_paired_errors.csv", index=False)
    grouped = frame.groupby(["group", "seed", "arm"], as_index=False)["absolute_error"].mean()
    pivot = grouped.pivot(index=["group", "seed"], columns="arm", values="absolute_error")
    if pivot.isna().any().any():
        raise ValueError("Incomplete arm/seed/plot comparison")
    by_group = pivot.groupby(level="group").mean()
    base_group = frame.groupby("group")["base_absolute_error"].mean()
    def comparison(left: str, right: str):
        difference = (by_group[right] - by_group[left]).to_numpy(dtype=float)
        low, high = _cluster_interval(difference, config.bootstrap_replicates, config.seed)
        return {"mae_reduction": float(difference.mean()),
                "bootstrap_95_percent_interval": [low, high],
                "plots": len(difference)}
    bands = (("0_to_2_5", -np.inf, 2.5), ("over_2_5_to_7_5", 2.5, 7.5),
             ("over_7_5_to_15", 7.5, 15), ("over_15", 15, np.inf))
    band_metrics = {}
    for name, lower, upper in bands:
        subset = frame[(frame["target"] > lower) & (frame["target"] <= upper)]
        band_metrics[name] = {"images": subset["filename"].nunique(),
                              "mean_mae_by_arm": subset.groupby("arm")["absolute_error"].mean().to_dict()}
    report = {"selection_split": "gold_validation", "seeds": list(config.fitting_seeds),
              "images": reference.shape[0], "plots": len(by_group),
              "plot_weighted_mae_by_arm": by_group.mean().to_dict(),
              "plot_weighted_mae_by_seed": {
                  str(seed): values.to_dict() for seed, values in
                  pivot.groupby(level="seed").mean().iterrows()
              },
              "frozen_base_plot_mae": float(base_group.mean()),
              "synthetic_vs_random": comparison("synthetic", "random"),
              "synthetic_vs_sham": comparison("synthetic", "sham"),
              "target_bands": band_metrics,
              "test_used_for_selection": False}
    primary = report["synthetic_vs_random"]
    low = band_metrics["0_to_2_5"]["mean_mae_by_arm"]
    report["objective_gates"] = {
        "plot_bootstrap_excludes_zero": primary["bootstrap_95_percent_interval"][0] > 0,
        "synthetic_beats_frozen_base": report["plot_weighted_mae_by_arm"]["synthetic"] <
        report["frozen_base_plot_mae"],
        "low_score_degradation_at_most_0_25":
        low.get("synthetic", float("inf")) - low.get("random", 0) <= 0.25,
        "practical_0_25_reduction_reached": primary["mae_reduction"] >= 0.25,
    }
    write_json(Path(config.run_dir) / "validation_comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
