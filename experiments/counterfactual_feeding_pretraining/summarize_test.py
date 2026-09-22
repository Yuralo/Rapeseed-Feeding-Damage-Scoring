"""Aggregate one validation-selected arm's five retrospective gold-test reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rapeseed_damage.artifacts import write_json

from .compare import _cluster_interval
from .config import Config, load_config
from .fit import run_dir


def run(config: Config, arm: str) -> dict:
    if arm not in {"random", "synthetic", "sham"}:
        raise ValueError("Unknown fitting arm")
    validation = Path(config.run_dir) / "validation_comparison.json"
    if not validation.is_file():
        raise FileNotFoundError("Complete the five-seed validation comparison before test reporting")
    reference = None
    rows = []
    seed_metrics = {}
    for seed in config.fitting_seeds:
        path = run_dir(config, arm, seed) / "test" / "predictions.csv"
        table = pd.read_csv(path, dtype={"plot_group_id": str}).sort_values("filename")
        table = table.reset_index(drop=True)
        required = {"filename", "plot_group_id", "target", "prediction", "base_prediction"}
        if required - set(table.columns) or table.empty:
            raise ValueError(f"Incomplete gold-test predictions: {path}")
        if table["filename"].duplicated().any() or table["plot_group_id"].isna().any():
            raise ValueError(f"Duplicate filenames or missing plots: {path}")
        numeric = table[["target", "prediction", "base_prediction"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Nonfinite gold-test predictions: {path}")
        if reference is None:
            reference = table[["filename", "plot_group_id", "target", "base_prediction"]].copy()
        elif (table["filename"].tolist() != reference["filename"].tolist() or
              table["plot_group_id"].tolist() != reference["plot_group_id"].tolist() or
              not np.allclose(table["target"], reference["target"], atol=1e-4) or
              not np.allclose(table["base_prediction"], reference["base_prediction"], atol=1e-4)):
            raise ValueError("Gold-test samples, targets, or frozen base differ across seeds")
        absolute = np.abs(table["prediction"] - table["target"])
        seed_metrics[str(seed)] = {
            "image_mae": float(absolute.mean()),
            "plot_weighted_mae": float(pd.DataFrame({
                "group": table["plot_group_id"], "error": absolute
            }).groupby("group")["error"].mean().mean()),
            "rmse": float(np.sqrt(np.square(table["prediction"] - table["target"]).mean())),
            "bias": float((table["prediction"] - table["target"]).mean()),
        }
        rows.append(pd.DataFrame({"seed": seed, "filename": table["filename"],
                                  "plot_group_id": table["plot_group_id"],
                                  "target": table["target"], "prediction": table["prediction"],
                                  "absolute_error": absolute}))
    pooled = pd.concat(rows, ignore_index=True)
    baseline = reference.copy()
    baseline["absolute_error"] = np.abs(baseline["base_prediction"] - baseline["target"])
    baseline_group = baseline.groupby("plot_group_id")["absolute_error"].mean()
    final_group = pooled.groupby("plot_group_id")["absolute_error"].mean()
    reduction = baseline_group - final_group
    low, high = _cluster_interval(reduction.to_numpy(), config.bootstrap_replicates, config.seed)
    bands = (("0_to_2_5", -np.inf, 2.5), ("over_2_5_to_7_5", 2.5, 7.5),
             ("over_7_5_to_15", 7.5, 15), ("over_15", 15, np.inf))
    band_metrics = {}
    for name, lower, upper in bands:
        initial = baseline[(baseline["target"] > lower) & (baseline["target"] <= upper)]
        final = pooled[(pooled["target"] > lower) & (pooled["target"] <= upper)]
        band_metrics[name] = {"images": len(initial),
                              "base_mae": float(initial["absolute_error"].mean())
                              if len(initial) else None,
                              "arm_mae": float(final["absolute_error"].mean())
                              if len(final) else None}
    report = {"arm_selected_on_validation": arm, "seeds": list(config.fitting_seeds),
              "images": len(baseline), "plots": len(baseline_group),
              "seed_metrics": seed_metrics,
              "mean_seed_image_mae": float(np.mean([entry["image_mae"]
                                                     for entry in seed_metrics.values()])),
              "mean_seed_plot_weighted_mae": float(final_group.mean()),
              "frozen_base_plot_weighted_mae": float(baseline_group.mean()),
              "plot_mae_reduction_vs_base": float(reduction.mean()),
              "plot_bootstrap_95_percent_interval": [low, high],
              "target_bands": band_metrics,
              "validation_comparison_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
              "test_used_for_selection": False,
              "warning": "Historical test results were visible during research; this is retrospective.",
              }
    root = Path(config.run_dir)
    pooled.to_csv(root / f"test_paired_errors_{arm}.csv", index=False)
    write_json(root / f"test_summary_{arm}.json", report)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    seeds = list(config.fitting_seeds)
    axes[0].plot(seeds, [seed_metrics[str(seed)]["plot_weighted_mae"] for seed in seeds],
                 "o-", label=arm, color="#54a24b")
    axes[0].axhline(baseline_group.mean(), color="#555555", linestyle="--", label="frozen base")
    axes[0].set(xlabel="Fitting seed", ylabel="Plot-weighted MAE", title="Historical gold test")
    axes[0].legend()
    x = np.arange(len(bands))
    axes[1].bar(x - .18, [band_metrics[name]["base_mae"]
                            if band_metrics[name]["base_mae"] is not None else np.nan
                            for name, _, _ in bands],
                width=.35, color="#555555", label="frozen base")
    axes[1].bar(x + .18, [band_metrics[name]["arm_mae"]
                            if band_metrics[name]["arm_mae"] is not None else np.nan
                            for name, _, _ in bands],
                width=.35, color="#54a24b", label=arm)
    axes[1].set(xticks=x, xticklabels=("0–2.5", "2.5–7.5", "7.5–15", ">15"),
                xlabel="Gold score", ylabel="Image MAE", title="Historical test by target band")
    axes[1].legend()
    figure.suptitle("Retrospective report; do not select from these results")
    figure.tight_layout()
    figure.savefig(root / f"test_summary_{arm}.png", dpi=170)
    plt.close(figure)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=("random", "synthetic", "sham"), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.arm), indent=2))


if __name__ == "__main__":
    main()
