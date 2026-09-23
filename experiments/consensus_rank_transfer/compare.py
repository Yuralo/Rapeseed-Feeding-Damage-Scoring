"""Strict matched-arm comparisons with plot-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .engine import ARMS
from .evaluate import OOD_COHORTS


def _path(config: Config, split: str, arm: str, seed: int, cohort: str | None) -> Path:
    directory = Path(config.run_dir) / "fits" / arm / f"seed_{seed}"
    if split == "ood":
        directory = directory / "ood" / str(cohort)
    else:
        directory = directory / split
    return directory / "predictions.csv"


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _compare_one(config: Config, split: str, cohort: str | None) -> tuple[dict, list[dict]]:
    tables = {(arm, seed): _read(_path(config, split, arm, seed, cohort))
              for arm in ARMS for seed in config.seeds}
    reference = tables[(ARMS[0], config.seeds[0])]
    keys = ("filename", "plot_group_id", "cohort_id", "source_path", "target",
            "base_prediction")
    for (arm, seed), rows in tables.items():
        if len(rows) != len(reference):
            raise ValueError(f"Unmatched image counts for {arm}/{seed}/{split}/{cohort}")
        for left, right in zip(reference, rows, strict=True):
            if any(left[key] != right[key] for key in keys):
                raise ValueError(f"Unmatched predictions or base in {arm}/{seed}/{split}/{cohort}")
    groups = np.array([row["plot_group_id"] for row in reference])
    targets = np.array([float(row["target"]) for row in reference])
    base = np.array([float(row["base_prediction"]) for row in reference])
    per_seed = {}
    predictions = {}
    for arm in ARMS:
        arm_predictions = np.array([[float(row["prediction"]) for row in tables[(arm, seed)]]
                                    for seed in config.seeds])
        predictions[arm] = arm_predictions
        per_seed[arm] = {str(seed): {"image_mae": float(np.mean(np.abs(pred - targets))),
                                     "plot_weighted_mae": float(np.mean([
                                         np.mean(np.abs(pred[groups == group] - targets[groups == group]))
                                         for group in np.unique(groups)]))}
                         for seed, pred in zip(config.seeds, arm_predictions, strict=True)}
    group_ids = np.unique(groups)
    group_error = {"base": np.array([np.mean(np.abs(base[groups == group] - targets[groups == group]))
                                     for group in group_ids])}
    for arm in ARMS:
        group_error[arm] = np.array([
            np.mean(np.abs(predictions[arm][:, groups == group] - targets[groups == group]))
            for group in group_ids
        ])
    rng = np.random.default_rng(42)
    sampled = rng.integers(0, len(group_ids), size=(config.bootstrap_replicates, len(group_ids)))
    comparisons = {}
    for control in ("base", "gold_only", "midpoint", "shuffled"):
        difference = group_error[control] - group_error["rank"]
        boot = difference[sampled].mean(axis=1)
        comparisons[f"rank_vs_{control}"] = {
            "mae_reduction": float(difference.mean()),
            "bootstrap_95_percent_interval": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    low = targets <= 2.5
    low_base = float(np.mean(np.abs(base[low] - targets[low]))) if low.any() else None
    low_rank = float(np.mean(np.abs(predictions["rank"][:, low] - targets[low]))) if low.any() else None
    primary = comparisons["rank_vs_base"]
    gates = {"rank_beats_base_by_0_25": primary["mae_reduction"] >= config.practical_mae_reduction,
             "rank_vs_base_interval_excludes_zero": primary["bootstrap_95_percent_interval"][0] > 0,
             "rank_beats_gold_only_by_0_25": comparisons["rank_vs_gold_only"]["mae_reduction"]
             >= config.practical_mae_reduction,
             "rank_vs_gold_only_interval_excludes_zero":
             comparisons["rank_vs_gold_only"]["bootstrap_95_percent_interval"][0] > 0,
             "rank_vs_shuffled_interval_excludes_zero":
             comparisons["rank_vs_shuffled"]["bootstrap_95_percent_interval"][0] > 0,
             "low_score_degradation_at_most_0_25": low_rank is not None and
             (low_rank - low_base) <= config.low_score_max_degradation}
    summary = {"split": split, "cohort": cohort, "images": len(reference),
               "plots": len(group_ids), "seeds": list(config.seeds),
               "label_quality": "single_rater_weak" if split == "ood" else "gold_historical",
               "base_plot_weighted_mae": float(group_error["base"].mean()),
               "mean_plot_weighted_mae": {arm: float(group_error[arm].mean()) for arm in ARMS},
               "per_seed": per_seed, "paired_comparisons": comparisons,
               "low_score": {"images": int(low.sum()), "base_mae": low_base,
                             "rank_mae": low_rank},
               "decision_gates": gates if split == "validation" else None,
               "confirmatory": False,
               "warning": ("OOD scores are single-rater weak labels and were not used for selection"
                           if split == "ood" else
                           "Historical gold labels have been inspected in earlier experiments")}
    rows = []
    for index, original in enumerate(reference):
        row = {key: original[key] for key in ("filename", "plot_group_id", "cohort_id",
                                                "source_path", "target", "base_prediction")}
        for arm in ARMS:
            row[f"{arm}_mean_prediction"] = float(predictions[arm][:, index].mean())
            row[f"{arm}_mean_absolute_error"] = float(
                np.abs(predictions[arm][:, index] - targets[index]).mean())
        row["base_absolute_error"] = float(abs(base[index] - targets[index]))
        rows.append(row)
    return summary, rows


def run(config: Config, split: str) -> dict:
    if split not in ("validation", "test", "ood"):
        raise ValueError("Unknown split")
    if split != "validation" and not (Path(config.run_dir) / "validation_comparison.json").is_file():
        raise FileNotFoundError("Run validation comparison first")
    cohorts = OOD_COHORTS if split == "ood" else (None,)
    results = {}
    for cohort in cohorts:
        summary, rows = _compare_one(config, split, cohort)
        name = f"ood_{cohort}" if cohort else split
        destination = Path(config.run_dir) / f"{name}_comparison.json"
        destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        with (destination.parent / f"{name}_matched_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        results[name] = summary
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/consensus_rank_transfer/config.toml")
    parser.add_argument("--split", choices=("validation", "test", "ood"), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.split), indent=2))


if __name__ == "__main__":
    main()
