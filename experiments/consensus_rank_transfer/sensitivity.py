"""Fixed-settings ten-point pair-separation sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from rapeseed_damage.checkpointing import save_checkpoint
from rapeseed_damage.reproducibility import resolve_device

from .config import Config, load_config
from .engine import (
    candidate_pairs,
    fit_once,
    load_pool,
    plot_mae,
    predict,
    subset_pool,
    target_weak_pool,
)
from .evaluate import OOD_COHORTS
from .reporting import save_predictions


def _main_rank_predictions(config: Config, split: str, seed: int, expected_names: np.ndarray,
                           cohort: str | None = None) -> np.ndarray:
    root = Path(config.run_dir) / "fits" / "rank" / f"seed_{seed}"
    path = root / ("ood" if cohort else split)
    if cohort:
        path = path / cohort
    with (path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row["filename"] for row in rows] != expected_names.tolist():
        raise ValueError(f"Primary rank and margin-10 samples differ for {split}/{cohort}")
    return np.array([float(row["prediction"]) for row in rows], dtype=np.float32)


def run(config: Config) -> dict:
    if config.rank_interval_margin != 5.0:
        raise ValueError("Sensitivity comparison expects the declared five-point primary rule")
    selection = json.loads((Path(config.run_dir) / "cv_selection.json").read_text())
    if not (Path(config.run_dir) / "validation_comparison.json").is_file():
        raise FileNotFoundError("Lock the main validation comparison before sensitivity analysis")
    choice = selection["selected"]["rank"]
    strict = replace(config, rank_interval_margin=10.0)
    gold = load_pool(config.run_dir, "finetune")
    weak = target_weak_pool(config.run_dir, config)
    validation = load_pool(config.run_dir, "validation")
    test = load_pool(config.run_dir, "test")
    all_weak = load_pool(config.run_dir, "pretrain")
    cohorts = {name: subset_pool(all_weak, np.flatnonzero(all_weak.cohort == name))
               for name in OOD_COHORTS}
    pools = {"validation": validation, "test": test,
             **{f"ood_{name}": pool for name, pool in cohorts.items()}}
    device = str(resolve_device("auto"))
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in pools}
    controls: dict[str, list[np.ndarray]] = {name: [] for name in pools}
    pair_counts = []
    for seed in config.seeds:
        model, history, info = fit_once(
            strict, gold, weak, np.arange(len(gold)), np.arange(len(weak)), arm="rank",
            seed=seed, rank_weight=choice["weak_weight"],
            residual_penalty=choice["residual_penalty"], midpoint_weight=0.0,
            epochs=choice["epoch"], device=device,
        )
        pair_counts.append(info["pairs"])
        destination = Path(config.run_dir) / "sensitivity_margin10" / f"seed_{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        first, second, weights, _ = candidate_pairs(weak, np.arange(len(weak)), strict, seed)
        with (destination / "sampled_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("higher_image", "lower_image", "weight"))
            for upper, lower, weight in zip(first, second, weights, strict=True):
                writer.writerow((weak.filename[upper], weak.filename[lower], float(weight)))
        save_checkpoint(destination / "model.pt", {
            "experiment": "consensus_rank_transfer_margin10", "version": 1,
            "seed": seed, "epoch": choice["epoch"], "margin": 10.0,
            "choice": {"weak_weight": choice["weak_weight"],
                       "residual_penalty": choice["residual_penalty"]},
            "model_state_dict": model.state_dict(),
        })
        (destination / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        for name, pool in pools.items():
            estimated, residual = predict(model, pool, device)
            location = destination / name
            save_predictions(location / "predictions.csv", pool, estimated, residual,
                             "rank_margin10", seed,
                             label_quality="single_rater_weak" if name.startswith("ood_")
                             else "gold_historical")
            control = _main_rank_predictions(config, "ood" if name.startswith("ood_") else name,
                                             seed, pool.filename,
                                             name[4:] if name.startswith("ood_") else None)
            predictions[name].append(estimated)
            controls[name].append(control)
    results = {}
    rng = np.random.default_rng(1005)
    for name, pool in pools.items():
        tighter = np.stack(predictions[name])
        primary = np.stack(controls[name])
        groups = np.unique(pool.group)
        by_plot = np.array([
            np.mean(np.abs(primary[:, pool.group == group] - pool.target[pool.group == group])) -
            np.mean(np.abs(tighter[:, pool.group == group] - pool.target[pool.group == group]))
            for group in groups
        ])
        sampled = rng.integers(0, len(groups), size=(config.bootstrap_replicates, len(groups)))
        results[name] = {
            "images": len(pool), "plots": len(groups),
            "primary_rank_plot_mae": float(np.mean([
                plot_mae(value, pool.target, pool.group) for value in primary])),
            "margin10_plot_mae": float(np.mean([
                plot_mae(value, pool.target, pool.group) for value in tighter])),
            "margin10_minus_primary_mae_reduction": float(by_plot.mean()),
            "bootstrap_95_percent_interval": np.quantile(by_plot[sampled].mean(axis=1),
                                                           [.025, .975]).tolist(),
            "label_quality": "single_rater_weak" if name.startswith("ood_") else "gold_historical",
        }
    summary = {"purpose": "Sensitivity only; no hyperparameter or arm selection",
               "primary_interval_margin": 5.0, "sensitivity_interval_margin": 10.0,
               "shared_selected_settings": {"epoch": choice["epoch"],
                                            "weak_weight": choice["weak_weight"],
                                            "residual_penalty": choice["residual_penalty"]},
               "pair_counts_by_seed": pair_counts, "results": results,
               "confirmatory": False}
    (Path(config.run_dir) / "sensitivity_margin10" / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/consensus_rank_transfer/config.toml")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
