"""Choose hyperparameters and training epochs using gold training plots only."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from rapeseed_damage.artifacts import environment_info
from rapeseed_damage.reproducibility import resolve_device

from .config import Config, load_config
from .engine import cv_folds, fit_once, load_pool, plot_mae, target_weak_pool


def _fingerprint(config: Config) -> str:
    source = Path(config.run_dir) / "frozen" / "summary.json"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def run(config: Config) -> dict:
    gold = load_pool(config.run_dir, "finetune")
    weak = target_weak_pool(config.run_dir, config)
    if set(gold.filename) & set(weak.filename):
        raise ValueError("Gold and weak training share exact images")
    assignments = cv_folds(gold.group, config.cv_folds, config.cv_seed)
    device = str(resolve_device("auto"))
    out = Path(config.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cv_folds.npz", filename=gold.filename, group=gold.group,
                        fold=assignments)
    options = {
        "gold_only": [(0.0, penalty) for penalty in config.residual_penalties],
        "midpoint": list(itertools.product(config.midpoint_weights, config.residual_penalties)),
        "rank": list(itertools.product(config.rank_weights, config.residual_penalties)),
    }
    candidates: dict[str, list[dict]] = {}
    selected: dict[str, dict] = {}
    fold_sizes = []
    for fold in range(config.cv_folds):
        hold = np.flatnonzero(assignments == fold)
        held_groups = set(gold.group[hold])
        allowed_weak = np.flatnonzero(~np.isin(weak.group, list(held_groups)))
        fold_sizes.append({"fold": fold, "gold_train": int(np.sum(assignments != fold)),
                           "gold_holdout": len(hold), "weak_train": len(allowed_weak),
                           "weak_excluded_same_plot": len(weak) - len(allowed_weak)})
    for arm, combinations in options.items():
        candidates[arm] = []
        for weak_weight, penalty in combinations:
            curves = []
            for fold in range(config.cv_folds):
                hold = np.flatnonzero(assignments == fold)
                train = np.flatnonzero(assignments != fold)
                weak_train = np.flatnonzero(~np.isin(weak.group, gold.group[hold]))
                _, history, _ = fit_once(
                    config, gold, weak, train, weak_train, arm=arm,
                    seed=config.cv_seed + fold, rank_weight=weak_weight if arm == "rank" else 0.0,
                    midpoint_weight=weak_weight if arm == "midpoint" else 0.0,
                    residual_penalty=penalty, epochs=config.max_epochs, validation=hold,
                    device=device,
                )
                curves.append([entry["cv_plot_mae"] for entry in history])
            if any(len(curve) != config.max_epochs + 1 for curve in curves):
                raise RuntimeError("CV fold did not complete every predeclared epoch")
            average = np.mean(np.asarray(curves, dtype=np.float64), axis=0)
            best_epoch = int(np.argmin(average))
            record = {"weak_weight": weak_weight, "residual_penalty": penalty,
                      "epoch": best_epoch, "cv_plot_mae": float(average[best_epoch]),
                      "mean_curve": average.tolist(), "fold_curves": curves}
            candidates[arm].append(record)
            print(f"CV {arm} weight={weak_weight:g} penalty={penalty:g}: "
                  f"epoch={best_epoch} plot MAE={average[best_epoch]:.4f}", flush=True)
        selected[arm] = min(candidates[arm], key=lambda item: (item["cv_plot_mae"],
                                                                   item["epoch"],
                                                                   item["weak_weight"]))
    selected["shuffled"] = {**selected["rank"], "selection_source": "rank"}
    baseline_fold_mae = [plot_mae(gold.base[assignments == fold], gold.target[assignments == fold],
                                  gold.group[assignments == fold])
                         for fold in range(config.cv_folds)]
    summary = {"selection_data": "finetune gold plots only", "config": config.to_dict(),
               "environment": environment_info(device, Path(__file__).resolve().parents[2]),
               "base_was_trained_on_all_cv_gold_images": True,
               "cv_interpretation": "Head-selection heuristic, not an unbiased out-of-sample estimate",
               "frozen_fingerprint": _fingerprint(config), "fold_sizes": fold_sizes,
               "base_cv_plot_mae": float(np.mean(baseline_fold_mae)),
               "base_fold_mae": baseline_fold_mae, "candidates": candidates,
               "selected": selected, "validation_used": False, "test_used": False,
               "ood_labels_used": False}
    (out / "cv_selection.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/consensus_rank_transfer/config.toml")
    args = parser.parse_args(argv)
    result = run(load_config(args.config))
    print(json.dumps({"selected": result["selected"],
                      "base_cv_plot_mae": result["base_cv_plot_mae"]}, indent=2))


if __name__ == "__main__":
    main()
