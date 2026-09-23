"""Fit each matched arm on all gold-training plots at the CV-selected epoch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rapeseed_damage.artifacts import environment_info
from rapeseed_damage.checkpointing import save_checkpoint
from rapeseed_damage.reproducibility import resolve_device

from .config import Config, load_config
from .engine import ARMS, candidate_pairs, fit_once, load_pool, predict, target_weak_pool
from .reporting import save_predictions


def _selection(config: Config) -> dict:
    path = Path(config.run_dir) / "cv_selection.json"
    selection = json.loads(path.read_text())
    fingerprint = hashlib.sha256((Path(config.run_dir) / "frozen" / "summary.json").read_bytes()).hexdigest()
    if selection["config"] != json.loads(json.dumps(config.to_dict())) or selection["frozen_fingerprint"] != fingerprint:
        raise ValueError("CV selection does not match the current config/frozen cache")
    if selection["validation_used"] or selection["test_used"] or selection["ood_labels_used"]:
        raise ValueError("CV selection improperly used evaluation labels")
    return selection


def run(config: Config, arm: str, seed: int) -> dict:
    if arm not in ARMS or seed not in config.seeds:
        raise ValueError("Unknown arm or seed")
    selection = _selection(config)
    choice = selection["selected"][arm]
    gold = load_pool(config.run_dir, "finetune")
    weak = target_weak_pool(config.run_dir, config)
    validation = load_pool(config.run_dir, "validation")
    if set(gold.group) & set(validation.group) or set(weak.group) & set(validation.group):
        raise ValueError("Validation shares plots with training")
    device = str(resolve_device("auto"))
    model, history, info = fit_once(
        config, gold, weak, np.arange(len(gold)), np.arange(len(weak)), arm=arm, seed=seed,
        rank_weight=choice["weak_weight"] if arm in ("rank", "shuffled") else 0.0,
        midpoint_weight=choice["weak_weight"] if arm == "midpoint" else 0.0,
        residual_penalty=choice["residual_penalty"], epochs=choice["epoch"], device=device,
    )
    prediction, residual = predict(model, validation, device)
    weak_prediction, _ = predict(model, weak, device)
    higher, lower, _, canonical_pairs = candidate_pairs(weak, np.arange(len(weak)), config,
                                                          seed, shuffled=False)
    pair_diagnostic = {
        "qualified_sampled_pairs": canonical_pairs["sampled_pairs"],
        "base_violation_fraction": float(np.mean(
            weak.base[higher] - weak.base[lower] < config.rank_prediction_margin)),
        "arm_violation_fraction": float(np.mean(
            weak_prediction[higher] - weak_prediction[lower] < config.rank_prediction_margin)),
    }
    if choice["epoch"] == 0 and not np.allclose(prediction, validation.base, atol=1e-6):
        raise RuntimeError("Zero-epoch model must exactly reproduce frozen base")
    destination = Path(config.run_dir) / "fits" / arm / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    if arm in ("rank", "shuffled"):
        first, second, weights, _ = candidate_pairs(
            weak, np.arange(len(weak)), config, seed, shuffled=arm == "shuffled"
        )
        with (destination / "sampled_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("first_image", "second_image", "first_plot", "second_plot",
                             "first_jlu", "first_gau", "second_jlu", "second_gau", "weight"))
            for left, right, weight in zip(first, second, weights, strict=True):
                writer.writerow((weak.filename[left], weak.filename[right],
                                 weak.group[left], weak.group[right],
                                 float(weak.score_jlu[left]), float(weak.score_gau[left]),
                                 float(weak.score_jlu[right]), float(weak.score_gau[right]),
                                 float(weight)))
    (destination / "environment.json").write_text(json.dumps(
        environment_info(device, Path(__file__).resolve().parents[2]), indent=2, sort_keys=True
    ) + "\n")
    checkpoint = {"experiment": "consensus_rank_transfer", "version": 1,
                  "arm": arm, "seed": seed, "epoch": choice["epoch"],
                  "choice": {"weak_weight": choice["weak_weight"],
                             "residual_penalty": choice["residual_penalty"]},
                  "config": config.to_dict(), "frozen_fingerprint": selection["frozen_fingerprint"],
                  "feature_dim": gold.features.shape[1], "model_state_dict": model.state_dict(),
                  "selection_data": "finetune gold plots only", "validation_used_for_selection": False,
                  "test_used_for_selection": False}
    save_checkpoint(destination / "model.pt", checkpoint)
    (destination / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    result = save_predictions(destination / "validation" / "predictions.csv", validation,
                              prediction, residual, arm, seed, label_quality="gold_historical")
    summary = {"arm": arm, "seed": seed, "epoch": choice["epoch"],
               "choice": {"weak_weight": choice["weak_weight"],
                          "residual_penalty": choice["residual_penalty"]},
               "train_gold_images": len(gold), "train_weak_images": len(weak),
               "pair_info": info["pairs"], "pair_diagnostic": pair_diagnostic,
               "validation": result,
               "test_evaluated": False}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/consensus_rank_transfer/config.toml")
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else config.seeds
    for arm in ARMS if args.arm == "all" else [args.arm]:
        for seed in seeds:
            result = run(config, arm, seed)
            print(f"{arm} seed={seed}: validation plot MAE "
                  f"{result['validation']['plot_weighted_mae']:.4f}", flush=True)


if __name__ == "__main__":
    main()
