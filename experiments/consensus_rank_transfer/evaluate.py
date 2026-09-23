"""Evaluate locked fits on historical gold test or matched weak OOD cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device

from .config import Config, load_config
from .engine import ARMS, ResidualHead, load_pool, predict, subset_pool
from .reporting import save_predictions

OOD_COHORTS = ("wg_insects_t1_bbch10", "dsv_asendorf_t1_bbch11")


def _load_model(config: Config, arm: str, seed: int, device: str) -> ResidualHead:
    path = Path(config.run_dir) / "fits" / arm / f"seed_{seed}" / "model.pt"
    state = load_checkpoint(path, device)
    fingerprint = hashlib.sha256((Path(config.run_dir) / "frozen" / "summary.json").read_bytes()).hexdigest()
    if (state.get("experiment"), state.get("version"), state.get("arm"), state.get("seed")) != (
        "consensus_rank_transfer", 1, arm, seed
    ):
        raise ValueError(f"Invalid rank-transfer checkpoint: {path}")
    if state["frozen_fingerprint"] != fingerprint or state["config"] != config.to_dict():
        raise ValueError(f"Checkpoint does not match current frozen cache/config: {path}")
    model = ResidualHead(state["feature_dim"], config.hidden_dim, config.max_residual).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model


def run(config: Config, split: str, arm: str, seed: int) -> dict:
    if split not in ("test", "ood") or arm not in ARMS or seed not in config.seeds:
        raise ValueError("Invalid evaluation split, arm, or seed")
    if not (Path(config.run_dir) / "validation_comparison.json").is_file():
        raise FileNotFoundError("Lock all validation arms first with compare --split validation")
    device = str(resolve_device("auto"))
    model = _load_model(config, arm, seed, device)
    if split == "test":
        pool = load_pool(config.run_dir, "test")
        if pool.features.shape[1] != model.layers[1].in_features:
            raise ValueError("Test feature dimension differs from checkpoint")
        prediction, residual = predict(model, pool, device)
        destination = Path(config.run_dir) / "fits" / arm / f"seed_{seed}" / "test"
        return save_predictions(destination / "predictions.csv", pool, prediction, residual,
                                arm, seed, label_quality="gold_historical")

    all_weak = load_pool(config.run_dir, "pretrain")
    results = {}
    for cohort in OOD_COHORTS:
        indices = (all_weak.cohort == cohort).nonzero()[0]
        if len(indices) == 0:
            raise ValueError(f"No cached OOD images in {cohort}")
        pool = subset_pool(all_weak, indices)
        prediction, residual = predict(model, pool, device)
        destination = Path(config.run_dir) / "fits" / arm / f"seed_{seed}" / "ood" / cohort
        results[cohort] = save_predictions(destination / "predictions.csv", pool,
                                           prediction, residual, arm, seed,
                                           label_quality="single_rater_weak")
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/consensus_rank_transfer/config.toml")
    parser.add_argument("--split", choices=("test", "ood"), required=True)
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else config.seeds
    for arm in ARMS if args.arm == "all" else [args.arm]:
        for seed in seeds:
            print(json.dumps({"split": args.split, "arm": arm, "seed": seed,
                              "result": run(config, args.split, arm, seed)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
