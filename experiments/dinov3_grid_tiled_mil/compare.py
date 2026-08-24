"""Paired candidate-versus-baseline comparison with a bootstrap confidence interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rapeseed_damage.artifacts import write_json

from .metrics import regression_metrics


def compare(candidate_path, baseline_path, *, bootstrap_samples: int, seed: int) -> dict:
    candidate = pd.read_csv(candidate_path)
    baseline = pd.read_csv(baseline_path)
    required = {"filename", "target", "prediction"}
    for label, table in (("candidate", candidate), ("baseline", baseline)):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{label} CSV is missing: {', '.join(sorted(missing))}")
        if table["filename"].duplicated().any():
            raise ValueError(f"{label} CSV contains duplicate filenames")
    paired = candidate[list(required)].merge(
        baseline[list(required)], on="filename", suffixes=("_candidate", "_baseline")
    )
    if len(paired) != len(candidate) or len(paired) != len(baseline):
        raise ValueError(
            f"Prediction manifests differ: candidate={len(candidate)}, baseline={len(baseline)}, "
            f"paired={len(paired)}"
        )
    if not np.allclose(paired["target_candidate"], paired["target_baseline"], atol=1e-5):
        raise ValueError("Paired target values differ")
    target = paired["target_candidate"].to_numpy(dtype=float)
    candidate_prediction = paired["prediction_candidate"].to_numpy(dtype=float)
    baseline_prediction = paired["prediction_baseline"].to_numpy(dtype=float)
    reduction = np.abs(baseline_prediction - target) - np.abs(candidate_prediction - target)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(reduction), size=(bootstrap_samples, len(reduction)))
    means = reduction[draws].mean(axis=1)
    report = {
        "samples": len(paired),
        "candidate": regression_metrics(target, candidate_prediction),
        "baseline": regression_metrics(target, baseline_prediction),
        "paired_absolute_error": {
            "mean_reduction": float(reduction.mean()),
            "median_reduction": float(np.median(reduction)),
            "improved_samples": int((reduction > 0).sum()),
            "unchanged_samples": int((reduction == 0).sum()),
            "worsened_samples": int((reduction < 0).sum()),
            "bootstrap_95_percent_ci_for_mean_reduction": [
                float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)),
            ],
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        },
    }
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    if arguments.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    report = compare(
        arguments.candidate,
        arguments.baseline,
        bootstrap_samples=arguments.bootstrap_samples,
        seed=arguments.seed,
    )
    if arguments.output:
        write_json(Path(arguments.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
