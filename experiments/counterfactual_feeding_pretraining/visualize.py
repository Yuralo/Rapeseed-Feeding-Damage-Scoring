"""Save the predeclared preparation, pretext, and gold-validation review figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config, load_config

COLORS = {"random": "#4c78a8", "synthetic": "#54a24b", "sham": "#e4a64e"}
ARMS = tuple(COLORS)
BANDS = ("0_to_2_5", "over_2_5_to_7_5", "over_7_5_to_15", "over_15")


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _preparation(config: Config, run_dir: Path) -> Path:
    summary = json.loads((run_dir / "preparation_summary.json").read_text())
    if summary["masks_only"]:
        raise ValueError("Full bite-pair preparation is required for review figures")
    cohorts = sorted(summary["by_cohort"])
    paired = np.asarray([summary["by_cohort"][name].get("paired", 0) for name in cohorts])
    failed = np.asarray([summary["by_cohort"][name].get("mask_failed", 0) +
                         summary["by_cohort"][name].get("pair_failed", 0)
                         for name in cohorts])
    coverage = paired / np.maximum(1, paired + failed)
    figure, axes = plt.subplots(1, 2, figsize=(15, max(4, .35 * len(cohorts) + 1)))
    y = np.arange(len(cohorts))
    axes[0].barh(y, paired, color="#54a24b", label="paired")
    axes[0].barh(y, failed, left=paired, color="#e45756", label="failed")
    axes[0].set(yticks=y, yticklabels=cohorts, xlabel="Images", title="Pair preparation by cohort")
    axes[0].legend()
    axes[1].barh(y, coverage, color="#4c78a8")
    axes[1].axvline(config.minimum_coverage_fraction, color="#e45756", linestyle="--",
                    label=f"gate {config.minimum_coverage_fraction:.0%}")
    axes[1].set(yticks=y, yticklabels=cohorts, xlim=(0, 1),
                xlabel="Accepted fraction", title="Cohort coverage")
    axes[1].legend()
    path = run_dir / "preparation_quality.png"
    _save(figure, path)
    return path


def _pretext(run_dir: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, mode in enumerate(("synthetic", "sham")):
        axis = axes[0, column]
        summary = json.loads((run_dir / "pretraining" / mode / "summary.json").read_text())
        diagnostics = summary["diagnostics"]
        levels = sorted(diagnostics["by_requested_level"], key=float)
        x = np.asarray([float(level) for level in levels]) * 100
        actual = np.asarray([diagnostics["by_requested_level"][level]["mean_realized_fraction"]
                             for level in levels]) * 100
        bite = [diagnostics["by_requested_level"][level]["mean_bite_delta"]
                for level in levels]
        sham = [diagnostics["by_requested_level"][level]["mean_sham_delta"]
                for level in levels]
        axis.plot(x, bite, "o-", color="#54a24b", label="bite response")
        axis.plot(x, sham, "o-", color="#e4a64e", label="sham response")
        axis.axhline(0, color="black", linewidth=.7)
        axis.set(xlabel="Requested leaf loss (%)", ylabel="Change in local evidence",
                 title=f"{mode} pretraining, held-out images")
        axis.text(.03, .97, f"realized: {', '.join(f'{value:.1f}%' for value in actual)}\n"
                  f"bite/loss r={diagnostics['realized_bite_delta_correlation']:.2f}",
                  transform=axis.transAxes, va="top", fontsize=8)
        axis.legend(loc="lower right")
        history = json.loads((run_dir / "pretraining" / mode / "history.json").read_text())
        epochs = [row["epoch"] for row in history]
        axes[1, column].plot(epochs, [row["train"]["loss"] for row in history],
                             label="train")
        axes[1, column].plot(epochs, [row["heldout"]["loss"] for row in history],
                             label="held-out")
        axes[1, column].set(xlabel="Epoch", ylabel="Pretext loss",
                            title=f"{mode} learning curve")
        axes[1, column].legend()
    path = run_dir / "pretext_diagnostics.png"
    _save(figure, path)
    return path


def _validation(run_dir: Path) -> Path:
    summary = json.loads((run_dir / "validation_comparison.json").read_text())
    errors = pd.read_csv(run_dir / "validation_paired_errors.csv")
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    by_seed = summary["plot_weighted_mae_by_seed"]
    seeds = list(map(int, by_seed))
    for arm in ARMS:
        axes[0].plot(seeds, [by_seed[str(seed)][arm] for seed in seeds], "o-",
                     color=COLORS[arm], label=arm)
    axes[0].axhline(summary["frozen_base_plot_mae"], color="#555555", linestyle="--",
                    label="frozen base")
    axes[0].set(xlabel="Fitting seed", ylabel="Plot-weighted MAE", title="Gold validation")
    axes[0].legend()
    plot_errors = errors.groupby(["group", "arm"])["absolute_error"].mean().unstack()
    gains = plot_errors["random"] - plot_errors["synthetic"]
    axes[1].hist(gains, bins=min(20, max(5, len(gains) // 3)), color=COLORS["synthetic"])
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set(xlabel="Random minus synthetic absolute error", ylabel="Plots",
                title="Paired plot gain across seeds")
    x = np.arange(len(BANDS))
    for index, arm in enumerate(ARMS):
        values = [summary["target_bands"][band]["mean_mae_by_arm"].get(arm, np.nan)
                  for band in BANDS]
        axes[2].bar(x + (index - 1) * .25, values, width=.23, color=COLORS[arm], label=arm)
    axes[2].set(xticks=x, xticklabels=("0–2.5", "2.5–7.5", "7.5–15", ">15"),
                ylabel="Image MAE", xlabel="Gold score", title="Error by target band")
    axes[2].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    path = run_dir / "validation_comparison.png"
    _save(figure, path)
    return path


def _training(config: Config, run_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, arm in zip(axes, ARMS):
        for seed in config.fitting_seeds:
            history_path = run_dir / "gold_fit" / arm / f"seed_{seed}" / "history.json"
            history = json.loads(history_path.read_text())
            axis.plot([row["epoch"] for row in history],
                      [row["plot_mae"] for row in history], alpha=.65,
                      label=f"seed {seed}")
        axis.set(xlabel="Epoch", title=arm)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Gold validation plot-weighted MAE")
    axes[-1].legend(fontsize=8)
    figure.suptitle("Gold fitting: all predeclared seeds and validation epochs")
    path = run_dir / "gold_training_curves.png"
    _save(figure, path)
    return path


def run(config: Config) -> list[str]:
    run_dir = Path(config.run_dir)
    return [str(_preparation(config, run_dir)), str(_pretext(run_dir)),
            str(_validation(run_dir)), str(_training(config, run_dir))]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
