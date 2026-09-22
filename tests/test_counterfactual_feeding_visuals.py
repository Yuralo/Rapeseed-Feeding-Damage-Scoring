"""Exercise the report figures with small, complete experiment summaries."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("matplotlib")

from experiments.counterfactual_feeding_pretraining.config import load_config
from experiments.counterfactual_feeding_pretraining.evaluate_ood import _plot
from experiments.counterfactual_feeding_pretraining.summarize_test import run as summarize_test
from experiments.counterfactual_feeding_pretraining.visualize import run


def test_all_review_figures_render(tmp_path):
    config = replace(load_config("experiments/counterfactual_feeding_pretraining/config.toml"),
                     run_dir=str(tmp_path))
    (tmp_path / "preparation_summary.json").write_text(json.dumps({
        "masks_only": False, "by_cohort": {
            "cohort_a": {"paired": 90, "mask_failed": 5, "pair_failed": 5},
            "cohort_b": {"paired": 80, "mask_failed": 10, "pair_failed": 10},
        },
    }))
    levels = {str(level): {"mean_realized_fraction": level,
                           "mean_bite_delta": level * 0.8,
                           "mean_sham_delta": level * 0.1}
              for level in config.severity_levels}
    for mode in ("synthetic", "sham"):
        path = tmp_path / "pretraining" / mode
        path.mkdir(parents=True)
        (path / "summary.json").write_text(json.dumps({
            "diagnostics": {"by_requested_level": levels,
                            "realized_bite_delta_correlation": .8},
        }))
        (path / "history.json").write_text(json.dumps([
            {"epoch": epoch, "train": {"loss": 1 / epoch},
             "heldout": {"loss": 1.1 / epoch}} for epoch in (1, 2, 3)
        ]))
    (tmp_path / "validation_comparison.json").write_text(json.dumps({
        "plot_weighted_mae_by_seed": {
            str(seed): {"random": 2.5, "synthetic": 2.2, "sham": 2.6}
            for seed in config.fitting_seeds
        },
        "frozen_base_plot_mae": 2.4,
        "target_bands": {
            band: {"mean_mae_by_arm": {"random": 2.5, "synthetic": 2.2, "sham": 2.6}}
            for band in ("0_to_2_5", "over_2_5_to_7_5", "over_7_5_to_15", "over_15")
        },
    }))
    pd.DataFrame({"group": ["a", "a", "a", "b", "b", "b"],
                  "arm": ["random", "synthetic", "sham"] * 2,
                  "absolute_error": [2.5, 2.1, 2.6, 3.0, 2.7, 3.1]}).to_csv(
        tmp_path / "validation_paired_errors.csv", index=False)
    for arm in ("random", "synthetic", "sham"):
        for seed in config.fitting_seeds:
            path = tmp_path / "gold_fit" / arm / f"seed_{seed}"
            path.mkdir(parents=True)
            (path / "history.json").write_text(json.dumps([
                {"epoch": epoch, "plot_mae": 2.5 - epoch * .1}
                for epoch in (0, 1, 2)
            ]))
    paths = run(config)
    assert len(paths) == 4
    for path in paths:
        with Image.open(path) as image:
            assert image.width > 100 and image.height > 100

    ood = {"cohorts": {"wg_insects_t1_bbch10": {
        "base_mae_vs_weak": 3.8, "base_bias_vs_weak": -2.0,
        "arms": {arm: {"mean_mae_vs_weak": 3.5 + index * .2,
                       "mean_bias_vs_weak": -1.8 + index * .1,
                       "seed_mae_vs_weak": [3.5 + index * .2] * 5}
                 for index, arm in enumerate(("random", "synthetic", "sham"))},
    }}}
    output = tmp_path / "ood.png"
    _plot(ood, output)
    with Image.open(output) as image:
        assert image.width > 100 and image.height > 100


def test_historical_test_summary_uses_all_seeds(tmp_path):
    config = replace(load_config("experiments/counterfactual_feeding_pretraining/config.toml"),
                     run_dir=str(tmp_path))
    (tmp_path / "validation_comparison.json").write_text("{}")
    for seed in config.fitting_seeds:
        path = tmp_path / "gold_fit" / "synthetic" / f"seed_{seed}" / "test"
        path.mkdir(parents=True)
        pd.DataFrame({"filename": ["a", "b", "c", "d"],
                      "plot_group_id": ["p1", "p1", "p2", "p2"],
                      "target": [1.0, 5.0, 10.0, 18.0],
                      "base_prediction": [2.0, 6.0, 11.0, 19.0],
                      "prediction": [1.5, 5.5, 10.5, 18.5]}).to_csv(
            path / "predictions.csv", index=False)
    report = summarize_test(config, "synthetic")
    assert report["plot_mae_reduction_vs_base"] == pytest.approx(.5)
    assert len(report["seed_metrics"]) == 5
    with Image.open(tmp_path / "test_summary_synthetic.png") as image:
        assert image.width > 100
