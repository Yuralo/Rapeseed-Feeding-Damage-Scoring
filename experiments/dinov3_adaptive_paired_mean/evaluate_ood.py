"""Matched-image OOD comparison with the earlier best-model probe."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.dinov3_grid_tiled_mil.metrics import regression_metrics
from experiments.dinov3_hierarchical_three_view_mil.data import (
    filter_usable_pretrain,
    load_manifest,
    verify_features,
)
from experiments.dinov3_plant_damage_mil.data import make_loader
from experiments.dinov3_plant_damage_mil.reporting import predict, save
from rapeseed_damage.artifacts import write_json

from .config import Config, load_config
from .data import _usable_patches, tables
from .evaluate import load_model
from .manifests import OOD_COHORTS


def _metrics(target, prediction):
    result = regression_metrics(target, prediction)
    result["mse"] = result["rmse"] ** 2
    result["bias"] = float(np.mean(np.asarray(prediction) - np.asarray(target)))
    return result


def _ood_table(config, cohort):
    base = replace(
        config.base,
        data=replace(config.base.data, pretrain_manifest=f"ood_{cohort}.csv"),
    )
    table = load_manifest(base, "pretrain")
    if set(table[config.base.data.cohort_column].astype(str)) != {cohort}:
        raise ValueError(f"Unexpected cohort in OOD manifest: {cohort}")
    if "gold" in set(table[config.base.data.supervision_tier_column].astype(str)):
        raise ValueError("OOD probe must use non-gold labels")
    return table


def run(config: Config, checkpoint: str | Path, output_dir: str | Path | None = None):
    model, scaler, device, state, _, _ = load_model(config, checkpoint)
    held = tables(config)
    used_names = set(
        state["train_filenames"] + state["validation_filenames"] + state["test_filenames"]
    )
    train_groups = set(
        pd.concat([held["pretrain"], held["finetune"]])[config.base.data.group_column].astype(str)
    )
    destination = Path(output_dir) if output_dir else Path(config.run_dir) / "ood_matched_probe"
    reports = {}
    for cohort in OOD_COHORTS:
        original = _ood_table(config, cohort)
        if used_names & set(original[config.base.data.filename_column].astype(str)):
            raise ValueError("OOD image is part of paired training/holdout")
        if train_groups & set(original[config.base.data.group_column].astype(str)):
            raise ValueError("OOD image shares a paired-training plot group")
        old_path = Path(config.reference_ood_dir) / cohort / "predictions.csv"
        old = pd.read_csv(old_path)
        if old["filename"].duplicated().any():
            raise ValueError(f"Duplicate filenames in earlier OOD predictions: {old_path}")
        if (
            original[config.base.data.filename_column].astype(str).tolist()
            != old["filename"].astype(str).tolist()
        ):
            raise ValueError("Frozen OOD rows differ from the earlier probe")
        selected, missing_base = filter_usable_pretrain(original, config.base)
        selected, missing_patches = _usable_patches(config, selected)
        missing_total = missing_base + len(missing_patches)
        if missing_total / len(original) > config.base.data.maximum_weak_failure_fraction:
            raise FileNotFoundError(f"{missing_total}/{len(original)} OOD features are missing")
        dimension = verify_features(selected, config.base)
        if dimension != int(state["feature_dim"]):
            raise ValueError("OOD feature dimension differs from checkpoint")
        loader = make_loader(selected, scaler, config, training=False, offset=4000)
        details = predict(model, loader, device, scaler)
        folder = destination / cohort
        report = save(details, scaler, folder, config)
        report["model"]["mse"] = report["model"]["rmse"] ** 2
        fresh = pd.DataFrame(
            {
                "filename": details.result.filenames,
                "target": details.result.targets,
                "new_prediction": details.result.predictions,
            }
        )
        matched = fresh.merge(
            old[["filename", "target", "prediction"]],
            on="filename",
            how="left",
            validate="one_to_one",
            suffixes=("", "_old"),
        )
        if matched["prediction"].isna().any() or len(matched) != len(fresh):
            raise ValueError("Could not pair every new OOD prediction with the earlier model")
        if not np.allclose(matched["target"], matched["target_old"], atol=1e-5):
            raise ValueError("OOD targets differ from the earlier probe")
        matched = matched.rename(columns={"prediction": "previous_prediction"})
        matched["new_absolute_error"] = np.abs(matched["new_prediction"] - matched["target"])
        matched["previous_absolute_error"] = np.abs(
            matched["previous_prediction"] - matched["target"]
        )
        matched["absolute_error_reduction"] = (
            matched["previous_absolute_error"] - matched["new_absolute_error"]
        )
        folder.mkdir(parents=True, exist_ok=True)
        matched.to_csv(folder / "paired_ood_comparison.csv", index=False)
        previous = _metrics(matched["target"], matched["previous_prediction"])
        current = _metrics(matched["target"], matched["new_prediction"])
        comparison = {
            "cohort": cohort,
            "reference_predictions": str(old_path),
            "reference_sample_count": len(original),
            "matched_sample_count": len(matched),
            "omitted_due_to_features": missing_total,
            "previous_model": previous,
            "paired_mean_model": current,
            "mae_reduction": previous["mae"] - current["mae"],
            "mse_reduction": previous["mse"] - current["mse"],
            "r2_increase": current["r2"] - previous["r2"],
            "improved_images": int((matched["absolute_error_reduction"] > 0).sum()),
            "worsened_images": int((matched["absolute_error_reduction"] < 0).sum()),
            "gold_standard": False,
        }
        report["matched_previous_ood_comparison"] = comparison
        write_json(folder / "summary.json", report)
        reports[cohort] = comparison
    summary = {
        "checkpoint": str(checkpoint),
        "cohorts": reports,
        "warning": (
            "OOD targets are weak/single-rater scores, not gold truth. The adapted frozen "
            "backbone may have seen these images without labels. OOD never selects checkpoints."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "ood_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(load_config(args.config), args.checkpoint, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
