"""Matched prediction tables and score diagnostics for the rank experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .engine import Pool, plot_mae


def metrics(pool: Pool, prediction: np.ndarray) -> dict:
    error = prediction - pool.target
    values = {"images": len(pool), "plots": len(np.unique(pool.group)),
              "image_mae": float(np.mean(np.abs(error))),
              "plot_weighted_mae": plot_mae(prediction, pool.target, pool.group),
              "rmse": float(np.sqrt(np.mean(np.square(error)))),
              "bias": float(np.mean(error))}
    bands = {"0_to_2_5": pool.target <= 2.5,
             "over_2_5_to_7_5": (pool.target > 2.5) & (pool.target <= 7.5),
             "over_7_5_to_15": (pool.target > 7.5) & (pool.target <= 15),
             "over_15": pool.target > 15}
    values["bands"] = {name: {"images": int(mask.sum()),
                              "mae": float(np.mean(np.abs(error[mask]))) if mask.any() else None,
                              "base_mae": float(np.mean(np.abs(pool.base[mask] - pool.target[mask])))
                              if mask.any() else None}
                       for name, mask in bands.items()}
    needed = pool.target - pool.base
    applied = prediction - pool.base
    values["correction_needed_correlation"] = (
        float(np.corrcoef(needed, applied)[0, 1])
        if np.std(needed) > 0 and np.std(applied) > 0 else None
    )
    values["calibration_slope"] = (
        float(np.polyfit(pool.target, prediction, 1)[0]) if np.std(pool.target) > 0 else None
    )
    return values


def save_predictions(path: Path, pool: Pool, prediction: np.ndarray,
                     residual: np.ndarray, arm: str, seed: int,
                     *, label_quality: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "plot_group_id", "cohort_id", "source_path", "target",
                         "base_prediction", "prediction", "residual", "absolute_error",
                         "base_absolute_error", "arm", "seed", "label_quality"))
        for index in range(len(pool)):
            writer.writerow((pool.filename[index], pool.group[index], pool.cohort[index],
                             pool.source_path[index], float(pool.target[index]),
                             float(pool.base[index]), float(prediction[index]), float(residual[index]),
                             float(abs(prediction[index] - pool.target[index])),
                             float(abs(pool.base[index] - pool.target[index])), arm, seed,
                             label_quality))
    summary = metrics(pool, prediction)
    summary.update({"arm": arm, "seed": seed, "label_quality": label_quality,
                    "base_plot_weighted_mae": plot_mae(pool.base, pool.target, pool.group),
                    "base_image_mae": float(np.mean(np.abs(pool.base - pool.target)))})
    (path.parent / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
