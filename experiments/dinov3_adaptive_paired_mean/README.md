# Best plant-damage MIL trained on two-rater means

This experiment reuses the strongest saved architecture: the frozen hierarchical
global + four-cell + SAM-plant model, plus its trainable high-resolution plant-patch
correction branch. The correction branch is freshly initialized so this is a clean
retrain rather than continued optimization of the earlier gold-only head.

- target: `(score_jlu + score_gau) / 2`;
- training: 325 gold-train + 381 discordant dual-rater images (706 total before any
  audited feature failures);
- checkpoint selection: 73 gold validation images;
- final in-distribution report: 72 untouched gold test images;
- excluded: 77 discordant images sharing plot groups with validation/test;
- OOD: the exact 100 DSV and 100 WG images used by the earlier OOD probe.

The frozen base checkpoint was trained only on the same 325 gold-train images and never
saw validation/test labels. Its target scaler is retained because its frozen output is
expressed in that coordinate system; this is only a fixed affine transformation, and
all reported MAE/MSE/RMSE/R² values are in original score units. The test and OOD sets
never select a checkpoint.

## Run

From the project root:

```bash
python -m experiments.dinov3_adaptive_paired_mean.manifests \
  --config experiments/dinov3_adaptive_paired_mean/config.toml

python -m experiments.dinov3_adaptive_paired_mean.prepare_features \
  --config experiments/dinov3_adaptive_paired_mean/config.toml

python -m experiments.dinov3_adaptive_paired_mean.train \
  --config experiments/dinov3_adaptive_paired_mean/config.toml

python -m experiments.dinov3_adaptive_paired_mean.prepare_features \
  --config experiments/dinov3_adaptive_paired_mean/config.toml \
  --include-ood

python -m experiments.dinov3_adaptive_paired_mean.evaluate_ood \
  --config experiments/dinov3_adaptive_paired_mean/config.toml \
  --checkpoint outputs/dinov3_adaptive_paired_mean/best_mse.pt
```

Feature preparation is resumable and validates existing caches. Do not pass
`--overwrite` unless preprocessing settings changed and you intentionally want to redo
SAM/DINO extraction.

Resume interrupted head training with:

```bash
python -m experiments.dinov3_adaptive_paired_mean.train \
  --config experiments/dinov3_adaptive_paired_mean/config.toml \
  --resume outputs/dinov3_adaptive_paired_mean/last.pt
```

Recreate the final gold test report without retraining:

```bash
python -m experiments.dinov3_adaptive_paired_mean.evaluate \
  --config experiments/dinov3_adaptive_paired_mean/config.toml \
  --checkpoint outputs/dinov3_adaptive_paired_mean/best_mse.pt \
  --split test
```

## Results to inspect

- `run_summary.json`: validation/test MAE, MSE, RMSE and R²;
- `gold_validation/summary.json`: exact metric changes against the previous best model;
- `gold_validation/paired_previous_best_comparison.csv`: image-level old/new errors;
- `gold_test/summary.json`: full untouched gold test analysis;
- `gold_test/predictions.csv`: per-image predictions and attention diagnostics;
- `ood_matched_probe/ood_summary.json`: paired OOD improvements versus the earlier model;
- `ood_matched_probe/<cohort>/paired_ood_comparison.csv`: image-level old/new errors.

OOD labels are weak/single-rater labels, not gold truth. Use OOD only as a diagnostic,
never for checkpoint selection or hyperparameter tuning.
