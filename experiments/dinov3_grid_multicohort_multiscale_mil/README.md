# Multi-cohort 3x3 + 4x4 MIL

This experiment keeps the compact multi-scale MIL architecture and changes the supervision
strategy:

1. pretrain the regression head on weak labels from DSV, WG, and non-gold Gross-Gerau images;
2. reload the best pretraining checkpoint and fine-tune on the curated 470-image gold subset;
3. select checkpoints using the gold validation manifest;
4. leave the gold test manifest untouched until an explicit evaluation command is run.

The 470 images are not mixed with the other scores as equivalent labels. They use the curated
JLU/GAU mean and weight 1.0. Remaining dual-scored and single-scored examples receive configurable
lower weights when the manifests are built. The dual-scored weight is reduced further as the
absolute JLU-GAU disagreement grows.

## 1. Build canonical manifests

```bash
python -m analysis.build_supervised_manifests \
  --root /home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage \
  --inventory outputs/dataset_inventory/dataset_images.csv \
  --output-dir outputs/dataset_manifests \
  --seed 42 \
  --gold-difference-threshold 5 \
  --dual-weak-weight 0.6 \
  --single-weak-weight 0.4
```

Inspect `outputs/dataset_manifests/manifest_summary.json` and
`outputs/dataset_manifests/score_join_issues.csv` before continuing. Every unmatched row should be
explained.

## 2. Audit preprocessing by cohort

This creates separate crop files and a CSV containing their paths; it does not create one enormous
contact sheet.

```bash
python -m experiments.dinov3_grid_multicohort_multiscale_mil.inspect_preprocessing \
  --config experiments/dinov3_grid_multicohort_multiscale_mil/config.toml \
  --samples-per-cohort 12
```

Open paths from
`outputs/dinov3_grid_multicohort_multiscale_mil/preprocessing_audit/preprocessing_audit.csv`, and
check `grid_failures.jsonl` if any sample failed.

## 3. Prepare frozen features

Smoke test both scales first:

```bash
python -m experiments.dinov3_grid_multicohort_multiscale_mil.prepare_features \
  --config experiments/dinov3_grid_multicohort_multiscale_mil/config.toml \
  --scale both \
  --limit 20
```

Then complete the resumable caches:

```bash
python -m experiments.dinov3_grid_multicohort_multiscale_mil.prepare_features \
  --config experiments/dinov3_grid_multicohort_multiscale_mil/config.toml \
  --scale both
```

## 4. Train

```bash
python -m experiments.dinov3_grid_multicohort_multiscale_mil.train \
  --config experiments/dinov3_grid_multicohort_multiscale_mil/config.toml \
  --from-scratch
```

Resume an interrupted stage with:

```bash
python -m experiments.dinov3_grid_multicohort_multiscale_mil.train \
  --config experiments/dinov3_grid_multicohort_multiscale_mil/config.toml \
  --resume outputs/dinov3_grid_multicohort_multiscale_mil/last.pt
```

Training writes the same regression, residual, prediction-example, attention, target, and history
artifacts as the earlier multiscale experiments. It does not evaluate the reserved test set.

## 5. Explicit final evaluation

```bash
python -m experiments.dinov3_grid_multicohort_multiscale_mil.evaluate \
  --config experiments/dinov3_grid_multicohort_multiscale_mil/config.toml \
  --checkpoint outputs/dinov3_grid_multicohort_multiscale_mil/best_mse.pt \
  --split test
```

Do not repeatedly inspect the reserved test metrics while choosing hyperparameters. Continue model
selection on `validation.csv`; use `test.csv` only after fixing the experiment choice.
