# Weak labels → all-gold validation

This experiment answers: **can the available non-gold scores train our existing
hierarchical three-view model when all 470 gold images are reserved for evaluation?**

It initializes a **new regression head**. It does not load the previous gold-trained
head/checkpoint, because that would leak gold labels into this experiment. It reuses
the same routed preprocessing, SAM plant crops, adapted frozen DINOv3 features, and
three-view architecture, so the supervised-data change is isolated.

## What the labels actually support

The canonical scored manifest has 2,733 images: 470 gold and 2,263 non-gold. Removing
every non-gold image that shares a physical plot or content hash with **any** gold
image leaves **1,987** training images:

- **184** have two scores; the target is their arithmetic mean.
- **1,803** have only one score; the target is that score.

Thus, a 2,000-image training run cannot use a two-scorer mean for every image. The
`all_weak` run uses all available scores, while the `dual_only` control trains on
only the 184 plot-isolated two-scorer means. The latter stays within the same GG
cohort as gold but is much smaller and has >5-point scorer disagreement.

Both runs evaluate on **all 470 gold images**. None of those gold images, nor a weak
image from a gold plot, enters the optimizer. Targets are normalized using weak
training targets only. The gold set selects checkpoints, so it is a validation set,
**not an unbiased final test set**. Do not compare its 470-image MAE directly to the
older 73-image validation MAE from models trained on 325 gold images.

## Run the main 1,987-image experiment

From the repository root on the machine that holds the images, adapted backbone and
existing feature cache:

```bash
python -m experiments.dinov3_weak_only_gold_validation.build_manifests \
  --config experiments/dinov3_weak_only_gold_validation/config_all_weak.toml

python -m experiments.dinov3_weak_only_gold_validation.prepare_features \
  --config experiments/dinov3_weak_only_gold_validation/config_all_weak.toml

python -m experiments.dinov3_weak_only_gold_validation.train \
  --config experiments/dinov3_weak_only_gold_validation/config_all_weak.toml
```

The feature preparation reuses valid cached three-view records and computes only
missing ones. It may run SAM3 and DINOv3 for weak images that were excluded from the
old split. Errors have per-image tracebacks under
`outputs/dinov3_weak_only_gold_validation_all_weak/feature_preparation/`.
If a small fraction of **weak training** images cannot be cropped, masked, or
embedded, training omits only those rows and records them in
`omitted_weak_feature_rows.csv`; `data_summary.json` reports the actual training
count. The configured weak-failure limit is enforced. **Every gold validation
image must still have a valid feature record**—none is silently dropped.

Inspect `run_summary.json`, `summary.json` (MSE-selected),
`best_mae_evaluation/summary.json`, `predictions.csv`, `worst_error_examples.png`,
`training_history.png`, and `weak_train_cohort_diagnostics.json`. The latter is an
*in-sample* fit to weak labels, not an external evaluation. The gold predictions
include filenames and target-range analysis.

Resume an interrupted run with:

```bash
python -m experiments.dinov3_weak_only_gold_validation.train \
  --config experiments/dinov3_weak_only_gold_validation/config_all_weak.toml \
  --resume outputs/dinov3_weak_only_gold_validation_all_weak/last.pt
```

## Two-scorer-only control

Repeat the three commands above with
`experiments/dinov3_weak_only_gold_validation/config_dual_only.toml`. This run uses
only 184 mean-of-two training targets and evaluates on **the same 470 gold images**.
It answers whether adding the 1,803 singly scored images helps despite their source
shift and noisier supervision.

## Interpretation

The main run mixes GG BBCH10 (184 dual-scored), WG BBCH10 (906 singly scored), and
DSV BBCH11 (897 singly scored). A poor result could reflect disagreement noise,
different imaging conditions, growth stage, or scoring differences—not just
insufficient model capacity. Compare the all-weak run to the dual-only control on the
same gold validation set, and examine the high-score tail and cohort diagnostics.
