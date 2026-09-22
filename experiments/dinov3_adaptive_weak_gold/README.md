# Adaptive MIL trained on weak labels, evaluated on all gold images

This is a direct architecture test of the strongest earlier 156-image model:
the routed DINOv3 **global + 3×3 overlapping context tiles + SAM plant crops**
with the original `SamAdaptiveMILRegressor` attention head and normalized-MSE
training. It is **not** the hierarchical global/2×2-cell/plant head used by the
previous weak-only experiment.

The data protocol is unchanged. The existing weak-only manifests exclude every
gold image and every weak image from the same physical plot. All 470 gold images
are reserved for validation/checkpoint selection; none enters the optimizer.
The two-scorer mean is used where both scores exist, and the available single
score is used for the remaining weak images. The head is initialized from
scratch—loading the previous gold-trained head would leak labels.

The previous adaptive result (2.50 MAE on a 156-image validation split) and this
run's 470-image gold validation result are **not directly comparable**. The
meaningful architecture comparison is the earlier hierarchical weak-only run
evaluated on these exact same 470 images.

The experiment reuses the existing routed three-view feature records for the
global image and SAM plant crops. It computes only new 3×3 tile embeddings from
the same processed image, applying EXIF orientation to raw photos. Thus it does
not rerun SAM3 or the grid detector and does not use the old broken IMG crops.

## Run on the training machine

The weak/gold manifests and three-view feature cache from the preceding
experiment must already be available. If needed, run the preceding package's
`build_manifests` command first.

```bash
python -m experiments.dinov3_adaptive_weak_gold.prepare_features \
  --config experiments/dinov3_adaptive_weak_gold/config.toml \
  --limit 20

python -m experiments.dinov3_adaptive_weak_gold.prepare_features \
  --config experiments/dinov3_adaptive_weak_gold/config.toml

python -m experiments.dinov3_adaptive_weak_gold.train \
  --config experiments/dinov3_adaptive_weak_gold/config.toml
```

The full preparation run is resumable: existing valid 3×3 caches are reused.
The six known unusable weak records are excluded; all gold features must be
present. Additional weak failures are logged and permitted only within the
configured 5% limit. Training writes `data_summary.json` and
`omitted_weak_feature_rows.csv` to make the actual sample count explicit.

The run produces `best_mse.pt`, `best_mae.pt`, `predictions.csv`, target-range
metrics, attention arrays/inspection, example and worst-error plots, learning
curves, and weak-train cohort diagnostics. The weak-train diagnostics are
in-sample, not external validation.

Resume with:

```bash
python -m experiments.dinov3_adaptive_weak_gold.train \
  --config experiments/dinov3_adaptive_weak_gold/config.toml \
  --resume outputs/dinov3_adaptive_weak_gold/last.pt
```

Re-evaluate the saved checkpoint without loading weak training features:

```bash
python -m experiments.dinov3_adaptive_weak_gold.evaluate \
  --config experiments/dinov3_adaptive_weak_gold/config.toml \
  --checkpoint outputs/dinov3_adaptive_weak_gold/best_mae.pt
```

Gold labels select the checkpoint, so this is validation rather than an
independent final test. Do not compare its MAE directly with old 73- or
156-image validation results.

To inspect the weak training labels visually, sample three images from each
source cohort (nine total):

```bash
python -m analysis.sample_weak_images --per-cohort 3
```

Open `outputs/random_weak_sample/index.html`. The same folder contains the
original images and `sampled_rows.csv`, including each target, its scorer
source, and the individual scorer values when two scores exist. This sampler
uses the weak-training manifest and never draws from the 470 gold images.
