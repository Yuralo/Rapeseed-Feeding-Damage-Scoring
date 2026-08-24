# Learned DINOv3 patch-attention regression

This is an independent experiment package built from the best normalized two-block run. It keeps
the same split, cached grid crops, augmentation, target normalization, optimizer recipe, final two
unfrozen transformer blocks, final backbone normalization, and regression objective.

The controlled architecture change is the representation passed to the regression head:

```text
previous: CLS + mean(patch tokens)
this run: CLS + mean(patch tokens) + gated-attention(patch tokens)
```

The attention pool uses a 128-dimensional gated MLP to assign a softmax weight to every final-layer
patch token. Both the attention module and regression head use the head learning rate. The final two
backbone blocks and final normalization use the smaller backbone learning rate.

The attention scorer starts at zero, which produces uniform patch weights on the first forward pass.
This gives the small-data run a stable mean-pooling starting point; spatial concentration must be
learned rather than inherited from random initial attention scores.

## Train

The original `config.toml` preserves the first outer-grid run. It is useful for reproducing that
checkpoint, but its crops can contain collector labels attached to the grid edge. New comparisons
should use `config_clean_inset.toml`. It applies a 7.5% projective inset on every grid edge and
shares those exact cleaned inputs with the matching two-block baseline.

Edit `data.dataset_dir` in both clean configs, install the requirements, and visually audit the
preprocessing before using the GPU:

```bash
python -m pip install -e .
python -m pip install -r experiments/dinov3_grid_patch_attention/requirements.txt

python -m experiments.dinov3_grid_patch_attention.inspect_preprocessing \
  --config experiments/dinov3_grid_patch_attention/config_clean_inset.toml \
  --count 12

python -m experiments.dinov3_grid_patch_attention.prepare_grid_cache \
  --config experiments/dinov3_grid_patch_attention/config_clean_inset.toml

python -m experiments.dinov3_grid_patch_attention.train \
  --config experiments/dinov3_grid_patch_attention/config_clean_inset.toml \
  --from-scratch
```

The inspection writes one compact JPEG triptych per sample under the
`preprocessing_inspection/` directory plus a JSON report. Each file shows the source image, old
outer-grid warp, and clean inset. It deliberately does not create one enormous contact sheet, so
the memory and file size stay bounded when inspecting many images. Confirm that every collector
label is absent and that useful plant content remains. If labels remain, raise the inset in both
clean configs and use a new cache/output name.
For a formerly failing boundary image, add `--filename IMAGE_NAME.jpg` to audit that exact sample;
the option can be repeated for several filenames.

Grid detection also has a conservative boundary-recovery fallback. It runs only after normal LSD
and combined LSD/Hough selection fail. When two strong adjacent bars predict a missing third bar
near a photo edge, it extrapolates that outer bar. Interior missing lines are not inferred, and
implausibly distant corners are rejected. Cache schema `v3` prevents older crops from being reused
after this detector change.

Train the matching cleaned baseline separately before interpreting an architecture difference:

```bash
python -m experiments.dinov3_grid_unfreeze2.train \
  --config experiments/dinov3_grid_unfreeze2/config_clean_inset.toml \
  --from-scratch
```

Both clean configs use `cache/grid_crops_1400_inset075`, the same split/seed, augmentation,
optimizer, and normalized targets. Their only intentional model difference is learned patch
attention. A cache key also includes the crop size and inset, so old outer-grid crops cannot be
silently reused.

Do not resume either clean run from an old checkpoint. Checkpoint validation rejects crop-size or
inset mismatches, and both models need to be retrained from scratch because their inputs changed.

Resume with the complete optimizer/scheduler/AMP/early-stop state:

```bash
python -m experiments.dinov3_grid_patch_attention.train \
  --config experiments/dinov3_grid_patch_attention/config_clean_inset.toml \
  --resume outputs/dinov3_grid_patch_attention_clean_inset075/last.pt
```

## Evaluate and inspect attention

```bash
python -m experiments.dinov3_grid_patch_attention.evaluate \
  --config experiments/dinov3_grid_patch_attention/config_clean_inset.toml \
  --checkpoint outputs/dinov3_grid_patch_attention_clean_inset075/best.pt
```

In addition to the regression metrics and predictions, evaluation saves:

- `prediction_examples.png`: the same plain target/prediction/error sheet as the baseline runs.
- `attention_examples.png`: patch-attention heatmaps over the exact cached grid crops.
- `attention_inspection.png`: severity-spanning original/overlay/top-patch panels.
- `patch_attention.npz`: every validation attention vector, filename, and patch-grid shape.
- Per-image normalized attention entropy and top-10%-patch mass in `predictions.csv`.
- Aggregate attention diagnostics in `metrics.json`.

Normalized entropy near `1` means nearly uniform attention; near `0` means concentrated attention.
For perfectly uniform attention, the top 10% of patches hold roughly 10% of the total mass. A larger
value indicates stronger localization. Heatmaps now use one fixed scale relative to uniform
attention: `1.0×` is neutral, `0.5×` is the low end, and `2.0×` is the high end. This prevents tiny
differences in almost-uniform weights from being stretched into apparently strong hotspots. The
title also reports entropy to four decimal places and the maximum uniform-relative weight.

The inspection panel selects images across the validation target range rather than simply taking the
first batch. Cyan boxes mark the patches with the largest learned pooling weights. These weights
show how the regression aggregator combined patch features; they are not the backbone's internal
self-attention and should not be treated as a causal explanation by themselves.

The experiment is useful only if MAE/RMSE/R² improve and the overlays focus on plausible plant or
damage regions. Better training loss with worse validation metrics indicates overfitting; extremely
concentrated attention on grid edges indicates a shortcut rather than useful localization. The
collector-label focus in the first run should therefore be treated as a preprocessing finding, not
as evidence that patch attention successfully localized feeding damage.
