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

The package shares the existing `cache/grid_crops_1400` files. Edit `data.dataset_dir` in
`config.toml`, install the requirements, and train from the repository root:

```bash
python -m pip install -e .
python -m pip install -r experiments/dinov3_grid_patch_attention/requirements.txt

python -m experiments.dinov3_grid_patch_attention.train \
  --config experiments/dinov3_grid_patch_attention/config.toml \
  --from-scratch
```

If the shared cache is incomplete, validate or build it first:

```bash
python -m experiments.dinov3_grid_patch_attention.prepare_grid_cache \
  --config experiments/dinov3_grid_patch_attention/config.toml
```

Resume with the complete optimizer/scheduler/AMP/early-stop state:

```bash
python -m experiments.dinov3_grid_patch_attention.train \
  --config experiments/dinov3_grid_patch_attention/config.toml \
  --resume outputs/dinov3_grid_patch_attention/last.pt
```

## Evaluate and inspect attention

```bash
python -m experiments.dinov3_grid_patch_attention.evaluate \
  --config experiments/dinov3_grid_patch_attention/config.toml \
  --checkpoint outputs/dinov3_grid_patch_attention/best.pt
```

In addition to the regression metrics and predictions, evaluation saves:

- `attention_examples.png`: patch-attention heatmaps over the exact cached grid crops.
- `attention_inspection.png`: severity-spanning original/overlay/top-patch panels.
- `patch_attention.npz`: every validation attention vector, filename, and patch-grid shape.
- Per-image normalized attention entropy and top-10%-patch mass in `predictions.csv`.
- Aggregate attention diagnostics in `metrics.json`.

Normalized entropy near `1` means nearly uniform attention; near `0` means concentrated attention.
For perfectly uniform attention, the top 10% of patches hold roughly 10% of the total mass. A larger
value indicates stronger localization. The plotted heatmap is min-max normalized per image for
visibility, so use the numeric diagnostics to distinguish real concentration from tiny visualized
differences.

The inspection panel selects images across the validation target range rather than simply taking the
first batch. Cyan boxes mark the patches with the largest learned pooling weights. These weights
show how the regression aggregator combined patch features; they are not the backbone's internal
self-attention and should not be treated as a causal explanation by themselves.

The experiment is useful only if MAE/RMSE/R² improve and the overlays focus on plausible plant or
damage regions. Better training loss with worse validation metrics indicates overfitting; extremely
concentrated attention on grid edges indicates a shortcut rather than useful localization.
