# DINOv3 3×3 + 4×4 multi-scale tiled MIL

This experiment learns one regression head from both spatial scales that already performed well:

```text
same clean 1400×1400 grid crop
        ├── 3×3 cached tiles → mean + dedicated gated attention
        ├── 4×4 cached tiles → mean + dedicated gated attention
        └── shared global representation
                              ↓
                    small regression head
```

The DINOv3 backbone remains frozen. Training reads the existing per-image 3×3 and 4×4 feature
caches, so it does not load or run DINOv3 and does not create a third cache. The feature projection
is shared across scales, while attention is scale-specific. Both scale means remain in the head so
all tiles contribute even when attention becomes concentrated.

## Required caches

If both earlier experiments have already been run, skip this section. Otherwise create the two
caches:

```bash
python -m experiments.dinov3_grid_tiled_mil.prepare_features \
  --config experiments/dinov3_grid_tiled_mil/config.toml

python -m experiments.dinov3_grid_tiled_mil.prepare_features \
  --config experiments/dinov3_grid_tiled_mil/config_4x4_small_regularized.toml
```

Cache verification happens before training and reports missing or stale records with their scale,
filename, and path.

## Train

```bash
python -m experiments.dinov3_grid_multiscale_tiled_mil.train \
  --config experiments/dinov3_grid_multiscale_tiled_mil/config.toml \
  --from-scratch
```

Resume an interrupted run:

```bash
python -m experiments.dinov3_grid_multiscale_tiled_mil.train \
  --config experiments/dinov3_grid_multiscale_tiled_mil/config.toml \
  --resume outputs/dinov3_grid_multiscale_3x3_4x4_mil_clean_inset075/last.pt
```

Training writes independent `best_mse.pt` and `best_mae.pt` checkpoints. The root artifacts are
generated from `best_mse.pt`; `best_mae_evaluation/` contains the complete MAE-selected evaluation.

## Evaluate a checkpoint

```bash
python -m experiments.dinov3_grid_multiscale_tiled_mil.evaluate \
  --config experiments/dinov3_grid_multiscale_tiled_mil/config.toml \
  --checkpoint outputs/dinov3_grid_multiscale_3x3_4x4_mil_clean_inset075/best_mae.pt \
  --output-dir outputs/dinov3_grid_multiscale_3x3_4x4_mil_clean_inset075/evaluation_mae
```

Important artifacts include `predictions.csv` with every 3×3 and 4×4 attention weight,
`multiscale_tile_attention.npz` with raw weights and boxes, and
`multiscale_attention_inspection.png` with the two heatmaps and top crops side by side.
