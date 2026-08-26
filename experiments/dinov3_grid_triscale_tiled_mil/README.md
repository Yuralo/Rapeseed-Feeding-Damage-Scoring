# DINOv3 3×3 + 4×4 + 5×5 tri-scale tiled MIL

This experiment combines the complementary behavior observed in the previous runs:

```text
same clean 1400×1400 grid crop
        ├── 3×3 cached tiles → mean only (plot-level context)
        ├── 4×4 cached tiles → mean + gated attention (regional evidence)
        ├── 5×5 cached tiles → mean + gated attention (individual-plant evidence)
        └── one averaged global representation
                              ↓
                    small regression head
```

The 3×3 attention from the preceding multi-scale experiment was nearly uniform, so it is removed
rather than duplicated alongside the 3×3 mean. DINOv3 remains frozen and is not loaded during
training.

## Required caches

The experiment reuses the existing caches:

- `cache/dinov3_grid_tiled_mil_features`
- `cache/dinov3_grid_tiled_mil_features_4x4`
- `cache/dinov3_grid_tiled_mil_features_5x5`

If all three earlier feature-extraction runs completed, no preparation command is needed. Training
audits every cache record and reports missing or stale records before creating the model.

## Train

```bash
python -m experiments.dinov3_grid_triscale_tiled_mil.train \
  --config experiments/dinov3_grid_triscale_tiled_mil/config.toml \
  --from-scratch
```

Resume:

```bash
python -m experiments.dinov3_grid_triscale_tiled_mil.train \
  --config experiments/dinov3_grid_triscale_tiled_mil/config.toml \
  --resume outputs/dinov3_grid_triscale_3x3_4x4_5x5_mil_clean_inset075/last.pt
```

Training independently saves `best_mse.pt` and `best_mae.pt`, evaluates both, and writes the
MAE-selected artifacts under `best_mae_evaluation/`.

## Evaluate a checkpoint

```bash
python -m experiments.dinov3_grid_triscale_tiled_mil.evaluate \
  --config experiments/dinov3_grid_triscale_tiled_mil/config.toml \
  --checkpoint outputs/dinov3_grid_triscale_3x3_4x4_5x5_mil_clean_inset075/best_mae.pt \
  --output-dir outputs/dinov3_grid_triscale_3x3_4x4_5x5_mil_clean_inset075/evaluation_mae
```

`triscale_attention_inspection.png` displays the 3×3 context layout and both learned attention
maps/top crops. `predictions.csv` contains every 4×4 and 5×5 attention weight, and
`triscale_tile_attention.npz` stores their raw arrays and all three tile layouts.
