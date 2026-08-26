# 4×4 + SAM-adaptive hybrid MIL

This controlled experiment combines the complementary fixed-grid and plant-centred representations
without changing DINOv3 or SAM. It trains a small head over six frozen representations:

```text
global + 3×3 mean + 4×4 mean + 4×4 gated attention
       + SAM plant mean + masked SAM plant attention
```

It reuses these existing caches:

- `cache/dinov3_grid_tiled_mil_features` for the global and 3×3 context features;
- `cache/dinov3_grid_tiled_mil_features_4x4` for fixed 4×4 features;
- `cache/dinov3_grid_sam_adaptive_mil_features` for adaptive plant features;
- `cache/sam3_masks_grid1400_inset075` for inspection plots.

No feature extraction or SAM inference is required if the earlier 3×3, 4×4, and adaptive runs were
prepared on the same dataset paths. Train from scratch with:

```bash
python -m experiments.dinov3_grid_4x4_sam_adaptive_hybrid_mil.train \
  --config experiments/dinov3_grid_4x4_sam_adaptive_hybrid_mil/config.toml \
  --from-scratch
```

The run is written to
`outputs/dinov3_grid_4x4_sam_adaptive_hybrid_mil_clean_inset075`. It saves independent best-MSE,
best-MAE, and last checkpoints; enriched prediction CSVs; target-range and calibration diagnostics;
representative and worst-error examples; and a combined 4×4/SAM attention inspection.

Resume:

```bash
python -m experiments.dinov3_grid_4x4_sam_adaptive_hybrid_mil.train \
  --config experiments/dinov3_grid_4x4_sam_adaptive_hybrid_mil/config.toml \
  --resume outputs/dinov3_grid_4x4_sam_adaptive_hybrid_mil_clean_inset075/last.pt
```

Evaluate an explicit checkpoint:

```bash
python -m experiments.dinov3_grid_4x4_sam_adaptive_hybrid_mil.evaluate \
  --config experiments/dinov3_grid_4x4_sam_adaptive_hybrid_mil/config.toml \
  --checkpoint outputs/dinov3_grid_4x4_sam_adaptive_hybrid_mil_clean_inset075/best_mse.pt
```

Regenerate reports for both selected checkpoints without retraining:

```bash
python -m experiments.dinov3_grid_4x4_sam_adaptive_hybrid_mil.report \
  --config experiments/dinov3_grid_4x4_sam_adaptive_hybrid_mil/config.toml
```
