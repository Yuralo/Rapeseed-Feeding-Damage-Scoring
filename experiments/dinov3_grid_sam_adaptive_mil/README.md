# SAM-guided plant-centred adaptive MIL

This experiment replaces fixed fine tiles with variable SAM-guided plant/group crops while retaining
the cached global view and 3×3 mean for plot-level context. Nearby SAM components are grouped,
nearest groups are merged when the configured cap is exceeded, and foreground coverage is checked
before DINO features are saved. Training uses global + 3×3 mean + plant mean + masked plant
attention. DINO and SAM remain frozen and are absent during head training.

First audit that the existing SAM masks are acceptable. Then create adaptive frozen features:

```bash
python -m experiments.dinov3_grid_sam_adaptive_mil.inspect_crops \
  --config experiments/dinov3_grid_sam_adaptive_mil/config.toml \
  --limit 24
```

This writes one moderate-size JPEG per image plus a JSON manifest; it does not create one giant
contact sheet. Check that boxes cover plants with useful surrounding context before extraction.

Then create adaptive frozen features:

```bash
python -m experiments.dinov3_grid_sam_adaptive_mil.prepare_features \
  --config experiments/dinov3_grid_sam_adaptive_mil/config.toml
```

Use `--limit 10` for a smoke test; rerun without the limit for the complete cache. The command
reports the number of crops and SAM-foreground coverage for every image.

Train:

```bash
python -m experiments.dinov3_grid_sam_adaptive_mil.train \
  --config experiments/dinov3_grid_sam_adaptive_mil/config.toml \
  --from-scratch
```

The output directory is `outputs/dinov3_grid_sam_adaptive_mil_clean_inset075`. It contains
independent `best_mse.pt` and `best_mae.pt` checkpoints, prediction CSVs, instance counts, mask
coverage, all plant attention weights, and `adaptive_attention_inspection.png`.

Resume:

```bash
python -m experiments.dinov3_grid_sam_adaptive_mil.train \
  --config experiments/dinov3_grid_sam_adaptive_mil/config.toml \
  --resume outputs/dinov3_grid_sam_adaptive_mil_clean_inset075/last.pt
```
