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
independent `best_mse.pt` and `best_mae.pt` checkpoints, regression and residual plots,
representative and worst-error image panels, SAM/attention diagnostic plots, target-range metrics,
and a prediction CSV with paths, errors, instance counts, mask coverage, attention statistics,
per-instance weights, and foreground-pixel counts.

Regenerate the complete analysis for both saved checkpoints without retraining:

```bash
python -m experiments.dinov3_grid_sam_adaptive_mil.report \
  --config experiments/dinov3_grid_sam_adaptive_mil/config.toml
```

Reports are written to `posthoc_reports/best_mse` and `posthoc_reports/best_mae` inside the run
directory, with a checkpoint comparison JSON beside them. Use `--output-dir` to place these
reports elsewhere.

Resume:

```bash
python -m experiments.dinov3_grid_sam_adaptive_mil.train \
  --config experiments/dinov3_grid_sam_adaptive_mil/config.toml \
  --resume outputs/dinov3_grid_sam_adaptive_mil_clean_inset075/last.pt
```
