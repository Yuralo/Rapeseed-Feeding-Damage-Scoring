# DINOv3 global + tiled MIL

This experiment tests whether full-image resizing is hiding small feeding-damage evidence. It uses
the same clean 7.5% inset grid crop as the current experiments, but represents every image with:

```text
one complete 1400×1400 view
          +
nine overlapping 560×560 tiles (3×3, 25% overlap)
          ↓
shared frozen DINOv3 encoder
          ↓
CLS + mean-patch representation for each view
          ↓
global representation + tile mean + learned gated tile attention
          ↓
normalized regression target
```

SAM is deliberately absent. The experiment answers one question: does retaining more local
resolution improve scoring? The DINOv3 backbone is frozen and its features are cached once, so
training the MIL head is fast and does not require loading DINOv3.

## 1. Install

```bash
python -m pip install -e .
python -m pip install -r experiments/dinov3_grid_tiled_mil/requirements.txt
```

Edit `data.dataset_dir` in `config.toml`. The existing clean grid cache is reused; make sure its
visual audit is acceptable before extracting features.

## 2. Create the feature cache

```bash
python -m experiments.dinov3_grid_tiled_mil.prepare_features \
  --config experiments/dinov3_grid_tiled_mil/config.toml
```

The command is resumable. It stores one compressed `.npz` per source image rather than one giant
file. Every record contains ten frozen feature vectors, nine tile boxes, and the exact processed
image path. Existing valid records are skipped. To rebuild them:

```bash
python -m experiments.dinov3_grid_tiled_mil.prepare_features \
  --config experiments/dinov3_grid_tiled_mil/config.toml \
  --overwrite
```

Use `--limit 5` for an initial extraction smoke test. A limited cache cannot be used for full
training; rerun without the limit afterward. Grid failures and DINO extraction failures are saved
as JSONL files in the configured output directory.

## 3. Train and resume

```bash
python -m experiments.dinov3_grid_tiled_mil.train \
  --config experiments/dinov3_grid_tiled_mil/config.toml \
  --from-scratch
```

```bash
python -m experiments.dinov3_grid_tiled_mil.train \
  --config experiments/dinov3_grid_tiled_mil/config.toml \
  --resume outputs/dinov3_grid_tiled_mil_clean_inset075/last.pt
```

Only the small MIL head is trained. Targets use the training split's mean and standard deviation,
and the checkpoint stores the exact filename split and scaler.

## 4. Evaluate

```bash
python -m experiments.dinov3_grid_tiled_mil.evaluate \
  --config experiments/dinov3_grid_tiled_mil/config.toml \
  --checkpoint outputs/dinov3_grid_tiled_mil_clean_inset075/best.pt
```

Important outputs include:

- `predictions.csv`, including the filename and all nine tile weights;
- `prediction_examples.png`, using the exact clean processed images;
- `tile_attention_examples.png`, with a tile-level heat map and top-tile rectangle;
- `tile_attention_inspection.png`, showing the tile layout, heat map, and top tile crop;
- `tile_attention.npz`, containing raw weights and tile boxes;
- `metrics.json`, `summary.json`, checkpoints, and training history.

## 5. Compare against the current best model

Use the same validation manifest and compare paired errors:

```bash
python -m experiments.dinov3_grid_tiled_mil.compare \
  --candidate outputs/dinov3_grid_tiled_mil_clean_inset075/predictions.csv \
  --baseline outputs/dinov3_grid_lora_patch_attention_sam_fusion/predictions.csv \
  --output outputs/dinov3_grid_tiled_mil_clean_inset075/paired_comparison.json
```

The comparison rejects mismatched filename manifests and reports a paired bootstrap 95% confidence
interval for mean absolute-error reduction. A positive reduction favors tiled MIL. Do not treat a
tiny aggregate difference like 0.009 MAE as a meaningful improvement when the interval contains
zero.

