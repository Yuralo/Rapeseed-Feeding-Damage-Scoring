# Hierarchical three-view MIL: weak labels → gold labels

This experiment tests two ideas without mixing their effects:

1. A hierarchical image representation:
   - one whole-image feature for spatial context;
   - four 2×2 cell features;
   - variable SAM3 plant crops, each assigned to a cell by its centre.
2. Reliability-weighted weak supervision before gold-standard fine-tuning.

The DINOv3 backbone is frozen and loaded from
`outputs/dinov3_routed_source_adaptation/adapted_backbone`. The trainable head first
contextualizes each plant with its owning cell and the global feature, pools plants inside
each cell, and then pools the four cells into the final regression representation.

## Data safety

- Target normalization is always fitted on the 325 gold training rows.
- Weak pretraining uses only `pretrain.csv` and its existing `sample_weight` reliability.
- Cohort-balanced sampling prevents the largest weak cohort from dominating.
- Weak pretraining uses weighted Huber loss; gold fine-tuning uses normalized MSE.
- The optimizer and learning-rate schedule are reset between stages.
- The 73-image gold validation split selects checkpoints.
- The 72-image gold test split is reserved and is never evaluated by `train.py`.
- Physical plot-group isolation is checked before training.

The same features and architecture are used by both configs, so the gold-only run is a
proper control for the weak-to-gold run.

## Run

Build the shared feature cache once. This is the expensive step because it runs routed
preprocessing, SAM3, and the adapted DINOv3 backbone over every supervised image:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m experiments.dinov3_hierarchical_three_view_mil.prepare_features \
  --config experiments/dinov3_hierarchical_three_view_mil/config_weak_then_gold.toml
```

SAM inference is capped at a 1400-pixel longest side and its mask is restored to the
prepared image coordinates afterward. If a particular image still exhausts CUDA memory,
the preparation command clears the allocator and retries at 1050 and then 700 pixels.
Mask PNGs use fast compression because the uncropped raw sources have large dimensions.

Train the gold-only control:

```bash
python -m experiments.dinov3_hierarchical_three_view_mil.train \
  --config experiments/dinov3_hierarchical_three_view_mil/config_gold_only.toml \
  --from-scratch
```

Train weak supervision followed by gold fine-tuning:

```bash
python -m experiments.dinov3_hierarchical_three_view_mil.train \
  --config experiments/dinov3_hierarchical_three_view_mil/config_weak_then_gold.toml \
  --from-scratch
```

Resume an interrupted run with its `last.pt`:

```bash
python -m experiments.dinov3_hierarchical_three_view_mil.train \
  --config experiments/dinov3_hierarchical_three_view_mil/config_weak_then_gold.toml \
  --resume outputs/dinov3_hierarchical_three_view_weak_then_gold/last.pt
```

Compare these files first:

- `outputs/dinov3_hierarchical_three_view_gold_only/summary.json`
- `outputs/dinov3_hierarchical_three_view_weak_then_gold/summary.json`
- each run's `predictions.csv`, `regression.png`, `worst_error_examples.png`, and
  `hierarchical_attention_inspection.png`
- `hierarchical_diagnostics.png` for plant count, cell focus, plant focus, and target-tail
  error relationships

Do not evaluate the test set until the architecture and supervision recipe have been
selected from validation. Then run exactly one final test evaluation:

```bash
python -m experiments.dinov3_hierarchical_three_view_mil.evaluate \
  --config experiments/dinov3_hierarchical_three_view_mil/config_weak_then_gold.toml \
  --checkpoint outputs/dinov3_hierarchical_three_view_weak_then_gold/best_mse.pt \
  --split test
```

## Failure policy

All gold features are mandatory. Weak feature failures may be skipped only while they are
at most 5% of the weak pool; the exact usable weak manifest is saved in the run directory.
Every preprocessing/SAM/feature exception is written with a traceback to
`feature_failures.jsonl`.
