# DINOv3 grid crop: final two blocks + final normalization

This is an independent experiment built from the working grid-cropped regression run. It keeps the
same DINOv3 feature representation and normalized MSE objective, but trains the regression head,
the final two transformer blocks, and the backbone's final normalization.

The defaults are a conservative RTX 3090 starting point:

- FP16 mixed precision, micro-batch 8, two-step accumulation (effective batch 16)
- backbone learning rate `1e-5`, head learning rate `3e-4`
- warmup followed by cosine decay, gradient clipping at `1.0`
- validation every epoch and early stopping after six non-improving evaluations
- mild flips and 5% color jitter on training images only
- epoch duration, throughput, first-epoch data/compute time, and peak CUDA memory logging

## Setup and crop validation

Install the repository and this experiment's dependencies, then edit `data.dataset_dir` in
`config.toml`:

```bash
python -m pip install -e .
python -m pip install -r experiments/dinov3_grid_unfreeze2/requirements.txt
```

Build every grid crop before using the GPU:

```bash
python -m experiments.dinov3_grid_unfreeze2.prepare_grid_cache \
  --config experiments/dinov3_grid_unfreeze2/config.toml
```

This command exits nonzero if any crop fails. Details and tracebacks go to
`outputs/dinov3_grid_unfreeze2/grid_failures.jsonl`; the summary goes to
`grid_cache_summary.json`. Use `--overwrite` only after intentionally changing the crop algorithm.
Training can lazily fill a missing cache, but precomputing makes failures visible before a long run
and keeps the GPU timing meaningful.

## Train, resume, and evaluate

Train from scratch:

```bash
python -m experiments.dinov3_grid_unfreeze2.train \
  --config experiments/dinov3_grid_unfreeze2/config.toml \
  --from-scratch
```

If batch 8 does not fit, change `batch_size` to `4` and
`gradient_accumulation_steps` to `4`; the effective batch remains 16. If memory is comfortable,
increase the micro-batch and reduce accumulation.

Resume after interruption (or after raising the total `epochs` value):

```bash
python -m experiments.dinov3_grid_unfreeze2.train \
  --config experiments/dinov3_grid_unfreeze2/config.toml \
  --resume outputs/dinov3_grid_unfreeze2/last.pt
```

Resume restores weights, optimizer, LR scheduler, AMP scaler, early-stopping state, split manifests,
history, and global optimizer step. Changing architecture settings is rejected. Changing optimizer
settings while resuming does not restart the optimizer recipe; use a fresh run for that comparison.

Evaluate the best checkpoint with exactly the same cached grid crops:

```bash
python -m experiments.dinov3_grid_unfreeze2.evaluate \
  --config experiments/dinov3_grid_unfreeze2/config.toml \
  --checkpoint outputs/dinov3_grid_unfreeze2/best.pt
```

`predictions.csv` records both original and processed paths, and `prediction_examples.png` opens the
processed cache directly. Training augmentation is never applied during validation or evaluation.

## Raw-target comparison

`config_raw_targets.toml` runs the same split, inputs, augmentation, model, optimizer, and seed but
passes the original damage scores directly to MSE. It writes to a different output directory and
shares the existing grid-crop cache:

```bash
python -m experiments.dinov3_grid_unfreeze2.train \
  --config experiments/dinov3_grid_unfreeze2/config_raw_targets.toml \
  --from-scratch
```

Evaluate it with its own config and checkpoint:

```bash
python -m experiments.dinov3_grid_unfreeze2.evaluate \
  --config experiments/dinov3_grid_unfreeze2/config_raw_targets.toml \
  --checkpoint outputs/dinov3_grid_unfreeze2_raw_targets/best.pt
```

Do not resume a normalized checkpoint with this config or evaluate one with it; checkpoint
validation deliberately rejects that mismatch. Compare MAE, RMSE, and R² directly between the two
runs. `objective_mse` is in normalized units for the original config and raw squared-score units for
this config, so its absolute value is not comparable between the runs.

## Four-block normalized-target comparison

`config_unfreeze4.toml` is identical to the normalized two-block `config.toml` except that it
unfreezes the final four transformer blocks. It uses its own output directory:

```bash
python -m experiments.dinov3_grid_unfreeze2.train \
  --config experiments/dinov3_grid_unfreeze2/config_unfreeze4.toml \
  --from-scratch
```

Evaluate it with:

```bash
python -m experiments.dinov3_grid_unfreeze2.evaluate \
  --config experiments/dinov3_grid_unfreeze2/config_unfreeze4.toml \
  --checkpoint outputs/dinov3_grid_unfreeze4/best.pt
```

The RTX 3090 starting micro-batch remains 8 for a controlled comparison. If this configuration runs
out of memory, change only `batch_size` to `4` and `gradient_accumulation_steps` to `4`, preserving
the effective batch size of 16.

The corresponding four-block experiment without target normalization uses
`config_raw_targets_unfreeze4.toml`:

```bash
python -m experiments.dinov3_grid_unfreeze2.train \
  --config experiments/dinov3_grid_unfreeze2/config_raw_targets_unfreeze4.toml \
  --from-scratch
```

Its output directory is `outputs/dinov3_grid_unfreeze4_raw_targets`.

## Interpreting the experiment

The default config tests a small fine-tuning recipe, not only the effect of unfreezing blocks,
because augmentation and the optimizer schedule also changed. For a cleaner comparison against the
existing grid experiment, set `augmentation.enabled = false`; keep the existing split seed and
processor unchanged.

Start by comparing validation MAE/RMSE/R² and checking the residual/example plots. Also inspect
`history.json` and `model_parameters.json` for runtime, VRAM, and the actual trainable proportion.
If this is clearly better without unstable validation loss, the next experiment can copy this folder
and increase `unfreeze_last_n_blocks`. LoRA is more useful later if full-block fine-tuning becomes
too expensive; it is deliberately not mixed into this first test.
