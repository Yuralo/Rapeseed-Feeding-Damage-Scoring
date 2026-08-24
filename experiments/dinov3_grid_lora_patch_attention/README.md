# DINOv3 LoRA + gated patch-attention regression

This package combines the two changes requested for this experiment:

1. rank-8 LoRA on every DINOv3 block's `q_proj` and `v_proj`;
2. learned gated attention over the final-layer patch tokens.

The regression representation is:

```text
CLS token + mean(patch tokens) + gated_attention(patch tokens)
```

The final DINOv3 normalization and the regression head are also trainable. All original
transformer attention/MLP weights remain frozen. Targets are z-score normalized from the training
split and transformed back to the score scale for reported MAE, RMSE, plots, and predictions.

This is a controlled combination of the existing experiments:

```text
LoRA control:      all-block q/v LoRA + final norm + CLS + patch mean
this experiment:   all-block q/v LoRA + final norm + CLS + patch mean + gated attention
```

The attention scorer starts with zero weights, so its initial distribution is exactly uniform.
Training must learn a useful departure from the patch mean instead of starting with a random
spatial preference.

## Important preprocessing check

This package uses the shared clean 7.5% inset grid-crop cache. A previously silent false-positive
grid crop has been found in the data, so do not trust a run until the complete preprocessing audit
contains no obviously broken crop. Inspect the filenames and crop images before training; deleting
or rebuilding a stale cache may be necessary after detector changes.

Build or validate this experiment's cache:

```bash
python -m experiments.dinov3_grid_lora_patch_attention.prepare_grid_cache \
  --config experiments/dinov3_grid_lora_patch_attention/config.toml
```

For a visual full-dataset audit:

```bash
python -m experiments.dinov3_grid_lora_patch_attention.inspect_preprocessing \
  --config experiments/dinov3_grid_lora_patch_attention/config.toml \
  --count 100000
```

## Install and train

Edit `data.dataset_dir` in `config.toml`, then install the project and experiment requirements:

```bash
python -m pip install -e .
python -m pip install -r experiments/dinov3_grid_lora_patch_attention/requirements.txt
```

Start from scratch:

```bash
python -m experiments.dinov3_grid_lora_patch_attention.train \
  --config experiments/dinov3_grid_lora_patch_attention/config.toml \
  --from-scratch
```

Resume:

```bash
python -m experiments.dinov3_grid_lora_patch_attention.train \
  --config experiments/dinov3_grid_lora_patch_attention/config.toml \
  --resume outputs/dinov3_grid_lora_patch_attention_clean_inset075/last.pt
```

Evaluate:

```bash
python -m experiments.dinov3_grid_lora_patch_attention.evaluate \
  --config experiments/dinov3_grid_lora_patch_attention/config.toml \
  --checkpoint outputs/dinov3_grid_lora_patch_attention_clean_inset075/best.pt
```

The checkpoint is deliberately compact: it contains the LoRA tensors, final norm,
patch-attention pooler, regression head, optimizer/scheduler state, target scaler, split
filenames, and configuration. Evaluation reloads the frozen pretrained DINOv3 weights and rejects
checkpoints from the LoRA-only or patch-attention-only packages.

## Outputs and comparison

The run saves standard metrics and prediction examples as well as:

- `attention_examples.png`: prediction examples with attention relative to uniform;
- `attention_inspection.png`: original, heat map, and top-attended patches side by side;
- `patch_attention.npz`: raw patch weights and grid dimensions;
- `predictions.csv`: filenames, predictions, residuals, and attention summaries;
- `model_parameters.json`: LoRA, attention, and total trainable parameter counts.

Compare first against `outputs/dinov3_grid_lora_clean_inset075`. That isolates the contribution of
gated patch pooling. Compare MAE, RMSE, R², and the image-level residuals—not training loss alone.
Also inspect whether attention follows plants/damage rather than labels, borders, or grid wire.

LoRA across every block still backpropagates through the full transformer, so activation memory is
not as small as the trainable parameter count suggests. The initial RTX 3090 setting is micro-batch
8 with accumulation 2. If it runs out of memory, change only `batch_size = 4` and
`gradient_accumulation_steps = 4`, preserving effective batch 16.
