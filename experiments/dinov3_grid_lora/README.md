# DINOv3 clean-grid LoRA regression

This experiment tests low-rank backbone adaptation before adding SAM or changing the regression
representation. It uses the same cleaned 7.5% inset crops, validation split, seed, mild
augmentation, normalized targets, `CLS + mean(patch tokens)` representation, regression head, and
training schedule as the clean two-block control.

The controlled adaptation change is:

```text
clean two-block control: fully train final two transformer blocks + final norm + head
this experiment:         rank-8 LoRA on q_proj/v_proj in every block + final norm + head
```

All original attention and MLP weights remain frozen. The model validates that it found exactly one
query and one value projection per transformer block before training. This prevents a Transformers
module-name change from silently producing a head-only experiment.

The initial settings are intentionally conservative for the small labeled data set:

- rank `8`, alpha `16`, adapter dropout `0.05`
- query and value projections in all DINOv3 transformer blocks
- adapter learning rate `1e-4`; head learning rate `3e-4`
- no adapter weight decay
- final backbone normalization remains trainable to match the two-block control

Checkpoints contain only trainable adapters, final normalization, head, and training state. The
pretrained frozen backbone is loaded again during resume/evaluation, keeping checkpoints much
smaller than full-backbone checkpoints.

LoRA reduces trainable parameters and optimizer/checkpoint memory, but adapters in every block
still require backpropagation through the full transformer. Activation memory can therefore be
higher than the final-two-block run. The RTX 3090 starting point remains micro-batch 8 with two-step
accumulation. If that runs out of memory, change only `batch_size` to `4` and
`gradient_accumulation_steps` to `4`, preserving effective batch 16.

## Prepare and train

Edit `data.dataset_dir` in `config.toml`, then install dependencies:

```bash
python -m pip install -e .
python -m pip install -r experiments/dinov3_grid_lora/requirements.txt
```

The clean cache is shared with the matching two-block and patch-attention runs. Build or validate it
before training:

```bash
python -m experiments.dinov3_grid_lora.prepare_grid_cache \
  --config experiments/dinov3_grid_lora/config.toml
```

Train from scratch:

```bash
python -m experiments.dinov3_grid_lora.train \
  --config experiments/dinov3_grid_lora/config.toml \
  --from-scratch
```

Resume an interrupted run:

```bash
python -m experiments.dinov3_grid_lora.train \
  --config experiments/dinov3_grid_lora/config.toml \
  --resume outputs/dinov3_grid_lora_clean_inset075/last.pt
```

Evaluate the best adapter:

```bash
python -m experiments.dinov3_grid_lora.evaluate \
  --config experiments/dinov3_grid_lora/config.toml \
  --checkpoint outputs/dinov3_grid_lora_clean_inset075/best.pt
```

The run saves the same metrics, residual plot, prediction examples, history, runtime, and VRAM
artifacts as the two-block control. Compare validation MAE, RMSE, and R² against
`outputs/dinov3_grid_unfreeze2_clean_inset075`; do not compare only training loss.

If rank 8 query/value LoRA is clearly better or at least competitive, the next LoRA-specific test
can expand targets to output projections or MLP layers. Do not combine that expansion with SAM or
gated pooling in the same run, because the source of any improvement would become ambiguous.
