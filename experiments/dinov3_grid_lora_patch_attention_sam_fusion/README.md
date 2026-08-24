# DINOv3 LoRA + patch attention + SAM fusion

This independent experiment tests whether explicit plant segmentation improves damage regression
beyond the clean 7.5%-inset LoRA + gated patch-attention model.

Each sample has three synchronized representations:

```text
clean grid crop ────────────────→ shared DINOv3 + LoRA + patch attention ─┐
SAM-masked clean crop ─────────→ same shared backbone and pooler ────────┼→ gated residual fusion
binary SAM plant mask (56×56) ─→ small CNN mask encoder ─────────────────┘
```

The DINOv3 backbone is not duplicated: both image branches use the same frozen base weights,
all-block query/value LoRA adapters, trainable final normalization, and gated patch-attention
pooler. The binary mask encoder and fusion module are intentionally small.

The ordinary original-image regression head remains present. The SAM fusion produces a residual
prediction whose final layer is initialized to zero. The recommended run warm-starts that original
branch from the completed LoRA + patch-attention checkpoint. It freezes the warm-started branch for
five epochs while fusion learns, then fine-tunes everything together.

The joint phase explicitly supervises the base prediction and discourages unnecessarily large SAM
corrections:

```text
loss = final MSE + 0.25 × base MSE + 0.01 × mean(SAM delta²)
```

SAM itself is always frozen and is never loaded by the training or evaluation process. Masks are
generated once and cached as PNG files with JSON sidecars.

## Controlled settings

The package keeps the same:

- labeled data, split seed, 7.5% grid inset, mild augmentation, and normalized targets;
- DINOv3 backbone and processor;
- rank-8 `q_proj`/`v_proj` LoRA across every transformer block;
- final normalization, patch-attention settings, learning rates, and effective batch size 16.

Because each sample now requires two backbone passes, the starting micro-batch is 4 with four-step
gradient accumulation. If a 3090 runs out of memory, use batch 2 with accumulation 8.

## 1. Install and configure

Edit `data.dataset_dir` in `config.toml`, then install:

```bash
python -m pip install -e .
python -m pip install -r \
  experiments/dinov3_grid_lora_patch_attention_sam_fusion/requirements.txt
```

SAM3 may require Hugging Face authentication or model access on the training machine. Resolve that
before preparing masks.

## 2. Prepare the clean grid crops

```bash
python -m experiments.dinov3_grid_lora_patch_attention_sam_fusion.prepare_grid_cache \
  --config experiments/dinov3_grid_lora_patch_attention_sam_fusion/config.toml
```

The known silent grid false-positive is not yet automatically rejected. SAM cannot repair incorrect
grid geometry, so inspect suspicious crops and filenames before trusting the experiment.

## 3. Generate the SAM cache

```bash
python -m experiments.dinov3_grid_lora_patch_attention_sam_fusion.prepare_sam_cache \
  --config experiments/dinov3_grid_lora_patch_attention_sam_fusion/config.toml
```

The default prompt is `green leaf`, with score threshold 0.25 and mask threshold 0.5. Cache
identity includes the source filename, grid settings, model, prompts, thresholds, and configured
foreground-quality bounds. Changing any of those settings creates a separate cache entry.

Masks outside the configured foreground range are saved for diagnosis but marked invalid, logged
to `sam_failures.jsonl`, and refused by training. Use `--overwrite` to regenerate the exact
configuration.

## 4. Inspect masks

```bash
python -m experiments.dinov3_grid_lora_patch_attention_sam_fusion.inspect_sam_masks \
  --config experiments/dinov3_grid_lora_patch_attention_sam_fusion/config.toml \
  --count 24
```

This creates one JPEG per sample under `sam_mask_inspection/`, rather than one enormous contact
sheet. Each file shows the clean crop, mask overlay, masked DINO input, filename, score, foreground
coverage, and mask validity. Specific files can be inspected with repeated `--filename` options.

Do not start training merely because cache generation exits successfully. Check that SAM includes
small or damaged plants rather than only bright healthy leaves, and that it excludes labels, wire,
and soil.

## 5. Train, resume, and evaluate

Warm-start from the earlier LoRA + patch-attention result:

```bash
python -m experiments.dinov3_grid_lora_patch_attention_sam_fusion.train \
  --config experiments/dinov3_grid_lora_patch_attention_sam_fusion/config.toml \
  --initialize-from \
  outputs/dinov3_grid_lora_patch_attention_clean_inset075/best.pt
```

This restores the exact control-model train/validation filenames and normalization statistics. A
plain `--from-scratch` mode remains available, but it does not use the staged warm-start strategy.

Resume:

```bash
python -m experiments.dinov3_grid_lora_patch_attention_sam_fusion.train \
  --config experiments/dinov3_grid_lora_patch_attention_sam_fusion/config.toml \
  --resume \
  outputs/dinov3_grid_lora_patch_attention_sam_fusion_warmstart_aux_clean_inset075/last.pt
```

Evaluate:

```bash
python -m experiments.dinov3_grid_lora_patch_attention_sam_fusion.evaluate \
  --config experiments/dinov3_grid_lora_patch_attention_sam_fusion/config.toml \
  --checkpoint \
  outputs/dinov3_grid_lora_patch_attention_sam_fusion_warmstart_aux_clean_inset075/best.pt
```

Checkpoints contain only trainable LoRA, normalization, patch-attention, mask-encoder, fusion, and
regression parameters plus training state. Evaluation reloads the frozen pretrained backbone and
rejects incompatible preprocessing, SAM, or architecture settings.

## Outputs and interpretation

In addition to regression metrics and filename-aware predictions, evaluation saves:

- automatic base-only versus final-model metrics and improved/worsened sample counts;
- base prediction and SAM residual for every image;
- original, masked, and binary-mask fusion weights;
- original-branch and masked-branch patch attention;
- mask foreground coverage;
- compressed raw diagnostic arrays;
- one five-panel JPEG per representative validation image under
  `sam_fusion_inspection/`.

Compare against `outputs/dinov3_grid_lora_patch_attention_clean_inset075`. A useful SAM result
should improve validation errors and show nontrivial but stable residual corrections. Fusion weights
alone do not measure causal importance because each representation has a learned projection; use
the saved SAM delta and image-level residuals as well.
