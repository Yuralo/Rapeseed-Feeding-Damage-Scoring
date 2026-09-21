# High-resolution plant-patch evidence (gold-only)

This is a controlled successor to `dinov3_hierarchical_three_view_mil`. It loads the
**gold-only** three-view checkpoint, freezes that complete scorer, and adds a small
trainable branch over 2×2 high-resolution patches inside each existing SAM plant box.
The branch starts with a **zero correction**, so its epoch-0 prediction is exactly the
three-view prediction. Only a validated improvement over that baseline is evidence
that the new local view helped.

Each plant patch is embedded separately with the same frozen, adapted DINOv3 backbone.
The SAM mask chooses plant-containing patches but does **not** erase RGB pixels: erasing
background pixels can also erase feeding holes. The trainable branch pools patch
evidence within plants, then combines plant evidence with the frozen model's plant
attention. Its correction is regularized. No SAM3 run or backbone fine-tuning is needed
for this experiment; the adapted backbone must be present for feature extraction.

**Important:** image-level scores do not identify lesion pixels. The red patch overlays
are *candidate local evidence*, not validated lesion segmentation or an estimate of
missing leaf area. If the overlays do not track feeding damage on visual inspection,
the next step is a small set of manually annotated damage regions, not treating these
maps as ground truth.

## Prerequisites

Use the same environment/dependencies as the three-view experiment. The machine needs:

- `outputs/dataset_manifests/{finetune,validation,test,pretrain}.csv`;
- the cached three-view features, processed images, and SAM masks;
- `outputs/dinov3_hierarchical_three_view_gold_only/best_mae.pt`;
- `outputs/dinov3_routed_source_adaptation/adapted_backbone` for *preparation* only.

All training/validation/test rows and plot-group isolation are checked against the
base checkpoint. Gold training is 325 images, validation is 73, and the 72-image test
split remains untouched during training. If the selected base checkpoint differs,
change only `base_checkpoint` in `config.toml` and run a fresh experiment directory.

## Run on the training machine

From the repository root:

```bash
python -m experiments.dinov3_plant_damage_mil.prepare_features \
  --config experiments/dinov3_plant_damage_mil/config.toml \
  --splits finetune validation
```

This is resumable: rerunning skips valid patch records. Per-image errors and tracebacks
go to `outputs/dinov3_plant_damage_mil_gold_only/patch_feature_failures.jsonl`.

```bash
python -m experiments.dinov3_plant_damage_mil.train \
  --config experiments/dinov3_plant_damage_mil/config.toml
```

Inspect `summary.json`, `best_mae_evaluation/summary.json`, `predictions.csv`,
`patch_evidence.csv`, `patch_evidence_inspection.png`, `regression.png`,
`worst_error_examples.png`, and `training_history.png`. The `paired_comparison` block
uses **the same validation images and frozen baseline checkpoint**; positive
`mae_improvement` means the new branch helped. Check target-range errors, especially
scores above 15, and inspect whether highlighted patches contain genuine feeding
damage. The saved best checkpoint may remain at epoch 0 if no improvement occurs.

Only after selecting the method using validation, prepare the reserved test features:

```bash
python -m experiments.dinov3_plant_damage_mil.prepare_features \
  --config experiments/dinov3_plant_damage_mil/config.toml \
  --splits test
```

Then evaluate once on the untouched test split:

```bash
python -m experiments.dinov3_plant_damage_mil.evaluate \
  --config experiments/dinov3_plant_damage_mil/config.toml \
  --checkpoint outputs/dinov3_plant_damage_mil_gold_only/best_mae.pt \
  --split test
```
