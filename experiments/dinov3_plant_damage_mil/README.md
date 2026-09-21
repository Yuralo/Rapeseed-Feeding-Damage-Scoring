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

## External-cohort (OOD) probe

The scored weak-label manifest includes WG insects at BBCH10 (closer shift) and DSV
Asendorf at BBCH11 (source **and** growth-stage shift). They are outside the gold
training cohort. Run a small target-blind sample from each first:

The patch-preparation step expects the *three-view* feature records for those images.
If it reports missing base records, build them first using the original package's
`prepare_features --config experiments/dinov3_hierarchical_three_view_mil/config_weak_then_gold.toml --split pretrain`.

```bash
python -m experiments.dinov3_plant_damage_mil.prepare_features \
  --config experiments/dinov3_plant_damage_mil/config.toml \
  --splits pretrain \
  --cohorts wg_insects_t1_bbch10 dsv_asendorf_t1_bbch11 \
  --limit-per-cohort 100

python -m experiments.dinov3_plant_damage_mil.evaluate_ood \
  --config experiments/dinov3_plant_damage_mil/config.toml \
  --checkpoint outputs/dinov3_plant_damage_mil_gold_only/best_mae.pt \
  --limit-per-cohort 100
```

The output is `outputs/dinov3_plant_damage_mil_gold_only/ood_weak_probe_100/` with a
separate report, predictions, and plots for each cohort. Re-run both commands without
`--limit-per-cohort` for the full scored OOD cohorts. The feature-preparation command
is resumable and only processes missing patch records. The gold test split is never
read by this OOD probe.

**Interpret carefully:** these external scores are single-rater/weak labels. Their
MAE is *agreement with that rater*, not directly comparable to gold validation MAE.
The two external cohorts also differ in collection conditions and, for DSV, growth
stage. Compare frozen baseline vs local branch **within each cohort** and inspect
predictions, bias, range and failure images. A reliable external accuracy estimate
will ultimately require a small newly double-scored gold set from each source.
The adapted DINOv3 backbone may already have seen these source cohorts *unlabeled*,
so this is an OOD test for the gold-trained scorer, not necessarily an entirely unseen
domain for the complete pipeline.
