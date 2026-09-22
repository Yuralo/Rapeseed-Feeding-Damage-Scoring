# Proposed experiment: learn the visual effect of a bite before fitting field scores

**Status:** runnable experiment package and pre-specified research design. The full image
collection, feature caches, adapted backbone, and saved model weights live on the training machine
and are not in this checkout. No full training run or accuracy improvement is claimed here.

## Run the package

Prerequisites on the training machine:

- the existing `outputs/dataset_manifests/{adaptation,finetune,validation,test}.csv` and original
  image storage;
- the gold-only frozen hierarchical checkpoint named in the plant-damage config;
- gold `finetune` and `validation` three-view and high-resolution patch caches from the existing
  packages;
- the routed adapted DINOv3 backbone and processor named in the hierarchical config;
- the existing experiment dependencies, including PyTorch, Transformers, SAM3, OpenCV, Pillow,
  NumPy, pandas, and matplotlib.

Use the training machine that has the original data and weights. From the repository root, activate
its experiment environment and install any missing package dependencies:

```bash
python -m pip install -e '.[dev,grid,segmentation]' \
  -r experiments/dinov3_grid_lora_patch_attention_sam_fusion/requirements.txt
```

Confirm the paths in `config.toml`, the referenced plant and hierarchical configs, and the raw
paths in the manifests. This checkout contains the manifest CSVs but **no raw images or base
checkpoint or adapted backbone weight files**, so it cannot execute the GPU experiment here.
Run the inventory before starting; it counts available images, model weights, and feature caches:

```bash
python -m experiments.counterfactual_feeding_pretraining.preflight \
  --config experiments/counterfactual_feeding_pretraining/config.toml
```

If the gold three-view and high-resolution patch feature caches are missing, prepare them before
fitting. These commands reuse the existing cache locations and are resumable:

```bash
python -m experiments.dinov3_hierarchical_three_view_mil.prepare_features \
  --config experiments/dinov3_hierarchical_three_view_mil/config_gold_only.toml \
  --split finetune --split validation

python -m experiments.dinov3_plant_damage_mil.prepare_features \
  --config experiments/dinov3_plant_damage_mil/config.toml \
  --splits finetune validation
```

All commands run from the repository root. First make a small technical smoke pass, then build
the complete pair cache. The full preparation runs SAM and DINO in separate passes to fit a
24 GB GPU and writes cohort counts, failures, and fixed target-blind QA examples.

```bash
python -m experiments.counterfactual_feeding_pretraining.prepare_pairs \
  --config experiments/counterfactual_feeding_pretraining/config.toml --limit 20

python -m experiments.counterfactual_feeding_pretraining.prepare_pairs \
  --config experiments/counterfactual_feeding_pretraining/config.toml
```

Inspect `outputs/counterfactual_feeding_pretraining/preparation_summary.json`,
`pair_preparation_failures.jsonl`, and the images under `quality_examples/`. Preparation is
resumable; an existing pair is identity-checked before reuse. The full run must reach the configured
coverage threshold. The 20-image smoke run cannot be used for a final pretraining run.

Train the damage pretext and artifact control. The 10% pretext holdout is assigned by plot group,
without using scores:

```bash
python -m experiments.counterfactual_feeding_pretraining.pretrain \
  --config experiments/counterfactual_feeding_pretraining/config.toml --mode synthetic

python -m experiments.counterfactual_feeding_pretraining.pretrain \
  --config experiments/counterfactual_feeding_pretraining/config.toml --mode sham
```

Review both `pretraining/<mode>/summary.json` files. A positive bite response with near-zero sham
response is the essential sanity check before using the synthetic initialization. Fit every
predeclared seed in every arm; the validation comparison refuses incomplete arm/seed sets:

```bash
for seed in 42 43 44 45 46; do
  for arm in random synthetic sham; do
    python -m experiments.counterfactual_feeding_pretraining.fit \
      --config experiments/counterfactual_feeding_pretraining/config.toml \
      --arm "$arm" --seed "$seed"
  done
done

python -m experiments.counterfactual_feeding_pretraining.compare \
  --config experiments/counterfactual_feeding_pretraining/config.toml

python -m experiments.counterfactual_feeding_pretraining.visualize \
  --config experiments/counterfactual_feeding_pretraining/config.toml
```

The comparison saves `validation_comparison.json` and per-image paired errors. Check its plot
bootstrap interval, the frozen-base reference, severity bands, and the worst-error images in each
arm's `validation/` folder. `visualize` saves preparation coverage, held-out bite/sham response
and pretext loss, gold fitting curves, and paired gold-validation figures. The full review also
includes the 25 target-blind synthetic
contact sheets per cohort and each seed's regression, worst-error, and patch-evidence figures.
Review the contact sheets manually; no metric can certify bite realism.

After the validation decision is locked, run the matched external-cohort probe. First prepare the
three-view and plant-patch caches for the **pretrain** manifest. The first OOD command below is a
100-image-per-cohort technical probe; repeat both feature-preparation and evaluation commands
without `--limit-per-cohort` for the full report. OOD evaluation checks that all three arms and
all five seeds use exactly the same images, weak targets, and frozen-base predictions:

```bash
python -m experiments.dinov3_hierarchical_three_view_mil.prepare_features \
  --config experiments/dinov3_hierarchical_three_view_mil/config_gold_only.toml \
  --split pretrain

python -m experiments.dinov3_plant_damage_mil.prepare_features \
  --config experiments/dinov3_plant_damage_mil/config.toml \
  --splits pretrain \
  --cohorts wg_insects_t1_bbch10 dsv_asendorf_t1_bbch11 \
  --limit-per-cohort 100

python -m experiments.counterfactual_feeding_pretraining.evaluate_ood \
  --config experiments/counterfactual_feeding_pretraining/config.toml \
  --limit-per-cohort 100
```

The OOD outputs include `ood_summary.json`, `matched_sample_manifest.csv`, `seed_metrics.csv`,
`matched_predictions.csv`, `ood_diagnostic.png`, and per-cohort/per-arm/per-seed regression,
worst-error, and patch-evidence views. WG and DSV scores are single-rater or weak labels, and the
backbone and synthetic pretext may have seen these images without scores. This probes scoring-head
shift and **agreement with those raters**, not independent gold accuracy. OOD labels must not select
the arm, seed, or checkpoint. A new blinded double-scored WG/DSV set is required for a confirmatory
OOD claim.

Only after the design decision is locked, explicitly make the historical gold test report. Replace
`synthetic` with the validation-selected arm and evaluate **all five** predeclared seeds:

```bash
python -m experiments.dinov3_plant_damage_mil.prepare_features \
  --config experiments/dinov3_plant_damage_mil/config.toml --splits test

for seed in 42 43 44 45 46; do
  python -m experiments.counterfactual_feeding_pretraining.evaluate \
    --config experiments/counterfactual_feeding_pretraining/config.toml \
    --arm synthetic --seed "$seed" --split test
done

python -m experiments.counterfactual_feeding_pretraining.summarize_test \
  --config experiments/counterfactual_feeding_pretraining/config.toml \
  --arm synthetic
```

Each test command uses a saved validation-selected checkpoint and marks the result retrospective.
`summarize_test` checks matched filenames, targets, and frozen-base predictions across seeds,
then saves the five-seed distribution, plot bootstrap interval, target-band results, and a figure.
Do not choose a single best test seed.

## Decision

Pretrain the existing *local plant-patch correction branch* on counterfactual feeding damage:
take a real rapeseed image, remove a known small amount of visible leaf tissue, and teach the
branch how much damage was added. Then fit the actual JLU/GAU field scores on the same gold
training images and with the same frozen hierarchical base used by
`dinov3_plant_damage_mil_gold_only`. The primary comparison is against an **identical correction
branch with random initialization**, trained on the identical real scores. This isolates whether
the counterfactual pretraining helps.

The synthetic task provides *relative* damage supervision, not a replacement numerical field
score. The real scoring scale is learned only from human-scored training images.

## Why this is the next distinct experiment

| Repository evidence | Consequence for this design |
| --- | --- |
| The local high-resolution branch reached 2.516 validation MAE on 73 gold images, but its frozen base alone reached 2.427; 41/73 images worsened. See `outputs/dinov3_plant_damage_mil_gold_only/summary.json`. | More local capacity alone has not solved the task. Test a damage-specific pretraining signal while holding capacity fixed. |
| The later paired-mean retrain reached 2.469 validation MAE and 2.920 on 72 gold test images. Its frozen base was 2.878 on that same test set. See `outputs/dinov3_adaptive_paired_mean/gold_test/summary.json`. | The corrected branch is still a small regression risk. Initialize it with a physical task and require a paired improvement over the unchanged base. |
| Validation error correlates with target (0.51 in the gold-only local run); the 72-image test contains only 10 scores above 15, where MAE was 4.80. | Report severity bands and score slope, not only overall MAE. More aggressive high-score weighting is a separate ablation. |
| The 470-image gold tier was selected by JLU–GAU difference ≤5 points. The excluded dual-scored tier had 15.79-point mean absolute disagreement and Pearson correlation 0.168. See `outputs/dataset_score_audit/summary.json`. | Do not treat discordant scorer means as equally precise ground truth. Keep them out of the primary supervised fit. |
| Weak-only training on 1,981 usable images yielded 4.370 MAE on all 470 gold images (`outputs/dinov3_adaptive_weak_gold/run_summary.json`). | Many images with uncertain scores did not remove the accuracy gap. Use the unlabeled pool for a damage-specific task, and reserve weak-score ranking for a separate ablation. |
| The canonical inventory contains 9,443 distinct images; 7,569 are in the conservative adaptation manifest, and the routed pipeline prepared 7,266 (303 failures). See `outputs/dataset_manifests/manifest_summary.json` and `outputs/dinov3_routed_source_adaptation/preparation_summary.json`. | Use the large safe pool for label-free synthetic pretraining. Audit failures by cohort and never silently count failed images as used. |
| Three views of a plot can differ substantially: median within-plot score range is 3.5–4.0 points in the gold splits; the 90th percentile is roughly 15–17 points. | Do not force different views from one plot to have identical predictions. Split by plot, but train on each image's own score. |

The prior search was mainly over image representations and pooling: raw/grid crops, partial
fine-tuning, LoRA, SAM fusion, fixed 3×3/4×4/5×5 tiles, hierarchical MIL, weak-score training,
and general domain adaptation. This proposal changes **what the local branch learns before it
sees a score**. The existing generic self-supervised adaptation did not reliably improve
downstream regression, so this pretext task must be tied to feeding morphology. The older
156-image validation numbers and the later 73/72-image gold split are different protocols and
must never be ranked together as if they shared a holdout set.

## Research basis and limits

- [Vieira et al. (2024), *An automatic method for estimating insect defoliation*](https://www.sciencedirect.com/science/article/pii/S2214317324000192)
  construct controlled bite-shaped tissue loss from real insect damage and estimate defoliation
  across 12 species. This supports the **physical perturbation**. Their synthetic damage is used
  to create reference severities for evaluation; it does **not** establish that our proposed
  pretraining will improve rapeseed field-score prediction.
- [Li et al. (CVPR 2021), *CutPaste*](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf)
  show that learning from synthetic local defects can improve fine-grained anomaly detection.
  Soil-revealing bites are a different defect, so this is a methodological analogy.
- [Eguskiza et al. (2026), *Latent diffusion meets semi-supervised learning*](https://www.sciencedirect.com/science/article/pii/S0168169926004357)
  report improved wheat disease-severity regression from synthetic images and unlabeled data.
  Their result is for yellow rust and diffusion-generated images, not insect feeding; it motivates
  a controlled trial here rather than predicting the size of an improvement.
- [Huang et al. (NeurIPS 2024), *RankUp*](https://proceedings.nips.cc/paper_files/paper/2024/hash/c26a8494fe31695db965ae8b7244b7c1-Abstract-Conference.html)
  demonstrate that pairwise ranking is useful in semi-supervised regression, including image age
  estimation. We use only the pairwise idea with **known within-image synthetic order**. Their
  distribution alignment assumes similar labeled and unlabeled target distributions; our cohorts
  violate that assumption, so it is intentionally omitted.

This is a falsifiable hypothesis, not a guarantee of the best score. A synthetic bite can be
recognized by a compositing seam rather than by missing tissue. The controls below are designed
to detect that failure.

## Data boundary

1. Use canonical SHA-256 image identities. Never train twice on copied files in the 470-image
   training subset or other duplicate folders.
2. Gold supervision: exactly the 325 rows of `finetune.csv`. Select the epoch using the existing
   73 rows of `validation.csv`. Keep the 72 rows of `test.csv` out of feature statistics, sampling,
   model selection, threshold selection, and visual quality reviews.
3. Synthetic pretraining: start from the 7,569 canonical rows in `adaptation.csv`. The manifest
   excludes *both* GG insect timepoints because cross-timepoint plot linkage is incomplete. The
   7,266 previously routed images are the immediate usable starting pool. Re-audit and, where
   possible, repair the 303 source-specific preprocessing failures before declaring the final
   count. No val/test GG image or potentially linked T2 GG image enters this stage, even without
   a score.
4. Balance pretraining exposure by source cohort so the 1,271-image GG sulfur set cannot dominate
   the 360-image WG T2 first-half set. An image already containing natural damage can still be
   used: the paired target is *additional* induced damage on that same image.
5. Fix random seeds, manifest hashes, source routes, segmentation settings, and the selected
   model before seeing the existing 72-image test result for the new run. The historical test
   results are already visible in this repository, so this test is now a **retrospective**
   comparison, not a fresh confirmatory test. A genuinely new gold set is needed for that claim.

## Stage A: construct a biologically relevant paired task

Use the same EXIF handling and source-specific routing as
`dinov3_routed_source_adaptation`: three audited raw-source folders remain full image; the other
folders use the 1400×1400, 7.5% inset quadrat. Obtain plant masks with the existing SAM3 prompt,
but inspect its performance on **every cohort** before large-scale generation. Reject a crop if
its plant foreground is too small to hold a visible bite, if the mask contains soil/card/grid, or
if the route selected only part of the quadrat. Log the reason and cohort.

For each accepted image, select the SAM plant box with most foreground, then its most
plant-covered high-resolution patch using the existing 2×2 patch geometry, and produce three views:

- **original:** unmodified crop;
- **bite:** an irregular interior hole or edge notch placed only where the plant mask says leaf,
  exposing nearby real soil texture from the *same* image;
- **sham:** the same compositing and interpolation operations outside the leaf mask, with no leaf
  pixels removed.

The implemented generator uses irregular contours with randomized scale and rotation. Tracing
real bite contours is a future fidelity upgrade and is not part of this comparison.
Sample added loss at 2%, 5%, 10%, and 20% of visible leaf foreground, but reject targets below a
minimum detectable pixel area. Record the **realized** removed-foreground fraction, not merely the
requested fraction. Never paste a bite
over a QR card or grid wire. Keep the original and its counterfactual in the same minibatch.

Quality gate before training: a fixed, random, stratified contact sheet of at least 25 pairs per
cohort, with original/bite/sham, mask, and measured fraction. Two reviewers should be unable to
identify most bites from a hard compositing edge alone. If the mask or realism gate fails for a
cohort, fix that generator or omit that cohort with a recorded count; do not silently lower the
quality bar. Report acceptance by cohort and damage level.

## Stage B: pretrain the local evidence branch

Keep the adapted DINOv3 backbone frozen and use the *same plant-patch feature extraction* as
`dinov3_plant_damage_mil`. This makes the resulting branch loadable into that experiment's
`input_norm`, `project`, `local`, and `patch_evidence` modules. The pretraining head produces a
scalar local severity `s(p)` for each patch. Feed each synthetic patch through the same local
module using its paired original plant embedding as the reference, matching the real model's
`patch_feature - plant_feature` input. Use image-paired supervision:

```
L_delta = Huber((s(bite) - s(original)), realized_removed_fraction)
L_order = softplus(margin - (s(bite) - s(original)))
L_sham  = Huber(s(sham) - s(original), 0)
L_total = L_delta + 0.25 L_order + 0.5 L_sham
```

The numerical fractions are **pretext units**. They are never inserted into the human-score
target column. Randomly alternate interior and margin loss. Preserve original/synthetic pairs
within each data split. Use 10% of the adaptation images, grouped by plot and stratified by
cohort, solely to check pretext generalization; their field scores, if any, are hidden.

The implemented diagnostic requires positive mean bite response, sham response less than half
the bite response in magnitude, and correlation above 0.3 between realized loss and predicted
change on held-out images. The sham-pretrained arm provides an additional artifact control.
If the bite branch learns compositing artifacts, stop before real-score fitting.

## Stage C: controlled real-score fit

Load the same frozen hierarchical gold-only checkpoint used in
`dinov3_plant_damage_mil_gold_only`. Train only the correction branch on 325 gold images with the
same crop cache, optimizer, target scaler, augmentations, epoch schedule, and residual penalty as
the random-initialization control. Initialize its local modules from Stage B. Set the final
`residual_head` to zero in both arms so epoch 0 exactly reproduces the frozen base. Checkpoint by
gold validation MAE, with MSE/RMSE reported as secondary measures. Evaluate the best frozen-base,
random-branch, and pretrained-branch predictions on **identical filenames**.

| Arm | Local branch initialization | Real-score training | Purpose |
| --- | --- | --- | --- |
| A | No branch | Existing 325 gold | Strong frozen-base floor |
| B | Random | Existing 325 gold | Capacity and training control; replicate local branch |
| C | Synthetic-bite pretraining | Existing 325 gold | Primary treatment versus B |
| D (diagnostic) | Sham/compositing pretraining | Existing 325 gold | Detect generic augmentation or seam advantage |

Run five fixed seeds for B–D. Use the same seeds and minibatch order across arms. Do not choose
the best seed. Predeclare one primary comparison: **C minus B in plot-weighted gold validation
MAE**; C versus A determines whether the correction is worth deploying at all. Compute paired
bootstrap intervals by `plot_group_id`, not by image, because multiple views of a plot are
correlated. Also report per-score-bin MAE (especially >15), RMSE, calibration slope, and changes
per image/plot. There should be no degradation greater than 0.25 score points in the 0–2.5 bin
when seeking high-score gains.

After the primary decision, a separately labeled extension can add the 381 plot-isolated
discordant GG images and the 1,805 DSV/WG singly scored images as **within-cohort ranking**
supervision. Form pairs only within the same rater/source and only when score gaps exceed a
predeclared uncertainty margin; never force absolute scales across cohorts or force different
views of one plot to agree. Compare this extension against the same extension without synthetic
pretraining. Do not mix it into the first C-vs-B test.

## What would count as evidence

Proceed beyond a pilot only if all of the following hold:

1. Source-balanced pretraining covers at least 95% of eligible routed images with documented
   image/mask QA; the two sham/artifact checks pass.
2. Across five paired seeds, C improves plot-weighted validation MAE over B and the frozen base;
   the plot-bootstrap 95% interval for the paired MAE reduction excludes zero. A practical target
   is at least 0.25 score points, but this is a target, not a promised gain.
3. High-score calibration improves without sacrificing the low-score band or making OOD
   predictions visibly unstable.
4. Once all design choices are fixed, make **one** historical 72-image test report. Label it
   retrospective because past experiments, including this design process, examined that split.

For a confirmatory result, obtain a new plot-disjoint, double-scored gold set from WG and DSV.
Set its size from a power calculation using paired plot-level error variation and the minimum
meaningful MAE reduction; stratify collection across the score range. Freeze the algorithm and
thresholds first, and evaluate once. This is also the only sound way to claim an OOD accuracy
improvement: the existing WG/DSV scores are single-rater and have different score calibration.

## Implementation sequence on the training machine

1. Recreate the published manifest counts and hashes. Confirm the 7,569/7,266 adaptation numbers
   and 325/73/72 gold split. Refuse to run on changed or overlapping manifests.
2. Build and review the stratified synthetic contact sheets; record mask and crop failure rates.
3. Cache original, bite, and sham DINOv3 patch features with source-image SHA, generator version,
   realized fraction, and random seed in every record. Make cache writes resumable and atomic.
4. Train Stage B with the frozen backbone; inspect the held-out synthetic diagnostics.
5. Train arms A–D and save per-image predictions, seed summaries, plot bootstrap results, and
   worst-error images. Select on the 73-image validation split only.
6. Lock the decision, then make the one retrospective test report and arrange the new blinded
   external gold set for a confirmatory result.

The proposed method is deliberately more expensive in feature preparation than another small
MIL head. Its value is that nearly every safe unscored image can teach the model what *additional
missing leaf tissue* looks like without requiring an invented field score.
