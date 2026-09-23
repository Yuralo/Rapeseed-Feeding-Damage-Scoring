# Proposed experiment: consensus-safe rank transfer

**Status:** runnable experiment package. No new model has been trained in this checkout, and no
gain is claimed. The original images, feature caches, and base weight file are on the training
machine.

## Run on the training machine

Use the same Python environment as the hierarchical three-view experiment, with PyTorch, NumPy,
pandas, scikit-learn, Pillow, and matplotlib installed. Run from the repository root:

```bash
python -m pip install -e '.[grid,segmentation]' \
  -r experiments/consensus_rank_transfer/requirements.txt
```

If you already use the environment from the earlier hierarchical or counterfactual runs, this
install step can be skipped. The original feature preparation also needs its existing SAM3 and
DINOv3 dependencies.

First check the paths and regenerate the manifest-only audit:

```bash
python -m experiments.consensus_rank_transfer.preflight
python -m experiments.consensus_rank_transfer.audit
```

`ready_for_prepare` must be true. If the hierarchical feature records for the pretrain, gold
finetune, validation, or test images are missing, build them with the original package. This
operation is resumable and verifies existing cache identities:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m experiments.dinov3_hierarchical_three_view_mil.prepare_features \
  --config experiments/dinov3_hierarchical_three_view_mil/config_gold_only.toml
```

Extract the **frozen** base prediction and pooled image representation once for all four splits.
This step validates the gold checkpoint, gold manifest identity, cached feature identity, and
plot separation. It writes `frozen/{pretrain,finetune,validation,test}.npz` under the new run
directory:

```bash
python -m experiments.consensus_rank_transfer.prepare
```

Choose all weights and training epochs using only grouped folds of the 325 gold training
images. A fold also excludes weak images from its held-out gold plots:

```bash
python -m experiments.consensus_rank_transfer.cv
```

The frozen base was originally trained on all 325 gold images. Thus this cross-validation is a
**head-selection heuristic**, not an unbiased performance estimate. Its purpose is to keep model
choices away from the historical validation/test labels. The subsequent validation comparison
is the first direct check of this new head, although that split was seen in earlier project work.

Fit all four matched arms for the five declared seeds, then create the locked validation
comparison and initial figures:

```bash
python -m experiments.consensus_rank_transfer.fit
python -m experiments.consensus_rank_transfer.compare --split validation
python -m experiments.consensus_rank_transfer.visualize
```

Review `validation_comparison.json`, especially the rank-versus-base and rank-versus-gold-only
plot-bootstrap intervals, practical 0.25-point gate, low-score degradation gate, seed spread,
and QA sheets. The command compares exactly matched filenames, plots, targets, and frozen-base
predictions across all arms. After the validation comparison is fixed, run the retrospective gold
test and all feature-usable images in both single-rater OOD cohorts, then regenerate all figures:

```bash
python -m experiments.consensus_rank_transfer.evaluate --split test
python -m experiments.consensus_rank_transfer.compare --split test
python -m experiments.consensus_rank_transfer.evaluate --split ood
python -m experiments.consensus_rank_transfer.compare --split ood
python -m experiments.consensus_rank_transfer.sensitivity
python -m experiments.consensus_rank_transfer.visualize
```

The main files are `cv_selection.json`, `validation_comparison.json`, `test_comparison.json`,
`ood_<cohort>_comparison.json`, each arm/seed's `metrics.json` and `predictions.csv`, and
`sensitivity_margin10/summary.json` and `figures/`. The detailed visual audit is in
`figures/index.html`; `figures/summary.json` lists every generated figure and missing optional
artifact. The visualizer reads saved result CSV/JSON files and runs without model checkpoints,
frozen feature arrays, or a GPU. To visualize a completed run in another location:

```bash
python -m experiments.consensus_rank_transfer.visualize --run-dir /path/to/completed/consensus_rank_transfer
```

If the saved `source_path` values point to another machine, add one or more local image trees:

```bash
python -m experiments.consensus_rank_transfer.visualize \
  --run-dir /path/to/completed/consensus_rank_transfer \
  --image-root /path/to/photographs
```

The report contains an all-arm outcome dashboard; CV search and fold trajectories; full-fit
training/loss curves; conservative pair qualification, sampled-pair margins, image degree,
rater disagreement and violation rates; all-arm calibration, signed error, score distributions,
severity bands with counts, error CDFs, paired plot changes and bootstrap intervals for each
available split; predeclared validation gates; per-seed plot MAE and correction spread;
cross-cohort OOD shift; and the
five-versus-ten-point pair-margin sensitivity. Per-split CSV files rank individual images,
unstable predictions, and plot-level gains/regressions. Contact sheets show the largest gains,
regressions, rank errors, and low-score regressions when photographs are accessible. Duplicate
image basenames in an `--image-root` are omitted rather than silently matched to the wrong photo.

The metric panels use **mean absolute error over seeds**, matching the comparison JSON; the
calibration panels show **mean predictions**. These are different operations and need not have
the same error. The visualizer checks image counts, plot counts, and plot-weighted MAE against the
saved comparison before rendering, and fails if they disagree. Missing optional CV, history,
sampling, sensitivity, or photograph sources are named in `figures/summary.json`. OOD agreement is reported
against the original single-rater values; it is not independent gold accuracy. The existing
gold test has been seen during prior research, so its result is also retrospective.

## Why this experiment

The synthetic-bite pretext was learned, but the validation gain over the random control was only
0.024 plot-weighted MAE (95% plot-bootstrap interval for the reduction: -0.062 to 0.115), and its
historical gold test was 0.046 worse than the frozen base. The low-score test band worsened by
0.397 MAE. More visual pretraining is therefore a weak bet without evidence that the induced
representation is aligned with how the raters score real plants.

The next unused information source is **which real images are clearly more damaged**, even when
raters disagree on the exact score. In the target GG cohort, 381 dual-rater images remain in the
weak-training manifest. Every one has a JLU/GAU score gap above the five-point gold threshold;
the median gap is 14.5. Replacing their two scores by an exact midpoint was already tried in the
paired-mean retrain and did not beat the frozen base on the historical test (2.920 versus 2.878
image MAE). The proposal is to extract only ordering relations that *both* scores support and
let the 325 gold training images determine the absolute score scale.

This is motivated by [RankUp](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c26a8494fe31695db965ae8b7244b7c1-Abstract-Conference.html),
which found that an auxiliary ranking task can help low-label regression, and by
[partial-label regression](https://ojs.aaai.org/index.php/AAAI/article/view/25871), which treats
imprecise real-valued labels as uncertain instead of collapsing them to a single value. Work on
[image aesthetics](https://openaccess.thecvf.com/content_ICCV_2019/html/Lee_Image_Aesthetic_Assessment_Based_on_Pairwise_Comparison__A_Unified_ICCV_2019_paper.html)
shows that comparisons can support score regression in another subjective vision task. None of
these papers establishes that the combination will improve rapeseed scoring; that is the test here.

## Feasibility already checked

Run from the repository root:

```bash
python3 -m experiments.consensus_rank_transfer.audit
```

The command writes `outputs/consensus_rank_transfer/feasibility.json`. It uses only the checked-in
manifests. Current results:

| Check | Result |
| --- | ---: |
| In-domain discordant dual-rater images | 381 in 194 plot groups |
| Median JLU/GAU gap | 14.5 score points |
| Cross-plot pairs whose entire score intervals do not overlap | 14,397 |
| Same, plus 5-point separation | 5,640 pairs; 322 images participate |
| Same, plus 10-point separation | 1,703 pairs; 197 images participate |
| Weak plot overlap with validation/test | 0 / 0 |
| Weak plot overlap with gold training | 130 plot groups, reported and allowed |

The unfiltered raters agree on only 59.6% of decided cross-plot orderings for these discordant
images. This is why **all pairwise orders are unsafe**. The interval-dominance rule is conservative:
image A may rank above B only when `min(JLU_A, GAU_A) > max(JLU_B, GAU_B) + margin`. The counts
measure candidate constraints, not their truth against an adjudicated latent score. Training
samples a capped subset so that prolific images and plot groups do not dominate. With the
declared 40-pair-per-image cap, severity-band cycling, and seed 42, the 5-point rule yields 2,510
training pairs involving 244 images before any feature failures or cross-validation exclusions.

## Model and training comparison

Keep the strongest existing frozen DINOv3 three-view feature extractor and frozen gold-only
prediction `b(x)`. Fit a small, zero-initialized residual head `r(x)` from the same cached
features, with prediction `b(x) + r(x)`. Bound the residual to five score points and regularize
its squared size; at initialization, the prediction is exactly the known baseline.
Do not re-train the backbone or use synthetic pairs in this study. The already adapted backbone
was trained from 7,266 usable images out of the 7,569-image adaptation manifest, so this arm
reuses the previous large-image-pool work without repeating the pretext run. The 1,805
single-rater weak images are from WG/DSV and remain separate external-cohort diagnostics.

Use the 325 gold images with plot-balanced robust regression to their existing consensus target.
For a qualifying weak pair `(higher, lower)`, add a squared hinge only when the predicted higher
score fails to exceed the lower score by one point. The conservative primary pair definition is
five-point interval separation. A ten-point definition is a predeclared sensitivity analysis.
The ranking term supplies order; the gold term supplies units and calibration. Never regress a
discordant image to its midpoint in the proposed arm.

Compare matched five-seed arms with the same architecture, features, gold loss, and declared
training-only selection procedure:

1. **Frozen base**: unchanged existing model.
2. **Gold-only residual**: measures the effect of extra capacity and optimization.
3. **Midpoint weak regression**: reproduces the obvious noisy-label alternative with this head.
4. **Consensus-safe ranking**: primary proposed arm.
5. **Shuffled-order control**: same number and score distribution of pairs, randomly reversed;
   tests whether any gain comes from real order information.

Use equal expected contribution per gold plot and per weak plot. Cap sampled pairs per image,
stratify by severity, and sample only across different plots. Set the rank-loss weight and head
regularization from a small, declared grid using **grouped cross-validation of the 325 gold
training images only**. When a gold plot is in a cross-validation holdout, also remove any weak
images from that plot from training. Never tune on the 73-image validation, 72-image historical
test, or OOD cohorts. Save fold assignments, sampled pair IDs, seed settings, and feature-failure
counts.

## Evaluation and decision

After the training-only choices are frozen, evaluate all five seeds on the 73 gold validation
images and then the 72 historical gold test images. Both sets have been inspected by previous
research, so label them **retrospective held-outs**, not fresh confirmatory evidence. The primary
metric is plot-weighted MAE; compare paired plot errors and their 95% cluster-bootstrap interval.
The proposed arm should beat both the frozen base and gold-only residual by at least 0.25 MAE,
with the paired interval for improvement above zero. This is a practical target, not a power
claim. Also require low-score (0–2.5) MAE to worsen by no more than 0.25, report >15 MAE, bias,
score calibration, and residual-versus-needed-correction correlation. Report all seeds, not only
the best checkpoint.

Visual outputs required for review: training/CV curves; rater-gap and accepted-pair coverage;
predicted-versus-gold and residual plots; plot-level paired error changes with bootstrap interval;
severity-band errors; and image contact sheets for the largest improvements, regressions, and
pairwise ranking violations. The same images and plots must appear in every arm's comparison.

Evaluate WG and DSV separately with matched samples and cohort-stratified plots. Their labels
are single-rater weak scores, so this is a **distribution-shift/score-agreement diagnostic**, not
OOD accuracy. Do not use their labels to choose the model. A claim of generalization needs a new,
blinded, double-scored and adjudicated OOD set with plot-level separation. The old GG test is
also historically visible; a new GG holdout would be needed for a genuinely prospective gain.
The frozen backbone was adapted using images from WG/DSV, so this OOD check probes the scoring
head under cohort shift rather than a wholly unseen visual domain.

## Why this is the next bet, and its failure mode

It acts on the strongest signal the last experiments exposed: label disagreement and calibration,
while retaining the visual encoder that already works reasonably well. It uses real plant images
and actual human assessments, but asks only for high-confidence order from noisy scores. The
largest risk is that interval dominance still encodes rater biases, or that the qualified pairs
overrepresent extreme plants. The shuffled-order arm, midpoint arm, accepted-pair coverage plot,
low-score safety gate, and independent gold evaluation are designed to reveal that failure.

If the qualified-order arm fails, the next action should be new blinded adjudications with visual
severity reference images, not another unverified pseudo-label scheme. In a related plant-severity
setting, [standard area diagrams improved rater concordance and interrater reliability](https://pubmed.ncbi.nlm.nih.gov/38277651/);
that supports improving label quality, though the published disease task differs from feeding
damage.
