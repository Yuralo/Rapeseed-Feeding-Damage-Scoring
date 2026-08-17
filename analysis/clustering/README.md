# Representation clustering comparison

This is an analysis workflow, not a training experiment. It extracts the same `CLS + mean patch`
features used by the regressors, optionally passes them through the learned regression head, and
clusters the resulting representations for the same ordered images.

Install the repository, the DINO experiment dependencies, and HDBSCAN:

```bash
python -m pip install -e .
python -m pip install -r experiments/dinov3_regression/requirements.txt
python -m pip install -r analysis/clustering/requirements.txt
```

Edit checkpoint paths in `analysis/clustering/config.toml`, then run from the repository root:

```bash
python -m analysis.clustering.compare \
  --config analysis/clustering/config.toml
```

For a quick smoke test, set `analysis.limit` to 50. Set it back to `0` for the full comparison.

Outputs are grouped by representation under `outputs/representation_clustering/`:

- normalized embeddings (`embeddings.npz`)
- per-image cluster assignments and PCA coordinates (`assignments.csv`)
- HDBSCAN and damage-score PCA plots
- nearest-neighbour, noise, and silhouette metrics
- a combined `comparison.csv`
- pairwise adjusted-Rand cluster agreement (`cluster_agreement.csv`)

The default comparison separates preprocessing effects from learned representation effects:

1. pretrained backbone on raw images;
2. trained baseline regression-head features on raw images;
3. pretrained backbone on grid crops;
4. trained grid-model regression-head features on grid crops.

The existing head-only checkpoints did not update DINO's backbone. Therefore, clustering their
backbone features would reproduce the corresponding pretrained-backbone result; use `feature =
"head"` to inspect what the learned regression head changed. When later checkpoints unfreeze DINO
blocks or use LoRA, add another representation with `feature = "backbone"`.

For example, a later two-block-unfrozen experiment would add:

```toml
[[representation]]
name = "grid_unfreeze2_backbone"
experiment_package = "experiments.dinov3_grid_unfreeze2"
experiment_config = "experiments/dinov3_grid_unfreeze2/config.toml"
checkpoint = "outputs/dinov3_grid_unfreeze2/best.pt"
preprocessing = "grid"
feature = "backbone"
```

Cluster metrics are descriptive. In particular, the learned head saw the regression labels during
training, so its nearest-neighbour target similarity is not an unbiased estimate of generalization.
Use the experiment's held-out evaluation metrics for that conclusion.

In `clustering.ipynb`, the whole comparison can be launched with:

```python
from analysis.clustering.compare import run

results = run("analysis/clustering/config.toml")
results
```
