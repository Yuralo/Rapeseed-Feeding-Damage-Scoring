# Rapeseed Feeding-Damage Scoring

An experiment-oriented repository for rapeseed feeding-damage research. The working prototype is
preserved in `index.ipynb`; runnable experiments live in independent folders and are free to use
different datasets, architectures, objectives, and training loops.

## Structure

```text
experiments/
  dinov3_regression/          # one complete, reproducible experiment
    config.toml
    config.py
    data.py
    model.py
    checkpoint.py
    metrics.py
    reporting.py
    setup.py
    train.py
    evaluate.py
  dinov3_grid_unfreeze2/      # cached grid crops + final-two-block fine-tuning
  dinov3_grid_patch_attention/ # CLS + mean patches + learned patch attention
  dinov3_grid_lora/           # rank-8 q/v LoRA across all DINOv3 blocks
  dinov3_grid_lora_patch_attention/ # all-block LoRA + gated patch pooling
  dinov3_grid_lora_patch_attention_sam_fusion/ # original + masked + mask fusion

src/rapeseed_damage/          # deliberately small shared toolbox
  artifacts.py                # JSON, environment, and Git metadata
  checkpointing.py            # atomic checkpoint file I/O only
  reproducibility.py          # seeds and device selection
  grid.py                     # grid/quadrat extraction
  segmentation.py             # optional SAM3 helper

index.ipynb                   # retained for interactive exploration
```

There is no universal model, dataset, trainer, objective, or evaluation interface. The shared code
only covers mechanics that do not constrain an experiment.

## Setup

Python 3.11–3.13 is recommended because PyTorch wheels may lag behind the newest Python release.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -r experiments/dinov3_regression/requirements.txt
```

The dataset is not part of this repository. Set `data.dataset_dir` in
`experiments/dinov3_regression/config.toml` on the training machine. The Hugging Face models must
either be downloadable on the first run or already be present in the local cache.

Experiment dependencies live beside the experiment instead of being imposed on the whole project.
Install `.[grid]`, `.[segmentation]`, or `.[viewer]` only when using those shared utilities.

## Run the DINOv3 experiment

Run experiment modules from the repository root; experiment folders are intentionally not installed
as part of the shared package.

Train from scratch:

```bash
python -m experiments.dinov3_regression.train \
  --config experiments/dinov3_regression/config.toml \
  --from-scratch
```

Resume after changing `training.epochs` to the new total:

```bash
python -m experiments.dinov3_regression.train \
  --config experiments/dinov3_regression/config.toml \
  --resume outputs/dinov3_baseline/last.pt
```

Evaluate a checkpoint:

```bash
python -m experiments.dinov3_regression.evaluate \
  --config experiments/dinov3_regression/config.toml \
  --checkpoint outputs/dinov3_baseline/best.pt
```

The partial fine-tuning experiment, including its crop-cache preparation command and RTX 3090
starting settings, is documented in `experiments/dinov3_grid_unfreeze2/README.md`.

The experiment saves checkpoints, the resolved config, metrics, predictions, plots, package and
hardware versions, and the current Git commit plus dirty state. Notebook-era checkpoints remain
loadable when their architecture matches the supplied experiment config. New checkpoints embed
the resolved config and environment/Git metadata so they remain attributable when copied elsewhere.

## Start a substantially different experiment

Copy the existing experiment directory as a starting point:

```bash
cp -R experiments/dinov3_regression experiments/my_new_experiment
```

Then rename the package and freely replace `data.py`, `model.py`, `train.py`, `metrics.py`, or any
other experiment file. If the new objective does not need target normalization, regression plots,
or the current checkpoint contents, delete them instead of extending a framework abstraction.

Use configuration files for actual run parameters and Git commits for code history. Branches are
useful for risky or long-running architecture changes, but a branch is not required for every
hyperparameter run. Each run records the commit automatically; a dirty run is explicitly marked in
`environment.json`.

Seeds and deterministic PyTorch settings remove the usual within-environment randomness. Exact
bit-for-bit values can still differ across PyTorch/CUDA versions or hardware, which is why the
environment is recorded with every run.
