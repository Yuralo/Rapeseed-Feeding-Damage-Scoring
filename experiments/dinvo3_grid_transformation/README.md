# DINOv3 regression experiment

This folder is the Python version of the regression portion of `index.ipynb`. It owns every choice
that is specific to the experiment: CSV schema, split, normalization, image processing, model,
MSE objective, optimizer, checkpoint contents, metrics, and plots.

Install its dependencies after installing the repository package:

```bash
python -m pip install -r experiments/dinov3_regression/requirements.txt
```

Train:

```bash
python -m experiments.dinov3_regression.train \
  --config experiments/dinov3_regression/config.toml \
  --from-scratch
```

Resume after increasing `training.epochs`:

```bash
python -m experiments.dinov3_regression.train \
  --config experiments/dinov3_regression/config.toml \
  --resume outputs/dinov3_baseline/last.pt
```

Evaluate:

```bash
python -m experiments.dinov3_regression.evaluate \
  --config experiments/dinov3_regression/config.toml \
  --checkpoint outputs/dinov3_baseline/best.pt
```

For a substantially different idea, copy this directory, rename it, and change or delete any file
you need. There is no shared trainer or required experiment interface.

Before training, validate grid detection on every image:

```bash
python -m experiments.dinvo3_grid_transformation.validate_preprocessing \
  --config experiments/dinvo3_grid_transformation/config.toml
```

Failures are written as structured JSON Lines to
`outputs/dinvo3_grid_transformation/grid_failures.jsonl`. Each record contains the filename, full
image path, dataset index, worker/process ID, exception type and message, pipeline stage, timestamp,
and complete traceback. Training still stops on a bad crop instead of silently changing the dataset.
