"""Dataset and split choices specific to this regression experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

from rapeseed_damage.reproducibility import seed_worker

from .config import Config


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: Sequence[float]) -> "TargetScaler":
        array = np.asarray(values, dtype=np.float32)
        mean, std = float(array.mean()), float(array.std())
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Training targets must have a non-zero finite standard deviation")
        return cls(mean, std)

    def transform(self, values: Sequence[float]) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse(self, values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


def image_path(config: Config, filename: str) -> Path:
    path = Path(config.data.dataset_dir) / filename
    return path if path.suffix else path.with_suffix(config.data.image_extension)


def load_scores(config: Config) -> pd.DataFrame:
    csv_path = Path(config.data.dataset_dir) / config.data.scores_file
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Scores file not found: {csv_path}. Update [data].dataset_dir in the config."
        )
    table = pd.read_csv(csv_path)
    required = {config.data.filename_column, config.data.target_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Scores CSV is missing column(s): {', '.join(sorted(missing))}")
    if table.empty:
        raise ValueError("Scores CSV contains no rows")
    table = table.copy()
    if table[config.data.filename_column].isna().any():
        raise ValueError("Filename column contains missing values")
    table[config.data.filename_column] = table[config.data.filename_column].astype(str)
    table[config.data.target_column] = pd.to_numeric(
        table[config.data.target_column], errors="raise"
    )
    if not np.isfinite(table[config.data.target_column].to_numpy(dtype=float)).all():
        raise ValueError("Target column contains missing or non-finite values")
    if config.data.verify_images:
        missing_images = [
            str(image_path(config, filename))
            for filename in table[config.data.filename_column]
            if not image_path(config, filename).is_file()
        ]
        if missing_images:
            preview = "\n".join(missing_images[:5])
            raise FileNotFoundError(
                f"{len(missing_images)} image(s) are missing. First paths:\n{preview}"
            )
    return table


def split_scores(
    table: pd.DataFrame,
    config: Config,
    training_filenames: Sequence[str] | None = None,
    validation_filenames: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the notebook split, or restore exact ordered checkpoint manifests."""
    if training_filenames is None or validation_filenames is None:
        train, validation = train_test_split(
            table,
            test_size=config.data.validation_fraction,
            random_state=config.data.split_seed,
        )
    else:
        names = table[config.data.filename_column].astype(str)
        if names.duplicated().any():
            raise ValueError("Checkpoint manifest restoration requires unique filenames")
        indexed = table.copy()
        indexed.index = names
        train_names = list(map(str, training_filenames))
        validation_names = list(map(str, validation_filenames))
        requested = train_names + validation_names
        unknown = set(requested) - set(names)
        if unknown:
            raise ValueError(
                "Checkpoint images are absent from the score table: "
                + ", ".join(sorted(unknown)[:5])
            )
        if len(requested) != len(set(requested)):
            raise ValueError("Checkpoint train/validation manifests overlap")
        train, validation = indexed.loc[train_names], indexed.loc[validation_names]
    if train.empty or validation.empty:
        raise ValueError("Training and validation splits must both contain samples")
    return train.reset_index(drop=True), validation.reset_index(drop=True)


class PlantDataset(Dataset):
    def __init__(self, table, normalized_targets, processor, config: Config):
        if len(table) != len(normalized_targets):
            raise ValueError("Table and target lengths differ")
        self.table = table.reset_index(drop=True)
        self.targets = np.asarray(normalized_targets, dtype=np.float32)
        self.processor = processor
        self.config = config

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, object]:
        filename = str(self.table.iloc[index][self.config.data.filename_column])
        path = image_path(self.config, filename)
        with Image.open(path) as source:
            image = source.convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "filename": filename,
            "image_path": str(path),
        }


class EpochSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, seed: int):
        self.dataset, self.seed, self.epoch = dataset, seed, 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())


def make_loaders(train, validation, scaler, processor, config: Config):
    train_dataset = PlantDataset(
        train,
        scaler.transform(train[config.data.target_column]),
        processor,
        config,
    )
    validation_dataset = PlantDataset(
        validation,
        scaler.transform(validation[config.data.target_column]),
        processor,
        config,
    )
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.runtime.pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": config.training.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=EpochSampler(train_dataset, config.training.seed),
        generator=torch.Generator().manual_seed(config.training.seed),
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.training.seed + 1),
        **common,
    )
    return train_loader, validation_loader
