"""Three synchronized SAM representations and deterministic experiment loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import random, uniform
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

from rapeseed_damage.reproducibility import seed_worker

from .config import Config
from .preprocessing import load_or_create_grid_crop, log_grid_failure
from .segmentation import load_cached_mask, log_sam_failure, make_masked_image


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    std: float
    enabled: bool = True
    training_mean: float | None = None

    @classmethod
    def fit(cls, values: Sequence[float]) -> "TargetScaler":
        array = np.asarray(values, dtype=np.float32)
        mean, std = float(array.mean()), float(array.std())
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Training targets must have a non-zero finite standard deviation")
        return cls(mean, std, enabled=True, training_mean=mean)

    def transform(self, values: Sequence[float]) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse(self, values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean

    @property
    def baseline_mean(self) -> float:
        return self.mean if self.training_mean is None else self.training_mean


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


def augment_pair(
    image: Image.Image,
    mask: Image.Image,
    config: Config,
) -> tuple[Image.Image, Image.Image]:
    settings = config.augmentation
    if not settings.enabled:
        return image, mask
    if random() < settings.horizontal_flip_probability:
        mirrored_image, mirrored_mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        image.close()
        mask.close()
        image, mask = mirrored_image, mirrored_mask
    if random() < settings.vertical_flip_probability:
        flipped_image, flipped_mask = ImageOps.flip(image), ImageOps.flip(mask)
        image.close()
        mask.close()
        image, mask = flipped_image, flipped_mask
    strength = settings.color_jitter_strength
    if strength:
        for enhancer in (
            ImageEnhance.Brightness,
            ImageEnhance.Contrast,
            ImageEnhance.Color,
        ):
            augmented = enhancer(image).enhance(uniform(1 - strength, 1 + strength))
            image.close()
            image = augmented
    return image, mask


def _mask_tensor(mask: Image.Image, size: int) -> torch.Tensor:
    resized = mask.convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


class PlantSamFusionDataset(Dataset):
    def __init__(self, table, normalized_targets, processor, config: Config, *, training: bool):
        if len(table) != len(normalized_targets):
            raise ValueError("Table and target lengths differ")
        self.table = table.reset_index(drop=True)
        self.targets = np.asarray(normalized_targets, dtype=np.float32)
        self.processor = processor
        self.config = config
        self.training = training

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, object]:
        filename = str(self.table.iloc[index][self.config.data.filename_column])
        source = image_path(self.config, filename)
        run_dir = Path(self.config.output.run_dir)
        grid_failure_log = run_dir / self.config.output.grid_failure_log
        sam_failure_log = run_dir / self.config.output.sam_failure_log
        image = mask = masked_image = None
        try:
            image, processed, _ = load_or_create_grid_crop(
                source,
                self.config.data.grid_cache_dir,
                size=self.config.data.grid_crop_size,
                inner_margin_fraction=self.config.data.grid_inner_margin_fraction,
            )
        except Exception as error:
            log_grid_failure(
                grid_failure_log,
                error=error,
                image_path=source,
                filename=filename,
                dataset_index=index,
            )
            raise RuntimeError(
                f"Grid crop failed for {filename!r}. Details: {grid_failure_log}"
            ) from error
        try:
            mask, mask_path, metadata = load_cached_mask(source, self.config)
            if self.training:
                image, mask = augment_pair(image, mask, self.config)
            masked_image = make_masked_image(
                image,
                mask,
                background_value=self.config.segmentation.background_value,
            )
            processed_pair = self.processor(
                images=[image, masked_image],
                return_tensors="pt",
            )["pixel_values"]
            mask_tensor = _mask_tensor(mask, self.config.model.mask_input_size)
            return {
                "original_pixel_values": processed_pair[0],
                "masked_pixel_values": processed_pair[1],
                "mask_values": mask_tensor,
                "mask_foreground_fraction": torch.tensor(
                    float(metadata["quality"]["foreground_fraction"]),
                    dtype=torch.float32,
                ),
                "target": torch.tensor(self.targets[index], dtype=torch.float32),
                "filename": filename,
                "source_image_path": str(source),
                "processed_image_path": str(processed),
                "mask_path": str(mask_path),
            }
        except Exception as error:
            log_sam_failure(
                sam_failure_log,
                error=error,
                image_path=source,
                filename=filename,
                dataset_index=index,
            )
            raise RuntimeError(
                f"SAM input preparation failed for {filename!r}. "
                f"Run prepare_sam_cache and inspect_sam_masks. Details: {sam_failure_log}"
            ) from error
        finally:
            for opened in (image, mask, masked_image):
                if opened is not None:
                    opened.close()


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
    train_dataset = PlantSamFusionDataset(
        train,
        scaler.transform(train[config.data.target_column]),
        processor,
        config,
        training=True,
    )
    validation_dataset = PlantSamFusionDataset(
        validation,
        scaler.transform(validation[config.data.target_column]),
        processor,
        config,
        training=False,
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
