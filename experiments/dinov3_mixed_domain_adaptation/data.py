"""Prepared-manifest loading and deterministic paired-view data loaders."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from random import Random

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from rapeseed_damage.reproducibility import seed_worker

from .augmentations import paired_views
from .config import Config
from .preprocessing import (
    PREPARED_SCHEMA_VERSION,
    RAW_TILED_MODE,
    choose_adaptation_tile,
    deserialize_tile_candidates,
    load_prepared_image,
)


def load_prepared_manifest(config: Config) -> list[dict[str, str]]:
    path = Path(config.data.prepared_manifest)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared manifest not found: {path}. Run prepare_inputs and inspect_preprocessing "
            "before training."
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "file_name",
        "cohort_id",
        "source_path",
        "width",
        "height",
        "input_mode",
        "tile_candidates",
        "prepared_schema_version",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    if not rows:
        raise ValueError(f"Prepared manifest is empty: {path}")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prepared manifest contains duplicate image IDs")
    allowed_modes = {RAW_TILED_MODE}
    bad_modes = sorted({row["input_mode"] for row in rows} - allowed_modes)
    if bad_modes:
        raise ValueError(f"Prepared manifest contains unsupported modes: {bad_modes}")
    bad_schema = sorted(
        {
            row["prepared_schema_version"]
            for row in rows
            if int(row["prepared_schema_version"]) != PREPARED_SCHEMA_VERSION
        }
    )
    if bad_schema:
        raise ValueError(
            "Prepared manifest schema mismatch; rebuild it with prepare_inputs. "
            f"Found versions: {bad_schema}"
        )
    expected_scales = set(config.tiles.grid_sizes)
    for row in rows:
        candidates = deserialize_tile_candidates(row["tile_candidates"])
        scales = {candidate.grid_size for candidate in candidates}
        if scales != expected_scales:
            raise ValueError(
                f"Prepared tile scales for {row['image_id']} are {sorted(scales)}, "
                f"but the config requests {sorted(expected_scales)}. Re-run prepare_inputs."
            )
    source_path = Path(config.data.manifest)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source adaptation manifest is missing: {source_path}")
    with source_path.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    source_ids = {row[config.data.id_column] for row in source_rows}
    prepared_ids = set(ids)
    if not prepared_ids <= source_ids:
        missing_ids = sorted(source_ids - prepared_ids)
        extra_ids = sorted(prepared_ids - source_ids)
        raise ValueError(
            "Prepared manifest contains images outside the adaptation manifest. "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}. "
            "Re-run prepare_inputs without --limit."
        )
    missing_ids = sorted(source_ids - prepared_ids)
    missing_fraction = len(missing_ids) / len(source_ids)
    if missing_fraction > config.data.maximum_excluded_fraction:
        raise ValueError(
            "Prepared manifest excludes too many adaptation images: "
            f"missing={len(missing_ids)}/{len(source_ids)} ({missing_fraction:.2%}), "
            f"limit={config.data.maximum_excluded_fraction:.2%}. Inspect the preparation log."
        )
    absent = [row["source_path"] for row in rows if not Path(row["source_path"]).is_file()]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} raw adaptation image(s) are missing. First paths:\n"
            + "\n".join(absent[:5])
        )
    return rows


def split_records(
    rows: Sequence[dict[str, str]],
    config: Config,
    *,
    training_ids: Sequence[str] | None = None,
    validation_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    indexed = {row["image_id"]: row for row in rows}
    if training_ids is not None or validation_ids is not None:
        if training_ids is None or validation_ids is None:
            raise ValueError("Checkpoint must contain both training and validation IDs")
        requested = list(training_ids) + list(validation_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("Checkpoint training and validation IDs overlap")
        unknown = sorted(set(requested) - set(indexed))
        if unknown:
            raise ValueError("Checkpoint references unknown image IDs: " + ", ".join(unknown[:5]))
        return (
            [indexed[image_id] for image_id in training_ids],
            [indexed[image_id] for image_id in validation_ids],
        )

    by_cohort: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cohort[row["cohort_id"]].append(row)
    training, validation = [], []
    for cohort, cohort_rows in sorted(by_cohort.items()):
        shuffled = list(cohort_rows)
        Random(f"{config.data.split_seed}:{cohort}").shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * config.data.validation_fraction))
        if validation_count >= len(shuffled):
            raise ValueError(f"Cohort {cohort!r} is too small for a train/validation split")
        validation.extend(shuffled[:validation_count])
        training.extend(shuffled[validation_count:])
    if not training or not validation:
        raise ValueError("Training and validation splits must both be non-empty")
    return training, validation


class PairedViewDataset(Dataset):
    def __init__(self, rows, processor, config: Config, *, training: bool):
        self.rows = [
            {**row, "_tile_candidates": deserialize_tile_candidates(row["tile_candidates"])}
            for row in rows
        ]
        self.processor = processor
        self.config = config
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item) -> dict[str, object]:
        if isinstance(item, (tuple, list)):
            epoch, index = map(int, item)
        else:
            epoch, index = 0, int(item)
        record = self.rows[index]
        image = load_prepared_image(record)
        tile = view_a = view_b = None
        try:
            rng = Random(f"{self.config.training.seed}:{epoch}:{record['image_id']}")
            selection = choose_adaptation_tile(
                record["_tile_candidates"],
                self.config,
                rng,
            )
            tile = image.crop(selection.box)
            view_a, view_b = paired_views(tile, self.config, rng)
            pixels = self.processor(images=[view_a, view_b], return_tensors="pt")[
                "pixel_values"
            ]
        finally:
            image.close()
            for opened in (tile, view_a, view_b):
                if opened is not None:
                    opened.close()
        return {
            "view_a": pixels[0],
            "view_b": pixels[1],
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "cohort_id": record["cohort_id"],
            "input_mode": record["input_mode"],
            "tile_grid_size": torch.tensor(selection.grid_size, dtype=torch.int64),
            "tile_row": torch.tensor(selection.row, dtype=torch.int64),
            "tile_column": torch.tensor(selection.column, dtype=torch.int64),
            "tile_sampling_strategy": selection.sampling_strategy,
            "tile_vegetation_fraction": torch.tensor(
                selection.vegetation_fraction, dtype=torch.float32
            ),
            "label_overlap_fraction": torch.tensor(
                selection.label_overlap_fraction, dtype=torch.float32
            ),
        }


class EpochSampler(Sampler[tuple[int, int]]):
    """Yield epoch-tagged indices so augmentation is resume-stable across workers."""

    def __init__(self, dataset: Dataset, seed: int):
        self.dataset, self.seed, self.epoch = dataset, seed, 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        return iter((self.epoch, index) for index in order)


def make_loaders(training, validation, processor, config: Config):
    training_dataset = PairedViewDataset(training, processor, config, training=True)
    validation_dataset = PairedViewDataset(validation, processor, config, training=False)
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.runtime.pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": config.training.num_workers > 0,
    }
    train_loader = DataLoader(
        training_dataset,
        sampler=EpochSampler(training_dataset, config.training.seed),
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
