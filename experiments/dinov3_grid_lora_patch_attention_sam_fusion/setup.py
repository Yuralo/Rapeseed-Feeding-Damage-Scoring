"""Data setup for three-representation SAM-fusion training and evaluation."""

from __future__ import annotations

from transformers import AutoImageProcessor

from .checkpoint import scaler_from
from .config import Config
from .data import TargetScaler, image_path, load_scores, make_loaders, split_scores
from .segmentation import load_cached_mask


def validate_sam_cache(table, config: Config) -> None:
    failures = []
    for _, row in table.iterrows():
        filename = str(row[config.data.filename_column])
        mask = None
        try:
            mask, _, _ = load_cached_mask(image_path(config, filename), config)
        except Exception as error:
            failures.append(f"{filename}: {type(error).__name__}: {error}")
        finally:
            if mask is not None:
                mask.close()
    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(
            f"SAM cache validation failed for {len(failures)} image(s). "
            "Run prepare_sam_cache and inspect_sam_masks before training. "
            f"First failures:\n{preview}"
        )


def prepare_data(config: Config, checkpoint=None):
    table = load_scores(config)
    validate_sam_cache(table, config)
    train, validation = split_scores(
        table,
        config,
        training_filenames=(checkpoint or {}).get("training_filenames"),
        validation_filenames=(checkpoint or {}).get("validation_filenames"),
    )
    scaler = (
        scaler_from(checkpoint)
        if checkpoint is not None
        else TargetScaler.fit(train[config.data.target_column])
    )
    processor = AutoImageProcessor.from_pretrained(config.model.processor)
    train_loader, validation_loader = make_loaders(
        train,
        validation,
        scaler,
        processor,
        config,
    )
    return table, train, validation, scaler, train_loader, validation_loader
