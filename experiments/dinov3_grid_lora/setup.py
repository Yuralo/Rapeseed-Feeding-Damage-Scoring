"""Data setup for LoRA training and checkpoint evaluation."""

from __future__ import annotations

from transformers import AutoImageProcessor

from .checkpoint import scaler_from
from .config import Config
from .data import TargetScaler, load_scores, make_loaders, split_scores


def prepare_data(config: Config, checkpoint=None):
    table = load_scores(config)
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
