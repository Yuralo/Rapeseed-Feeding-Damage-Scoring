"""Generic atomic PyTorch checkpoint I/O.

Checkpoint contents belong to each experiment; this module only handles files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(path: str | Path, device) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a dictionary checkpoint: {checkpoint_path}")
    return state

