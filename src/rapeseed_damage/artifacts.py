"""Small, experiment-agnostic helpers for recording run artifacts."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def git_state(repository: str | Path = ".") -> dict[str, Any]:
    """Record the commit and dirty state without making Git a run dependency."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment_info(device=None, repository: str | Path = ".") -> dict[str, Any]:
    """Collect versions lazily so the shared module does not require ML packages."""
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_state(repository),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "device": str(device) if device is not None else None,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": (
                    torch.backends.cudnn.version() if torch.cuda.is_available() else None
                ),
            }
        )
    except ImportError:
        result["torch"] = None
    try:
        import transformers

        result["transformers"] = transformers.__version__
    except ImportError:
        result["transformers"] = None
    return result
