"""Standard-library-only configuration for representation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AnalysisSettings:
    output_dir: str
    device: str = "auto"
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 0
    pca_components: int = 50
    min_cluster_size: int = 10
    min_samples: int = 5
    limit: int = 0


@dataclass(frozen=True)
class RepresentationSettings:
    name: str
    experiment_package: str
    experiment_config: str
    checkpoint: str
    preprocessing: str
    feature: str


def load_analysis_config(path: str | Path):
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    analysis = AnalysisSettings(**raw["analysis"])
    representations = [RepresentationSettings(**item) for item in raw["representation"]]
    if not representations:
        raise ValueError("At least one [[representation]] is required")
    names = [item.name for item in representations]
    if len(names) != len(set(names)):
        raise ValueError("Representation names must be unique")
    for item in representations:
        if item.preprocessing not in {"raw", "grid"}:
            raise ValueError(f"Unsupported preprocessing for {item.name}: {item.preprocessing}")
        if item.feature not in {"backbone", "head"}:
            raise ValueError(f"Unsupported feature stage for {item.name}: {item.feature}")
        if item.feature == "head" and not item.checkpoint:
            raise ValueError(f"Head features require a checkpoint: {item.name}")
    return analysis, representations

