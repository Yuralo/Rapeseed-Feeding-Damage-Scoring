"""Small residual head, guarded pair construction, and plot-grouped fitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import Config

ARMS = ("gold_only", "midpoint", "rank", "shuffled")


@dataclass
class Pool:
    features: np.ndarray
    base: np.ndarray
    target: np.ndarray
    group: np.ndarray
    filename: np.ndarray
    cohort: np.ndarray
    source_path: np.ndarray
    score_jlu: np.ndarray
    score_gau: np.ndarray

    def __len__(self) -> int:
        return len(self.target)


def load_pool(run_dir: str | Path, split: str) -> Pool:
    path = Path(run_dir) / "frozen" / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen embeddings: {path}; run prepare")
    summary_path = Path(run_dir) / "frozen" / "summary.json"
    summary = json.loads(summary_path.read_text())
    expected = summary["splits"][split]["npz_sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Frozen {split} embeddings changed since preparation")
    with np.load(path, allow_pickle=False) as data:
        values = {name: data[name].copy() for name in Pool.__dataclass_fields__}
    pool = Pool(**values)
    if any(len(getattr(pool, name)) != len(pool) for name in Pool.__dataclass_fields__):
        raise ValueError(f"Different array lengths in {path}")
    if not np.isfinite(pool.features).all() or not np.isfinite(pool.base).all():
        raise ValueError(f"Nonfinite feature or base in {path}")
    return pool


def subset_pool(pool: Pool, indices: np.ndarray) -> Pool:
    return Pool(**{name: getattr(pool, name)[indices].copy()
                   for name in Pool.__dataclass_fields__})


def target_weak_pool(run_dir: str | Path, config: Config) -> Pool:
    all_weak = load_pool(run_dir, "pretrain")
    selected = np.flatnonzero((all_weak.cohort == config.target_cohort)
                              & np.isfinite(all_weak.score_jlu)
                              & np.isfinite(all_weak.score_gau))
    if not len(selected):
        raise ValueError("No in-domain dual-rater weak images in prepared cache")
    return subset_pool(all_weak, selected)


class ResidualHead(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, maximum: float):
        super().__init__()
        self.maximum = maximum
        self.layers = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden_dim),
                                    nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.maximum * torch.tanh(self.layers(features).squeeze(-1))
        # No output clamp: epoch zero reproduces the frozen base exactly, including rare
        # scores outside the nominal range. Bound only the learned correction.
        return base + residual, residual


def plot_weights(groups: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return (weights / weights.sum()).astype(np.float32)


def cv_folds(groups: np.ndarray, count: int, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    if len(unique) < count:
        raise ValueError("Fewer plot groups than CV folds")
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    assigned = {group: index % count for index, group in enumerate(shuffled)}
    return np.array([assigned[group] for group in groups], dtype=np.int16)


def candidate_pairs(weak: Pool, selected: np.ndarray, config: Config, seed: int,
                    *, shuffled: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return global weak indices for interval-dominant, different-plot pairs."""
    selected = np.asarray(selected, dtype=np.int64)
    low = np.minimum(weak.score_jlu, weak.score_gau)
    high = np.maximum(weak.score_jlu, weak.score_gau)
    sampling_band = np.digitize((low + high) / 2, [2.5, 7.5, 15.0])
    rng = np.random.default_rng(seed)

    def interleave(indices) -> list[int]:
        queues = [list(map(int, rng.permutation([i for i in indices if sampling_band[i] == band])))
                  for band in range(4)]
        ordered: list[int] = []
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.pop())
        return ordered

    by_high: dict[int, list[int]] = {}
    raw_count = 0
    for left_pos, left in enumerate(selected):
        for right in selected[left_pos + 1:]:
            if weak.group[left] == weak.group[right]:
                continue
            if low[left] > high[right] + config.rank_interval_margin:
                upper, lower = int(left), int(right)
            elif low[right] > high[left] + config.rank_interval_margin:
                upper, lower = int(right), int(left)
            else:
                continue
            by_high.setdefault(upper, []).append(lower)
            raw_count += 1
    pairs = []
    degree: dict[int, int] = {}
    for upper in interleave(by_high):
        if degree.get(int(upper), 0) >= config.max_pairs_per_image:
            continue
        for lower in interleave(by_high[int(upper)]):
            if degree.get(int(lower), 0) >= config.max_pairs_per_image:
                continue
            pairs.append((int(upper), int(lower)))
            degree[int(upper)] = degree.get(int(upper), 0) + 1
            degree[int(lower)] = degree.get(int(lower), 0) + 1
            if degree[int(upper)] >= config.max_pairs_per_image:
                break
    if not pairs:
        raise ValueError("No qualified weak pairs remain after filtering")
    pairs = np.asarray(pairs, dtype=np.int64)
    high_band_counts = np.bincount(sampling_band[pairs[:, 0]], minlength=4).tolist()
    low_band_counts = np.bincount(sampling_band[pairs[:, 1]], minlength=4).tolist()
    weights = plot_weights(weak.group[pairs[:, 0]])
    if shuffled:
        reverse = rng.random(len(pairs)) < 0.5
        pairs[reverse] = pairs[reverse][:, ::-1]
    first, second = pairs[:, 0], pairs[:, 1]
    return first, second, weights, {
        "raw_qualified_pairs": raw_count, "sampled_pairs": len(pairs),
        "participating_images": len(set(first.tolist()) | set(second.tolist())),
        "maximum_pairs_per_image": max(degree.values()),
        "higher_midpoint_band_pairs": high_band_counts,
        "lower_midpoint_band_pairs": low_band_counts,
        "shuffled": shuffled,
    }


def plot_mae(prediction: np.ndarray, target: np.ndarray, groups: np.ndarray) -> float:
    weights = plot_weights(groups)
    return float(np.sum(weights * np.abs(prediction - target)))


def fit_once(config: Config, gold: Pool, weak: Pool, gold_train: np.ndarray,
             weak_train: np.ndarray, *, arm: str, seed: int, rank_weight: float,
             residual_penalty: float, midpoint_weight: float, epochs: int,
             validation: np.ndarray | None = None, patience: int | None = None,
             device: str = "cpu") -> tuple[ResidualHead, list[dict], dict]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    combined_features = np.concatenate([gold.features, weak.features]).astype(np.float32)
    combined_base = np.concatenate([gold.base, weak.base]).astype(np.float32)
    x = torch.from_numpy(combined_features).to(device)
    b = torch.from_numpy(combined_base).to(device)
    gold_train = np.asarray(gold_train, dtype=np.int64)
    weak_train = np.asarray(weak_train, dtype=np.int64)
    if not len(gold_train):
        raise ValueError("Gold training fold is empty")
    if arm != "gold_only" and not len(weak_train):
        raise ValueError("Weak training fold is empty")
    gidx = torch.from_numpy(gold_train).to(device)
    widx = torch.from_numpy(weak_train + len(gold)).to(device)
    gy = torch.from_numpy(gold.target[gold_train].astype(np.float32)).to(device)
    gw = torch.from_numpy(plot_weights(gold.group[gold_train])).to(device)
    wy = torch.from_numpy(((weak.score_jlu[weak_train] + weak.score_gau[weak_train]) / 2)
                          .astype(np.float32)).to(device)
    ww = torch.from_numpy(plot_weights(weak.group[weak_train])).to(device)
    pair_info: dict = {}
    if arm in ("rank", "shuffled"):
        hi, lo, pw, pair_info = candidate_pairs(weak, weak_train, config, seed,
                                                 shuffled=arm == "shuffled")
        p_hi = torch.from_numpy(hi + len(gold)).to(device)
        p_lo = torch.from_numpy(lo + len(gold)).to(device)
        p_w = torch.from_numpy(pw).to(device)
    model = ResidualHead(x.shape[1], config.hidden_dim, config.max_residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    history = []
    best = (plot_mae(gold.base[validation], gold.target[validation], gold.group[validation])
            if validation is not None else float("inf"))
    best_epoch = 0
    stale = 0
    if validation is not None:
        history.append({"epoch": 0, "cv_plot_mae": best,
                        "train_plot_mae": plot_mae(gold.base[gold_train], gold.target[gold_train],
                                                    gold.group[gold_train])})
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, residual = model(x, b)
        gold_loss = torch.sum(gw * F.smooth_l1_loss(prediction[gidx], gy, beta=1.0,
                                                    reduction="none"))
        regularizer = torch.square(residual[torch.cat([gidx, widx])]).mean()
        loss = gold_loss + residual_penalty * regularizer
        weak_loss = torch.tensor(0.0, device=device)
        if arm == "midpoint":
            weak_loss = torch.sum(ww * F.smooth_l1_loss(prediction[widx], wy, beta=1.0,
                                                        reduction="none"))
            loss = loss + midpoint_weight * weak_loss
        elif arm in ("rank", "shuffled"):
            weak_loss = torch.sum(p_w * torch.square(torch.relu(
                config.rank_prediction_margin - (prediction[p_hi] - prediction[p_lo]))))
            loss = loss + rank_weight * weak_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            fitted, fitted_residual = model(x, b)
        fitted_np = fitted.detach().cpu().numpy()
        row = {"epoch": epoch, "loss": float(loss.detach().cpu()),
               "gold_loss": float(gold_loss.detach().cpu()),
               "weak_loss": float(weak_loss.detach().cpu()),
               "residual_rms": float(torch.sqrt(torch.square(fitted_residual).mean()).cpu()),
               "train_plot_mae": plot_mae(fitted_np[gold_train], gold.target[gold_train],
                                           gold.group[gold_train])}
        if validation is not None:
            row["cv_plot_mae"] = plot_mae(fitted_np[validation], gold.target[validation],
                                           gold.group[validation])
            if row["cv_plot_mae"] < best - 1e-5:
                best, best_epoch, stale = row["cv_plot_mae"], epoch, 0
            else:
                stale += 1
        history.append(row)
        if validation is not None and patience is not None and stale >= patience:
            break
    return model, history, {"pairs": pair_info, "best_cv_epoch": best_epoch,
                            "best_cv_plot_mae": best if validation is not None else None}


def predict(model: ResidualHead, pool: Pool, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        prediction, residual = model(torch.from_numpy(pool.features.astype(np.float32)).to(device),
                                     torch.from_numpy(pool.base.astype(np.float32)).to(device))
    return prediction.cpu().numpy(), residual.cpu().numpy()
