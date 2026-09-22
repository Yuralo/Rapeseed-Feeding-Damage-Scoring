"""Train the exact plant-patch local modules on measured bite/sham DINO pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from experiments.dinov3_grid_tiled_mil.runtime import configure_acceleration
from experiments.dinov3_hierarchical_three_view_mil.checkpoint import validate_for as validate_base
from experiments.dinov3_plant_damage_mil.model import PlantDamageRegressor
from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything, seed_worker

from .config import Config, load_config
from .prepare_pairs import load_pair

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PREFIXES = ("input_norm.", "project.", "local.", "patch_evidence.")


def local_state(model: PlantDamageRegressor) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if key.startswith(LOCAL_PREFIXES)}


def load_local_state(model: PlantDamageRegressor, state: dict) -> None:
    expected = set(local_state(model))
    if set(state) != expected:
        raise ValueError(f"Local pretraining tensors differ: missing={sorted(expected-set(state))}, "
                         f"extra={sorted(set(state)-expected)}")
    model.load_state_dict(state, strict=False)


def local_score(model: PlantDamageRegressor, patch: torch.Tensor,
                plant: torch.Tensor) -> torch.Tensor:
    """Exactly the local evidence path used in PlantDamageRegressor.forward."""
    patch_projection = model.project(model.input_norm(patch.float()))
    plant_projection = model.project(model.input_norm(plant.float()))
    local = model.local(torch.cat([patch_projection, patch_projection - plant_projection], dim=-1))
    return model.patch_evidence(local).squeeze(-1)


class PairDataset(Dataset):
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        record = load_pair(Path(row["pair_path"]), row["identity"])
        level = int(row["level_index"])
        return {
            "plant": torch.from_numpy(np.asarray(record["plant_feature"], dtype=np.float32)),
            "original": torch.from_numpy(np.asarray(record["original_feature"], dtype=np.float32)),
            "bite": torch.from_numpy(np.asarray(record["bite_features"][level], dtype=np.float32)),
            "sham": torch.from_numpy(np.asarray(record["sham_features"][level], dtype=np.float32)),
            "realized": torch.tensor(float(record["realized_fractions"][level]), dtype=torch.float32),
            "requested": torch.tensor(float(record["requested_fractions"][level]), dtype=torch.float32),
        }


def _split_rows(config: Config, allow_partial: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    summary_path = Path(config.run_dir) / "preparation_summary.json"
    index_path = Path(config.run_dir) / "pair_index.csv"
    if not summary_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Run prepare_pairs before pretraining")
    summary = json.loads(summary_path.read_text())
    if summary["masks_only"]:
        raise ValueError("Pair features have not been prepared")
    if summary.get("preparation_config") != json.loads(json.dumps(asdict(config))):
        raise ValueError("Pair preparation config changed; regenerate the pair index")
    full_count = len(pd.read_csv(config.adaptation_manifest))
    if not allow_partial and summary["requested_images"] != full_count:
        raise ValueError("Pair preparation was a limited smoke run; rerun the full manifest")
    if not allow_partial and summary["coverage_fraction"] < config.minimum_coverage_fraction:
        raise ValueError("Pair coverage is below the configured threshold")
    if summary["adaptation_manifest_sha256"] != hashlib.sha256(
        Path(config.adaptation_manifest).read_bytes()
    ).hexdigest():
        raise ValueError("Adaptation manifest changed after pair preparation")
    table = pd.read_csv(index_path, dtype={"plot_group_id": str}, keep_default_na=False)
    if table.empty:
        raise ValueError("Pair index is empty")
    # A physical plot stays in one side; missing QR groups fall back to image identity.
    keys = [str(row.plot_group_id or row.sha256)
            for row in table.itertuples()]
    holdout = np.asarray([
        int(hashlib.sha256(f"{config.seed}:{key}".encode()).hexdigest()[:12], 16) /
        float(16**12) < config.synthetic_holdout_fraction for key in keys
    ])
    train, validation = table.loc[~holdout].reset_index(drop=True), table.loc[holdout].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Synthetic train/holdout split is empty")
    groups_train = {key for key, value in zip(keys, holdout) if not value}
    groups_val = {key for key, value in zip(keys, holdout) if value}
    if groups_train & groups_val:
        raise ValueError("Synthetic plot groups overlap across splits")
    return train, validation, summary["adaptation_manifest_sha256"]


def _loader(rows, config, training):
    dataset = PairDataset(rows)
    sampler = None
    if training:
        counts = Counter(rows["cohort_id"])
        weights = torch.tensor([1 / counts[cohort] for cohort in rows["cohort_id"]],
                               dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True,
                                        generator=torch.Generator().manual_seed(config.seed + 17))
    return DataLoader(dataset, batch_size=config.pretrain_batch_size, shuffle=False,
                      sampler=sampler, num_workers=config.pretrain_workers,
                      pin_memory=config.plant.base.runtime.pin_memory,
                      worker_init_fn=seed_worker, persistent_workers=config.pretrain_workers > 0)


def pair_loss(model, batch: dict, config: Config, mode: str):
    plant = batch["plant"]
    original = local_score(model, batch["original"], plant)
    bite = local_score(model, batch["bite"], plant)
    sham = local_score(model, batch["sham"], plant)
    if mode == "synthetic":
        positive, negative = bite, sham
    elif mode == "sham":
        positive, negative = sham, bite
    else:
        raise ValueError(f"Unknown pretraining mode: {mode}")
    delta = positive - original
    negative_delta = negative - original
    realized = batch["realized"].float()
    regression = F.smooth_l1_loss(delta, realized, beta=config.delta_huber_beta)
    order = F.softplus(config.order_margin - delta).mean()
    control = F.smooth_l1_loss(negative_delta, torch.zeros_like(negative_delta),
                               beta=config.delta_huber_beta)
    loss = regression + config.order_weight * order + config.sham_weight * control
    return loss, {"delta_loss": regression, "order_loss": order, "control_loss": control,
                  "bite_delta": (bite - original).mean(), "sham_delta": (sham - original).mean(),
                  "realized": realized.mean()}


def _epoch(model, loader, config, device, mode, optimizer=None):
    model.train(optimizer is not None)
    model.base.eval()
    total = Counter()
    seen = 0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, details = pair_loss(model, moved, config, mode)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                )
                optimizer.step()
            size = len(moved["realized"])
            total["loss"] += float(loss.detach()) * size
            for key, value in details.items():
                total[key] += float(value.detach()) * size
            seen += size
    return {key: value / seen for key, value in total.items()}


def _diagnostics(model, loader, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            original = local_score(model, moved["original"], moved["plant"])
            bite = local_score(model, moved["bite"], moved["plant"])
            sham = local_score(model, moved["sham"], moved["plant"])
            for requested, realized, bite_delta, sham_delta in zip(
                moved["requested"].cpu().numpy(), moved["realized"].cpu().numpy(),
                (bite - original).cpu().numpy(), (sham - original).cpu().numpy()
            ):
                rows.append((float(requested), float(realized), float(bite_delta),
                             float(sham_delta)))
    array = np.asarray(rows)
    levels = {}
    for requested in sorted(set(np.round(array[:, 0], 3))):
        selected = np.isclose(array[:, 0], requested, atol=1e-3)
        levels[str(requested)] = {"count": int(selected.sum()),
                                  "mean_bite_delta": float(array[selected, 2].mean()),
                                  "mean_sham_delta": float(array[selected, 3].mean()),
                                  "mean_realized_fraction": float(array[selected, 1].mean())}
    bite_mean = float(array[:, 2].mean())
    sham_mean = float(array[:, 3].mean())
    correlation = (float(np.corrcoef(array[:, 1], array[:, 2])[0, 1])
                   if len(array) > 2 and array[:, 1].std() > 0 and array[:, 2].std() > 0
                   else 0.0)
    passes = bite_mean > 0 and abs(sham_mean) < bite_mean * 0.5 and correlation > 0.3
    return {"samples": len(rows), "mean_bite_delta": bite_mean,
            "mean_sham_delta": sham_mean, "realized_bite_delta_correlation": correlation,
            "by_requested_level": levels, "passes_synthetic_gate": bool(passes)}


def run(config: Config, mode: str = "synthetic", *, allow_partial: bool = False) -> dict:
    if mode not in {"synthetic", "sham"}:
        raise ValueError("mode must be synthetic or sham")
    seed_everything(config.seed, config.plant.base.runtime.deterministic)
    device = resolve_device(config.plant.base.runtime.device)
    configure_acceleration(config.plant.base, device)
    train, validation, manifest_hash = _split_rows(config, allow_partial)
    train_loader = _loader(train, config, training=True)
    val_loader = _loader(validation, config, training=False)
    base_state = load_checkpoint(config.plant.base_checkpoint, device)
    validate_base(base_state, config.plant.base)
    dimension = int(base_state["feature_dim"])
    model = PlantDamageRegressor(dimension, config.plant, base_state["model_state_dict"]).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.input_norm, model.project, model.local, model.patch_evidence):
        module.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.pretrain_learning_rate, weight_decay=config.plant.weight_decay,
    )
    output = Path(config.run_dir) / "pretraining" / mode
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "environment.json", environment_info(device, ROOT))
    write_json(output / "config.json", asdict(config))
    best = float("inf")
    patience = 0
    history = []
    for epoch in range(1, config.pretrain_epochs + 1):
        training = _epoch(model, train_loader, config, device, mode, optimizer)
        heldout = _epoch(model, val_loader, config, device, mode)
        history.append({"epoch": epoch, "train": training, "heldout": heldout})
        state = {"experiment": "counterfactual_feeding_pretraining", "version": 1,
                 "mode": mode, "epoch": epoch, "local_state_dict": local_state(model),
                 "feature_dim": dimension, "config": asdict(config),
                 "plant_config": asdict(config.plant), "manifest_sha256": manifest_hash,
                 "heldout": heldout, "train_rows": len(train), "heldout_rows": len(validation)}
        save_checkpoint(output / "last.pt", state)
        if heldout["loss"] < best - 1e-5:
            best = heldout["loss"]
            patience = 0
            save_checkpoint(output / "best.pt", state)
        else:
            patience += 1
        write_json(output / "history.json", history)
        print(f"{mode} epoch {epoch:02d}: train={training['loss']:.4f} "
              f"holdout={heldout['loss']:.4f} patience={patience}", flush=True)
        if patience >= config.pretrain_patience:
            break
    selected = load_checkpoint(output / "best.pt", device)
    load_local_state(model, selected["local_state_dict"])
    diagnostics = _diagnostics(model, val_loader, device)
    report = {"mode": mode, "best_epoch": selected["epoch"],
              "heldout": selected["heldout"], "train_rows": len(train),
              "heldout_rows": len(validation), "checkpoint": str(output / "best.pt"),
              "diagnostics": diagnostics}
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("synthetic", "sham"), default="synthetic")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Smoke tests only; the final comparison requires full pair coverage")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), args.mode,
                         allow_partial=args.allow_partial), indent=2))


if __name__ == "__main__":
    main()
