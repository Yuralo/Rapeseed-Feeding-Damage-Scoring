"""Tests for measured counterfactuals and the safety of the pretext split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from experiments.counterfactual_feeding_pretraining.config import load_config
from experiments.counterfactual_feeding_pretraining.prepare_pairs import _manifest
from experiments.counterfactual_feeding_pretraining.pretrain import _split_rows, pair_loss
from experiments.counterfactual_feeding_pretraining.synthesis import make_pair
from experiments.dinov3_hierarchical_three_view_mil.model import HierarchicalThreeViewRegressor
from experiments.dinov3_plant_damage_mil.model import PlantDamageRegressor

CONFIG = "experiments/counterfactual_feeding_pretraining/config.toml"


def test_bite_removes_only_leaf_and_sham_edits_only_soil():
    rng = np.random.default_rng(19)
    rgb = rng.integers(50, 120, size=(256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (128, 128), (43, 32), 0, 0, 360, 1, -1)
    leaf = mask.astype(bool)
    rgb[leaf] = (65, 180, 70)
    box = (65, 65, 190, 190)
    for fraction in (0.02, 0.05, 0.10, 0.20):
        pair = make_pair(Image.fromarray(rgb), leaf, box, fraction, rng)
        local_leaf = leaf[box[1]:box[3], box[0]:box[2]]
        original = rgb[box[1]:box[3], box[0]:box[2]]
        bite = np.asarray(pair.bite)
        sham = np.asarray(pair.sham)
        assert 0.65 * fraction <= pair.realized_fraction <= 1.35 * fraction
        assert np.all(pair.removed_mask <= local_leaf)
        assert np.all(pair.sham_mask <= ~local_leaf)
        assert np.array_equal(bite[~pair.removed_mask], original[~pair.removed_mask])
        assert np.array_equal(sham[local_leaf], original[local_leaf])
        assert pair.sham_mask.sum() >= 0.8 * pair.removed_mask.sum()


def test_adaptation_manifest_rejects_gold_holdout_and_gg(tmp_path):
    config = load_config(CONFIG)
    holdout = pd.read_csv("outputs/dataset_manifests/validation.csv").iloc[0]
    path = tmp_path / "adaptation.csv"
    row = {"image_id": "fake", "absolute_path": "/does/not/exist.jpg",
           "relative_path": "external/fake.jpg", "sha256": holdout["sha256"],
           "cohort_id": "wg_insects_t1_bbch10", "plot_group_id": "safe_group"}
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlaps gold holdout"):
        _manifest(replace(config, adaptation_manifest=str(path)))
    row["sha256"] = "a" * 64
    row["cohort_id"] = "gg_insects_t2_bbch13"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="timepoints"):
        _manifest(replace(config, adaptation_manifest=str(path)))


def test_pretext_holdout_keeps_plot_views_together(tmp_path):
    config = load_config(CONFIG)
    manifest = tmp_path / "adaptation.csv"
    rows = [{"image_id": f"image_{index}", "cohort_id": "wg_insects_t1_bbch10",
             "plot_group_id": f"plot_{index // 3}", "sha256": f"{index:064x}"}
            for index in range(300)]
    pd.DataFrame(rows).to_csv(manifest, index=False)
    output = tmp_path / "run"
    output.mkdir()
    index = pd.DataFrame({**row, "pair_path": "/fake.npz", "identity": "x",
                          "level_index": 0, "relative_path": f"{row['image_id']}.jpg",
                          "requested_fraction": .02, "realized_fraction": .02,
                          "kind": "interior"} for row in rows)
    index.to_csv(output / "pair_index.csv", index=False)
    (output / "preparation_summary.json").write_text(json.dumps({
        "masks_only": False, "requested_images": len(rows), "coverage_fraction": 1.0,
        "preparation_config": asdict(changed := replace(
            config, adaptation_manifest=str(manifest), run_dir=str(output))),
        "adaptation_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }))
    train, heldout, _ = _split_rows(changed, allow_partial=False)
    assert len(train) + len(heldout) == len(rows)
    assert not set(train["plot_group_id"]) & set(heldout["plot_group_id"])


def test_pretext_loss_trains_local_evidence_without_updating_frozen_base():
    config = load_config(CONFIG)
    base = HierarchicalThreeViewRegressor(8, config.plant.base)
    model = PlantDamageRegressor(8, config.plant, base.state_dict())
    batch = {"plant": torch.randn(4, 8), "original": torch.randn(4, 8),
             "bite": torch.randn(4, 8), "sham": torch.randn(4, 8),
             "realized": torch.tensor([.02, .05, .10, .20])}
    loss, details = pair_loss(model, batch, config, "synthetic")
    assert torch.isfinite(loss)
    assert set(details) >= {"bite_delta", "sham_delta", "realized"}
    loss.backward()
    assert model.patch_evidence.weight.grad is not None
    assert model.local[1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.base.parameters())
