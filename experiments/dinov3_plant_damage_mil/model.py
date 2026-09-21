"""Frozen three-view scorer plus trainable local plant-patch correction."""

from __future__ import annotations

import torch
from torch import nn

from experiments.dinov3_hierarchical_three_view_mil.model import HierarchicalThreeViewRegressor

from .config import Config


class PlantDamageRegressor(nn.Module):
    """Learn local evidence from image scores; maps are candidates, not lesion labels."""

    def __init__(self, feature_dim: int, config: Config, base_state_dict: dict):
        super().__init__()
        self.base = HierarchicalThreeViewRegressor(feature_dim, config.base)
        self.base.load_state_dict(base_state_dict)
        self.base.requires_grad_(False)
        self.base.eval()
        self.feature_dim = feature_dim
        hidden = config.hidden_dim
        self.input_norm = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU())
        self.local = nn.Sequential(
            nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.patch_attention = nn.Linear(hidden, 1)
        self.patch_evidence = nn.Linear(hidden, 1)
        self.residual_head = nn.Linear(3, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, batch: dict, *, return_attention: bool = False):
        global_feature = batch["global_feature"]
        cells = batch["cell_features"]
        plants = batch["plant_features"]
        plant_valid = batch["plant_valid"].bool()
        patch_valid = batch["patch_valid"].bool() & plant_valid[:, :, None]
        if not patch_valid.any(dim=-1)[plant_valid].all():
            raise ValueError("Each real SAM plant must have a valid high-resolution patch")
        with torch.no_grad():
            baseline, baseline_attention = self.base(
                global_feature, cells, plants, plant_valid,
                batch["plant_cell_indices"], return_attention=True,
            )
        patch_projection = self.project(self.input_norm(batch["patch_features"].float()))
        plant_projection = self.project(self.input_norm(plants.float()))[:, :, None, :]
        local = self.local(torch.cat([
            patch_projection, patch_projection - plant_projection.expand_as(patch_projection)
        ], dim=-1))
        scores = self.patch_evidence(local).squeeze(-1)
        logits = self.patch_attention(local).squeeze(-1)
        coverage = batch["patch_foreground_fraction"].float().clamp_min(0.01)
        logits = logits + coverage.log()
        logits = logits.masked_fill(~patch_valid, -1e4)
        patch_weights = torch.softmax(logits, dim=-1) * patch_valid.float()
        patch_weights = patch_weights / patch_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        plant_evidence = (scores * patch_weights).sum(dim=-1)
        plant_evidence = plant_evidence.masked_fill(~plant_valid, 0)
        base_weights = baseline_attention["plant_weights"].detach()
        mean_evidence = (plant_evidence * plant_valid.float()).sum(dim=-1) / plant_valid.sum(dim=-1)
        attended_evidence = (plant_evidence * base_weights).sum(dim=-1)
        top_evidence = plant_evidence.masked_fill(~plant_valid, -1e4).max(dim=-1).values
        residual = self.residual_head(torch.stack([
            mean_evidence, attended_evidence, top_evidence
        ], dim=-1)).squeeze(-1)
        output = baseline + residual
        if not return_attention:
            return output
        return output, {
            **baseline_attention,
            "base_prediction": baseline,
            "residual": residual,
            "patch_weights": patch_weights,
            "patch_evidence": scores.masked_fill(~patch_valid, 0),
            "plant_evidence": plant_evidence,
        }

    def parameter_summary(self) -> dict:
        return {
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen_three_view_parameters": sum(p.numel() for p in self.base.parameters()),
            "views": "global + four cells + SAM plants + local high-resolution plant patches",
            "interpretation": "Patch maps are weakly supervised local evidence, not verified lesion masks",
        }
