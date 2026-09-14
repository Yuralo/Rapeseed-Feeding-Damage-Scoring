"""Hierarchical plant-within-cell, cell-within-image regression head."""

from __future__ import annotations

import torch
from torch import nn

from .config import Config


class GatedScore(nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float, temperature: float):
        super().__init__()
        self.value = nn.Linear(dimension, hidden)
        self.control = nn.Linear(dimension, hidden)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.temperature = temperature
        nn.init.zeros_(self.score.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gated = torch.tanh(self.value(values)) * torch.sigmoid(self.control(values))
        return self.score(self.dropout(gated)).squeeze(-1).float() / self.temperature


class HierarchicalThreeViewRegressor(nn.Module):
    """Fuse global context, four cell views, and SAM plants assigned to each cell."""

    def __init__(self, feature_dim: int, config: Config):
        super().__init__()
        projection = config.model.projection_dim
        self.feature_dim = int(feature_dim)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
        )
        context_dim = 3 * projection
        self.plant_delta = nn.Sequential(nn.Linear(context_dim, projection), nn.Tanh())
        self.plant_gate = nn.Sequential(nn.Linear(context_dim, projection), nn.Sigmoid())
        self.plant_attention = GatedScore(
            projection,
            config.model.attention_hidden_dim,
            config.model.attention_dropout,
            config.model.attention_temperature,
        )
        self.cell_delta = nn.Sequential(nn.Linear(context_dim, projection), nn.Tanh())
        self.cell_gate = nn.Sequential(nn.Linear(context_dim, projection), nn.Sigmoid())
        self.cell_attention = GatedScore(
            projection,
            config.model.attention_hidden_dim,
            config.model.attention_dropout,
            config.model.attention_temperature,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(3 * projection),
            nn.Linear(3 * projection, config.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.head_hidden_dim, 1),
        )

    def forward(
        self,
        global_feature: torch.Tensor,
        cell_features: torch.Tensor,
        plant_features: torch.Tensor,
        plant_valid: torch.Tensor,
        plant_cell_indices: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        if global_feature.ndim != 2 or global_feature.shape[-1] != self.feature_dim:
            raise ValueError("Unexpected global feature shape")
        if cell_features.shape[1:] != (4, self.feature_dim):
            raise ValueError("Expected exactly four cell features")
        if plant_features.shape[-1] != self.feature_dim:
            raise ValueError("Unexpected plant feature dimension")
        if plant_valid.shape != plant_features.shape[:2]:
            raise ValueError("Plant mask shape does not match plant features")
        if plant_cell_indices.shape != plant_valid.shape:
            raise ValueError("Plant cell-index shape does not match plant mask")
        if not plant_valid.any(dim=1).all():
            raise ValueError("Every image requires at least one valid SAM plant")

        global_projected = self.projection(self.input_norm(global_feature.float()))
        cells = self.projection(self.input_norm(cell_features.float()))
        plants = self.projection(self.input_norm(plant_features.float()))
        global_for_plants = global_projected[:, None, :].expand_as(plants)
        owning_cells = torch.gather(
            cells,
            1,
            plant_cell_indices.clamp(0, 3).unsqueeze(-1).expand(-1, -1, cells.shape[-1]),
        )
        plant_context = torch.cat([plants, owning_cells, global_for_plants], dim=-1)
        contextual_plants = plants + self.plant_gate(plant_context) * self.plant_delta(plant_context)
        plant_logits = self.plant_attention(contextual_plants)

        cell_plant_pools, within_cell_weights, cells_with_plants = [], [], []
        for cell_index in range(4):
            membership = plant_valid & (plant_cell_indices == cell_index)
            has_plants = membership.any(dim=1)
            safe_logits = plant_logits.masked_fill(~membership, float("-inf"))
            safe_logits = torch.where(has_plants[:, None], safe_logits, torch.zeros_like(safe_logits))
            weights = torch.softmax(safe_logits, dim=1) * membership.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            pooled = (weights.unsqueeze(-1) * contextual_plants.float()).sum(dim=1)
            cell_plant_pools.append(pooled)
            within_cell_weights.append(weights)
            cells_with_plants.append(has_plants)
        plant_pools = torch.stack(cell_plant_pools, dim=1).to(cells.dtype)
        cell_has_plants = torch.stack(cells_with_plants, dim=1)
        global_for_cells = global_projected[:, None, :].expand_as(cells)
        cell_context = torch.cat([cells, plant_pools, global_for_cells], dim=-1)
        plant_gate = cell_has_plants.unsqueeze(-1).to(cells.dtype)
        contextual_cells = cells + plant_gate * self.cell_gate(cell_context) * self.cell_delta(
            cell_context
        )
        cell_weights = torch.softmax(self.cell_attention(contextual_cells), dim=1)
        attended_cells = (cell_weights.unsqueeze(-1) * contextual_cells.float()).sum(dim=1)
        pooled = torch.cat(
            [global_projected, contextual_cells.mean(dim=1), attended_cells.to(cells.dtype)], dim=-1
        )
        prediction = self.head(pooled).squeeze(-1)
        if not return_attention:
            return prediction

        plant_weights = torch.zeros_like(plant_logits)
        for cell_index, local_weights in enumerate(within_cell_weights):
            plant_weights = plant_weights + local_weights * cell_weights[:, cell_index : cell_index + 1]
        plant_weights = plant_weights * plant_valid.float()
        plant_weights = plant_weights / plant_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return prediction, {
            "cell_weights": cell_weights,
            "plant_weights": plant_weights,
            "cells_with_plants": cell_has_plants,
        }

    def parameter_summary(self) -> dict[str, object]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": total,
            "frozen_backbone_parameters_loaded_during_training": 0,
            "frozen_feature_dimension": self.feature_dim,
            "views": "global + four 2x2 cells + SAM plant crops",
            "hierarchy": "plant-with-owning-cell context -> cells -> image",
        }


__all__ = ["HierarchicalThreeViewRegressor"]
