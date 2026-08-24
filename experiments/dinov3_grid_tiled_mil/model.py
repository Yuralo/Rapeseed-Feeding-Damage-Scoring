"""Small global-plus-tiled MIL head for cached frozen DINOv3 features."""

from __future__ import annotations

import torch
from torch import nn

from .config import Config


class GatedTileAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float, temperature: float):
        super().__init__()
        self.value_gate = nn.Linear(feature_dim, hidden_dim)
        self.control_gate = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.temperature = temperature
        nn.init.zeros_(self.score.weight)

    def forward(self, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated = torch.tanh(self.value_gate(tiles)) * torch.sigmoid(self.control_gate(tiles))
        logits = self.score(self.dropout(gated)).squeeze(-1) / self.temperature
        weights = torch.softmax(logits.float(), dim=1)
        attended = torch.sum(weights.unsqueeze(-1) * tiles.float(), dim=1).to(tiles.dtype)
        return attended, weights


class GlobalTiledMILRegressor(nn.Module):
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
        self.tile_attention = GatedTileAttention(
            projection,
            config.model.attention_hidden_dim,
            config.model.attention_dropout,
            config.model.attention_temperature,
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(3 * projection),
            nn.Linear(3 * projection, config.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.head_hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, *, return_attention: bool = False):
        if features.ndim != 3 or features.shape[1] < 2:
            raise ValueError("Expected [batch, one_global_plus_tiles, feature_dim]")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected frozen feature dimension {self.feature_dim}, got {features.shape[-1]}"
            )
        projected = self.projection(self.input_norm(features.float()))
        global_view = projected[:, 0]
        tiles = projected[:, 1:]
        tile_mean = tiles.mean(dim=1)
        attended_tiles, weights = self.tile_attention(tiles)
        prediction = self.regression_head(
            torch.cat([global_view, tile_mean, attended_tiles], dim=-1)
        ).squeeze(-1)
        return (prediction, weights) if return_attention else prediction

    def parameter_summary(self) -> dict[str, int | float | str]:
        total = sum(parameter.numel() for parameter in self.parameters())
        attention = sum(parameter.numel() for parameter in self.tile_attention.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": total,
            "frozen_backbone_parameters_loaded_during_training": 0,
            "attention_parameters": attention,
            "frozen_feature_dimension": self.feature_dim,
            "pooled_representations": "global+tile_mean+gated_tile_attention",
        }
