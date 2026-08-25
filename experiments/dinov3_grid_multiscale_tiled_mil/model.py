"""Shared-projection, scale-specific-attention MIL regression head."""

from __future__ import annotations

import torch
from torch import nn

from experiments.dinov3_grid_tiled_mil.model import GatedTileAttention

from .config import Config


class MultiScaleTiledMILRegressor(nn.Module):
    def __init__(self, feature_dim: int, config: Config):
        super().__init__()
        projection = config.model.projection_dim
        self.feature_dim = int(feature_dim)
        self.coarse_tiles = config.coarse.rows * config.coarse.columns
        self.fine_tiles = config.fine.rows * config.fine.columns
        self.input_norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
        )
        attention_arguments = (
            projection,
            config.model.attention_hidden_dim,
            config.model.attention_dropout,
            config.model.attention_temperature,
        )
        self.coarse_attention = GatedTileAttention(*attention_arguments)
        self.fine_attention = GatedTileAttention(*attention_arguments)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(5 * projection),
            nn.Linear(5 * projection, config.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.head_hidden_dim, 1),
        )

    def _validate(self, features: torch.Tensor, expected_tiles: int, name: str) -> None:
        expected_views = expected_tiles + 1
        if features.ndim != 3 or features.shape[1] != expected_views:
            raise ValueError(
                f"Expected {name} features shaped [batch, {expected_views}, feature_dim]"
            )
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected frozen feature dimension {self.feature_dim}, got {features.shape[-1]}"
            )

    def forward(
        self,
        coarse_features: torch.Tensor,
        fine_features: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        self._validate(coarse_features, self.coarse_tiles, "coarse")
        self._validate(fine_features, self.fine_tiles, "fine")
        coarse = self.projection(self.input_norm(coarse_features.float()))
        fine = self.projection(self.input_norm(fine_features.float()))
        # Both caches contain the same global view. Averaging avoids arbitrarily preferring one
        # extraction while still exposing only one global representation to the head.
        global_view = 0.5 * (coarse[:, 0] + fine[:, 0])
        coarse_tiles, fine_tiles = coarse[:, 1:], fine[:, 1:]
        coarse_attended, coarse_weights = self.coarse_attention(coarse_tiles)
        fine_attended, fine_weights = self.fine_attention(fine_tiles)
        pooled = torch.cat(
            [
                global_view,
                coarse_tiles.mean(dim=1),
                coarse_attended,
                fine_tiles.mean(dim=1),
                fine_attended,
            ],
            dim=-1,
        )
        prediction = self.regression_head(pooled).squeeze(-1)
        if return_attention:
            return prediction, coarse_weights, fine_weights
        return prediction

    def parameter_summary(self) -> dict[str, int | str]:
        total = sum(parameter.numel() for parameter in self.parameters())
        attention = sum(
            parameter.numel()
            for module in (self.coarse_attention, self.fine_attention)
            for parameter in module.parameters()
        )
        return {
            "total_parameters": total,
            "trainable_parameters": total,
            "frozen_backbone_parameters_loaded_during_training": 0,
            "attention_parameters": attention,
            "frozen_feature_dimension": self.feature_dim,
            "coarse_tiles_per_image": self.coarse_tiles,
            "fine_tiles_per_image": self.fine_tiles,
            "pooled_representations": (
                "global+coarse_mean+coarse_attention+fine_mean+fine_attention"
            ),
        }
