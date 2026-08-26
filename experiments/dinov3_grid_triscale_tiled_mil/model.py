"""Context mean plus regional/local gated-attention MIL regression head."""

from __future__ import annotations

import torch
from torch import nn

from experiments.dinov3_grid_tiled_mil.model import GatedTileAttention

from .config import Config


class TriScaleTiledMILRegressor(nn.Module):
    def __init__(self, feature_dim: int, config: Config):
        super().__init__()
        projection = config.model.projection_dim
        self.feature_dim = int(feature_dim)
        self.context_tiles = config.context.rows * config.context.columns
        self.regional_tiles = config.regional.rows * config.regional.columns
        self.local_tiles = config.local.rows * config.local.columns
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
        self.regional_attention = GatedTileAttention(*attention_arguments)
        self.local_attention = GatedTileAttention(*attention_arguments)
        # global + 3x3 mean + 4x4 mean/attention + 5x5 mean/attention
        self.regression_head = nn.Sequential(
            nn.LayerNorm(6 * projection),
            nn.Linear(6 * projection, config.model.head_hidden_dim),
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
        context_features: torch.Tensor,
        regional_features: torch.Tensor,
        local_features: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        self._validate(context_features, self.context_tiles, "context")
        self._validate(regional_features, self.regional_tiles, "regional")
        self._validate(local_features, self.local_tiles, "local")
        context = self.projection(self.input_norm(context_features.float()))
        regional = self.projection(self.input_norm(regional_features.float()))
        local = self.projection(self.input_norm(local_features.float()))
        # Each cache contains the same global view; expose one averaged representation.
        global_view = (context[:, 0] + regional[:, 0] + local[:, 0]) / 3.0
        context_tiles = context[:, 1:]
        regional_tiles = regional[:, 1:]
        local_tiles = local[:, 1:]
        regional_attended, regional_weights = self.regional_attention(regional_tiles)
        local_attended, local_weights = self.local_attention(local_tiles)
        pooled = torch.cat(
            [
                global_view,
                context_tiles.mean(dim=1),
                regional_tiles.mean(dim=1),
                regional_attended,
                local_tiles.mean(dim=1),
                local_attended,
            ],
            dim=-1,
        )
        prediction = self.regression_head(pooled).squeeze(-1)
        if return_attention:
            return prediction, regional_weights, local_weights
        return prediction

    def parameter_summary(self) -> dict[str, int | str]:
        total = sum(parameter.numel() for parameter in self.parameters())
        attention = sum(
            parameter.numel()
            for module in (self.regional_attention, self.local_attention)
            for parameter in module.parameters()
        )
        return {
            "total_parameters": total,
            "trainable_parameters": total,
            "frozen_backbone_parameters_loaded_during_training": 0,
            "attention_parameters": attention,
            "frozen_feature_dimension": self.feature_dim,
            "context_tiles_per_image": self.context_tiles,
            "regional_tiles_per_image": self.regional_tiles,
            "local_tiles_per_image": self.local_tiles,
            "pooled_representations": (
                "global+3x3_mean+4x4_mean+4x4_attention+5x5_mean+5x5_attention"
            ),
        }
