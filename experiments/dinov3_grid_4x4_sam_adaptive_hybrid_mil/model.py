"""Shared-projection fixed-tile plus SAM-adaptive hybrid MIL head."""

import torch
from torch import nn

from experiments.dinov3_grid_sam_adaptive_mil.model import MaskedGatedAttention
from experiments.dinov3_grid_tiled_mil.model import GatedTileAttention

from .config import Config


class HybridMILRegressor(nn.Module):
    def __init__(self, feature_dim: int, config: Config):
        super().__init__()
        projection = config.model.projection_dim
        self.feature_dim = int(feature_dim)
        self.context_views = 1 + config.context.rows * config.context.columns
        self.fine_views = 1 + config.fine.rows * config.fine.columns
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
        self.fine_attention = GatedTileAttention(*attention_arguments)
        self.plant_attention = MaskedGatedAttention(*attention_arguments)
        self.head = nn.Sequential(
            nn.LayerNorm(6 * projection),
            nn.Linear(6 * projection, config.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.head_hidden_dim, 1),
        )

    def forward(self, context, fine, instances, valid, *, return_attention=False):
        if context.ndim != 3 or context.shape[1:] != (self.context_views, self.feature_dim):
            raise ValueError("Unexpected 3x3 context feature shape")
        if fine.ndim != 3 or fine.shape[1:] != (self.fine_views, self.feature_dim):
            raise ValueError("Unexpected 4x4 feature shape")
        if instances.ndim != 3 or instances.shape[-1] != self.feature_dim:
            raise ValueError("Unexpected adaptive-instance feature shape")
        if valid.shape != instances.shape[:2]:
            raise ValueError("Adaptive-instance validity mask shape differs from features")
        context = self.projection(self.input_norm(context.float()))
        fine = self.projection(self.input_norm(fine.float()))
        instances = self.projection(self.input_norm(instances.float()))
        global_view = 0.5 * (context[:, 0] + fine[:, 0])
        context_tiles = context[:, 1:]
        fine_tiles = fine[:, 1:]
        fine_attended, fine_weights = self.fine_attention(fine_tiles)
        valid_float = valid.unsqueeze(-1).float()
        plant_mean = (instances * valid_float).sum(1) / valid_float.sum(1).clamp_min(1)
        plant_attended, plant_weights = self.plant_attention(instances, valid)
        pooled = torch.cat(
            [
                global_view,
                context_tiles.mean(1),
                fine_tiles.mean(1),
                fine_attended,
                plant_mean,
                plant_attended,
            ],
            dim=-1,
        )
        prediction = self.head(pooled).squeeze(-1)
        if return_attention:
            return prediction, fine_weights, plant_weights
        return prediction

    def parameter_summary(self):
        total = sum(item.numel() for item in self.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": total,
            "frozen_backbone_parameters_loaded_during_training": 0,
            "frozen_feature_dimension": self.feature_dim,
            "context_tiles_per_image": self.context_views - 1,
            "fine_tiles_per_image": self.fine_views - 1,
            "pooled_representations": (
                "global+3x3_mean+4x4_mean+4x4_attention+plant_mean+plant_attention"
            ),
        }


__all__ = ["HybridMILRegressor"]

