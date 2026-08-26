"""Global/context plus masked variable-instance attention regression head."""

import torch
from torch import nn

from .config import Config


class MaskedGatedAttention(nn.Module):
    def __init__(self, dim, hidden, dropout, temperature):
        super().__init__()
        self.value = nn.Linear(dim, hidden)
        self.control = nn.Linear(dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.temperature = temperature
        nn.init.zeros_(self.score.weight)

    def forward(self, instances, valid):
        if not valid.any(dim=1).all():
            raise ValueError("Every image requires at least one valid plant instance")
        gated = torch.tanh(self.value(instances)) * torch.sigmoid(self.control(instances))
        logits = self.score(self.dropout(gated)).squeeze(-1) / self.temperature
        logits = logits.float().masked_fill(~valid, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        attended = (weights.unsqueeze(-1) * instances.float()).sum(dim=1).to(instances.dtype)
        return attended, weights


class SamAdaptiveMILRegressor(nn.Module):
    def __init__(self, feature_dim: int, config: Config):
        super().__init__()
        projection = config.model.projection_dim
        self.feature_dim = feature_dim
        self.context_views = 1 + config.context.rows * config.context.columns
        self.input_norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection), nn.GELU(), nn.Dropout(config.model.dropout)
        )
        self.instance_attention = MaskedGatedAttention(
            projection,
            config.model.attention_hidden_dim,
            config.model.attention_dropout,
            config.model.attention_temperature,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(4 * projection),
            nn.Linear(4 * projection, config.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.head_hidden_dim, 1),
        )

    def forward(self, context, instances, valid, *, return_attention=False):
        if context.shape[1] != self.context_views or context.shape[-1] != self.feature_dim:
            raise ValueError("Unexpected context feature shape")
        if instances.shape[-1] != self.feature_dim or valid.shape != instances.shape[:2]:
            raise ValueError("Unexpected adaptive-instance feature/mask shape")
        context = self.projection(self.input_norm(context.float()))
        instances = self.projection(self.input_norm(instances.float()))
        valid_float = valid.unsqueeze(-1).float()
        instance_mean = (instances * valid_float).sum(1) / valid_float.sum(1).clamp_min(1)
        attended, weights = self.instance_attention(instances, valid)
        pooled = torch.cat([context[:, 0], context[:, 1:].mean(1), instance_mean, attended], dim=-1)
        prediction = self.head(pooled).squeeze(-1)
        return (prediction, weights) if return_attention else prediction

    def parameter_summary(self):
        total = sum(item.numel() for item in self.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": total,
            "frozen_backbone_parameters_loaded_during_training": 0,
            "frozen_feature_dimension": self.feature_dim,
            "pooled_representations": "global+3x3_mean+plant_mean+plant_attention",
        }
