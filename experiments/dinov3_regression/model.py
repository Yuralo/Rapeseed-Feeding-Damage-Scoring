"""The model architecture used by this experiment."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .config import Config


class DinoV3Regressor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.freeze_backbone = config.model.freeze_backbone
        self.backbone = AutoModel.from_pretrained(config.model.backbone)
        hidden_size = self.backbone.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, config.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.head_hidden_dim, 1),
        )
        self.backbone.requires_grad_(not self.freeze_backbone)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        cls_token = tokens[:, 0]
        registers = getattr(self.backbone.config, "num_register_tokens", 0)
        pooled_patches = tokens[:, 1 + registers :].mean(dim=1)
        return torch.cat([cls_token, pooled_patches], dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                features = self.extract_features(pixel_values)
        else:
            features = self.extract_features(pixel_values)
        return self.regression_head(features).squeeze(-1)


def make_optimizer(model: DinoV3Regressor, config: Config):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

