"""DINOv3 regressor with only its final blocks and normalization trainable."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoModel

from .config import Config


BLOCK_PATHS = ("encoder.layer", "encoder.layers", "layers", "layer", "blocks")
NORM_PATHS = ("layernorm", "norm", "encoder.layernorm", "encoder.norm")


def _nested_module(root: nn.Module, path: str):
    value = root
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _find_blocks(backbone: nn.Module) -> tuple[Sequence[nn.Module], str]:
    for path in BLOCK_PATHS:
        value = _nested_module(backbone, path)
        if isinstance(value, (nn.ModuleList, nn.Sequential, list, tuple)) and len(value):
            return value, path
    children = ", ".join(name for name, _ in backbone.named_children()) or "<none>"
    raise RuntimeError(
        "Could not locate the DINOv3 transformer blocks. "
        f"Top-level modules are: {children}. Tried: {', '.join(BLOCK_PATHS)}"
    )


def _find_final_norm(backbone: nn.Module) -> tuple[nn.Module, str]:
    for path in NORM_PATHS:
        value = _nested_module(backbone, path)
        if isinstance(value, nn.Module):
            return value, path
    children = ", ".join(name for name, _ in backbone.named_children()) or "<none>"
    raise RuntimeError(
        "Could not locate the DINOv3 final normalization. "
        f"Top-level modules are: {children}. Tried: {', '.join(NORM_PATHS)}"
    )


class DinoV3Regressor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.unfreeze_last_n_blocks = config.model.unfreeze_last_n_blocks
        self.unfreeze_final_norm = config.model.unfreeze_final_norm
        self.backbone = AutoModel.from_pretrained(config.model.backbone)
        hidden_size = self.backbone.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, config.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.head_hidden_dim, 1),
        )

        blocks, self.block_path = _find_blocks(self.backbone)
        if self.unfreeze_last_n_blocks > len(blocks):
            raise ValueError(
                f"Requested {self.unfreeze_last_n_blocks} trainable blocks, "
                f"but {self.block_path} contains only {len(blocks)}"
            )
        self.backbone.requires_grad_(False)
        for block in blocks[len(blocks) - self.unfreeze_last_n_blocks :]:
            block.requires_grad_(True)
        if self.unfreeze_final_norm:
            final_norm, self.final_norm_path = _find_final_norm(self.backbone)
            final_norm.requires_grad_(True)
        else:
            self.final_norm_path = None

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        cls_token = tokens[:, 0]
        registers = getattr(self.backbone.config, "num_register_tokens", 0)
        pooled_patches = tokens[:, 1 + registers :].mean(dim=1)
        return torch.cat([cls_token, pooled_patches], dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.extract_features(pixel_values)).squeeze(-1)

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode while trainable tail modules train."""
        super().train(mode)
        if mode:
            self.backbone.eval()
            blocks, _ = _find_blocks(self.backbone)
            for block in blocks[len(blocks) - self.unfreeze_last_n_blocks :]:
                block.train(True)
            if self.unfreeze_final_norm:
                final_norm, _ = _find_final_norm(self.backbone)
                final_norm.train(True)
            self.regression_head.train(True)
        return self

    def parameter_summary(self) -> dict[str, int | float | str | None]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_percentage": 100.0 * trainable / total,
            "block_path": self.block_path,
            "final_norm_path": self.final_norm_path,
        }


def _optimizer_group(parameters, *, lr: float, weight_decay: float, name: str):
    values = list(parameters)
    return {"params": values, "lr": lr, "weight_decay": weight_decay, "group_name": name}


def make_optimizer(model: DinoV3Regressor, config: Config):
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        (scope, decay): [] for scope in ("backbone", "head") for decay in (False, True)
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scope = "head" if name.startswith("regression_head.") else "backbone"
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups[(scope, use_decay)].append(parameter)

    optimizer_groups = []
    for scope in ("backbone", "head"):
        learning_rate = getattr(config.training, f"{scope}_learning_rate")
        weight_decay = getattr(config.training, f"{scope}_weight_decay")
        if groups[(scope, True)]:
            optimizer_groups.append(
                _optimizer_group(
                    groups[(scope, True)],
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    name=f"{scope}_decay",
                )
            )
        if groups[(scope, False)]:
            optimizer_groups.append(
                _optimizer_group(
                    groups[(scope, False)],
                    lr=learning_rate,
                    weight_decay=0.0,
                    name=f"{scope}_no_decay",
                )
            )
    if not optimizer_groups:
        raise RuntimeError("The model has no trainable parameters")
    return torch.optim.AdamW(optimizer_groups)
