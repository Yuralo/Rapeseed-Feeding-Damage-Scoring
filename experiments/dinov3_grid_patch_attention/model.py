"""DINOv3 regression with gated attention over final-layer patch tokens."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoModel

from .config import Config


BLOCK_PATHS = (
    "model",
    "model.layers",
    "model.layer",
    "model.blocks",
    "model.encoder.layers",
    "model.encoder.layer",
    "encoder.layers",
    "encoder.layer",
    "layers",
    "layer",
    "blocks",
)
NORM_PATHS = (
    "norm",
    "layernorm",
    "model.norm",
    "model.layernorm",
    "encoder.norm",
    "encoder.layernorm",
)
BLOCK_CONTAINER_NAMES = {"blocks", "layer", "layers"}


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
    candidates = []
    for path, module in backbone.named_modules():
        if (
            path
            and path.rsplit(".", 1)[-1] in BLOCK_CONTAINER_NAMES
            and isinstance(module, (nn.ModuleList, nn.Sequential))
            and len(module)
        ):
            candidates.append((module, path))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        longest = max(len(module) for module, _ in candidates)
        longest_candidates = [item for item in candidates if len(item[0]) == longest]
        if len(longest_candidates) == 1:
            return longest_candidates[0]
    children = ", ".join(name for name, _ in backbone.named_children()) or "<none>"
    discovered = ", ".join(f"{path} ({len(module)})" for module, path in candidates)
    raise RuntimeError(
        "Could not locate the DINOv3 transformer blocks. "
        f"Top-level modules are: {children}. Tried: {', '.join(BLOCK_PATHS)}. "
        f"Recursively discovered block-like containers: {discovered or '<none>'}"
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


class GatedPatchAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_size: int, dropout: float, temperature: float):
        super().__init__()
        self.value_gate = nn.Linear(hidden_size, attention_size)
        self.control_gate = nn.Linear(hidden_size, attention_size)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(attention_size, 1, bias=False)
        self.temperature = temperature
        nn.init.zeros_(self.score.weight)

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated = torch.tanh(self.value_gate(patch_tokens)) * torch.sigmoid(
            self.control_gate(patch_tokens)
        )
        logits = self.score(self.dropout(gated)).squeeze(-1) / self.temperature
        weights = torch.softmax(logits.float(), dim=1)
        attended = torch.sum(weights.unsqueeze(-1) * patch_tokens.float(), dim=1).to(
            patch_tokens.dtype
        )
        return attended, weights


class DinoV3PatchAttentionRegressor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.unfreeze_last_n_blocks = config.model.unfreeze_last_n_blocks
        self.unfreeze_final_norm = config.model.unfreeze_final_norm
        self.backbone = AutoModel.from_pretrained(config.model.backbone)
        hidden_size = self.backbone.config.hidden_size
        self.patch_attention = GatedPatchAttention(
            hidden_size,
            config.model.attention_hidden_dim,
            config.model.attention_dropout,
            config.model.attention_temperature,
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(3 * hidden_size),
            nn.Linear(3 * hidden_size, config.model.head_hidden_dim),
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

    def extract_features(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        cls_token = tokens[:, 0]
        registers = getattr(self.backbone.config, "num_register_tokens", 0)
        patch_tokens = tokens[:, 1 + registers :]
        mean_patches = patch_tokens.mean(dim=1)
        attended_patches, attention_weights = self.patch_attention(patch_tokens)
        features = torch.cat([cls_token, mean_patches, attended_patches], dim=-1)
        return features, attention_weights

    def forward(self, pixel_values: torch.Tensor, *, return_attention: bool = False):
        features, attention_weights = self.extract_features(pixel_values)
        predictions = self.regression_head(features).squeeze(-1)
        return (predictions, attention_weights) if return_attention else predictions

    def attention_grid(self, pixel_values: torch.Tensor, patch_count: int) -> tuple[int, int]:
        patch_size = getattr(self.backbone.config, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            patch_height, patch_width = map(int, patch_size)
        else:
            patch_height = patch_width = int(patch_size)
        rows = pixel_values.shape[-2] // patch_height
        columns = pixel_values.shape[-1] // patch_width
        if rows * columns != patch_count:
            side = int(round(patch_count**0.5))
            if side * side != patch_count:
                raise RuntimeError(
                    f"Cannot map {patch_count} patch tokens to an image grid; "
                    f"processor tensor is {tuple(pixel_values.shape[-2:])}, "
                    f"patch size is {patch_size}"
                )
            rows = columns = side
        return rows, columns

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.backbone.eval()
            blocks, _ = _find_blocks(self.backbone)
            for block in blocks[len(blocks) - self.unfreeze_last_n_blocks :]:
                block.train(True)
            if self.unfreeze_final_norm:
                final_norm, _ = _find_final_norm(self.backbone)
                final_norm.train(True)
            self.patch_attention.train(True)
            self.regression_head.train(True)
        return self

    def parameter_summary(self) -> dict[str, int | float | str | None]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        attention = sum(parameter.numel() for parameter in self.patch_attention.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_percentage": 100.0 * trainable / total,
            "attention_parameters": attention,
            "block_path": self.block_path,
            "final_norm_path": self.final_norm_path,
            "pooled_representations": "cls+mean_patches+attention_patches",
        }


def _optimizer_group(parameters, *, lr: float, weight_decay: float, name: str):
    return {
        "params": list(parameters),
        "lr": lr,
        "weight_decay": weight_decay,
        "group_name": name,
    }


def make_optimizer(model: DinoV3PatchAttentionRegressor, config: Config):
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        (scope, decay): [] for scope in ("backbone", "head") for decay in (False, True)
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scope = (
            "head"
            if name.startswith(("patch_attention.", "regression_head."))
            else "backbone"
        )
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
