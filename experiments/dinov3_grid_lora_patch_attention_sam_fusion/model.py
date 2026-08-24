"""Shared-backbone DINOv3 LoRA with original, masked, and binary-mask fusion."""

from __future__ import annotations

from collections.abc import Sequence

from peft import LoraConfig, get_peft_model
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


def _targeted_linear_modules(
    backbone: nn.Module,
    targets: tuple[str, ...] | list[str],
) -> dict[str, list[str]]:
    matches = {target: [] for target in targets}
    for name, module in backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for target in targets:
            if name == target or name.endswith(f".{target}"):
                matches[target].append(name)
    return matches


class GatedPatchAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_size: int,
        dropout: float,
        temperature: float,
    ):
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


class MaskEncoder(nn.Module):
    def __init__(self, input_size: int, embedding_dim: int):
        super().__init__()
        self.input_size = input_size
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(64, embedding_dim)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError(f"Expected masks shaped [batch, 1, H, W], got {masks.shape}")
        return self.projection(self.features(masks).flatten(1))


class ResidualRepresentationFusion(nn.Module):
    """Fuse three representations while initially leaving the base prediction unchanged."""

    def __init__(
        self,
        image_dim: int,
        mask_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.image_projection = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
        )
        self.mask_projection = nn.Sequential(
            nn.LayerNorm(mask_dim),
            nn.Linear(mask_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        original_features: torch.Tensor,
        masked_features: torch.Tensor,
        mask_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representations = torch.stack(
            (
                self.image_projection(original_features),
                self.image_projection(masked_features),
                self.mask_projection(mask_features),
            ),
            dim=1,
        )
        gate_logits = self.gate(representations.flatten(1))
        gate_weights = torch.softmax(gate_logits.float(), dim=1)
        fused = torch.sum(
            gate_weights.unsqueeze(-1) * representations.float(),
            dim=1,
        ).to(representations.dtype)
        return self.delta_head(fused).squeeze(-1), gate_weights


class DinoV3LoRAPatchAttentionSamFusionRegressor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        base_backbone = AutoModel.from_pretrained(config.model.backbone)
        hidden_size = int(base_backbone.config.hidden_size)
        blocks, self.block_path = _find_blocks(base_backbone)
        targets = tuple(config.model.lora_target_modules)
        matches = _targeted_linear_modules(base_backbone, targets)
        invalid_counts = {
            target: len(names)
            for target, names in matches.items()
            if len(names) != len(blocks)
        }
        if invalid_counts:
            discovered = {target: names for target, names in matches.items()}
            raise RuntimeError(
                "LoRA target discovery did not find exactly one target per DINOv3 block. "
                f"blocks={len(blocks)}, counts={invalid_counts}, discovered={discovered}"
            )

        base_backbone.requires_grad_(False)
        final_norm = None
        if config.model.train_final_norm:
            final_norm, self.final_norm_path = _find_final_norm(base_backbone)
        else:
            self.final_norm_path = None
        adapter_config = LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=list(targets),
            bias="none",
            inference_mode=False,
            init_lora_weights=True,
        )
        self.backbone = get_peft_model(base_backbone, adapter_config)
        if final_norm is not None:
            final_norm.requires_grad_(True)

        self.target_module_names = [name for names in matches.values() for name in names]
        self.patch_attention = GatedPatchAttention(
            hidden_size,
            config.model.attention_hidden_dim,
            config.model.attention_dropout,
            config.model.attention_temperature,
        )
        feature_size = 3 * hidden_size
        self.regression_head = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, config.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.head_hidden_dim, 1),
        )
        self.mask_encoder = MaskEncoder(
            config.model.mask_input_size,
            config.model.mask_embedding_dim,
        )
        self.fusion = ResidualRepresentationFusion(
            feature_size,
            config.model.mask_embedding_dim,
            config.model.fusion_hidden_dim,
            config.model.fusion_dropout,
        )
        self._validate_trainable_backbone_parameters()

    def _validate_trainable_backbone_parameters(self) -> None:
        unexpected = [
            name
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
            and "lora_" not in name
            and not (
                self.final_norm_path is not None
                and f".{self.final_norm_path}." in f".{name}."
            )
        ]
        if unexpected:
            raise RuntimeError(
                "Unexpected fully trainable backbone parameters after LoRA injection: "
                + ", ".join(unexpected)
            )

    def extract_features(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        cls_token = tokens[:, 0]
        base_config = self.backbone.get_base_model().config
        registers = int(getattr(base_config, "num_register_tokens", 0))
        patch_tokens = tokens[:, 1 + registers :]
        mean_patches = patch_tokens.mean(dim=1)
        attended_patches, attention_weights = self.patch_attention(patch_tokens)
        features = torch.cat([cls_token, mean_patches, attended_patches], dim=-1)
        return features, attention_weights

    def forward(
        self,
        original_pixel_values: torch.Tensor,
        masked_pixel_values: torch.Tensor,
        mask_values: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ):
        original_features, original_attention = self.extract_features(
            original_pixel_values
        )
        masked_features, masked_attention = self.extract_features(masked_pixel_values)
        mask_features = self.mask_encoder(mask_values.float())
        base_predictions = self.regression_head(original_features).squeeze(-1)
        delta, fusion_weights = self.fusion(
            original_features,
            masked_features,
            mask_features,
        )
        predictions = base_predictions + delta
        if not return_diagnostics:
            return predictions
        return predictions, {
            "original_attention": original_attention,
            "masked_attention": masked_attention,
            "fusion_weights": fusion_weights,
            "base_predictions": base_predictions,
            "fusion_delta": delta,
        }

    def attention_grid(self, pixel_values: torch.Tensor, patch_count: int) -> tuple[int, int]:
        base_config = self.backbone.get_base_model().config
        patch_size = getattr(base_config, "patch_size", 16)
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
        """Keep frozen backbone stochastic layers off while LoRA dropout remains active."""
        super().train(mode)
        if mode:
            self.backbone.eval()
            for name, module in self.backbone.named_modules():
                if "lora_dropout" in name:
                    module.train(True)
            self.patch_attention.train(True)
            self.regression_head.train(True)
            self.mask_encoder.train(True)
            self.fusion.train(True)
        return self

    def adaptation_state_dict(self) -> dict[str, torch.Tensor]:
        """Return every trainable adaptation/fusion tensor for compact checkpoints."""
        trainable = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        complete = self.state_dict()
        missing = sorted(trainable - set(complete))
        if missing:
            raise RuntimeError(
                "Trainable parameters missing from state_dict: " + ", ".join(missing)
            )
        return {name: complete[name] for name in sorted(trainable)}

    def load_adaptation_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        required = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        missing = sorted(required - set(state))
        unexpected = sorted(set(state) - set(self.state_dict()))
        if missing or unexpected:
            raise ValueError(
                "SAM-fusion checkpoint parameter mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError(
                "Unexpected SAM-fusion checkpoint keys: "
                + ", ".join(result.unexpected_keys)
            )

    def parameter_summary(self) -> dict[str, int | float | str | list[str] | None]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        adapter = sum(
            parameter.numel()
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad and "lora_" in name
        )
        attention = sum(parameter.numel() for parameter in self.patch_attention.parameters())
        mask_encoder = sum(parameter.numel() for parameter in self.mask_encoder.parameters())
        fusion = sum(parameter.numel() for parameter in self.fusion.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_percentage": 100.0 * trainable / total,
            "adapter_parameters": adapter,
            "attention_parameters": attention,
            "mask_encoder_parameters": mask_encoder,
            "fusion_parameters": fusion,
            "target_module_count": len(self.target_module_names),
            "target_module_names": self.target_module_names,
            "block_path": self.block_path,
            "final_norm_path": self.final_norm_path,
            "pooled_representations": (
                "original(cls+mean+attention)+masked(cls+mean+attention)+binary_mask"
            ),
        }


def _optimizer_group(parameters, *, lr: float, weight_decay: float, name: str):
    return {
        "params": list(parameters),
        "lr": lr,
        "weight_decay": weight_decay,
        "group_name": name,
    }


def make_optimizer(
    model: DinoV3LoRAPatchAttentionSamFusionRegressor,
    config: Config,
):
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        (scope, decay): [] for scope in ("adapter", "head") for decay in (False, True)
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scope = (
            "head"
            if name.startswith(
                ("regression_head.", "patch_attention.", "mask_encoder.", "fusion.")
            )
            else "adapter"
        )
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups[(scope, use_decay)].append(parameter)

    optimizer_groups = []
    for scope in ("adapter", "head"):
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
        raise RuntimeError("The SAM-fusion LoRA model has no trainable parameters")
    return torch.optim.AdamW(optimizer_groups)
