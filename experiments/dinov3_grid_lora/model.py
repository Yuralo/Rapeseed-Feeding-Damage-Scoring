"""DINOv3 regression with LoRA adapters across all transformer blocks."""

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


class DinoV3LoRARegressor(nn.Module):
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
        self.regression_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, config.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.head_hidden_dim, 1),
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

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        cls_token = tokens[:, 0]
        base_config = self.backbone.get_base_model().config
        registers = int(getattr(base_config, "num_register_tokens", 0))
        pooled_patches = tokens[:, 1 + registers :].mean(dim=1)
        return torch.cat([cls_token, pooled_patches], dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.extract_features(pixel_values)).squeeze(-1)

    def train(self, mode: bool = True):
        """Keep frozen backbone stochastic layers off while LoRA dropout remains active."""
        super().train(mode)
        if mode:
            self.backbone.eval()
            for name, module in self.backbone.named_modules():
                if "lora_dropout" in name:
                    module.train(True)
            self.regression_head.train(True)
        return self

    def adaptation_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only adapters, final norm, and regression head for compact checkpoints."""
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
                "LoRA checkpoint parameter mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError(
                "Unexpected LoRA checkpoint keys: " + ", ".join(result.unexpected_keys)
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
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_percentage": 100.0 * trainable / total,
            "adapter_parameters": adapter,
            "target_module_count": len(self.target_module_names),
            "target_module_names": self.target_module_names,
            "block_path": self.block_path,
            "final_norm_path": self.final_norm_path,
            "pooled_representations": "cls+mean_patches",
        }


def _optimizer_group(parameters, *, lr: float, weight_decay: float, name: str):
    return {
        "params": list(parameters),
        "lr": lr,
        "weight_decay": weight_decay,
        "group_name": name,
    }


def make_optimizer(model: DinoV3LoRARegressor, config: Config):
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        (scope, decay): [] for scope in ("adapter", "head") for decay in (False, True)
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scope = "head" if name.startswith("regression_head.") else "adapter"
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
        raise RuntimeError("The LoRA model has no trainable parameters")
    return torch.optim.AdamW(optimizer_groups)
