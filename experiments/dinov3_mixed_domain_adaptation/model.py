"""Frozen-teacher DINOv3 adaptation with LoRA across every transformer block."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import functional
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
NORM_PATHS = ("norm", "layernorm", "model.norm", "model.layernorm", "encoder.norm")


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
    candidates = [
        (module, path)
        for path, module in backbone.named_modules()
        if path.rsplit(".", 1)[-1] in {"blocks", "layer", "layers"}
        and isinstance(module, (nn.ModuleList, nn.Sequential))
        and len(module)
    ]
    if candidates:
        longest = max(len(module) for module, _ in candidates)
        winners = [item for item in candidates if len(item[0]) == longest]
        if len(winners) == 1:
            return winners[0]
    children = ", ".join(name for name, _ in backbone.named_children()) or "<none>"
    raise RuntimeError(
        "Could not locate DINOv3 transformer blocks. "
        f"Top-level modules: {children}; tried {', '.join(BLOCK_PATHS)}"
    )


def _find_final_norm(backbone: nn.Module) -> tuple[nn.Module, str]:
    for path in NORM_PATHS:
        value = _nested_module(backbone, path)
        if isinstance(value, nn.Module):
            return value, path
    raise RuntimeError(f"Could not locate DINOv3 final norm; tried {', '.join(NORM_PATHS)}")


def _targeted_linear_modules(backbone, targets) -> dict[str, list[str]]:
    matches = {target: [] for target in targets}
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Linear):
            for target in targets:
                if name == target or name.endswith(f".{target}"):
                    matches[target].append(name)
    return matches


def _representation(backbone, pixels: torch.Tensor) -> torch.Tensor:
    tokens = backbone(pixel_values=pixels).last_hidden_state
    cls_token = tokens[:, 0]
    config = (
        backbone.get_base_model().config if hasattr(backbone, "get_base_model") else backbone.config
    )
    registers = int(getattr(config, "num_register_tokens", 0))
    mean_patches = tokens[:, 1 + registers :].mean(dim=1)
    return torch.cat([cls_token, mean_patches], dim=-1)


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 1.0 - functional.cosine_similarity(left.float(), right.float(), dim=-1).mean()


class DinoV3DomainAdapter(nn.Module):
    def __init__(self, config: Config, *, include_teacher: bool = True):
        super().__init__()
        base_student = AutoModel.from_pretrained(config.model.backbone)
        blocks, self.block_path = _find_blocks(base_student)
        targets = tuple(config.model.lora_target_modules)
        matches = _targeted_linear_modules(base_student, targets)
        invalid = {
            target: len(names) for target, names in matches.items() if len(names) != len(blocks)
        }
        if invalid:
            raise RuntimeError(
                "LoRA target discovery expected one target per block. "
                f"blocks={len(blocks)}, invalid_counts={invalid}, matches={matches}"
            )
        base_student.requires_grad_(False)
        final_norm = None
        if config.model.train_final_norm:
            final_norm, self.final_norm_path = _find_final_norm(base_student)
        else:
            self.final_norm_path = None
        self.student = get_peft_model(
            base_student,
            LoraConfig(
                r=config.model.lora_rank,
                lora_alpha=config.model.lora_alpha,
                lora_dropout=config.model.lora_dropout,
                target_modules=list(targets),
                bias="none",
                inference_mode=False,
                init_lora_weights=True,
            ),
        )
        if final_norm is not None:
            final_norm.requires_grad_(True)
        self.teacher = None
        if include_teacher:
            self.teacher = AutoModel.from_pretrained(config.model.backbone)
            self.teacher.requires_grad_(False)
            self.teacher.eval()
        self.config = config
        self.target_module_names = [name for names in matches.values() for name in names]
        self._validate_trainable_parameters()

    def _validate_trainable_parameters(self) -> None:
        unexpected = [
            name
            for name, parameter in self.student.named_parameters()
            if parameter.requires_grad
            and "lora_" not in name
            and not (
                self.final_norm_path is not None and f".{self.final_norm_path}." in f".{name}."
            )
        ]
        if unexpected:
            raise RuntimeError("Unexpected trainable student parameters: " + ", ".join(unexpected))

    def representations(self, view_a: torch.Tensor, view_b: torch.Tensor):
        student = _representation(self.student, torch.cat([view_a, view_b], dim=0))
        student_a, student_b = student.chunk(2)
        if self.teacher is None:
            raise RuntimeError("The frozen teacher was omitted; training losses are unavailable")
        with torch.no_grad():
            teacher = _representation(self.teacher, torch.cat([view_a, view_b], dim=0))
            teacher_a, teacher_b = teacher.chunk(2)
        return student_a, student_b, teacher_a, teacher_b

    def losses(self, view_a: torch.Tensor, view_b: torch.Tensor):
        student_a, student_b, teacher_a, teacher_b = self.representations(view_a, view_b)
        cross = 0.5 * (
            cosine_distance(student_a, teacher_b) + cosine_distance(student_b, teacher_a)
        )
        anchor = 0.5 * (
            cosine_distance(student_a, teacher_a) + cosine_distance(student_b, teacher_b)
        )
        total = (
            self.config.objective.cross_view_weight * cross
            + self.config.objective.same_view_anchor_weight * anchor
        )
        diagnostics = {
            "loss": total,
            "cross_view_loss": cross,
            "anchor_loss": anchor,
            "student_view_cosine": functional.cosine_similarity(
                student_a.float(), student_b.float(), dim=-1
            ).mean(),
            "student_teacher_cosine": 1.0 - anchor,
            "feature_std": torch.cat([student_a.float(), student_b.float()], dim=0)
            .std(dim=0, unbiased=False)
            .mean(),
        }
        return diagnostics

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.student.eval()
            for name, module in self.student.named_modules():
                if "lora_dropout" in name:
                    module.train(True)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def adaptation_state_dict(self) -> dict[str, torch.Tensor]:
        complete = self.state_dict()
        trainable = {
            f"student.{name}"
            for name, parameter in self.student.named_parameters()
            if parameter.requires_grad
        }
        missing = sorted(trainable - set(complete))
        if missing:
            raise RuntimeError(
                "Trainable parameters missing from state_dict: " + ", ".join(missing)
            )
        return {name: complete[name] for name in sorted(trainable)}

    def load_adaptation_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        required = {
            f"student.{name}"
            for name, parameter in self.student.named_parameters()
            if parameter.requires_grad
        }
        missing = sorted(required - set(state))
        unexpected = sorted(set(state) - set(self.state_dict()))
        if missing or unexpected:
            raise ValueError(
                f"Adaptation checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError("Unexpected checkpoint keys: " + ", ".join(result.unexpected_keys))

    def parameter_summary(self) -> dict[str, object]:
        student_total = sum(parameter.numel() for parameter in self.student.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.student.parameters() if parameter.requires_grad
        )
        return {
            "student_parameters": student_total,
            "trainable_parameters": trainable,
            "trainable_percentage": 100.0 * trainable / student_total,
            "teacher_parameters": (
                sum(parameter.numel() for parameter in self.teacher.parameters())
                if self.teacher is not None
                else 0
            ),
            "target_module_count": len(self.target_module_names),
            "target_module_names": self.target_module_names,
            "block_path": self.block_path,
            "final_norm_path": self.final_norm_path,
            "representation": "cls+mean_patches",
        }


def make_optimizer(model: DinoV3DomainAdapter, config: Config):
    decay, no_decay = [], []
    for parameter in model.student.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    groups = []
    if decay:
        groups.append(
            {
                "params": decay,
                "weight_decay": config.training.weight_decay,
                "group_name": "adapter_decay",
            }
        )
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "group_name": "adapter_no_decay"})
    if not groups:
        raise RuntimeError("The adaptation model has no trainable parameters")
    return torch.optim.AdamW(groups, lr=config.training.learning_rate)
