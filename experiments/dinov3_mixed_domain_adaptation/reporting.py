"""Lightweight adaptation training plots."""

from __future__ import annotations

from pathlib import Path


def save_history_plot(history: dict[str, list], path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.reshape(-1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(history["val_epochs"], history["val_loss"], marker="o", label="validation")
    axes[0].set(title="Teacher-distillation objective", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(history["val_epochs"], history["val_cross_view_loss"], label="cross-view")
    axes[1].plot(history["val_epochs"], history["val_anchor_loss"], label="teacher anchor")
    axes[1].set(title="Validation loss components", xlabel="Epoch")
    axes[1].legend()
    axes[2].plot(
        history["val_epochs"], history["val_student_teacher_cosine"], label="student-teacher"
    )
    axes[2].plot(history["val_epochs"], history["val_student_view_cosine"], label="paired views")
    axes[2].set(title="Representation cosine", xlabel="Epoch", ylim=(0, 1.02))
    axes[2].legend()
    axes[3].plot(history["val_epochs"], history["val_feature_std"], label="feature std")
    axes[3].set(title="Collapse diagnostic", xlabel="Epoch", ylabel="Mean feature std")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
