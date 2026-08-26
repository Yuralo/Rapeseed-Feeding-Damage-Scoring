"""Train mixed-source DINOv3 LoRA adaptation against a frozen teacher."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoImageProcessor

from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .checkpoint import payload, validate_for
from .config import Config, load_config
from .data import load_prepared_manifest, make_loaders, split_records
from .model import DinoV3DomainAdapter, make_optimizer
from .reporting import save_history_plot
from .runtime import (
    autocast_context,
    configure_acceleration,
    learning_rates,
    make_grad_scaler,
    make_scheduler,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
METRIC_KEYS = (
    "loss",
    "cross_view_loss",
    "anchor_loss",
    "student_view_cosine",
    "student_teacher_cosine",
    "feature_std",
)


def empty_history() -> dict[str, list]:
    return {
        "train_loss": [],
        "train_cross_view_loss": [],
        "train_anchor_loss": [],
        "val_epochs": [],
        "val_loss": [],
        "val_cross_view_loss": [],
        "val_anchor_loss": [],
        "val_student_view_cosine": [],
        "val_student_teacher_cosine": [],
        "val_feature_std": [],
        "learning_rates": [],
        "epoch_seconds": [],
        "peak_cuda_memory_gb": [],
    }


def _move(batch, device):
    return (
        batch["view_a"].to(device, non_blocking=True),
        batch["view_b"].to(device, non_blocking=True),
    )


def evaluate(model, loader, config: Config, device) -> dict[str, float]:
    model.eval()
    totals = Counter()
    samples = 0
    with torch.no_grad():
        for batch in loader:
            view_a, view_b = _move(batch, device)
            with autocast_context(config, device):
                metrics = model.losses(view_a, view_b)
            batch_size = view_a.shape[0]
            for key in METRIC_KEYS:
                totals[key] += float(metrics[key].detach().item()) * batch_size
            totals["label_overlap_fraction"] += float(batch["label_overlap_fraction"].sum().item())
            samples += batch_size
    if not samples:
        raise ValueError("Validation loader is empty")
    return {key: float(value / samples) for key, value in totals.items()} | {"samples": samples}


def _checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    grad_scaler,
    epoch,
    metrics,
    config,
    training,
    validation,
    history,
    best_loss,
    global_step,
    patience,
    stopped_early,
    environment,
):
    return payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=grad_scaler,
        epoch=epoch,
        metrics=metrics,
        config=config,
        training_ids=[row["image_id"] for row in training],
        validation_ids=[row["image_id"] for row in validation],
        history=history,
        best_validation_loss=best_loss,
        training_state={
            "global_step": global_step,
            "evaluations_without_improvement": patience,
            "stopped_early": stopped_early,
        },
        environment=environment,
    )


def run(config: Config, resume: str | Path | None = None) -> dict:
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    configure_acceleration(config, device)
    state = load_checkpoint(resume, device) if resume else None
    if state:
        validate_for(state, config)
    rows = load_prepared_manifest(config)
    training, validation = split_records(
        rows,
        config,
        training_ids=(state or {}).get("training_ids"),
        validation_ids=(state or {}).get("validation_ids"),
    )
    processor = AutoImageProcessor.from_pretrained(config.model.processor)
    train_loader, validation_loader = make_loaders(training, validation, processor, config)
    model = DinoV3DomainAdapter(config).to(device)
    optimizer = make_optimizer(model, config)
    scheduler, total_steps, warmup_steps = make_scheduler(optimizer, config, len(train_loader))
    grad_scaler = make_grad_scaler(config, device)

    start_epoch, best_loss, global_step, patience = 1, float("inf"), 0, 0
    history = empty_history()
    if state:
        model.load_adaptation_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("grad_scaler_state_dict"):
            grad_scaler.load_state_dict(state["grad_scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state["best_validation_loss"])
        history.update(state.get("history") or {})
        training_state = state.get("training_state") or {}
        global_step = int(training_state.get("global_step", 0))
        patience = int(training_state.get("evaluations_without_improvement", 0))
        print(f"Resuming at epoch {start_epoch} from {resume}", flush=True)

    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_info(device, REPOSITORY_ROOT)
    parameters = model.parameter_summary()
    split_summary = {
        "total": len(rows),
        "training": len(training),
        "validation": len(validation),
        "modes": dict(Counter(row["preprocessing_mode"] for row in rows)),
        "training_cohorts": dict(Counter(row["cohort_id"] for row in training)),
        "validation_cohorts": dict(Counter(row["cohort_id"] for row in validation)),
    }
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "model_parameters.json", parameters)
    write_json(run_dir / "split_summary.json", split_summary)
    print(
        f"Prepared images={len(rows)} | train={len(training)} | val={len(validation)} | "
        f"modes={split_summary['modes']}",
        flush=True,
    )
    print(
        f"Trainable student parameters={parameters['trainable_parameters']:,} "
        f"({parameters['trainable_percentage']:.3f}%) | total optimizer steps={total_steps} | "
        f"warmup={warmup_steps}",
        flush=True,
    )
    effective_batch = config.training.batch_size * config.training.gradient_accumulation_steps
    print(f"Effective batch size={effective_batch} | device={device}", flush=True)

    best_path = run_dir / config.output.best_checkpoint_name
    last_path = run_dir / config.output.last_checkpoint_name
    stopped_early = False
    for epoch in range(start_epoch, config.training.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = Counter()
        samples = optimizer_steps = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        for batch_index, batch in enumerate(train_loader):
            view_a, view_b = _move(batch, device)
            accumulation = config.training.gradient_accumulation_steps
            window_start = batch_index - batch_index % accumulation
            window_size = min(accumulation, len(train_loader) - window_start)
            with autocast_context(config, device):
                metrics = model.losses(view_a, view_b)
                backward_loss = metrics["loss"] / window_size
            grad_scaler.scale(backward_loss).backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(
                train_loader
            )
            if should_step:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    config.training.gradient_clip_norm,
                )
                scale_before = grad_scaler.get_scale()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                if grad_scaler.get_scale() >= scale_before:
                    scheduler.step()
                    global_step += 1
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
            batch_size = view_a.shape[0]
            for key in METRIC_KEYS:
                totals[key] += float(metrics[key].detach().item()) * batch_size
            samples += batch_size
        epoch_seconds = perf_counter() - started
        train_metrics = {key: float(totals[key] / samples) for key in METRIC_KEYS}
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_cross_view_loss"].append(train_metrics["cross_view_loss"])
        history["train_anchor_loss"].append(train_metrics["anchor_loss"])
        history["learning_rates"].append(learning_rates(optimizer))
        history["epoch_seconds"].append(epoch_seconds)
        history["peak_cuda_memory_gb"].append(peak_memory)
        message = (
            f"Epoch {epoch:03d}/{config.training.epochs} | train={train_metrics['loss']:.5f} "
            f"| {epoch_seconds:.1f}s | {samples / epoch_seconds:.1f} images/s | "
            f"peak={peak_memory:.2f} GB | steps={optimizer_steps}"
        )
        should_evaluate = epoch % config.training.eval_every == 0 or epoch == config.training.epochs
        if should_evaluate:
            metrics = evaluate(model, validation_loader, config, device)
            history["val_epochs"].append(epoch)
            for key in (
                "loss",
                "cross_view_loss",
                "anchor_loss",
                "student_view_cosine",
                "student_teacher_cosine",
                "feature_std",
            ):
                history[f"val_{key}"].append(metrics[key])
            improved = metrics["loss"] < best_loss - config.training.early_stopping_min_delta
            if improved:
                best_loss, patience = metrics["loss"], 0
            else:
                patience += 1
            stopped_early = patience >= config.training.early_stopping_patience
            checkpoint = _checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_scaler=grad_scaler,
                epoch=epoch,
                metrics=metrics,
                config=config,
                training=training,
                validation=validation,
                history=history,
                best_loss=best_loss,
                global_step=global_step,
                patience=patience,
                stopped_early=stopped_early,
                environment=environment,
            )
            save_checkpoint(last_path, checkpoint)
            if improved:
                save_checkpoint(best_path, checkpoint)
            message += (
                f" | val={metrics['loss']:.5f} | teacher cosine="
                f"{metrics['student_teacher_cosine']:.4f} | feature std="
                f"{metrics['feature_std']:.4f} | patience="
                f"{patience}/{config.training.early_stopping_patience}"
            )
        print(message, flush=True)
        write_json(run_dir / "history.json", history)
        if config.output.save_plots:
            save_history_plot(history, run_dir / "training_history.png")
        if stopped_early:
            print(f"Early stopping after epoch {epoch}; best validation loss={best_loss:.5f}")
            break

    if not best_path.is_file():
        raise RuntimeError("Training did not produce a best checkpoint")
    best_state = load_checkpoint(best_path, device)
    model.load_adaptation_state_dict(best_state["model_state_dict"])
    final_metrics = evaluate(model, validation_loader, config, device)
    summary = {
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "train_samples": len(training),
        "validation_samples": len(validation),
        "completed_epochs": len(history["train_loss"]),
        "global_optimizer_steps": global_step,
        "stopped_early": stopped_early,
        "validation": final_metrics,
        "model_parameters": parameters,
        "next_step": "Export the best checkpoint, then use that directory as backbone and processor.",
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume")
    mode.add_argument("--from-scratch", action="store_true")
    arguments = parser.parse_args(argv)
    report = run(load_config(arguments.config), arguments.resume)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
