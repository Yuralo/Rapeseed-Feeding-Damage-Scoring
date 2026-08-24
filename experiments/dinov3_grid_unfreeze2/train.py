"""Train the grid-cropped final-two-block DINOv3 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint, save_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .checkpoint import payload, validate_for
from .config import Config, load_config
from .metrics import predict
from .model import DinoV3Regressor, make_optimizer
from .reporting import save_evaluation, save_history_plot, save_label_plot
from .runtime import (
    autocast_context,
    configure_acceleration,
    learning_rates,
    make_grad_scaler,
    make_scheduler,
)
from .setup import prepare_data

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def empty_history() -> dict[str, list]:
    return {
        "train_loss": [],
        "val_epochs": [],
        "val_loss": [],
        "val_mae": [],
        "val_r2": [],
        "learning_rates": [],
        "epoch_seconds": [],
        "data_seconds": [],
        "compute_seconds": [],
        "samples_per_second": [],
        "peak_cuda_memory_gb": [],
        "optimizer_steps": [],
    }


def restore_history(state) -> dict[str, list]:
    history = empty_history()
    for key, values in (state.get("history") or {}).items():
        if key in history:
            history[key] = values
    return history


def _checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    grad_scaler,
    epoch,
    metrics,
    target_scaler,
    config,
    train_names,
    validation_names,
    history,
    best_loss,
    global_step,
    evaluations_without_improvement,
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
        scaler=target_scaler,
        config=config,
        training_filenames=train_names,
        validation_filenames=validation_names,
        history=history,
        best_validation_loss=best_loss,
        training_state={
            "global_step": global_step,
            "evaluations_without_improvement": evaluations_without_improvement,
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

    table, train, validation, target_scaler, train_loader, validation_loader = prepare_data(
        config, state
    )
    model = DinoV3Regressor(config).to(device)
    optimizer = make_optimizer(model, config)
    scheduler, total_optimizer_steps, warmup_steps = make_scheduler(
        optimizer, config, len(train_loader)
    )
    grad_scaler = make_grad_scaler(config, device)

    start_epoch, history, best_loss = 1, empty_history(), float("inf")
    global_step, evaluations_without_improvement = 0, 0
    if state:
        model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict"):
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("grad_scaler_state_dict"):
            grad_scaler.load_state_dict(state["grad_scaler_state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        history = restore_history(state)
        best_loss = float(state.get("best_validation_loss", state.get("val_loss", best_loss)))
        training_state = state.get("training_state") or {}
        global_step = int(training_state.get("global_step", 0))
        evaluations_without_improvement = int(
            training_state.get("evaluations_without_improvement", 0)
        )
        print(f"Resuming at epoch {start_epoch} from {resume}", flush=True)

    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_environment = environment_info(device, REPOSITORY_ROOT)
    parameter_summary = model.parameter_summary()
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "environment.json", run_environment)
    write_json(run_dir / "model_parameters.json", parameter_summary)
    print(
        f"Trainable: {parameter_summary['trainable_parameters']:,} / "
        f"{parameter_summary['total_parameters']:,} "
        f"({parameter_summary['trainable_percentage']:.2f}%) | "
        f"blocks={parameter_summary['block_path']} | norm={parameter_summary['final_norm_path']}",
        flush=True,
    )
    print(
        f"Effective batch size: "
        f"{config.training.batch_size * config.training.gradient_accumulation_steps} | "
        f"optimizer steps: {total_optimizer_steps} | warmup steps: {warmup_steps}",
        flush=True,
    )
    print(f"Grid crop cache: {Path(config.data.grid_cache_dir).resolve()}", flush=True)
    target_mode = (
        f"z-score (mean={target_scaler.mean:.4f}, std={target_scaler.std:.4f})"
        if target_scaler.enabled
        else "raw scores (identity transform)"
    )
    print(f"Target processing: {target_mode}", flush=True)
    print(
        f"Grid preprocessing failures: {run_dir / config.output.grid_failure_log}", flush=True
    )
    if config.output.save_plots:
        save_label_plot(table[config.data.target_column].to_numpy(), run_dir / "targets.png")

    train_names = train[config.data.filename_column].astype(str).tolist()
    validation_names = validation[config.data.filename_column].astype(str).tolist()
    best_path = run_dir / config.output.best_checkpoint_name
    last_path = run_dir / config.output.last_checkpoint_name
    best_saved = best_path.is_file()
    stopped_early = False

    for epoch in range(start_epoch, config.training.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, samples, epoch_optimizer_steps = 0.0, 0, 0
        data_seconds, compute_seconds = 0.0, 0.0
        profile_epoch = epoch <= config.runtime.profile_first_n_epochs
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        epoch_started = perf_counter()
        previous_batch_finished = epoch_started

        for batch_index, batch in enumerate(train_loader):
            batch_ready = perf_counter()
            data_seconds += batch_ready - previous_batch_finished
            if profile_epoch and device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_started = perf_counter()

            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = batch["target"].float().to(device, non_blocking=True)
            accumulation = config.training.gradient_accumulation_steps
            window_start = batch_index - batch_index % accumulation
            window_size = min(accumulation, len(train_loader) - window_start)
            with autocast_context(config, device):
                predictions = model(pixels)
                raw_loss = torch.nn.functional.mse_loss(predictions.float(), targets)
                backward_loss = raw_loss / window_size
            grad_scaler.scale(backward_loss).backward()

            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
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
                step_was_skipped = grad_scaler.get_scale() < scale_before
                if not step_was_skipped:
                    scheduler.step()
                    global_step += 1
                    epoch_optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)

            total_loss += raw_loss.detach().item() * targets.numel()
            samples += targets.numel()
            if profile_epoch and device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_finished = perf_counter()
            compute_seconds += batch_finished - compute_started
            previous_batch_finished = batch_finished

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = perf_counter() - epoch_started
        train_loss = total_loss / samples
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        )
        history["train_loss"].append(float(train_loss))
        history["learning_rates"].append(learning_rates(optimizer))
        history["epoch_seconds"].append(float(epoch_seconds))
        history["data_seconds"].append(float(data_seconds) if profile_epoch else None)
        history["compute_seconds"].append(float(compute_seconds) if profile_epoch else None)
        history["samples_per_second"].append(float(samples / epoch_seconds))
        history["peak_cuda_memory_gb"].append(float(peak_memory_gb))
        history["optimizer_steps"].append(epoch_optimizer_steps)
        message = (
            f"Epoch {epoch:03d}/{config.training.epochs} | train {train_loss:.5f} | "
            f"{epoch_seconds:.1f}s | {samples / epoch_seconds:.1f} samples/s | "
            f"peak {peak_memory_gb:.2f} GB"
        )

        should_evaluate = epoch % config.training.eval_every == 0 or epoch == config.training.epochs
        if should_evaluate:
            result = predict(model, validation_loader, device, target_scaler, config)
            metrics = result.metrics()
            history["val_epochs"].append(epoch)
            history["val_loss"].append(metrics["objective_mse"])
            history["val_mae"].append(metrics["mae"])
            history["val_r2"].append(metrics["r2"])
            improved = (
                metrics["objective_mse"]
                < best_loss - config.training.early_stopping_min_delta
            )
            if improved:
                best_loss = metrics["objective_mse"]
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            stopped_early = (
                evaluations_without_improvement >= config.training.early_stopping_patience
            )
            checkpoint = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_scaler=grad_scaler,
                epoch=epoch,
                metrics=metrics,
                target_scaler=target_scaler,
                config=config,
                train_names=train_names,
                validation_names=validation_names,
                history=history,
                best_loss=best_loss,
                global_step=global_step,
                evaluations_without_improvement=evaluations_without_improvement,
                stopped_early=stopped_early,
                environment=run_environment,
            )
            save_checkpoint(last_path, checkpoint)
            if improved:
                save_checkpoint(best_path, checkpoint)
                best_saved = True
            message += (
                f" | val objective MSE {metrics['objective_mse']:.5f}"
                f" | MAE {metrics['mae']:.3f} | R² {metrics['r2']:.3f}"
                f" | patience {evaluations_without_improvement}/"
                f"{config.training.early_stopping_patience}"
            )
        print(message, flush=True)
        write_json(run_dir / "history.json", history)
        if config.output.save_plots:
            save_history_plot(history, run_dir / "training_history.png")
        if stopped_early:
            print(f"Early stopping after epoch {epoch}; best val loss {best_loss:.5f}", flush=True)
            break

    if best_saved:
        evaluation_checkpoint = best_path
    elif resume:
        previous_best = Path(resume).parent / config.output.best_checkpoint_name
        evaluation_checkpoint = previous_best if previous_best.is_file() else Path(resume)
    else:
        raise RuntimeError("Training completed without producing a best checkpoint")
    best_state = load_checkpoint(evaluation_checkpoint, device)
    model.load_state_dict(best_state["model_state_dict"])
    report = save_evaluation(
        predict(model, validation_loader, device, target_scaler, config),
        target_scaler,
        run_dir,
        config,
    )
    report.update(
        {
            "best_checkpoint": str(evaluation_checkpoint),
            "last_checkpoint": str(last_path if last_path.is_file() else resume),
            "train_samples": len(train),
            "validation_samples": len(validation),
            "device": str(device),
            "stopped_early": stopped_early,
            "completed_epochs": len(history["train_loss"]),
            "global_optimizer_steps": global_step,
            "model_parameters": parameter_summary,
        }
    )
    write_json(run_dir / "summary.json", report)
    return report


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
