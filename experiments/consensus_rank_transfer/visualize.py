"""Research figures and image-level QA contact sheets for every available split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .engine import candidate_pairs, predict, target_weak_pool


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _contact_sheet(rows: list[dict], path: Path, title: str, *, columns: int = 4) -> int:
    from PIL import Image, ImageDraw, ImageOps

    width, image_height, label_height = 340, 235, 75
    if not rows:
        return 0
    count = min(len(rows), 12)
    height = ((count + columns - 1) // columns) * (image_height + label_height) + 38
    canvas = Image.new("RGB", (columns * width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black")
    loaded = 0
    for index, row in enumerate(rows[:count]):
        source = Path(row["source_path"])
        if not source.is_file():
            continue
        with Image.open(source) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (width - 12, image_height - 10))
        x = (index % columns) * width
        y = 38 + (index // columns) * (image_height + label_height)
        canvas.paste(image, (x + 5, y + 5))
        label = (f"target {float(row['target']):.1f}  base {float(row['base_prediction']):.1f}\n"
                 f"rank {float(row['rank_mean_prediction']):.1f}  "
                 f"ΔMAE {float(row['base_absolute_error'])-float(row['rank_mean_absolute_error']):+.1f}\n"
                 f"{Path(row['filename']).name}")
        draw.multiline_text((x + 5, y + image_height + 2), label, fill="black")
        loaded += 1
    if loaded:
        canvas.save(path)
    return loaded


def _plot_split(rows: list[dict], split: str, destination: Path) -> dict:
    import matplotlib.pyplot as plt

    target = np.array([float(row["target"]) for row in rows])
    base = np.array([float(row["base_prediction"]) for row in rows])
    rank = np.array([float(row["rank_mean_prediction"]) for row in rows])
    groups = np.array([row["plot_group_id"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    axes[0].scatter(target, base, s=18, alpha=.65, label="frozen base")
    axes[0].scatter(target, rank, s=18, alpha=.65, label="rank")
    axes[0].plot([0, max(target.max(), rank.max())], [0, max(target.max(), rank.max())],
                 color="black", linewidth=1)
    axes[0].set(xlabel="Reference score", ylabel="Predicted score", title=f"{split}: score calibration")
    axes[0].legend()
    bins = ((0, 2.5), (2.5, 7.5), (7.5, 15), (15, float("inf")))
    labels = ("0–2.5", "2.5–7.5", "7.5–15", ">15")
    positions = np.arange(len(bins))
    for offset, key, name in ((-.2, "base_prediction", "base"), (.2, "rank_mean_prediction", "rank")):
        errors = []
        for lower, upper in bins:
            mask = (target >= lower if lower == 0 else target > lower) & (target <= upper)
            values = np.array([float(row[key]) for row in rows])
            errors.append(float(np.mean(np.abs(values[mask] - target[mask]))) if mask.any() else np.nan)
        axes[1].bar(positions + offset, errors, width=.38, label=name)
    axes[1].set_xticks(positions, labels)
    axes[1].set(ylabel="Image MAE", title="Severity bands")
    axes[1].legend()
    delta = np.array([np.mean(np.abs(base[groups == group] - target[groups == group])) -
                      np.mean(np.abs(rank[groups == group] - target[groups == group]))
                      for group in np.unique(groups)])
    axes[2].bar(np.arange(len(delta)), np.sort(delta), color=np.where(np.sort(delta) >= 0,
                                                                     "#3c8d66", "#c55a5a"))
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set(xlabel="Plot groups sorted by benefit", ylabel="Base − rank plot MAE",
                title="Plot-level paired changes")
    axes[3].scatter(target - base, rank - base, s=18, alpha=.7)
    axes[3].axhline(0, color="black", linewidth=1)
    axes[3].axvline(0, color="black", linewidth=1)
    axes[3].set(xlabel="Correction needed (target − base)",
                ylabel="Correction applied (rank − base)", title="Does the correction follow error?")
    fig.tight_layout()
    fig.savefig(destination / f"{split}_diagnostics.png", dpi=160)
    plt.close(fig)
    improvement = np.abs(base - target) - np.abs(rank - target)
    best = [rows[i] for i in np.argsort(improvement)[-12:][::-1]]
    worst = [rows[i] for i in np.argsort(improvement)[:12]]
    return {"improvements_loaded": _contact_sheet(best, destination / f"{split}_improvements.png",
                                                  f"{split}: largest improvements"),
            "regressions_loaded": _contact_sheet(worst, destination / f"{split}_regressions.png",
                                                 f"{split}: largest regressions")}


def _pair_violations(config: Config, weak, destination: Path) -> dict:
    from PIL import Image, ImageDraw, ImageOps

    from rapeseed_damage.reproducibility import resolve_device

    from .evaluate import _load_model

    device = str(resolve_device("auto"))
    model = _load_model(config, "rank", config.seeds[0], device)
    prediction, _ = predict(model, weak, device)
    hi, lo, _, info = candidate_pairs(weak, np.arange(len(weak)), config,
                                      config.seeds[0], shuffled=False)
    shortfall = config.rank_prediction_margin - (prediction[hi] - prediction[lo])
    selected = np.argsort(shortfall)[-8:][::-1]
    canvas = Image.new("RGB", (900, 8 * 170 + 35), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Largest qualified-order violations, first rank seed", fill="black")
    loaded = 0
    for pair_index in selected:
        if shortfall[pair_index] <= 0:
            continue
        y = 35 + loaded * 170
        for column, index in enumerate((hi[pair_index], lo[pair_index])):
            source = Path(str(weak.source_path[index]))
            if not source.is_file():
                continue
            with Image.open(source) as opened:
                picture = ImageOps.contain(opened.convert("RGB"), (310, 130))
            x = column * 445
            canvas.paste(picture, (x + 5, y))
            draw.text((x + 5, y + 133),
                      f"{'higher' if column == 0 else 'lower'}: "
                      f"[{min(weak.score_jlu[index], weak.score_gau[index]):.1f}, "
                      f"{max(weak.score_jlu[index], weak.score_gau[index]):.1f}] "
                      f"pred {prediction[index]:.1f}", fill="black")
        loaded += 1
    if loaded:
        canvas.crop((0, 0, 900, 35 + loaded * 170)).save(destination / "pair_violations.png")
    return {"qualified_pairs_sampled": info["sampled_pairs"],
            "violations": int(np.sum(shortfall > 0)), "examples_loaded": loaded}


def run(config: Config) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(config.run_dir) / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    selection = json.loads((Path(config.run_dir) / "cv_selection.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in ("gold_only", "midpoint", "rank"):
        choice = selection["selected"][arm]
        ax.plot(choice["mean_curve"], label=f"{arm} (selected epoch {choice['epoch']})")
    ax.axhline(selection["base_cv_plot_mae"], color="black", linestyle="--", label="frozen base")
    ax.set(xlabel="Epoch", ylabel="Gold train-CV plot MAE", title="Grouped cross-validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "cv_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, color in (("gold_only", "#3b79a2"), ("midpoint", "#dc9b40"),
                       ("rank", "#398c61"), ("shuffled", "#a05596")):
        curves = []
        for seed in config.seeds:
            path = Path(config.run_dir) / "fits" / arm / f"seed_{seed}" / "history.json"
            if not path.is_file():
                continue
            history = json.loads(path.read_text())
            curve = [entry["train_plot_mae"] for entry in history if entry["epoch"] > 0]
            if curve:
                ax.plot(np.arange(1, len(curve) + 1), curve, color=color, alpha=.18)
                curves.append(curve)
        if curves:
            ax.plot(np.arange(1, len(curves[0]) + 1), np.mean(curves, axis=0),
                    color=color, linewidth=2, label=arm)
    ax.set(xlabel="Epoch", ylabel="Gold training plot MAE", title="Matched full-training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "fit_curves.png", dpi=160)
    plt.close(fig)

    weak = target_weak_pool(config.run_dir, config)
    gaps = np.abs(weak.score_jlu - weak.score_gau)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(gaps, bins=25, color="#576fa0")
    axes[0].set(xlabel="Absolute JLU–GAU difference", ylabel="Images", title="Weak-label disagreement")
    margins = [0.0, 5.0, 10.0]
    counts = []
    # The audit records all possible pairs; the fit caps each high image separately.
    audit = json.loads((Path(config.run_dir) / "feasibility.json").read_text())
    for margin in margins:
        counts.append(audit["margin_audit"][str(margin)]["interval_dominance_pairs"])
    axes[1].bar([str(int(x)) for x in margins], counts, color="#3c8d66")
    axes[1].set(xlabel="Extra interval separation (points)", ylabel="Qualified pairs",
                title="Accepted-pair coverage")
    fit_summary_path = (Path(config.run_dir) / "fits" / "rank" /
                        f"seed_{config.seeds[0]}" / "summary.json")
    if fit_summary_path.is_file():
        pair_info = json.loads(fit_summary_path.read_text())["pair_info"]
        labels = ("0–2.5", "2.5–7.5", "7.5–15", ">15")
        positions = np.arange(4)
        axes[2].bar(positions - .2, pair_info["higher_midpoint_band_pairs"],
                    width=.4, label="higher")
        axes[2].bar(positions + .2, pair_info["lower_midpoint_band_pairs"],
                    width=.4, label="lower")
        axes[2].set_xticks(positions, labels)
        axes[2].set(xlabel="Weak midpoint band for sampling only", ylabel="Sampled pairs",
                    title="Severity coverage after cap")
        axes[2].legend()
    else:
        axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(destination / "rater_and_pair_coverage.png", dpi=160)
    plt.close(fig)

    summary = {"figures": ["cv_curves.png", "fit_curves.png",
                           "rater_and_pair_coverage.png"], "contact_sheets": {}}
    if (Path(config.run_dir) / "fits" / "rank" / f"seed_{config.seeds[0]}" / "model.pt").is_file():
        summary["pair_violations"] = _pair_violations(config, weak, destination)
        if summary["pair_violations"]["examples_loaded"]:
            summary["figures"].append("pair_violations.png")
    for name in ("validation", "test", *(f"ood_{cohort}" for cohort in
                                          ("wg_insects_t1_bbch10", "dsv_asendorf_t1_bbch11"))):
        path = Path(config.run_dir) / f"{name}_matched_predictions.csv"
        if not path.is_file():
            continue
        summary["contact_sheets"][name] = _plot_split(_csv(path), name, destination)
        summary["figures"].append(f"{name}_diagnostics.png")
        comparison = json.loads((Path(config.run_dir) / f"{name}_comparison.json").read_text())
        controls = ("base", "gold_only", "midpoint", "shuffled")
        values = [comparison["paired_comparisons"][f"rank_vs_{control}"] for control in controls]
        centers = np.array([value["mae_reduction"] for value in values])
        lower = np.array([value["bootstrap_95_percent_interval"][0] for value in values])
        upper = np.array([value["bootstrap_95_percent_interval"][1] for value in values])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(np.arange(len(controls)), centers, yerr=[centers - lower, upper - centers],
                    fmt="o", capsize=4, color="#3c8d66")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(np.arange(len(controls)), controls)
        ax.set(ylabel="Plot MAE reduction for rank (95% plot bootstrap)",
               title=f"{name}: paired comparisons")
        fig.tight_layout()
        fig.savefig(destination / f"{name}_bootstrap.png", dpi=160)
        plt.close(fig)
        summary["figures"].append(f"{name}_bootstrap.png")
    sensitivity_path = Path(config.run_dir) / "sensitivity_margin10" / "summary.json"
    if sensitivity_path.is_file():
        sensitivity = json.loads(sensitivity_path.read_text())["results"]
        names = list(sensitivity)
        primary = [sensitivity[name]["primary_rank_plot_mae"] for name in names]
        tighter = [sensitivity[name]["margin10_plot_mae"] for name in names]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x - .18, primary, width=.35, label="5-point primary")
        ax.bar(x + .18, tighter, width=.35, label="10-point sensitivity")
        ax.set_xticks(x, [name.replace("ood_", "weak ") for name in names], rotation=12)
        ax.set(ylabel="Plot-weighted MAE", title="Pair-separation sensitivity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / "pair_margin_sensitivity.png", dpi=160)
        plt.close(fig)
        summary["figures"].append("pair_margin_sensitivity.png")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/consensus_rank_transfer/config.toml")
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
