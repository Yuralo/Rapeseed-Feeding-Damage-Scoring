"""Result-first visual audit of the consensus-safe rank-transfer experiment.

Reads saved CSV/JSON artifacts. Checkpoints, frozen features, and a GPU are not needed.
Original photographs are optional; unavailable photographs are listed in the report.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import Config, load_config

ARMS = ("base", "gold_only", "midpoint", "rank", "shuffled")
COLORS = {"base": "#333333", "gold_only": "#3379a6", "midpoint": "#d08b32",
          "rank": "#29815a", "shuffled": "#96549b"}
BANDS = ((0, 2.5), (2.5, 7.5), (7.5, 15), (15, np.inf))
BAND_NAMES = ("0–2.5", "2.5–7.5", "7.5–15", ">15")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, output: Path, summary: dict, description: str) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)
    summary["figures"].append(output.name)
    summary["descriptions"][output.name] = description


def _values(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _pred(rows: list[dict], arm: str) -> np.ndarray:
    return _values(rows, "base_prediction" if arm == "base" else f"{arm}_mean_prediction")


def _band_mask(target: np.ndarray, index: int) -> np.ndarray:
    low, high = BANDS[index]
    return (target >= low if index == 0 else target > low) & (target <= high)


def _plot_error_by_group(error: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    return unique, np.asarray([np.mean(error[groups == group])
                               for group in unique])


def _summary_dashboard(comparisons: dict, destination: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    names = [name for name in ("validation", "test") if name in comparisons]
    names.extend(sorted(name for name in comparisons if name not in names))
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(max(11, len(names) * 2.5), 9), sharex=True)
    for i, arm in enumerate(ARMS):
        y = [comparisons[name]["base_plot_weighted_mae"] if arm == "base" else
             comparisons[name]["mean_plot_weighted_mae"][arm] for name in names]
        axes[0].scatter(x + (i - 2) * .13, y, color=COLORS[arm], label=arm, s=65)
    axes[0].set(ylabel="Plot-weighted MAE (lower is better)", title="Matched-arm outcomes by split")
    axes[0].legend(ncol=len(ARMS), fontsize=9)
    for i, name in enumerate(names):
        report = comparisons[name]
        value = report["paired_comparisons"]["rank_vs_base"]
        low, high = value["bootstrap_95_percent_interval"]
        center = value["mae_reduction"]
        axes[1].errorbar(i, center, yerr=[[center - low], [high - center]],
                         fmt="o", capsize=5, color=COLORS["rank"])
        axes[1].text(i, high + .04, f"n={report['images']} / {report['plots']} plots",
                     ha="center", fontsize=8)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(ylabel="Base − rank plot MAE\n95% plot bootstrap", xlabel="Evaluation split")
    axes[1].set_xticks(x, [name.replace("ood_", "OOD: ") for name in names], rotation=12)
    fig.suptitle("Historical gold splits and single-rater OOD agreement are different evidence", fontsize=13)
    _save(fig, destination / "outcome_overview.png", summary,
          "All matched arms and the paired rank-versus-base plot bootstrap. OOD labels are single-rater weak scores.")


def _cohort_shift(rows_by_split: dict[str, list[dict]], destination: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    if not any(name.startswith("ood_") for name in rows_by_split):
        return
    names = [name for name in ("validation", "test") if name in rows_by_split]
    names.extend(sorted(name for name in rows_by_split if name not in names))
    labels = [name.replace("ood_", "OOD: ") for name in names]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    quantities = (
        ("target", lambda rows: _values(rows, "target"), "Reference score distribution"),
        ("base", lambda rows: _pred(rows, "base"), "Frozen-base score distribution"),
        ("correction", lambda rows: _pred(rows, "rank") - _pred(rows, "base"), "Rank correction distribution"),
        ("bias", lambda rows: _pred(rows, "rank") - _values(rows, "target"), "Rank signed error distribution"),
    )
    for axis, (_, function, title) in zip(axes.flat, quantities, strict=True):
        values = [function(rows_by_split[name]) for name in names]
        result = axis.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=True,
                              medianprops={"color": "black"})
        for box, name in zip(result["boxes"], names, strict=True):
            box.set_facecolor("#c7e2d1" if name.startswith("ood_") else "#b8d5e8")
        axis.set(title=title, ylabel="Score points")
        axis.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Cohort shift: blue = historical gold, green = single-rater OOD")
    _save(fig, destination / "cohort_shift.png", summary,
          "Reference and prediction distributions, rank corrections, and signed errors across cohorts; OOD scores are weak labels.")


def _experiment_design(config: Config, destination: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 7))
    audit_path = Path(config.run_dir) / "feasibility.json"
    gold_count = (_read_json(audit_path).get("gold_counts", {}).get("finetune")
                  if audit_path.is_file() else None)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.13",
                                    facecolor=color, edgecolor="#344", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "#456", "lw": 1.5})

    box(.3, 4.9, 3.2, 1.1, "Frozen DINOv3 three-view\nfeatures + base score", "#dfe8ee")
    box(.3, 2.6, 3.2, 1.1, f"{gold_count if gold_count is not None else 'Gold'} training images\nplot-balanced score loss", "#cfe0ee")
    box(.3, .5, 3.2, 1.1, "Discordant dual-rater images\ninterval-safe cross-plot pairs", "#d7ecdd")
    box(5.1, 2.3, 3.6, 2.0, f"Matched residual head\ngold-only · midpoint · rank\nshuffled-order control\n{len(config.seeds)} seeds", "#f2ead3")
    box(10.1, 4.9, 3.4, 1.1, "Grouped gold-train CV\nselect weights + epoch", "#ece0f0")
    box(10.1, 2.55, 3.4, 1.2, "Historical gold validation/test\npaired plot bootstrap", "#cfe0ee")
    box(10.1, .45, 3.4, 1.2, "WG + DSV OOD cohorts\nsingle-rater agreement only", "#d7ecdd")
    arrow((3.6, 5.45), (5, 3.75))
    arrow((3.6, 3.15), (5, 3.2))
    arrow((3.6, 1.05), (5, 2.55))
    arrow((8.8, 3.75), (10, 5.2))
    arrow((8.8, 3.3), (10, 3.15))
    arrow((8.8, 2.75), (10, 1.05))
    ax.text(7, 6.35, "Consensus-safe rank transfer: data flow and evidence boundaries",
            fontsize=15, ha="center", weight="bold")
    ax.text(7, .1, f"Rank pair margin {config.rank_interval_margin:g}; prediction margin "
            f"{config.rank_prediction_margin:g}; max {config.max_pairs_per_image} pairs/image. "
            "OOD labels and historical test are not used for head selection.",
            fontsize=9, ha="center")
    _save(fig, destination / "experiment_design.png", summary,
          "Data flow, matched arms, training-only grouped selection, historical evaluation, and OOD score-agreement boundary.")


def _validation_gates(config: Config, comparison: dict, destination: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    gates = comparison.get("decision_gates")
    if not gates:
        return
    pairs = comparison["paired_comparisons"]
    base = pairs["rank_vs_base"]
    gold = pairs["rank_vs_gold_only"]
    shuffled = pairs["rank_vs_shuffled"]
    low = comparison["low_score"]
    rows = (
        ("Rank gain vs base", base["mae_reduction"], f"≥ {config.practical_mae_reduction:g}",
         gates["rank_beats_base_by_0_25"]),
        ("Rank/base interval lower bound", base["bootstrap_95_percent_interval"][0], "> 0",
         gates["rank_vs_base_interval_excludes_zero"]),
        ("Rank gain vs gold-only", gold["mae_reduction"], f"≥ {config.practical_mae_reduction:g}",
         gates["rank_beats_gold_only_by_0_25"]),
        ("Rank/gold interval lower bound", gold["bootstrap_95_percent_interval"][0], "> 0",
         gates["rank_vs_gold_only_interval_excludes_zero"]),
        ("Rank/shuffled interval lower bound", shuffled["bootstrap_95_percent_interval"][0], "> 0",
         gates["rank_vs_shuffled_interval_excludes_zero"]),
        ("Low-score MAE degradation", low["rank_mae"] - low["base_mae"] if
         low["rank_mae"] is not None and low["base_mae"] is not None else None,
         f"≤ {config.low_score_max_degradation:g}", gates["low_score_degradation_at_most_0_25"]),
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    cells = [[label, "n/a" if value is None else f"{value:+.3f}", rule, "PASS" if passed else "FAIL"]
             for label, value, rule, passed in rows]
    table = ax.table(cellText=cells, colLabels=["Predeclared check", "Observed", "Rule", "Result"],
                     cellLoc="left", loc="center", colWidths=[.55, .15, .12, .12])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.15)
    for i, (_, _, _, passed) in enumerate(rows, start=1):
        table[i, 3].set_facecolor("#cde9d7" if passed else "#f4cece")
    ax.set_title("Validation decision gates (historical gold; retrospective)", pad=20)
    _save(fig, destination / "validation_gates.png", summary,
          "Every predeclared practical-effect, bootstrap, shuffled-control, and low-score safety gate with observed values.")


def _cv(selection: dict, destination: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, arm in zip(axes.flat[:3], ("gold_only", "midpoint", "rank"), strict=True):
        choice = selection["selected"][arm]
        for candidate in selection["candidates"][arm]:
            axis.plot(candidate["mean_curve"], color=COLORS[arm], alpha=.13)
        for fold in choice["fold_curves"]:
            axis.plot(fold, color=COLORS[arm], alpha=.3, linestyle=":")
        axis.plot(choice["mean_curve"], color=COLORS[arm], linewidth=2.5, label="selected mean")
        axis.scatter(choice["epoch"], choice["cv_plot_mae"], color="black", s=36, zorder=3)
        axis.axhline(selection["base_cv_plot_mae"], color="black", linestyle="--", alpha=.6)
        axis.set(title=f"{arm}: weight={choice['weak_weight']:g}, penalty={choice['residual_penalty']:g}",
                 xlabel="Epoch (0 = frozen base)", ylabel="Held-fold plot MAE")
        axis.legend(fontsize=8)
    fold_sizes = selection.get("fold_sizes", [])
    axis = axes.flat[3]
    if fold_sizes:
        x = np.arange(len(fold_sizes))
        axis.bar(x - .2, [f["gold_holdout"] for f in fold_sizes], width=.4, label="gold holdout")
        axis.bar(x + .2, [f["weak_excluded_same_plot"] for f in fold_sizes], width=.4,
                 label="weak excluded")
        axis.set_xticks(x, [str(f["fold"]) for f in fold_sizes])
        axis.set(xlabel="Fold", ylabel="Images", title="Grouped holdouts and weak-image exclusions")
        axis.legend(fontsize=8)
    fig.suptitle("CV is a head-selection heuristic: the frozen base saw all CV gold images")
    _save(fig, destination / "cv_curves.png", summary,
          "All search candidates, selected fold trajectories, selected epoch, and grouped-fold exclusions. Baseline CV line is not unbiased.")


def _fit_dynamics(run_dir: Path, seeds: tuple[int, ...], destination: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = (("train_plot_mae", "Gold training plot MAE"), ("gold_loss", "Gold loss"),
               ("weak_loss", "Weak loss"), ("residual_rms", "Residual RMS"))
    any_history = False
    for arm in ARMS[1:]:
        for seed in seeds:
            path = run_dir / "fits" / arm / f"seed_{seed}" / "history.json"
            if not path.is_file():
                continue
            history = _read_json(path)
            any_history = True
            for axis, (key, _) in zip(axes.flat, metrics, strict=True):
                x = [entry["epoch"] for entry in history if key in entry]
                y = [entry[key] for entry in history if key in entry]
                if x:
                    axis.plot(x, y, color=COLORS[arm], alpha=.45, linewidth=1.2)
                    axis.scatter(x[-1], y[-1], color=COLORS[arm], s=12)
    if not any_history:
        plt.close(fig)
        summary["missing"].append("fit histories")
        return
    for axis, (_, label) in zip(axes.flat, metrics, strict=True):
        axis.set(xlabel="Epoch", ylabel=label, title=label)
    for arm in ARMS[1:]:
        axes.flat[0].plot([], [], color=COLORS[arm], label=arm)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Full-fit trajectories: one line per available seed; endpoints are selected epochs")
    _save(fig, destination / "fit_curves.png", summary,
          "Training dynamics of all arms and seeds, including weak loss and residual magnitude. Training curves do not measure generalization.")


def _weak_evidence(run_dir: Path, seeds: tuple[int, ...], destination: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    audit_path = run_dir / "feasibility.json"
    pair_path = run_dir / "fits" / "rank" / f"seed_{seeds[0]}" / "sampled_pairs.csv"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    available = False
    if audit_path.is_file():
        audit = _read_json(audit_path)
        margins = sorted(audit["margin_audit"], key=float)
        axes[0, 0].bar(margins, [audit["margin_audit"][m]["interval_dominance_pairs"] for m in margins],
                       color=COLORS["rank"])
        axes[0, 0].set(xlabel="Interval separation margin", ylabel="Qualified candidate pairs",
                       title="Candidate pool before training cap")
        coverage = [audit["margin_audit"][m].get("participating_images") for m in margins]
        if all(value is not None for value in coverage):
            axes[0, 1].bar(margins, coverage, color=COLORS["gold_only"])
            axes[0, 1].set(xlabel="Interval separation margin", ylabel="Participating images",
                           title="How much of the weak pool contributes?")
        else:
            axes[0, 1].axis("off")
        available = True
    else:
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")
    if pair_path.is_file():
        pairs = _read_csv(pair_path)
        if not pairs:
            raise ValueError(f"Empty rank sampled-pair file: {pair_path}")
        gap = [min(float(r["first_jlu"]), float(r["first_gau"])) -
               max(float(r["second_jlu"]), float(r["second_gau"])) for r in pairs]
        axes[1, 0].hist(gap, bins=25, color=COLORS["rank"])
        axes[1, 0].set(xlabel="Conservative interval gap", ylabel="Sampled pairs",
                       title=f"Actual sampled pair margins (seed {seeds[0]})")
        degree = Counter(value for row in pairs for value in (row["first_image"], row["second_image"]))
        axes[1, 1].hist(list(degree.values()), bins=min(20, max(degree.values())), color=COLORS["midpoint"])
        axes[1, 1].set(xlabel="Sampled-pair degree per image", ylabel="Images",
                       title="Training exposure concentration")
        available = True
        with (destination / "pair_sampling_audit.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("image", "sampled_pair_degree"))
            writer.writerows(sorted(degree.items(), key=lambda item: (-item[1], item[0])))
        scores = {}
        for row in pairs:
            for side in ("first", "second"):
                scores[row[f"{side}_image"]] = (float(row[f"{side}_jlu"]), float(row[f"{side}_gau"]))
        jlu, gau = np.asarray(list(scores.values())).T
        fig, disagreement_axes = plt.subplots(1, 2, figsize=(12, 5))
        disagreement_axes[0].scatter(jlu, gau, s=18, alpha=.55, color=COLORS["gold_only"])
        maximum = max(float(jlu.max()), float(gau.max()), 1)
        disagreement_axes[0].plot([0, maximum], [0, maximum], "k--", linewidth=1)
        disagreement_axes[0].set(xlabel="JLU score", ylabel="GAU score",
                                 title=f"Selected weak images (n={len(scores)})")
        disagreement_axes[1].hist(np.abs(jlu - gau), bins=25, color=COLORS["midpoint"])
        disagreement_axes[1].set(xlabel="Absolute rater score gap", ylabel="Selected weak images",
                                 title="Disagreement among pair participants")
        fig.suptitle("Dual-rater evidence in the sampled training graph; selection biased")
        _save(fig, destination / "sampled_rater_disagreement.png", summary,
              "JLU-versus-GAU scores and disagreement for unique sampled pair participants. These are not the entire weak pool.")
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        summary["missing"].append("rank sampled_pairs.csv")
    if available:
        fig.suptitle("Dual-rater interval ordering: qualification, coverage, and sampled concentration")
        _save(fig, destination / "rater_and_pair_coverage.png", summary,
              "Candidate pair counts and coverage at each margin; actual sampled-pair interval gaps and image degrees.")
    else:
        plt.close(fig)
    diagnostics = []
    for arm in ARMS[1:]:
        for seed in seeds:
            path = run_dir / "fits" / arm / f"seed_{seed}" / "summary.json"
            if path.is_file():
                row = _read_json(path).get("pair_diagnostic", {})
                if "arm_violation_fraction" in row:
                    diagnostics.append((arm, seed, row))
    if diagnostics:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, arm in enumerate(ARMS[1:]):
            rows = [(seed, d) for name, seed, d in diagnostics if name == arm]
            for seed, d in rows:
                ax.plot([i - .13, i + .13], [d["base_violation_fraction"], d["arm_violation_fraction"]],
                        color=COLORS[arm], alpha=.55)
                ax.scatter(i + .13, d["arm_violation_fraction"], color=COLORS[arm], s=25)
        ax.set_xticks(range(4), ARMS[1:])
        ax.set(ylabel="Fraction of qualified sampled orders violated", title="Before → after, each arm and seed")
        _save(fig, destination / "pair_violation_by_seed.png", summary,
              "Qualified-order violation fractions for each seed. This is a training-pool diagnostic, not held-out accuracy.")


def _split_figures(name: str, rows: list[dict], comparison: dict, destination: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    target = _values(rows, "target")
    groups = np.asarray([row["plot_group_id"] for row in rows])
    predictions = {arm: _pred(rows, arm) for arm in ARMS}
    # Reported MAE averages each seed's absolute error; it is generally not
    # equal to the error of the displayed mean prediction.
    error = {arm: _values(rows, "base_absolute_error" if arm == "base"
                          else f"{arm}_mean_absolute_error") for arm in ARMS}
    title = name.replace("ood_", "OOD: ")
    quality = "single-rater weak agreement" if name.startswith("ood_") else "historical gold"
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    extent = max(1, float(np.max([target, *predictions.values()])))
    for arm in ARMS:
        axes[0, 0].scatter(target, predictions[arm], s=10, alpha=.36, color=COLORS[arm], label=arm)
        axes[0, 1].scatter(target, predictions[arm] - target, s=10, alpha=.36, color=COLORS[arm], label=arm)
    axes[0, 0].plot([0, extent], [0, extent], "k--", linewidth=1)
    axes[0, 0].set(xlabel="Reference score", ylabel="Predicted score", title="Calibration and range")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set(xlabel="Reference score", ylabel="Prediction − reference", title="Signed error versus severity")
    offsets = np.linspace(-.32, .32, len(ARMS))
    for arm, offset in zip(ARMS, offsets, strict=True):
        axes[1, 0].bar(np.arange(4) + offset,
                       [np.mean(error[arm][_band_mask(target, j)]) if _band_mask(target, j).any() else np.nan
                        for j in range(4)], width=.13, label=arm, color=COLORS[arm])
        axes[1, 1].hist(predictions[arm], bins=20, histtype="step", linewidth=1.7,
                        color=COLORS[arm], label=arm)
    axes[1, 0].set_xticks(range(4), [f"{label}\nn={_band_mask(target, j).sum()}" for j, label in enumerate(BAND_NAMES)])
    axes[1, 0].set(xlabel="Reference score band", ylabel="Image MAE", title="Severity band errors and sample sizes")
    axes[1, 1].hist(target, bins=20, histtype="step", linewidth=2, linestyle="--", color="black", label="reference")
    axes[1, 1].set(xlabel="Score", ylabel="Images", title="Predicted-score distribution and reference shift")
    axes[1, 1].legend(ncol=2, fontsize=8)
    fig.suptitle(f"{title}: {quality} ({len(rows)} images, {len(np.unique(groups))} plots)")
    _save(fig, destination / f"{name}_diagnostics.png", summary,
          "All-arm mean prediction calibration, signed errors, per-seed-averaged severity MAE with counts, and score distributions.")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    improvement = error["base"] - error["rank"]
    axes[0, 0].scatter(target - predictions["base"], predictions["rank"] - predictions["base"],
                       c=improvement, cmap="RdYlGn", vmin=-max(1, np.max(abs(improvement))),
                       vmax=max(1, np.max(abs(improvement))), s=24, alpha=.75)
    axes[0, 0].axhline(0, color="black", linewidth=1)
    axes[0, 0].axvline(0, color="black", linewidth=1)
    axes[0, 0].set(xlabel="Correction needed: reference − base", ylabel="Correction applied: rank − base",
                   title="Direction and size of correction")
    plot_ids, base_plot = _plot_error_by_group(error["base"], groups)
    _, rank_plot = _plot_error_by_group(error["rank"], groups)
    plot_gain = base_plot - rank_plot
    axes[0, 1].scatter(base_plot, rank_plot, c=plot_gain, cmap="RdYlGn", s=35)
    lim = max(base_plot.max(), rank_plot.max(), 1)
    axes[0, 1].plot([0, lim], [0, lim], "k--", linewidth=1)
    axes[0, 1].set(xlabel="Base plot MAE", ylabel="Rank plot MAE", title="Each plot group is one point")
    ordered = np.sort(plot_gain)
    axes[1, 0].bar(np.arange(len(ordered)), ordered,
                   color=[COLORS["rank"] if x >= 0 else "#c55a5a" for x in ordered])
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set(xlabel="Plot groups sorted by rank benefit", ylabel="Base − rank plot MAE",
                   title=f"Improved: {(plot_gain > 0).sum()} / {len(plot_gain)} plots")
    for arm in ARMS:
        ordered_error = np.sort(error[arm])
        axes[1, 1].plot(ordered_error, (np.arange(len(rows)) + 1) / len(rows),
                        color=COLORS[arm], label=arm)
    axes[1, 1].set(xlabel="Absolute error threshold", ylabel="Fraction of images ≤ threshold",
                   title="Absolute-error CDF")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"{title}: where gains and regressions occur ({quality})")
    _save(fig, destination / f"{name}_error_anatomy.png", summary,
          "Correction needed versus applied, paired plot errors, ordered plot gains, and all-arm error CDFs.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    controls = ("base", "gold_only", "midpoint", "shuffled")
    records = [comparison["paired_comparisons"][f"rank_vs_{arm}"] for arm in controls]
    centers = np.asarray([v["mae_reduction"] for v in records])
    lows = np.asarray([v["bootstrap_95_percent_interval"][0] for v in records])
    highs = np.asarray([v["bootstrap_95_percent_interval"][1] for v in records])
    axes[0].errorbar(np.arange(4), centers, yerr=[centers - lows, highs - centers], fmt="o",
                     capsize=5, color=COLORS["rank"])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(range(4), controls, rotation=12)
    axes[0].set(ylabel="Control − rank plot MAE", title="95% paired plot-bootstrap intervals")
    for j in range(4):
        mask = _band_mask(target, j)
        if mask.any():
            axes[1].scatter(j + .12, np.mean(error["rank"][mask] - error["base"][mask]),
                            color=COLORS["rank"], s=max(25, mask.sum() * 2))
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(range(4), [f"{label}\nn={_band_mask(target, j).sum()}" for j, label in enumerate(BAND_NAMES)])
    axes[1].set(ylabel="Rank − base image MAE", title="Severity-specific change; dot area shows count")
    fig.suptitle(f"{title}: paired evidence and low-score safety ({quality})")
    _save(fig, destination / f"{name}_bootstrap.png", summary,
          "Rank versus every control with saved paired plot-bootstrap intervals and severity-specific changes.")

    with (destination / f"{name}_plot_changes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("plot_group_id", "images", "base_mae", "rank_mae", "base_minus_rank_mae"))
        for group, before, after, gain in sorted(zip(plot_ids, base_plot, rank_plot, plot_gain, strict=True),
                                                 key=lambda item: item[3]):
            writer.writerow((group, int((groups == group).sum()), before, after, gain))
    with (destination / f"{name}_image_changes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "plot_group_id", "target", "base_prediction", "rank_prediction",
                         "base_absolute_error", "rank_absolute_error", "base_minus_rank_absolute_error", "source_path"))
        for i in np.argsort(improvement):
            row = rows[i]
            writer.writerow((row["filename"], row["plot_group_id"], target[i], predictions["base"][i],
                             predictions["rank"][i], error["base"][i], error["rank"][i],
                             improvement[i], row["source_path"]))


def _seed_figures(run_dir: Path, name: str, comparison: dict, destination: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    seeds = [str(seed) for seed in comparison["seeds"]]
    for arm in ARMS[1:]:
        values = [comparison["per_seed"][arm][seed]["plot_weighted_mae"] for seed in seeds]
        axes[0].plot(seeds, values, "o-", color=COLORS[arm], label=arm)
    axes[0].axhline(comparison["base_plot_weighted_mae"], color="black", linestyle="--", label="base")
    axes[0].set(xlabel="Seed", ylabel="Plot-weighted MAE", title="Every trained seed")
    axes[0].legend(fontsize=8)
    corrections = []
    expected = None
    for seed in seeds:
        path = run_dir / "fits" / "rank" / f"seed_{seed}"
        path = path / "ood" / name[4:] / "predictions.csv" if name.startswith("ood_") else path / name / "predictions.csv"
        if not path.is_file():
            continue
        rows = _read_csv(path)
        identity = [(r["filename"], r["plot_group_id"], r["target"], r["base_prediction"]) for r in rows]
        if expected is None:
            expected = identity
        elif identity != expected:
            raise ValueError(f"Unmatched rank seeds in {name}")
        corrections.append(_values(rows, "prediction") - _values(rows, "base_prediction"))
    if len(corrections) >= 2:
        spread = np.std(np.stack(corrections), axis=0)
        axes[1].hist(spread, bins=20, color=COLORS["rank"])
        axes[1].set(xlabel="Std. dev. of rank correction over seeds", ylabel="Images",
                    title=f"Image-level seed instability (median {np.median(spread):.2f})")
        with (destination / f"{name}_seed_instability.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("filename", "correction_std_across_seeds"))
            writer.writerows(sorted(zip((item[0] for item in expected), spread, strict=True),
                                    key=lambda item: -item[1]))
    else:
        axes[1].axis("off")
    fig.suptitle(f"{name}: seed robustness and correction spread")
    _save(fig, destination / f"{name}_seeds.png", summary,
          "Per-seed plot-weighted MAE for all arms and distribution of image-level rank correction spread.")


def _image_lookup(image_roots: tuple[Path, ...]) -> dict[str, Path]:
    lookup = {}
    basename_lookup = {}
    ambiguous = set()
    ambiguous_relative = set()
    for root in image_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            relative = str(path.relative_to(root))
            if relative in lookup:
                ambiguous_relative.add(relative)
            else:
                lookup[relative] = path
            if path.name in basename_lookup:
                ambiguous.add(path.name)
            else:
                basename_lookup[path.name] = path
    for name in ambiguous:
        basename_lookup.pop(name, None)
    for name in ambiguous_relative:
        lookup.pop(name, None)
    for name, path in basename_lookup.items():
        lookup.setdefault(name, path)
    return lookup


def _contact_sheet(rows: list[dict], indices: np.ndarray, path: Path, title: str,
                   image_lookup: dict[str, Path]) -> int:
    from PIL import Image, ImageDraw, ImageOps

    found = []
    for index in indices:
        row = rows[int(index)]
        original = Path(row["source_path"])
        image_path = (original if original.is_file() else
                      image_lookup.get(row["filename"], image_lookup.get(Path(row["filename"]).name)))
        if image_path is not None and image_path.is_file():
            found.append((row, image_path))
            if len(found) == 12:
                break
    if not found:
        return 0
    width, height = 370, 330
    canvas = Image.new("RGB", (4 * width, ((len(found) + 3) // 4) * height + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black")
    for i, (row, image_path) in enumerate(found):
        with Image.open(image_path) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (width - 12, 245))
        x, y = (i % 4) * width, 40 + (i // 4) * height
        canvas.paste(image, (x + 5, y + 5))
        gain = float(row["base_absolute_error"]) - float(row["rank_mean_absolute_error"])
        label = (f"target {float(row['target']):.1f} | base {float(row['base_prediction']):.1f} | "
                 f"rank {float(row['rank_mean_prediction']):.1f}\n"
                 f"base - rank error {gain:+.2f} | {row['plot_group_id']}\n{row['filename']}")
        draw.multiline_text((x + 5, y + 254), label, fill="black")
    canvas.save(path)
    return len(found)


def _image_sheets(name: str, rows: list[dict], destination: Path, lookup: dict[str, Path],
                  summary: dict) -> None:
    target = _values(rows, "target")
    base = _pred(rows, "base")
    rank = _pred(rows, "rank")
    improvement = np.abs(base - target) - np.abs(rank - target)
    selections = {
        "improvements": np.argsort(improvement)[::-1],
        "regressions": np.argsort(improvement),
        "largest_rank_errors": np.argsort(np.abs(rank - target))[::-1],
        "low_score": np.flatnonzero(target <= 2.5)[np.argsort(improvement[target <= 2.5])],
    }
    for label, indices in selections.items():
        filename = f"{name}_{label}.png"
        loaded = _contact_sheet(rows, indices, destination / filename, f"{name}: {label}", lookup)
        summary["contact_sheets"][filename] = loaded
        if loaded:
            summary["figures"].append(filename)
            summary["descriptions"][filename] = f"Original-image evidence: {label.replace('_', ' ')}."


def _pair_violation_sheet(config: Config, destination: Path, lookup: dict[str, Path],
                          summary: dict) -> None:
    """Optional model-based image evidence; the main report never depends on it."""
    from PIL import Image, ImageDraw, ImageOps

    run_dir = Path(config.run_dir)
    checkpoint = run_dir / "fits" / "rank" / f"seed_{config.seeds[0]}" / "model.pt"
    if (not checkpoint.is_file() or not (run_dir / "frozen" / "pretrain.npz").is_file()
            or not (run_dir / "frozen" / "summary.json").is_file()):
        summary["missing"].append("rank checkpoint and frozen pretrain cache for pair-violation photographs")
        return
    from rapeseed_damage.reproducibility import resolve_device

    from .engine import candidate_pairs, predict, target_weak_pool
    from .evaluate import _load_model

    weak = target_weak_pool(config.run_dir, config)
    device = str(resolve_device("auto"))
    model = _load_model(config, "rank", config.seeds[0], device)
    prediction, _ = predict(model, weak, device)
    hi, lo, _, info = candidate_pairs(weak, np.arange(len(weak)), config,
                                      config.seeds[0], shuffled=False)
    shortfall = config.rank_prediction_margin - (prediction[hi] - prediction[lo])
    selected = np.argsort(shortfall)[::-1]
    examples = []
    for pair_index in selected:
        if shortfall[pair_index] <= 0 or len(examples) >= 8:
            break
        images = []
        for index in (hi[pair_index], lo[pair_index]):
            original = Path(str(weak.source_path[index]))
            filename = str(weak.filename[index])
            path = original if original.is_file() else lookup.get(filename, lookup.get(Path(filename).name))
            images.append(path)
        if all(path is not None and path.is_file() for path in images):
            examples.append((int(pair_index), images))
    summary["pair_violations"] = {"qualified_pairs_sampled": int(info["sampled_pairs"]),
                                  "violations": int(np.sum(shortfall > 0)),
                                  "examples_loaded": len(examples)}
    if not examples:
        return
    canvas = Image.new("RGB", (900, 35 + 175 * len(examples)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Largest qualified-order violations, first rank seed", fill="black")
    for row_index, (pair_index, images) in enumerate(examples):
        y = 35 + row_index * 175
        for column, (index, path) in enumerate(zip((hi[pair_index], lo[pair_index]), images, strict=True)):
            with Image.open(path) as opened:
                picture = ImageOps.contain(opened.convert("RGB"), (310, 130))
            x = column * 445
            canvas.paste(picture, (x + 5, y))
            low = min(weak.score_jlu[index], weak.score_gau[index])
            high = max(weak.score_jlu[index], weak.score_gau[index])
            draw.text((x + 5, y + 133),
                      f"{'higher' if column == 0 else 'lower'} [{low:.1f}, {high:.1f}] "
                      f"pred {prediction[index]:.1f}; shortfall {shortfall[pair_index]:.1f}", fill="black")
    canvas.save(destination / "pair_violations.png")
    summary["figures"].append("pair_violations.png")
    summary["descriptions"]["pair_violations.png"] = "Largest remaining violations of qualified dual-rater order, with source images."


def _sensitivity(run_dir: Path, destination: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    path = run_dir / "sensitivity_margin10" / "summary.json"
    if not path.is_file():
        summary["missing"].append("sensitivity_margin10/summary.json")
        return
    results = _read_json(path)["results"]
    names = list(results)
    x = np.arange(len(names))
    primary = [results[name]["primary_rank_plot_mae"] for name in names]
    strict = [results[name]["margin10_plot_mae"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - .18, primary, width=.35, label="5-point", color=COLORS["rank"])
    axes[0].bar(x + .18, strict, width=.35, label="10-point", color=COLORS["midpoint"])
    axes[0].set_xticks(x, [name.replace("ood_", "OOD: ") for name in names], rotation=12)
    axes[0].set(ylabel="Plot-weighted MAE", title="Fixed-settings margin sensitivity")
    axes[0].legend()
    centers = np.asarray([results[name]["margin10_minus_primary_mae_reduction"] for name in names])
    lows = np.asarray([results[name]["bootstrap_95_percent_interval"][0] for name in names])
    highs = np.asarray([results[name]["bootstrap_95_percent_interval"][1] for name in names])
    axes[1].errorbar(x, centers, yerr=[centers - lows, highs - centers], fmt="o", capsize=5,
                     color=COLORS["midpoint"])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x, [name.replace("ood_", "OOD: ") for name in names], rotation=12)
    axes[1].set(ylabel="5-point − 10-point plot MAE", title="Paired plot-bootstrap interval")
    _save(fig, destination / "pair_margin_sensitivity.png", summary,
          "Five versus ten point pair interval separation at fixed settings, with paired plot-bootstrap intervals.")


def _html_report(destination: Path, comparisons: dict, summary: dict) -> None:
    lines = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
             "<title>Consensus rank transfer visual audit</title>",
             "<style>body{font:16px system-ui;max-width:1300px;margin:2rem auto;padding:0 1rem;color:#202828}img{max-width:100%;border:1px solid #ddd}figure{margin:2rem 0}figcaption{margin:.5rem 0;color:#444}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.45rem;text-align:left}nav a{margin-right:1rem}</style>",
             "<h1>Consensus-safe rank transfer: visual audit</h1>",
             "<p>Historical gold validation/test are retrospective. OOD scores are single-rater weak labels; their plots show score agreement under cohort shift.</p>",
             "<nav><a href='#results'>Results</a><a href='#methods'>Experiment</a><a href='#images'>Image evidence</a><a href='#files'>Downloads</a></nav>",
             "<h2 id='results'>Results</h2><table><tr><th>Split</th><th>Images / plots</th><th>Base MAE</th><th>Rank MAE</th><th>Rank gain vs base [95% interval]</th><th>Label quality</th></tr>"]
    names = [name for name in ("validation", "test") if name in comparisons]
    names.extend(sorted(name for name in comparisons if name not in names))
    for name in names:
        report = comparisons[name]
        p = report["paired_comparisons"]["rank_vs_base"]
        interval = p["bootstrap_95_percent_interval"]
        lines.append(f"<tr><td>{html.escape(name)}</td><td>{report['images']} / {report['plots']}</td>"
                     f"<td>{report['base_plot_weighted_mae']:.3f}</td>"
                     f"<td>{report['mean_plot_weighted_mae']['rank']:.3f}</td>"
                     f"<td>{p['mae_reduction']:+.3f} [{interval[0]:+.3f}, {interval[1]:+.3f}]</td>"
                     f"<td>{html.escape(report['label_quality'])}</td></tr>")
    lines.append("</table>")
    if "validation" in comparisons and comparisons["validation"].get("decision_gates"):
        lines.append("<h3>Predeclared validation gates</h3><ul>")
        for key, value in comparisons["validation"]["decision_gates"].items():
            lines.append(f"<li>{html.escape(key)}: <strong>{'pass' if value else 'fail'}</strong></li>")
        lines.append("</ul>")
    for heading, files in (("Results", [f for f in summary["figures"] if f.startswith(("outcome_", "cohort_", "validation_", "test_", "ood_")) and f not in summary["contact_sheets"]]),
                           ("Experiment", [f for f in summary["figures"] if f.startswith(("experiment_", "cv_", "fit_", "rater_", "pair_", "sampled_"))]),
                           ("Image evidence", [f for f in summary["figures"] if f in summary["contact_sheets"]])):
        lines.append(f"<h2 id='{heading.lower().split()[0]}'>{heading}</h2>")
        if not files:
            lines.append("<p>No applicable artifacts were available.</p>")
        for file in files:
            lines.append(f"<figure><a href='{html.escape(file)}'><img loading='lazy' src='{html.escape(file)}' alt='{html.escape(file)}'></a>"
                         f"<figcaption><strong>{html.escape(file)}</strong> — {html.escape(summary['descriptions'].get(file, ''))}</figcaption></figure>")
    lines.append("<h2 id='files'>Audit files and limitations</h2><ul>")
    for file in sorted(destination.glob("*.csv")):
        lines.append(f"<li><a href='{html.escape(file.name)}'>{html.escape(file.name)}</a></li>")
    lines.append("<li><a href='summary.json'>summary.json</a></li></ul>")
    if summary["missing"]:
        lines.append("<p>Missing optional artifacts: " + html.escape(", ".join(summary["missing"])) + ".</p>")
    if not any(summary["contact_sheets"].values()):
        lines.append("<p>Original photographs were unavailable at the saved source paths or supplied image roots. CSV case rankings remain available.</p>")
    lines.append("</html>")
    (destination / "index.html").write_text("\n".join(lines), encoding="utf-8")


def run(config: Config, *, image_roots: tuple[Path, ...] = ()) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    run_dir = Path(config.run_dir)
    comparison_paths = sorted(run_dir.glob("*_comparison.json"))
    if not comparison_paths:
        raise FileNotFoundError(f"No completed comparison JSON files in {run_dir}; run compare first or set --run-dir")
    comparisons = {path.name.removesuffix("_comparison.json"): _read_json(path) for path in comparison_paths}
    required = {name: run_dir / f"{name}_matched_predictions.csv" for name in comparisons}
    absent = [str(path) for path in required.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError("Matched prediction CSVs are required: " + ", ".join(absent))
    rows_by_split = {name: _read_csv(path) for name, path in required.items()}
    for name, rows in rows_by_split.items():
        if len(rows) != comparisons[name]["images"]:
            raise ValueError(f"{name}: CSV and comparison image counts differ")
        if not rows:
            raise ValueError(f"{name}: empty matched predictions")
        groups = np.asarray([row["plot_group_id"] for row in rows])
        if len(np.unique(groups)) != comparisons[name]["plots"]:
            raise ValueError(f"{name}: CSV and comparison plot counts differ")
        for arm in ARMS:
            column = "base_absolute_error" if arm == "base" else f"{arm}_mean_absolute_error"
            _, errors = _plot_error_by_group(_values(rows, column), groups)
            reported = (comparisons[name]["base_plot_weighted_mae"] if arm == "base" else
                        comparisons[name]["mean_plot_weighted_mae"][arm])
            if not np.isclose(errors.mean(), reported, rtol=1e-5, atol=1e-5):
                raise ValueError(f"{name}/{arm}: matched CSV does not reproduce reported plot MAE")
    destination = run_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"figures": [], "descriptions": {}, "contact_sheets": {}, "missing": [],
               "splits": {name: {"images": len(rows), "plots": len({r['plot_group_id'] for r in rows}),
                                  "label_quality": comparisons[name]["label_quality"]}
                          for name, rows in rows_by_split.items()},
               "image_roots": [str(root) for root in image_roots]}
    _experiment_design(config, destination, summary)
    _summary_dashboard(comparisons, destination, summary)
    _cohort_shift(rows_by_split, destination, summary)
    if "validation" in comparisons:
        _validation_gates(config, comparisons["validation"], destination, summary)
    selection_path = run_dir / "cv_selection.json"
    if selection_path.is_file():
        _cv(_read_json(selection_path), destination, summary)
    else:
        summary["missing"].append("cv_selection.json")
    _fit_dynamics(run_dir, config.seeds, destination, summary)
    _weak_evidence(run_dir, config.seeds, destination, summary)
    lookup = _image_lookup(image_roots)
    for name, rows in rows_by_split.items():
        _split_figures(name, rows, comparisons[name], destination, summary)
        _seed_figures(run_dir, name, comparisons[name], destination, summary)
        _image_sheets(name, rows, destination, lookup, summary)
    _pair_violation_sheet(config, destination, lookup, summary)
    _sensitivity(run_dir, destination, summary)
    missing_images = sum(len(rows) for rows in rows_by_split.values()) if not any(summary["contact_sheets"].values()) else None
    if missing_images is not None:
        summary["missing"].append(f"original evaluation images for contact sheets ({missing_images} rows across splits)")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _html_report(destination, comparisons, summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/consensus_rank_transfer/config.toml")
    parser.add_argument("--run-dir", type=Path, help="Completed run directory; overrides config run_dir")
    parser.add_argument("--image-root", type=Path, action="append", default=[],
                        help="Optional image tree for contact sheets; repeat for multiple roots")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.run_dir:
        config = replace(config, run_dir=str(args.run_dir))
    print(json.dumps(run(config, image_roots=tuple(args.image_root)), indent=2))


if __name__ == "__main__":
    main()
