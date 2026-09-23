"""Audit whether discordant dual-rater images contain usable ordering evidence.

Only the existing manifest CSVs are needed. This does not train a model or read the
original images, so the audit can run in the lightweight research checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "outputs" / "dataset_manifests"
DEFAULT_OUTPUT = ROOT / "outputs" / "consensus_rank_transfer" / "feasibility.json"
TARGET_COHORT = "gg_insects_t1_bbch10"


def read_manifest(name: str) -> list[dict[str, str]]:
    with (MANIFESTS / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def score_interval(row: dict[str, str]) -> tuple[float, float]:
    return tuple(sorted((float(row["score_jlu"]), float(row["score_gau"]))))


def pair_evidence(rows: list[dict[str, str]], margin: float) -> dict[str, int | float]:
    eligible_images: set[str] = set()
    high_images: set[str] = set()
    low_images: set[str] = set()
    counts = Counter()
    for left_index, left in enumerate(rows):
        left_low, left_high = score_interval(left)
        for right in rows[left_index + 1 :]:
            if left["plot_group_id"] == right["plot_group_id"]:
                continue
            right_low, right_high = score_interval(right)
            counts["cross_plot_pairs"] += 1
            left_diff = float(left["score_jlu"]) - float(right["score_jlu"])
            right_diff = float(left["score_gau"]) - float(right["score_gau"])
            if left_diff * right_diff > 0:
                counts["rater_order_agreement"] += 1
            elif left_diff * right_diff < 0:
                counts["rater_order_opposition"] += 1
            else:
                counts["rater_order_tie"] += 1

            if left_low > right_high + margin:
                high, low = left, right
            elif right_low > left_high + margin:
                high, low = right, left
            else:
                continue
            counts["interval_dominance_pairs"] += 1
            eligible_images.update((left["image_id"], right["image_id"]))
            high_images.add(high["image_id"])
            low_images.add(low["image_id"])

    n_decided = counts["rater_order_agreement"] + counts["rater_order_opposition"]
    return {
        **counts,
        "margin_points": margin,
        "participating_images": len(eligible_images),
        "images_as_higher": len(high_images),
        "images_as_lower": len(low_images),
        "rater_order_agreement_fraction_non_ties": (
            counts["rater_order_agreement"] / n_decided if n_decided else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifests = {name: read_manifest(name) for name in ("pretrain", "finetune", "validation", "test")}
    weak = [
        row
        for row in manifests["pretrain"]
        if row["cohort_id"] == TARGET_COHORT and row["score_jlu"] and row["score_gau"]
    ]
    weak_plots = {row["plot_group_id"] for row in weak}
    split_plots = {
        name: {row["plot_group_id"] for row in manifests[name]}
        for name in ("finetune", "validation", "test")
    }
    overlap = {name: len(weak_plots & plots) for name, plots in split_plots.items()}
    # Related images in the gold training plots are allowed, and reported. The
    # validation and test plot groups must remain isolated from weak training.
    if overlap["validation"] or overlap["test"]:
        raise RuntimeError(f"Weak/evaluation plot leakage: {overlap}")

    widths = [high - low for row in weak for low, high in [score_interval(row)]]
    report = {
        "source": "outputs/dataset_manifests/{pretrain,finetune,validation,test}.csv",
        "target_cohort": TARGET_COHORT,
        "discordant_dual_rater_images": len(weak),
        "weak_plot_groups": len(weak_plots),
        "weak_gold_plot_overlap": overlap,
        "score_interval_width_median": median(widths),
        "score_interval_width_mean": sum(widths) / len(widths),
        "weak_images_with_both_raters_above_15": sum(
            min(float(row["score_jlu"]), float(row["score_gau"])) > 15 for row in weak
        ),
        "margin_audit": {str(margin): pair_evidence(weak, margin) for margin in (0.0, 5.0, 10.0)},
        "gold_counts": {name: len(manifests[name]) for name in ("finetune", "validation", "test")},
        "interpretation": (
            "This counts candidate pairwise constraints only. It does not establish that "
            "their ordering matches adjudicated damage severity or that model accuracy will improve."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
