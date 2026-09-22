import csv
import json

import pytest
from PIL import Image

from analysis.plot_score_audit import build_summary, run

FIELDS = (
    "image_id", "cohort_id", "supervision_tier", "score_jlu", "score_gau",
    "target", "target_source",
)


def _fixtures():
    inventory = {
        "totals": {"canonical_image_paths": 4, "discovered_image_paths": 5,
                   "duplicate_copies": 1},
        "cohorts": [{"cohort_id": "gg", "canonical_images": 2},
                    {"cohort_id": "wg", "canonical_images": 2}],
    }
    scored = [
        {"image_id": "g", "cohort_id": "gg", "supervision_tier": "gold",
         "score_jlu": "2", "score_gau": "4", "target": "3", "target_source": "mean_of_two_scorers"},
        {"image_id": "w", "cohort_id": "gg", "supervision_tier": "dual_weak",
         "score_jlu": "20", "score_gau": "8", "target": "14", "target_source": "mean_of_two_scorers"},
        {"image_id": "s", "cohort_id": "wg", "supervision_tier": "single_weak",
         "score_jlu": "", "score_gau": "", "target": "1", "target_source": "single_scorer"},
    ]
    weak = [scored[1], scored[2]]
    return inventory, scored, weak


def test_reconciles_unique_images_and_scorer_counts():
    inventory, scored, weak = _fixtures()
    result = build_summary(inventory, scored, weak)
    assert result["canonical_images"] == 4
    assert result["scored_images"] == 3
    assert result["unscored_images"] == 1
    assert result["supervision_tiers"] == {
        "gold": 1, "dual_weak": 1, "single_weak": 1,
    }
    assert result["dual_scored_images"] == 2
    assert result["weak_run_target_sources"] == {
        "mean_of_two_scorers": 1, "single_scorer": 1,
    }
    assert result["scorers"]["dual_weak"]["jlu_minus_gau_mean"] == 12
    assert result["scorers"]["gold"]["absolute_difference_median"] == 2


def test_invalid_gold_agreement_is_rejected():
    inventory, scored, weak = _fixtures()
    scored[0]["score_gau"] = "10"
    with pytest.raises(ValueError, match="Gold row exceeds"):
        build_summary(inventory, scored, weak)


def test_creates_three_nonempty_figures_and_auditable_tables(tmp_path):
    inventory, scored, weak = _fixtures()
    inventory_file = tmp_path / "inventory.json"
    inventory_file.write_text(json.dumps(inventory), encoding="utf-8")
    for name, rows in (("scored", scored), ("weak", weak)):
        with (tmp_path / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    output = tmp_path / "audit"
    result = run(inventory_file, tmp_path / "scored.csv", tmp_path / "weak.csv", output)
    assert result["scored_images"] == 3
    assert (output / "summary.json").is_file()
    assert (output / "cohort_breakdown.csv").is_file()
    for name in ("dataset_composition.png", "score_distributions.png",
                 "scorer_comparison.png"):
        with Image.open(output / name) as image:
            assert image.width >= 1500
            assert image.height >= 850
