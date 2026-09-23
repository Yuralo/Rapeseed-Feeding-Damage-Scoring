"""Small end-to-end check of guarded ranking, fold selection, and matched evaluation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from experiments.consensus_rank_transfer.compare import run as compare
from experiments.consensus_rank_transfer.config import load_config
from experiments.consensus_rank_transfer.cv import run as select
from experiments.consensus_rank_transfer.engine import (
    ARMS,
    Pool,
    candidate_pairs,
    load_pool,
    target_weak_pool,
)
from experiments.consensus_rank_transfer.evaluate import run as evaluate
from experiments.consensus_rank_transfer.fit import run as fit
from experiments.consensus_rank_transfer.sensitivity import run as sensitivity


def _pool(count: int, prefix: str, *, offset: float = 0,
          cohort: str = "gg_insects_t1_bbch10") -> Pool:
    x = np.arange(count, dtype=np.float32)
    score = x / 2 + offset
    return Pool(
        features=np.stack([x / 10, np.sin(x), np.cos(x), np.ones(count)], axis=1),
        base=score.copy(), target=score + np.sin(x) / 3,
        group=np.array([f"{prefix}_plot_{i // 2}" for i in range(count)]),
        filename=np.array([f"{prefix}_{i}.jpg" for i in range(count)]),
        cohort=np.array([cohort] * count), source_path=np.array(["/tmp/absent.jpg"] * count),
        score_jlu=score + np.where(x % 2 == 0, 8, 0),
        score_gau=score + np.where(x % 2 == 0, 0, 8),
    )


def _concat(*pools: Pool) -> Pool:
    return Pool(**{field: np.concatenate([getattr(pool, field) for pool in pools])
                   for field in Pool.__dataclass_fields__})


def test_rank_pipeline_and_cache_identity(tmp_path):
    config = replace(
        load_config("experiments/consensus_rank_transfer/config.toml"),
        run_dir=str(tmp_path), seeds=(42,), cv_folds=2, max_epochs=3,
        rank_weights=(0.1,), residual_penalties=(0.01,), midpoint_weights=(0.1,),
        bootstrap_replicates=40,
    )
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    pools = {
        "finetune": _pool(20, "gold"),
        "validation": _pool(8, "validation"),
        "test": _pool(8, "test"),
        "pretrain": _concat(
            _pool(60, "weak", offset=10),
            _pool(6, "wg", cohort="wg_insects_t1_bbch10"),
            _pool(6, "dsv", cohort="dsv_asendorf_t1_bbch11"),
        ),
    }
    summary = {"splits": {}}
    for split, pool in pools.items():
        path = frozen / f"{split}.npz"
        np.savez_compressed(path, **{field: getattr(pool, field)
                                     for field in Pool.__dataclass_fields__})
        summary["splits"][split] = {"npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (frozen / "summary.json").write_text(json.dumps(summary))

    weak = target_weak_pool(tmp_path, config)
    higher, lower, weights, pair_info = candidate_pairs(weak, np.arange(len(weak)), config, 42)
    assert pair_info["sampled_pairs"] > 0
    assert np.all(np.minimum(weak.score_jlu[higher], weak.score_gau[higher]) >
                  np.maximum(weak.score_jlu[lower], weak.score_gau[lower]) +
                  config.rank_interval_margin)
    assert np.all(weak.group[higher] != weak.group[lower])
    shuffled_high, shuffled_low, shuffled_weights, _ = candidate_pairs(
        weak, np.arange(len(weak)), config, 42, shuffled=True
    )
    assert np.array_equal(weights, shuffled_weights)
    assert np.array_equal(np.sort(np.stack([higher, lower], axis=1), axis=1),
                          np.sort(np.stack([shuffled_high, shuffled_low], axis=1), axis=1))

    selection = select(config)
    assert selection["validation_used"] is False
    assert selection["base_was_trained_on_all_cv_gold_images"] is True
    for arm in ARMS:
        fit(config, arm, 42)
    validation = compare(config, "validation")["validation"]
    assert validation["images"] == 8
    assert set(validation["decision_gates"]) == {
        "rank_beats_base_by_0_25", "rank_vs_base_interval_excludes_zero",
        "rank_beats_gold_only_by_0_25", "rank_vs_gold_only_interval_excludes_zero",
        "rank_vs_shuffled_interval_excludes_zero",
        "low_score_degradation_at_most_0_25",
    }
    for arm in ARMS:
        evaluate(config, "test", arm, 42)
        evaluate(config, "ood", arm, 42)
    assert compare(config, "test")["test"]["images"] == 8
    ood = compare(config, "ood")
    assert ood["ood_wg_insects_t1_bbch10"]["images"] == 6
    assert ood["ood_dsv_asendorf_t1_bbch11"]["label_quality"] == "single_rater_weak"
    assert sensitivity(config)["results"]["validation"]["images"] == 8

    if importlib.util.find_spec("matplotlib") is not None:
        from experiments.consensus_rank_transfer.visualize import run as visualize

        (tmp_path / "feasibility.json").write_text(json.dumps({"margin_audit": {
            "0.0": {"interval_dominance_pairs": 100},
            "5.0": {"interval_dominance_pairs": 50},
            "10.0": {"interval_dominance_pairs": 10},
        }}))
        figures = visualize(config)
        assert "validation_diagnostics.png" in figures["figures"]
        assert "ood_wg_insects_t1_bbch10_bootstrap.png" in figures["figures"]
        assert "pair_margin_sensitivity.png" in figures["figures"]
        assert (tmp_path / "figures" / "cv_curves.png").is_file()
        assert (tmp_path / "figures" / "index.html").is_file()
        assert (tmp_path / "figures" / "validation_error_anatomy.png").is_file()
        assert (tmp_path / "figures" / "validation_plot_changes.csv").is_file()
        assert (tmp_path / "figures" / "sampled_rater_disagreement.png").is_file()
        assert len(figures["splits"]) == 4

        # A result-only copy can be audited without frozen features or checkpoints.
        portable = tmp_path / "portable"
        portable.mkdir()
        for artifact in tmp_path.glob("*_comparison.json"):
            shutil.copy2(artifact, portable / artifact.name)
        for artifact in tmp_path.glob("*_matched_predictions.csv"):
            shutil.copy2(artifact, portable / artifact.name)
        from PIL import Image

        image_root = tmp_path / "photographs"
        image_root.mkdir()
        Image.new("RGB", (64, 64), "green").save(image_root / "validation_0.jpg")
        portable_figures = visualize(replace(config, run_dir=str(portable)), image_roots=(image_root,))
        assert "validation_improvements.png" in portable_figures["figures"] or \
            "validation_regressions.png" in portable_figures["figures"]
        assert "cv_selection.json" in portable_figures["missing"]
        assert (portable / "figures" / "index.html").is_file()

    # The frozen arrays are part of the experiment identity, not a mutable side input.
    with (frozen / "finetune.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed since preparation"):
        load_pool(tmp_path, "finetune")
