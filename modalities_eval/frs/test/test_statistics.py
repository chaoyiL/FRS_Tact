import numpy as np
import pytest

from modalities_eval.frs import statistics
from modalities_eval.frs.statistics import sample_error_rows
from modalities_eval.frs.statistics import summarize_rows


GT = np.zeros((4, 1, 1), dtype=np.float32)
VLA = np.ones((4, 1, 1), dtype=np.float32)
FULL = np.zeros((4, 1, 1), dtype=np.float32)
CONDITIONS = {
    "baseline_fixed": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32).reshape(4, 1, 1),
    "current_only": np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32).reshape(4, 1, 1),
}
METADATA = [
    {"episode_index": 10, "dataset_index": 100},
    {"episode_index": 10, "dataset_index": 101},
    {"episode_index": 20, "dataset_index": 200},
    {"episode_index": 20, "dataset_index": 201},
]
ORIGINAL_GATE = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
COUNTERFACTUAL_GATES = {
    "baseline_fixed": np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
    "current_only": np.array([0.4, 0.4, 0.6, 0.6], dtype=np.float32),
}


def test_sample_error_rows_records_paired_per_sample_metrics():
    rows = sample_error_rows(
        full=FULL,
        conditions=CONDITIONS,
        vla=VLA,
        gt=GT,
        metadata=METADATA,
        original_gate=ORIGINAL_GATE,
        counterfactual_gates=COUNTERFACTUAL_GATES,
    )

    assert len(rows) == 8
    assert rows[0] == {
        "episode_index": 10,
        "dataset_index": 100,
        "condition": "baseline_fixed",
        "original_gate": pytest.approx(0.1),
        "counterfactual_gate": pytest.approx(0.9),
        "mse_gt": pytest.approx(1.0),
        "mse_vla": pytest.approx(0.0),
        "mse_vla_gt": pytest.approx(1.0),
        "gt_gain": pytest.approx(0.0),
        "repair_success": False,
        "contribution": pytest.approx(1.0),
    }


def test_summary_uses_episode_clusters_and_original_gate_regions():
    rows = sample_error_rows(
        full=FULL,
        conditions=CONDITIONS,
        vla=VLA,
        gt=GT,
        metadata=METADATA,
        original_gate=ORIGINAL_GATE,
        counterfactual_gates=COUNTERFACTUAL_GATES,
    )

    summary = summarize_rows(rows, bootstrap_samples=100, bootstrap_seed=7)

    assert summary["baseline_fixed"]["mean_contribution"] == pytest.approx(0.5)
    assert summary["baseline_fixed"]["high"]["sample_count"] == 2
    assert summary["baseline_fixed"]["high"]["mean_contribution"] == pytest.approx(0.0)
    assert summary["baseline_fixed"]["low"]["mean_contribution"] == pytest.approx(1.0)
    assert summary["baseline_fixed"]["sample_count"] == 4
    assert summary["baseline_fixed"]["episode_count"] == 2


def test_summary_uses_configured_original_gate_boundaries_for_three_strata():
    gates = [0.3, 0.300001, 0.699999, 0.7]
    rows = [
        {
            "condition": "gate_test",
            "source_index": 0,
            "episode_index": index,
            "original_gate": gate,
            "counterfactual_gate": 1.0 - gate,
            "contribution": float(index),
            "gt_gain": 0.0,
            "mse_gt": 0.0,
            "mse_vla": 0.0,
            "mse_vla_gt": 0.0,
            "repair_success": False,
        }
        for index, gate in enumerate(gates)
    ]

    summary = summarize_rows(
        rows,
        bootstrap_samples=10,
        bootstrap_seed=3,
        rank_low_gate_threshold=0.3,
        rank_high_gate_threshold=0.7,
    )["gate_test"]

    assert summary["gate_thresholds"] == {"low": 0.3, "high": 0.7}
    assert summary["low"]["sample_count"] == 1
    assert summary["low"]["mean_original_gate"] == pytest.approx(0.3)
    assert summary["transition"]["sample_count"] == 2
    assert summary["transition"]["mean_contribution"] == pytest.approx(1.5)
    assert summary["high"]["sample_count"] == 1
    assert summary["high"]["mean_original_gate"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("low", "high"),
    [(-0.1, 0.7), (0.3, 1.1), (0.7, 0.7), (float("nan"), 0.7)],
)
def test_summary_rejects_invalid_gate_thresholds(low, high):
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        summarize_rows(
            [],
            bootstrap_samples=10,
            rank_low_gate_threshold=low,
            rank_high_gate_threshold=high,
        )


def test_summary_bootstrap_is_deterministic_and_clustered_by_episode():
    rows = sample_error_rows(
        full=FULL,
        conditions=CONDITIONS,
        vla=VLA,
        gt=GT,
        metadata=METADATA,
        original_gate=ORIGINAL_GATE,
        counterfactual_gates=COUNTERFACTUAL_GATES,
    )

    first = summarize_rows(rows, bootstrap_samples=100, bootstrap_seed=7)
    second = summarize_rows(rows, bootstrap_samples=100, bootstrap_seed=7)

    assert first == second
    ci = first["baseline_fixed"]["mean_contribution_ci"]
    assert ci["lower"] == pytest.approx(0.0)
    assert ci["upper"] == pytest.approx(1.0)


def test_summary_cluster_bootstrap_resamples_whole_unequal_sized_episodes():
    gt = np.zeros((4, 1, 1), dtype=np.float32)
    rows = sample_error_rows(
        full=np.zeros_like(gt),
        conditions={
            "baseline_fixed": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(
                4, 1, 1
            )
        },
        vla=np.ones_like(gt),
        gt=gt,
        metadata=[
            {"episode_index": 1},
            {"episode_index": 2},
            {"episode_index": 2},
            {"episode_index": 2},
        ],
        original_gate=np.full(4, 0.1, dtype=np.float32),
        counterfactual_gates={"baseline_fixed": np.zeros(4, dtype=np.float32)},
    )

    summary = summarize_rows(rows, bootstrap_samples=1000, bootstrap_seed=11)

    ci = summary["baseline_fixed"]["mean_contribution_ci"]
    assert ci["lower"] == pytest.approx(0.0)
    assert ci["upper"] == pytest.approx(1.0)


def test_vectorized_bootstrap_matches_explicit_row_weighted_shared_draw_reference():
    rows = []
    for episode, gate, values in (
        (0, 0.2, ((1.0, 10.0, False), (3.0, 30.0, True))),
        (1, 0.5, ((5.0, 50.0, True),)),
        (2, 0.8, ((7.0, 70.0, False), (9.0, 90.0, True), (11.0, 110.0, True))),
    ):
        for contribution, gt_gain, repair_success in values:
            rows.append(
                {
                    "condition": "counterfactual",
                    "source": "source/a",
                    "episode_index": episode,
                    "original_gate": gate,
                    "counterfactual_gate": 1.0 - gate,
                    "contribution": contribution,
                    "gt_gain": gt_gain,
                    "mse_gt": contribution + 1.0,
                    "mse_vla": contribution + 2.0,
                    "mse_vla_gt": contribution + 3.0,
                    "repair_success": repair_success,
                }
            )

    samples = 37
    seed = 19
    actual = summarize_rows(
        rows,
        bootstrap_samples=samples,
        bootstrap_seed=seed,
        rank_low_gate_threshold=0.3,
        rank_high_gate_threshold=0.7,
    )["counterfactual"]
    rng = np.random.default_rng(seed)

    def explicit_intervals(stratum_rows):
        clusters = {}
        for row in stratum_rows:
            clusters.setdefault((row["source"], row["episode_index"]), []).append(row)
        grouped = tuple(clusters.values())
        indices = rng.integers(0, len(grouped), size=(samples, len(grouped)))

        def interval(metric):
            draws = [
                np.mean(
                    [
                        float(row[metric])
                        for cluster_index in selected
                        for row in grouped[cluster_index]
                    ]
                )
                for selected in indices
            ]
            lower, upper = np.quantile(draws, (0.025, 0.975))
            return {"lower": float(lower), "upper": float(upper)}

        return {
            **{f"mean_{metric}_ci": interval(metric) for metric in statistics._MEAN_METRICS},
            "repair_success_rate_ci": interval("repair_success"),
        }

    expected = {
        "all": explicit_intervals(rows),
        "low": explicit_intervals([row for row in rows if row["original_gate"] <= 0.3]),
        "transition": explicit_intervals(
            [row for row in rows if 0.3 < row["original_gate"] < 0.7]
        ),
        "high": explicit_intervals([row for row in rows if row["original_gate"] >= 0.7]),
    }

    for stratum, reference in expected.items():
        observed = actual if stratum == "all" else actual[stratum]
        for key, interval in reference.items():
            assert observed[key] == pytest.approx(interval)


def test_bootstrap_metric_access_scales_with_rows_not_draws(monkeypatch):
    accesses = 0

    class CountingRow(dict):
        def __getitem__(self, key):
            nonlocal accesses
            if key == "contribution":
                accesses += 1
            return super().__getitem__(key)

    monkeypatch.setattr(statistics, "_MEAN_METRICS", ("contribution",))
    rows = [
        CountingRow(
            condition="counterfactual",
            source_index=0,
            episode_index=index // 4,
            original_gate=0.1,
            contribution=float(index),
            repair_success=index % 2 == 0,
        )
        for index in range(80)
    ]

    summarize_rows(rows, bootstrap_samples=200, bootstrap_seed=5)

    # These rows participate in the overall and low strata. Pre-aggregation
    # reads each metric once per stratum, independent of the number of draws.
    assert accesses <= 2 * len(rows)


def test_summary_clusters_same_episode_index_separately_per_source():
    rows = sample_error_rows(
        full=np.zeros((2, 1, 1), dtype=np.float32),
        conditions={"baseline_fixed": np.ones((2, 1, 1), dtype=np.float32)},
        vla=np.ones((2, 1, 1), dtype=np.float32),
        gt=np.zeros((2, 1, 1), dtype=np.float32),
        metadata=[
            {"source_index": 0, "episode_index": 0},
            {"source_index": 1, "episode_index": 0},
        ],
        original_gate=np.array([0.1, 0.1], dtype=np.float32),
        counterfactual_gates={"baseline_fixed": np.zeros(2, dtype=np.float32)},
    )

    summary = summarize_rows(rows, bootstrap_samples=10, bootstrap_seed=0)

    assert summary["baseline_fixed"]["episode_count"] == 2


def test_summary_falls_back_to_source_when_source_index_is_explicitly_none():
    rows = [
        {
            "condition": "baseline_fixed",
            "source": source,
            "source_index": None,
            "episode_index": 0,
            "original_gate": 0.1,
            "counterfactual_gate": 0.0,
            "contribution": contribution,
            "gt_gain": 0.0,
            "mse_gt": 0.0,
            "mse_vla": 0.0,
            "mse_vla_gt": 0.0,
            "repair_success": False,
        }
        for source, contribution in (("source/a", 0.0), ("source/b", 1.0))
    ]

    summary = summarize_rows(rows, bootstrap_samples=20, bootstrap_seed=0)

    assert summary["baseline_fixed"]["episode_count"] == 2
