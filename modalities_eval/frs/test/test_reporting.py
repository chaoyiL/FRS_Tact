import csv
import json
import warnings

import pytest

from modalities_eval.frs.reporting import write_report


ROWS = [
    {
        "cache_index": 11,
        "source": "source/a",
        "source_index": 0,
        "source_cache_index": 1,
        "dataset_index": 101,
        "episode_index": 7,
        "condition": "baseline_fixed",
        "original_gate": 0.2,
        "counterfactual_gate": 0.2,
        "mse_gt": 3.0,
        "mse_vla": 2.0,
        "mse_vla_gt": 4.0,
        "gt_gain": 1.0,
        "repair_success": True,
        "contribution": 2.0,
    },
    {
        "cache_index": 12,
        "source": "source/a",
        "source_index": 0,
        "source_cache_index": 2,
        "dataset_index": 102,
        "episode_index": 7,
        "condition": "baseline_fixed",
        "original_gate": 0.8,
        "counterfactual_gate": 0.2,
        "mse_gt": 1.0,
        "mse_vla": 2.0,
        "mse_vla_gt": 4.0,
        "gt_gain": 3.0,
        "repair_success": True,
        "contribution": 4.0,
    },
    {
        "cache_index": 13,
        "source": "source/b",
        "source_index": 1,
        "source_cache_index": 1,
        "dataset_index": 201,
        "episode_index": 7,
        "condition": "baseline_fixed",
        "original_gate": 0.3,
        "counterfactual_gate": 0.3,
        "mse_gt": 5.0,
        "mse_vla": 2.0,
        "mse_vla_gt": 4.0,
        "gt_gain": -1.0,
        "repair_success": False,
        "contribution": 6.0,
    },
]

PROVENANCE = {
    "status": "configuration_only",
    "strong_content_hashes_verified": False,
    "override_used": True,
    "warning": "Array and encoder contents are not verified.",
}


def test_write_report_writes_required_artifacts_with_stable_columns(tmp_path):
    paths = write_report(
        ROWS,
        output_dir=tmp_path,
        bootstrap_samples=50,
        bootstrap_seed=3,
        rank_low_gate_threshold=0.25,
        rank_high_gate_threshold=0.75,
        provenance=PROVENANCE,
    )

    assert set(paths) == {"per_sample", "per_episode", "summary", "plot"}
    assert all(path.exists() for path in paths.values())

    with paths["per_sample"].open(newline="", encoding="utf-8") as file:
        per_sample = csv.DictReader(file)
        assert per_sample.fieldnames == [
            "cache_index",
            "source",
            "source_index",
            "source_cache_index",
            "dataset_index",
            "episode_index",
            "condition",
            "original_gate",
            "counterfactual_gate",
            "mse_gt",
            "mse_vla",
            "mse_vla_gt",
            "gt_gain",
            "repair_success",
            "contribution",
        ]
        assert list(per_sample)[0]["condition"] == "baseline_fixed"

    with paths["per_episode"].open(newline="", encoding="utf-8") as file:
        per_episode = list(csv.DictReader(file))
    assert len(per_episode) == 2
    assert per_episode[0]["source"] == "source/a"
    assert per_episode[0]["episode_index"] == "7"
    assert float(per_episode[0]["contribution"]) == pytest.approx(3.0)
    assert per_episode[1]["source"] == "source/b"


def test_write_report_emits_standard_json_with_null_empty_gate_strata(tmp_path):
    low_only_rows = [row for row in ROWS if row["original_gate"] < 0.5]

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        paths = write_report(
            low_only_rows,
            output_dir=tmp_path,
            bootstrap_samples=10,
            bootstrap_seed=0,
            rank_low_gate_threshold=0.3,
            rank_high_gate_threshold=0.7,
            provenance=PROVENANCE,
        )

    contents = paths["summary"].read_text(encoding="utf-8")
    assert "NaN" not in contents
    assert "Infinity" not in contents
    summary = json.loads(contents)
    assert summary["provenance"] == PROVENANCE
    assert summary["baseline_fixed"]["high"]["sample_count"] == 0
    assert summary["baseline_fixed"]["high"]["mean_contribution"] is None
    assert summary["baseline_fixed"]["high"]["mean_contribution_ci"] == {
        "lower": None,
        "upper": None,
    }
