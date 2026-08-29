from __future__ import annotations

import pandas as pd
import pytest

from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    canonicalize_action_dataframe,
    canonicalize_action_mapping,
)


def test_converter_canonicalizes_action_before_writing_v30_files() -> None:
    features = {
        "observation.state": {"shape": [2]},
        "actions": {"shape": [2]},
    }
    frame = pd.DataFrame({"observation.state": [[0.0, 0.0]], "actions": [[1.0, 2.0]]})

    canonical_features = canonicalize_action_mapping(features, source="info.json")
    canonical_frame = canonicalize_action_dataframe(frame, source="episode.parquet")

    assert list(canonical_features) == ["observation.state", "action"]
    assert list(canonical_frame.columns) == ["observation.state", "action"]


@pytest.mark.parametrize(
    "value",
    [
        {"actions": {}, "action": {}},
        pd.DataFrame({"actions": [[1.0]], "action": [[1.0]]}),
    ],
)
def test_converter_refuses_action_key_collisions(value: object) -> None:
    with pytest.raises(ValueError, match="Both 'actions' and 'action'"):
        if isinstance(value, pd.DataFrame):
            canonicalize_action_dataframe(value, source="episode.parquet")
        else:
            canonicalize_action_mapping(value, source="info.json")
