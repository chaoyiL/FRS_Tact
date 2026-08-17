from __future__ import annotations

import numpy as np
import pytest

from deploy_pi05_frs.frs_protocol import (
    FRSChunkStart,
    FRSProtocolError,
    FRSSteerRequest,
    parse_frs_server_message,
)


def _block_start() -> dict[str, object]:
    return {
        "type": "frs_chunk_start",
        "obs_seq": 2,
        "chunk_id": 3,
        "observation": {"observation.state": np.zeros((20,), dtype=np.float32)},
        "observation_timestamp": 100.0,
        "control_dt": 0.05,
        "action_horizon": 50,
        "execution_mode": "block",
        "action_timestamps": None,
        "nominal_chunk_end": None,
    }


def test_parse_block_chunk_start() -> None:
    message = parse_frs_server_message(_block_start())

    assert isinstance(message, FRSChunkStart)
    assert message.action_horizon == 50
    assert message.execution_mode == "block"


def test_parse_steer_request_preserves_observation() -> None:
    observation = {"observation.state": np.ones((20,), dtype=np.float32)}
    message = parse_frs_server_message(
        {
            "type": "frs_steer_request",
            "chunk_id": 3,
            "request_id": 4,
            "action_index": 5,
            "target_timestamp": None,
            "protection_applied": False,
            "observation": observation,
        }
    )

    assert isinstance(message, FRSSteerRequest)
    assert message.observation is observation


@pytest.mark.parametrize(
    "message",
    [
        {**_block_start(), "chunk_id": True},
        {**_block_start(), "action_horizon": 0},
        {**_block_start(), "unexpected": True},
    ],
)
def test_reject_malformed_messages(message: dict[str, object]) -> None:
    with pytest.raises(FRSProtocolError):
        parse_frs_server_message(message)
