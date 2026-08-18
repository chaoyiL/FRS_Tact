from __future__ import annotations

import pytest

from deploy_pi05_frs.bridge_client import RobotBridgeClient


def _client_with_message(message):
    client = object.__new__(RobotBridgeClient)
    client._receive = lambda timeout=None: message
    return client


def test_receive_action_ack_accepts_matching_sequence():
    _client_with_message({"type": "action_ack", "obs_seq": 7}).receive_action_ack(7, 1.0)


@pytest.mark.parametrize(
    "message",
    [
        {"type": "obs", "obs_seq": 7},
        {"type": "action_ack", "obs_seq": 6},
        {"type": "action_ack", "obs_seq": True},
    ],
)
def test_receive_action_ack_rejects_wrong_message_or_sequence(message):
    with pytest.raises(RuntimeError):
        _client_with_message(message).receive_action_ack(7, 1.0)
