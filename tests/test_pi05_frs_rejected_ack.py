from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from deploy_pi05 import remote_client
from deploy_pi05.frs_protocol import FRSChunkEnd, FRSChunkStart, FRSSteerAck, FRSSteerRequest


class ScriptedBridge:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent_actions = []

    def receive_frs_message(self, _timeout):
        return next(self.messages)

    def send_frs_chunk_ready(self, *_args):
        return None

    def send_frs_steer_action(self, chunk_id, request_id, action_index, action, *, trace):
        self.sent_actions.append((chunk_id, request_id, action_index, action.copy(), trace))


class FakeFRSRuntime:
    def __init__(self):
        self.policy = SimpleNamespace(
            config=SimpleNamespace(action_horizon=2, state_dim=1, robot_action_dim=2)
        )
        self.ended_chunks = []

    def begin_chunk(self, chunk_id, _observation, _task, *, seed, num_steps):
        return SimpleNamespace(chunk_id=chunk_id)

    def steer_action(self, chunk_id, request_id, _observation, action_index):
        return SimpleNamespace(
            chunk_id=chunk_id,
            request_id=request_id,
            action_index=action_index,
            selected_action=np.array([action_index, action_index], dtype=np.float32),
        )

    def end_chunk(self, chunk_id):
        self.ended_chunks.append(chunk_id)


class RejectedAckTest(unittest.TestCase):
    @patch.object(remote_client, "_safe_trace", return_value=None)
    @patch.object(remote_client, "submit_observation")
    @patch.object(remote_client, "prepare_observation", return_value={})
    def test_rejected_action_is_logged_and_next_action_is_processed(self, *_mocks):
        bridge = ScriptedBridge(
            [
                FRSChunkStart(0, 1, {}, 0.0, 0.1, 2, "block", None, None),
                FRSSteerRequest(1, 40, 0, None, False, {}),
                FRSSteerAck(1, 40, 0, "rejected", None),
                FRSSteerRequest(1, 41, 1, None, False, {}),
                FRSSteerAck(1, 41, 1, "scheduled", 1.0),
                FRSChunkEnd(1, "exhausted", 1, 0),
            ]
        )
        frs = FakeFRSRuntime()

        with self.assertLogs(remote_client.LOGGER, level="WARNING") as logs:
            remote_client._run_frs(
                bridge,
                frs,
                task="pick tube",
                image_keys=(),
                observation_timeout_s=1.0,
                action_ack_timeout_s=1.0,
                seed=0,
                sample_steps=1,
                max_chunks=1,
                saver=None,
            )

        self.assertEqual([sent[2] for sent in bridge.sent_actions], [0, 1])
        self.assertEqual(frs.ended_chunks, [1])
        self.assertIn("Skipping robot-rejected FRS action (1, 40, 0)", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
