import importlib.util
import sys
import tempfile
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

policy_module = ModuleType("deploy_pi05.policy")
policy_module.Pi05RemotePolicy = object
previous_policy_module = sys.modules.get("deploy_pi05.policy")
sys.modules["deploy_pi05.policy"] = policy_module
try:
    remote_client_spec = importlib.util.spec_from_file_location(
        "deploy_pi05._remote_client_frs_test",
        Path(__file__).parents[1] / "deploy_pi05" / "remote_client.py",
    )
    assert remote_client_spec is not None
    assert remote_client_spec.loader is not None
    remote_client = importlib.util.module_from_spec(remote_client_spec)
    sys.modules[remote_client_spec.name] = remote_client
    remote_client_spec.loader.exec_module(remote_client)
finally:
    del sys.modules[remote_client_spec.name]
    if previous_policy_module is None:
        del sys.modules["deploy_pi05.policy"]
    else:
        sys.modules["deploy_pi05.policy"] = previous_policy_module
from deploy_pi05.deployment import SINGLE_RIGHT_ARM_PROFILE
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
    def __init__(self, *, profile=None, action_dim=2, selected_action=None):
        self.policy = SimpleNamespace(
            config=SimpleNamespace(
                action_horizon=2,
                state_dim=7 if profile == SINGLE_RIGHT_ARM_PROFILE else 20,
                robot_action_dim=action_dim,
                state_action_profile=profile,
            )
        )
        self.ended_chunks = []
        self.begin_observations = []
        self.steer_observations = []
        self.selected_action = (
            np.asarray(selected_action, dtype=np.float32)
            if selected_action is not None
            else None
        )

    def begin_chunk(self, chunk_id, observation, _task, *, seed, num_steps):
        self.begin_observations.append(observation)
        return SimpleNamespace(chunk_id=chunk_id)

    def steer_action(self, chunk_id, request_id, observation, action_index):
        self.steer_observations.append(observation)
        selected_action = (
            self.selected_action
            if self.selected_action is not None
            else np.array([action_index, action_index], dtype=np.float32)
        )
        return SimpleNamespace(
            chunk_id=chunk_id,
            request_id=request_id,
            action_index=action_index,
            selected_action=selected_action,
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


class RightArmProfileAdapterTest(unittest.TestCase):
    def test_single_right_profile_projects_warmup_observation_before_frs(self):
        raw_state = np.arange(20, dtype=np.float32)
        warmup_observations = []

        class FakeBridge:
            def __init__(self, **_kwargs):
                pass

            @staticmethod
            def receive_observation(*, timeout):
                del timeout
                return 1, {"observation.state": raw_state}

            @staticmethod
            def send_state(*_args, **_kwargs):
                pass

        class FakePolicy:
            robot_image_keys = ()

            def __init__(self, _config):
                self.config = SimpleNamespace(
                    state_dim=7,
                    state_action_profile=SINGLE_RIGHT_ARM_PROFILE,
                )

        class FakeFRSRuntime:
            tactile_keys = ()

            def __init__(self, _config, *, policy, **_kwargs):
                self.policy = policy

            @staticmethod
            def reset_episode(observation):
                warmup_observations.append(observation)

            @staticmethod
            def warmup(observation, _task, *, seed, sample_steps):
                del seed, sample_steps
                warmup_observations.append(observation)

        config = {
            "connection": {
                "address": "robot.example",
                "port": 8000,
                "observation_timeout_s": 1.0,
                "action_ack_timeout_s": 1.0,
            },
            "observation": {"language_prompt": "pick tube"},
            "runtime": {"warmup_runs": 1, "auto_start": True},
            "seed": 0,
            "num_steps": 1,
            "frs": {},
            "logging": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "deploy_pi05_frs_right.yaml"
            config_path.write_bytes(b"test config")
            with patch.object(
                remote_client, "load_deployment_config_bytes", return_value=config
            ), patch.object(
                remote_client,
                "make_policy_config",
                return_value=SimpleNamespace(state_dim=7),
            ), patch.object(remote_client, "_jax_runtime", return_value=("cpu", ())), patch.object(
                remote_client, "print_startup_summary"
            ), patch.object(remote_client, "Pi05RemotePolicy", FakePolicy), patch.object(
                remote_client, "FRSRuntime", FakeFRSRuntime
            ), patch.object(remote_client, "RobotBridgeClient", FakeBridge), patch.object(
                remote_client,
                "parse_gripper_hysteresis_config",
                return_value=object(),
            ), patch.object(remote_client, "parse_task_switch", return_value=0), patch.object(
                remote_client, "parse_task1_motion_gain_config", return_value=object()
            ), patch.object(
                remote_client,
                "start_observation_saver",
                return_value=None,
            ), patch.object(
                remote_client, "cleanup_deployment_resources"
            ), patch.object(remote_client, "_run_frs"):
                remote_client.run(config_path, max_iterations_override=1)

        self.assertEqual(len(warmup_observations), 2)
        for observation in warmup_observations:
            np.testing.assert_array_equal(observation["observation.state"], raw_state[7:14])

    def test_single_right_profile_projects_frs_observations_and_expands_selected_action(self):
        chunk_state = np.arange(20, dtype=np.float32)
        steer_state = np.arange(20, 40, dtype=np.float32)
        selected_right_action = np.arange(10, dtype=np.float32)
        bridge = ScriptedBridge(
            [
                FRSChunkStart(
                    0,
                    1,
                    {"observation.state": chunk_state},
                    0.0,
                    0.1,
                    2,
                    "block",
                    None,
                    None,
                ),
                FRSSteerRequest(
                    1,
                    40,
                    0,
                    None,
                    False,
                    {"observation.state": steer_state},
                ),
                FRSSteerAck(1, 40, 0, "scheduled", 1.0),
                FRSChunkEnd(1, "exhausted", 1, 0),
            ]
        )
        frs = FakeFRSRuntime(
            profile=SINGLE_RIGHT_ARM_PROFILE,
            action_dim=10,
            selected_action=selected_right_action,
        )

        with patch.object(remote_client, "_safe_trace", return_value=None), patch.object(
            remote_client,
            "prepare_observation",
            side_effect=lambda observation, **_kwargs: observation,
        ):
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

        np.testing.assert_array_equal(
            frs.begin_observations[0]["observation.state"], chunk_state[7:14]
        )
        np.testing.assert_array_equal(
            frs.steer_observations[0]["observation.state"], steer_state[7:14]
        )
        expected_wire_action = np.zeros((20,), dtype=np.float32)
        expected_wire_action[3] = 1.0
        expected_wire_action[7] = 1.0
        expected_wire_action[9] = steer_state[6]
        expected_wire_action[10:] = selected_right_action
        np.testing.assert_array_equal(bridge.sent_actions[0][3], expected_wire_action)

    def test_dual_profile_preserves_frs_observations_and_selected_action(self):
        chunk_state = np.arange(20, dtype=np.float32)
        steer_state = np.arange(20, 40, dtype=np.float32)
        selected_action = np.arange(20, dtype=np.float32)
        bridge = ScriptedBridge(
            [
                FRSChunkStart(
                    0,
                    1,
                    {"observation.state": chunk_state},
                    0.0,
                    0.1,
                    2,
                    "block",
                    None,
                    None,
                ),
                FRSSteerRequest(
                    1,
                    40,
                    0,
                    None,
                    False,
                    {"observation.state": steer_state},
                ),
                FRSSteerAck(1, 40, 0, "scheduled", 1.0),
                FRSChunkEnd(1, "exhausted", 1, 0),
            ]
        )
        frs = FakeFRSRuntime(action_dim=20, selected_action=selected_action)

        with patch.object(remote_client, "_safe_trace", return_value=None), patch.object(
            remote_client,
            "prepare_observation",
            side_effect=lambda observation, **_kwargs: observation,
        ):
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

        np.testing.assert_array_equal(
            frs.begin_observations[0]["observation.state"], chunk_state
        )
        np.testing.assert_array_equal(
            frs.steer_observations[0]["observation.state"], steer_state
        )
        np.testing.assert_array_equal(bridge.sent_actions[0][3], selected_action)


if __name__ == "__main__":
    unittest.main()
