"""State/history alignment and request timing for the action-origin baseline."""

from types import SimpleNamespace
import numpy as np
import pytest
import torch
from train_baseline_act_inter.config import RIGHT_TACTILE_KEYS, TACTILE_KEYS
from train_baseline_act_inter.data import BaselineCacheDataset
from train_baseline_act_inter.model import (
    DirectDecoderConfig,
    fireflow_integrate_velocity,
)
from deploy_baseline_act_inter.runtime import DirectDecoderRuntime


def test_history_is_oldest_first_and_clamps_at_episode_start():
    features = np.arange(70, dtype=np.float32)[:, None, None] * np.ones((1, 4, 512), np.float32)
    identity = dict(repo_id="demo", revision=None, root="/tmp/demo")
    tactile = SimpleNamespace(
        metadata=dict(total_frames=70, tactile_keys=TACTILE_KEYS, dataset_identity=identity),
        get_many=lambda indices: features[indices],
    )
    actions = SimpleNamespace(
        manifest=dict(dataset_identity=identity),
        dataset_indices=np.array([50, 65]),
        episode_starts=np.array([50, 50]),
        states=np.arange(40, dtype=np.float32).reshape(2, 20),
        coarse_actions=np.zeros((2, 50, 20), np.float32),
        expert_actions=np.zeros((2, 50, 20), np.float32),
        valid_masks=np.ones((2, 50), bool),
        episode_indices=np.array([1, 1]),
        indices=lambda split: np.array([0, 1]),
    )
    dataset = BaselineCacheDataset(actions, tactile, "train")
    np.testing.assert_array_equal(dataset[0]["tactile"][:, 0, 0], np.full(10, 50))
    np.testing.assert_array_equal(dataset[1]["tactile"][:, 0, 0], [50, 50, 50, 50, 50, 53, 56, 59, 62, 65])
    np.testing.assert_array_equal(dataset[1]["state"], actions.states[1])
    np.testing.assert_array_equal(dataset[1]["baseline_tactile"], features[50])


def test_fireflow_uses_vla_origin_and_reference_time_schedule():
    observations = []
    start = torch.full((1, 50, 10), 2.0)

    def velocity(x, t):
        observations.append((x.clone(), float(t[0])))
        return torch.ones_like(x)

    end = fireflow_integrate_velocity(velocity, start, num_steps=10)
    torch.testing.assert_close(end, torch.full_like(start, 3.0))
    torch.testing.assert_close(observations[0][0], start, rtol=0, atol=0)
    np.testing.assert_allclose(
        [time for _, time in observations],
        [0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        atol=1e-6,
    )


@pytest.mark.parametrize("width,keys,state_dim", [(10, RIGHT_TACTILE_KEYS, 7), (20, TACTILE_KEYS, 20)])
def test_runtime_reads_latest_state_preserves_history_across_chunks_and_deduplicates(width, keys, state_dim):
    calls = []

    class Policy:
        samples = 0

        def predict_action_chunk(self, observation, task, **kwargs):
            self.samples += 1
            return np.full((1, 50, width), 0.2, np.float32)

        def normalize_state(self, state):
            return np.asarray(state, np.float32) * 2

        def unnormalize_actions(self, actions):
            return np.asarray(actions, np.float32)

    class Encoder:
        tactile_keys = keys

        def encode(self, obs):
            return np.full((1, len(keys), 512), obs[keys[0]][0, 0, 0], np.float32)

    class Decoder:
        config = DirectDecoderConfig(action_dim=width, tactile_keys=keys)

        def decode(self, coarse, tactile, *, state):
            calls.append((coarse.copy(), tactile.copy(), state.copy()))
            return coarse + 0.1

    def obs(value):
        return {
            "observation.state": np.full(20, value, np.float32),
            **{key: np.full((2, 2, 3), value, np.uint8) for key in keys},
        }

    policy = Policy()
    runtime = DirectDecoderRuntime(
        policy=policy,
        tactile_encoder=Encoder(),
        decoder=Decoder(),
        action_dim=width,
        max_normalized_action_abs=8,
        max_normalized_delta_rms=4,
    )
    runtime.reset_episode(obs(1))
    runtime.begin_chunk(1, obs(1), "task")
    result = runtime.steer_action(1, 1, obs(2), 0)
    assert runtime.steer_action(1, 1, obs(2), 0) is result
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][2], np.full((1, state_dim), 4, np.float32))
    assert calls[0][1].shape == (1, 10, len(keys), 512)
    np.testing.assert_array_equal(calls[0][1][0, -1], np.full((len(keys), 512), 2, np.float32))
    changed = obs(2)
    changed["observation.state"][-1] = 99
    with pytest.raises(ValueError, match="duplicate"):
        runtime.steer_action(1, 1, changed, 0)
    runtime.end_chunk(1)
    runtime.begin_chunk(2, obs(3), "task")
    for index in range(4):
        runtime.steer_action(2, index + 2, obs(index + 4), index)
    assert policy.samples == 2
    # Three requests back is value 4, retained in the previous history deque.
    assert calls[-1][1][0, -2, 0, 0] == 4
    np.testing.assert_allclose(calls[-1][0], 0.2)
    np.testing.assert_allclose(result.decoded_normalized[..., 9::10], 0.2)
