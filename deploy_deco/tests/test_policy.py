import numpy as np
import pytest
import torch

from deploy_deco.policy import DECOPolicy


def test_uint8_image_is_converted_to_unit_float():
    image = np.full((4, 5, 3), 255, dtype=np.uint8)
    converted = DECOPolicy._image(image, "camera")
    assert converted.dtype == np.float32
    assert np.all(converted == 1.0)


def test_out_of_range_float_image_is_rejected():
    image = np.full((4, 5, 3), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        DECOPolicy._image(image, "camera")


def _policy(*, bread_phase: bool) -> tuple[DECOPolicy, list[int]]:
    calls: list[int] = []

    class Model:
        def __call__(self, *inputs):
            calls.append(len(inputs))
            return torch.zeros((1, 2, 3), dtype=torch.float32)

    policy = object.__new__(DECOPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.model = Model()
    policy.metadata = {"phase_count": 2} if bread_phase else {}
    policy.phase_count = 2 if bread_phase else None
    policy.image_keys = ("camera0", "camera1")
    policy.state_dim = 3
    policy.action_horizon = 2
    policy.action_dim = 3
    policy.expected_sample_hz = 20.0
    return policy, calls


def _observation():
    return {
        "camera0": np.zeros((4, 5, 3), dtype=np.uint8),
        "camera1": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation.state": np.zeros(3, dtype=np.float32),
    }


def test_bread_phase_policy_requires_phase_and_invokes_three_input_torchscript():
    policy, calls = _policy(bread_phase=True)
    with pytest.raises(ValueError, match="phase_id"):
        policy.predict(_observation(), seed=1)
    action = policy.predict(_observation(), seed=1, phase_id=1)
    assert action.shape == (2, 3)
    assert calls == [3]


@pytest.mark.parametrize("phase_id", [-1, 2, True, "0"])
def test_bread_phase_policy_rejects_invalid_phase_ids(phase_id):
    policy, _ = _policy(bread_phase=True)
    with pytest.raises(ValueError, match="phase_id"):
        policy.predict(_observation(), seed=1, phase_id=phase_id)


def test_regular_policy_keeps_two_input_torchscript_call():
    policy, calls = _policy(bread_phase=False)
    policy.predict(_observation(), seed=1)
    assert calls == [2]
