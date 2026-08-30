import numpy as np
import pytest
import torch

from deploy_deco.artifact import STAGE2_EXPORT_FORMAT, TACTILE_FIELD_ORDER
import deploy_deco.policy as policy_module
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


def _policy(
    *, bread_phase: bool = False, uses_tactile: bool = False
) -> tuple[DECOPolicy, list[int], list[tuple[torch.Tensor, ...]]]:
    calls: list[int] = []
    captured: list[tuple[torch.Tensor, ...]] = []

    class Model:
        def __call__(self, *inputs):
            calls.append(len(inputs))
            captured.append(inputs)
            return torch.full((1, 2, 3), 3.0, dtype=torch.float32)

    policy = object.__new__(DECOPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.model = Model()
    policy.metadata = {
        "format": STAGE2_EXPORT_FORMAT,
        "input": {
            "images": [1, 2, 3, 224, 224],
            "tactile_images": [1, 4, 3, 224, 224],
        },
    } if uses_tactile else ({"phase_count": 2} if bread_phase else {})
    policy.phase_count = 2 if bread_phase else None
    policy.uses_tactile = uses_tactile
    policy.tactile_keys = TACTILE_FIELD_ORDER if uses_tactile else ()
    policy.visual_hw = (224, 224) if uses_tactile else None
    policy.tactile_hw = (224, 224) if uses_tactile else None
    policy.image_keys = ("camera0", "camera1")
    policy.state_dim = 3
    policy.action_horizon = 2
    policy.action_dim = 3
    policy.expected_sample_hz = 20.0
    return policy, calls, captured


def _observation():
    return {
        "camera0": np.zeros((4, 5, 3), dtype=np.uint8),
        "camera1": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation.state": np.zeros(3, dtype=np.float32),
    }


def _stage2_observation(*, visual_hw=(224, 224), tactile_hw=(224, 224)):
    observation = {
        "camera0": np.zeros((*visual_hw, 3), dtype=np.uint8),
        "camera1": np.zeros((*visual_hw, 3), dtype=np.uint8),
        "observation.state": np.zeros(3, dtype=np.float32),
    }
    for key, value in zip(TACTILE_FIELD_ORDER, (10, 20, 30, 40), strict=True):
        observation[key] = np.full((*tactile_hw, 3), value, dtype=np.uint8)
    return observation


def test_bread_phase_policy_requires_phase_and_invokes_three_input_torchscript():
    policy, calls, _ = _policy(bread_phase=True)
    with pytest.raises(ValueError, match="phase_id"):
        policy.predict(_observation(), seed=1)
    action = policy.predict(_observation(), seed=1, phase_id=1)
    assert action.shape == (2, 3)
    assert calls == [3]


@pytest.mark.parametrize("phase_id", [-1, 2, True, "0"])
def test_bread_phase_policy_rejects_invalid_phase_ids(phase_id):
    policy, _, _ = _policy(bread_phase=True)
    with pytest.raises(ValueError, match="phase_id"):
        policy.predict(_observation(), seed=1, phase_id=phase_id)


def test_regular_policy_keeps_two_input_torchscript_call():
    policy, calls, _ = _policy()
    policy.predict(_observation(), seed=1)
    assert calls == [2]


def test_stage2_policy_stacks_tactile_streams_in_contract_order():
    policy, calls, captured = _policy(uses_tactile=True)
    action = policy.predict(_stage2_observation(), seed=1)
    assert action.shape == (2, 3)
    assert calls == [3]
    assert captured[0][0].shape == (1, 2, 3, 224, 224)
    assert captured[0][1].shape == (1, 4, 3, 224, 224)
    np.testing.assert_allclose(
        captured[0][1][0, :, 0, 0, 0].cpu().numpy(),
        np.array([10, 20, 30, 40], dtype=np.float32) / 255.0,
    )


def test_stage2_policy_rejects_missing_tactile_key_before_model_call():
    policy, calls, _ = _policy(uses_tactile=True)
    observation = _stage2_observation()
    del observation[TACTILE_FIELD_ORDER[0]]
    with pytest.raises(ValueError, match="missing keys"):
        policy.predict(observation, seed=1)
    assert calls == []


@pytest.mark.parametrize("visual_hw,tactile_hw", [((223, 224), (224, 224)), ((256, 256), (224, 224)), ((224, 224), (223, 224)), ((224, 224), (256, 256))])
def test_stage2_policy_rejects_non_contract_image_shapes_before_model_call(visual_hw, tactile_hw):
    policy, calls, _ = _policy(uses_tactile=True)
    with pytest.raises(ValueError, match="Stage 2"):
        policy.predict(_stage2_observation(visual_hw=visual_hw, tactile_hw=tactile_hw), seed=1)
    assert calls == []


def test_stage2_policy_rejects_phase_conditioning(monkeypatch):
    metadata = {
        "format": STAGE2_EXPORT_FORMAT,
        "phase_count": 2,
        "camera_names": ["camera0", "camera1"],
        "input": {
            "images": [1, 2, 3, 224, 224],
            "tactile_images": [1, 4, 3, 224, 224],
            "observation": [1, 3],
        },
        "output": {"action": [1, 2, 3]},
        "expected_sample_hz": 20.0,
    }

    monkeypatch.setattr(policy_module, "load_torchscript", lambda *args, **kwargs: (object(), metadata))
    with pytest.raises(ValueError, match="Stage 2 tactile"):
        DECOPolicy("unused", device="cpu")
