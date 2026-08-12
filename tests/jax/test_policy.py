from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.flax import save_file as save_safetensors_file

from modalities_eval.utils import EvalObservation
from train_smolvla import policy as policy_module
from train_smolvla.modeling import PrefixContext
from train_smolvla.policy import JaxSmolVLAPolicy
from utils import source_model
from utils.source_model import reverse_integrate_actions


@pytest.fixture
def observation() -> dict[str, np.ndarray]:
    return {"observation.state": np.zeros(3, dtype=np.float32)}


@pytest.fixture
def policy() -> JaxSmolVLAPolicy:
    prepare_calls: list[tuple[object, str]] = []

    class RecordingPreprocessor:
        def prepare(self, observation, task):
            prepare_calls.append((observation, task))
            return {
                "images": jnp.zeros((1, 1, 3, 8, 8), dtype=jnp.float32),
                "image_masks": jnp.ones((1, 1), dtype=jnp.bool_),
                "language_tokens": jnp.ones((1, 2), dtype=jnp.int32),
                "language_masks": jnp.ones((1, 2), dtype=jnp.bool_),
                "state": jnp.zeros((1, 3), dtype=jnp.float32),
            }

        def unnormalize_actions(self, actions):
            raise AssertionError("reverse_action_chunk must not unnormalize actions")

    class ConstantVelocityModel:
        def build_prefix_context(
            self,
            params,
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
        ):
            del params, images, image_masks, language_tokens, language_masks
            return PrefixContext(
                pad_mask=jnp.ones((state.shape[0], 1), dtype=jnp.bool_),
                cache=(),
            )

        def denoise_step(self, params, context, x_t, timestep):
            del params, context, timestep
            return jnp.full_like(x_t, 0.25)

    result = object.__new__(JaxSmolVLAPolicy)
    result.config = SimpleNamespace(chunk_size=2, action_dim=3, max_action_dim=4)
    result.params = {}
    result.model = ConstantVelocityModel()
    result.preprocessor = RecordingPreprocessor()
    result.prepare_calls = prepare_calls
    return result


def _write_config(path: Path, **overrides: object) -> None:
    config = {
        "chunk_size": 2,
        "n_action_steps": 2,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [3]},
            "observation.images.camera1": {"type": "VISUAL", "shape": [3, 8, 8]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [3]}},
    }
    config.update(overrides)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_policy_loads_local_visual_checkpoint_and_samples_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(tmp_path)
    save_safetensors_file(
        {"checkpoint.marker": np.asarray([7.0], dtype=np.float32)},
        tmp_path / "model.safetensors",
    )
    calls: dict[str, object] = {}

    class OfflinePreprocessor:
        def __init__(self, checkpoint, config, **kwargs):
            calls["preprocessor"] = (Path(checkpoint), config, kwargs)

        def prepare(self, observation, task):
            calls["prepare"] = (observation, task)
            return {
                "images": jnp.zeros((1, 1, 3, 8, 8), dtype=jnp.float32),
                "image_masks": jnp.ones((1, 1), dtype=jnp.bool_),
                "language_tokens": jnp.ones((1, 2), dtype=jnp.int32),
                "language_masks": jnp.ones((1, 2), dtype=jnp.bool_),
                "state": jnp.zeros((1, 3), dtype=jnp.float32),
            }

        def unnormalize_actions(self, actions):
            return actions

    class RecordingModel:
        def sample_actions(self, params, *args, **kwargs):
            calls["params"] = params
            calls["sample_kwargs"] = kwargs
            return jnp.full((1, 2, 3), 5.0, dtype=jnp.float32)

    monkeypatch.setattr(policy_module, "JaxSmolVLAPreprocessor", OfflinePreprocessor)
    policy = JaxSmolVLAPolicy.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(policy.model, policy_module.JaxSmolVLA)
    policy.model = RecordingModel()

    actions = policy.predict_action_chunk(
        {"observation.state": np.zeros(3, dtype=np.float32)},
        "pick cube",
        noise=jnp.zeros((1, 2, 32), dtype=jnp.float32),
        jit=False,
    )

    assert policy.checkpoint == tmp_path.resolve()
    assert float(policy.params["checkpoint.marker"][0]) == 7.0
    assert calls["preprocessor"][2]["local_files_only"] is True
    assert calls["prepare"][1] == "pick cube"
    np.testing.assert_array_equal(actions, np.full((1, 2, 3), 5.0, dtype=np.float32))


def test_policy_rejects_tactile_checkpoint_through_visual_config_entry(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        use_tactile_encoder=True,
        tactile_keys=["observation.images.tactile_left_0"],
    )

    with pytest.raises(ValueError, match="train_vtsmolvla"):
        JaxSmolVLAPolicy.from_pretrained(tmp_path, local_files_only=True)


@pytest.mark.parametrize("solver", ["euler", "fireflow", "slerpflow"])
def test_reverse_action_chunk_supports_all_solvers(policy, observation, solver) -> None:
    actions = jnp.zeros((1, policy.config.chunk_size, policy.config.action_dim), dtype=jnp.float16)

    result = policy.reverse_action_chunk(
        observation,
        "pick the tube",
        actions,
        num_steps=4,
        solver=solver,
    )

    assert result.shape == actions.shape
    assert result.dtype == jnp.float32
    assert bool(jnp.isfinite(result).all())
    assert policy.prepare_calls == [(observation, "pick the tube")]


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2, 3),
        (1, 1, 3),
        (1, 2, 4),
    ],
)
def test_reverse_action_chunk_rejects_wrong_normalized_shape(policy, observation, shape) -> None:
    with pytest.raises(ValueError, match="normalized_actions must have shape"):
        policy.reverse_action_chunk(
            observation,
            "pick the tube",
            jnp.zeros(shape, dtype=jnp.float32),
            num_steps=4,
            solver="euler",
        )

    assert policy.prepare_calls == []


@pytest.mark.parametrize("invalid_value", [jnp.nan, jnp.inf])
def test_reverse_action_chunk_rejects_nonfinite_normalized_actions(
    policy,
    observation,
    invalid_value,
) -> None:
    actions = jnp.zeros((1, policy.config.chunk_size, policy.config.action_dim))
    actions = actions.at[0, 0, 0].set(invalid_value)

    with pytest.raises(ValueError, match="normalized_actions must be finite"):
        policy.reverse_action_chunk(
            observation,
            "pick the tube",
            actions,
            num_steps=4,
            solver="euler",
        )

    assert policy.prepare_calls == []


@pytest.mark.parametrize(
    "invalid_result",
    [
        jnp.zeros((1, 1, 3), dtype=jnp.float32),
        jnp.full((1, 2, 3), jnp.nan, dtype=jnp.float32),
    ],
)
def test_reverse_action_chunk_rejects_invalid_reverse_output(
    policy,
    observation,
    monkeypatch,
    invalid_result,
) -> None:
    monkeypatch.setattr(
        policy_module,
        "reverse_integrate_prepared_actions",
        lambda *args, **kwargs: invalid_result,
    )

    with pytest.raises(RuntimeError, match="invalid normalized chunk"):
        policy.reverse_action_chunk(
            observation,
            "pick the tube",
            jnp.zeros((1, 2, 3), dtype=jnp.float32),
            num_steps=4,
            solver="euler",
        )


def test_reverse_integrate_actions_eval_wrapper_matches_prepared_core(policy, observation) -> None:
    prepared = policy.preprocessor.prepare(observation, "pick the tube")
    eval_observation = EvalObservation(
        images=prepared["images"],
        image_masks=prepared["image_masks"],
        language_tokens=prepared["language_tokens"],
        language_masks=prepared["language_masks"],
        state=prepared["state"],
        image_keys=("observation.images.camera1",),
    )
    actions = jnp.arange(6, dtype=jnp.float32).reshape(1, 2, 3) / 10.0

    wrapped = reverse_integrate_actions(
        policy,
        eval_observation,
        actions,
        num_steps=4,
        solver="fireflow",
    )
    prepared_result = source_model.reverse_integrate_prepared_actions(
        policy,
        prepared,
        actions,
        num_steps=4,
        solver="fireflow",
    )

    np.testing.assert_allclose(wrapped, prepared_result, rtol=0.0, atol=0.0)
