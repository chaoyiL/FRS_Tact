from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.flax import save_file as save_safetensors_file

from modalities_eval.utils import EvalObservation
from train_smolvla import policy as policy_module
from train_smolvla.modeling import PrefixContext
from train_smolvla.policy import JaxSmolVLAPolicy
from utils import source_flow
from utils import source_model
from utils.source_model import reverse_integrate_actions

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def observation() -> dict[str, np.ndarray]:
    return {"observation.state": np.zeros(3, dtype=np.float32)}


@pytest.fixture
def policy() -> JaxSmolVLAPolicy:
    prepare_calls: list[tuple[object, str]] = []
    prepared_batch = {
        "images": jnp.full((1, 1, 3, 8, 8), 0.125, dtype=jnp.float32),
        "image_masks": jnp.ones((1, 1), dtype=jnp.bool_),
        "language_tokens": jnp.asarray([[2, 7]], dtype=jnp.int32),
        "language_masks": jnp.asarray([[True, False]], dtype=jnp.bool_),
        "state": jnp.asarray([[0.25, 0.5, 0.75]], dtype=jnp.float32),
    }

    class RecordingPreprocessor:
        def prepare(self, observation, task):
            prepare_calls.append((observation, task))
            return prepared_batch

        def unnormalize_actions(self, actions):
            raise AssertionError("reverse_action_chunk must not unnormalize actions")

    class RecordingInputDependentModel:
        def __init__(self):
            self.prefix_records: list[dict[str, np.ndarray]] = []
            self.denoise_records: list[dict[str, np.ndarray]] = []

        def _record_prefix(self, images, image_masks, language_tokens, language_masks, state):
            self.prefix_records.append(
                {
                    "images": np.asarray(images),
                    "image_masks": np.asarray(image_masks),
                    "language_tokens": np.asarray(language_tokens),
                    "language_masks": np.asarray(language_masks),
                    "state": np.asarray(state),
                }
            )

        def _record_denoise(self, parameter, context_value, pad_mask, x_t, timestep):
            self.denoise_records.append(
                {
                    "parameter": np.asarray(parameter),
                    "context_value": np.asarray(context_value),
                    "pad_mask": np.asarray(pad_mask),
                    "x_t": np.asarray(x_t),
                    "timestep": np.asarray(timestep),
                }
            )

        def build_prefix_context(
            self,
            params,
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
        ):
            del params
            jax.debug.callback(
                self._record_prefix,
                images,
                image_masks,
                language_tokens,
                language_masks,
                state,
                ordered=True,
            )
            batch_size = state.shape[0]

            def batch_sum(value):
                return jnp.asarray(value, dtype=jnp.float32).reshape(batch_size, -1).sum(axis=-1)

            context_value = (
                0.001 * batch_sum(images)
                + 0.01 * batch_sum(image_masks)
                + 0.02 * batch_sum(language_tokens)
                + 0.03 * batch_sum(language_masks)
                + 0.04 * batch_sum(state)
            )
            return PrefixContext(
                pad_mask=language_masks,
                cache=((context_value[:, None], (2.0 * context_value)[:, None]),),
            )

        def denoise_step(self, params, context, x_t, timestep):
            context_value = context.cache[0][0]
            jax.debug.callback(
                self._record_denoise,
                params["velocity_bias"],
                context_value,
                context.pad_mask,
                x_t,
                timestep,
                ordered=True,
            )
            time = timestep[:, None, None]
            conditioning = context_value[:, :, None]
            padded_width = jnp.asarray(x_t.shape[-1], dtype=jnp.float32)
            return (
                0.05 * x_t
                + params["velocity_bias"]
                + conditioning
                + jnp.square(time)
                + 0.01 * padded_width
            )

    result = object.__new__(JaxSmolVLAPolicy)
    result.config = SimpleNamespace(chunk_size=2, action_dim=3, max_action_dim=4)
    result.params = {"velocity_bias": jnp.asarray(0.2, dtype=jnp.float32)}
    result.model = RecordingInputDependentModel()
    result.preprocessor = RecordingPreprocessor()
    result.prepare_calls = prepare_calls
    result.prepared_batch = prepared_batch
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


def test_policy_reverse_core_import_does_not_load_evaluation_stack() -> None:
    script = """
import sys

class BlockEvaluationImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "modalities_eval" or fullname.startswith("modalities_eval."):
            raise RuntimeError(f"evaluation import is forbidden: {fullname}")
        return None

sys.meta_path.insert(0, BlockEvaluationImports())
import train_smolvla.policy
import utils.source_flow
assert not any(name == "modalities_eval" or name.startswith("modalities_eval.") for name in sys.modules)
assert "EvalObservation" not in vars(utils.source_flow)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


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
    jax.block_until_ready(result)

    assert result.shape == actions.shape
    assert result.dtype == jnp.float32
    assert bool(jnp.isfinite(result).all())
    assert policy.prepare_calls == [(observation, "pick the tube")]
    assert len(policy.model.prefix_records) == 1
    expected_timesteps = {
        "euler": [0.0, 0.25, 0.5, 0.75],
        "fireflow": [0.0, 0.125, 0.375, 0.625, 0.875],
        "slerpflow": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    np.testing.assert_allclose(
        [float(record["timestep"][0]) for record in policy.model.denoise_records],
        expected_timesteps[solver],
        rtol=0.0,
        atol=1e-7,
    )
    for record in policy.model.denoise_records:
        assert record["x_t"].shape == (1, 2, policy.config.max_action_dim)
        np.testing.assert_array_equal(record["x_t"][..., -1], np.zeros((1, 2)))
        np.testing.assert_array_equal(record["pad_mask"], policy.prepared_batch["language_masks"])
        np.testing.assert_allclose(record["parameter"], policy.params["velocity_bias"])
        np.testing.assert_allclose(record["context_value"], [[0.304]], rtol=0.0, atol=1e-6)


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
    assert (
        source_model.reverse_integrate_prepared_actions
        is source_flow.reverse_integrate_prepared_actions
    )
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
    jax.block_until_ready(wrapped)
    assert len(policy.model.prefix_records) == 1
    for field in ("images", "image_masks", "language_tokens", "language_masks", "state"):
        np.testing.assert_array_equal(policy.model.prefix_records[0][field], prepared[field])
    for record in policy.model.denoise_records:
        np.testing.assert_allclose(record["context_value"], [[0.304]], rtol=0.0, atol=1e-6)

    prepared_result = source_flow.reverse_integrate_prepared_actions(
        policy,
        prepared,
        actions,
        num_steps=4,
        solver="fireflow",
    )
    jax.block_until_ready(prepared_result)

    np.testing.assert_allclose(wrapped, prepared_result, rtol=0.0, atol=0.0)
