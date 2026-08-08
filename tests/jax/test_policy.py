from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.flax import save_file as save_safetensors_file

from train_smolvla import policy as policy_module
from train_smolvla.policy import JaxSmolVLAPolicy


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
