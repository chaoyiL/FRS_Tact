"""Tests for the vendored openpi training stack as this repo wires it up.

These cover the pieces that are *not* verbatim upstream -- the config registry, the pick_tube data
config and its policy transforms, and the multi-dataset rename plumbing in `data_loader` -- plus a
couple of upstream behaviours the repo depends on (LoRA weights being filled in rather than
demanded from the base checkpoint). Nothing here touches a GPU, a dataset, or the network.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lerobot.policies.pi05_jax import model, policy_config, transforms
from lerobot.policies.pi05_jax.model import IMAGE_KEYS, ModelType
from lerobot.policies.pi05_jax.pi0_config import Pi0Config
from lerobot.policies.pi05_jax.policies.pick_tube_policy import PickTubeInputs, PickTubeOutputs
from lerobot.policies.pi05_jax.training import (
    config as _config,
    data_loader as _data_loader,
    optimizer as _optimizer,
    weight_loaders,
)


def test_registered_configs_match_the_pi05_base_checkpoint_geometry() -> None:
    config = _config.get_config("pi05_pick_tube")

    assert config.model.model_type is ModelType.PI05
    # action_horizon 50 is not a placeholder: it is what pi05_base actually restores with.
    assert (config.model.action_dim, config.model.action_horizon, config.model.max_token_len) == (32, 50, 200)
    assert config.weight_loader.params_path.endswith("pi05_base/params")
    # LoRA fine-tune: EMA off, and the freeze filter must actually freeze something.
    assert config.ema_decay is None
    assert config.freeze_filter is not None
    assert len(config.data.sources) == 4


def test_unknown_config_name_suggests_a_close_match() -> None:
    with pytest.raises(ValueError, match="Did you mean"):
        _config.get_config("pi05_pick_tub")


def test_pick_tube_data_config_builds_the_openpi_transform_chain(tmp_path) -> None:
    config = _config.get_config("pi05_pick_tube")
    # No norm stats on disk -> `_load_norm_stats` logs and returns None rather than raising.
    data_config = config.data.create(tmp_path, config.model)

    assert data_config.norm_stats is None
    assert data_config.asset_id == "pick_tube"
    # pi0.5 discretizes state into the prompt assuming [-1, 1]; that requires quantile norm.
    assert data_config.use_quantile_norm is True

    repack = data_config.repack_transforms.inputs[0]
    assert isinstance(repack, transforms.RepackTransform)
    assert repack.structure["image"] == {
        "left_wrist_0_rgb": "observation.images.camera1",
        "right_wrist_0_rgb": "observation.images.camera2",
    }

    assert isinstance(data_config.data_transforms.inputs[0], PickTubeInputs)
    assert isinstance(data_config.data_transforms.outputs[0], PickTubeOutputs)

    model_steps = data_config.model_transforms.inputs
    tokenize = next(step for step in model_steps if isinstance(step, transforms.TokenizePrompt))
    assert tokenize.discrete_state_input is True
    pad = next(step for step in model_steps if isinstance(step, transforms.PadStatesAndActions))
    assert pad.model_action_dim == config.model.action_dim


def test_pick_tube_inputs_masks_off_the_absent_third_person_camera() -> None:
    # LeRobot hands back float32 CHW in [0, 1]; the model needs uint8 HWC.
    frame = np.zeros((3, 240, 320), dtype=np.float32)
    frame[0] = 1.0
    out = PickTubeInputs(model_type=ModelType.PI05)(
        {
            "image": {"left_wrist_0_rgb": frame, "right_wrist_0_rgb": frame},
            "state": np.zeros(20, dtype=np.float32),
            "actions": np.zeros((50, 20), dtype=np.float32),
            "prompt": np.asarray("pick the tube"),
        }
    )

    assert set(out["image"]) == set(IMAGE_KEYS)
    assert out["image"]["left_wrist_0_rgb"].shape == (240, 320, 3)
    assert out["image"]["left_wrist_0_rgb"].dtype == np.uint8
    # Red channel came first in CHW; after the transpose it must be channel index 0 of HWC.
    assert out["image"]["left_wrist_0_rgb"][0, 0, 0] == 255
    assert bool(out["image_mask"]["left_wrist_0_rgb"]) is True
    assert bool(out["image_mask"]["base_0_rgb"]) is False
    np.testing.assert_array_equal(out["image"]["base_0_rgb"], np.zeros((240, 320, 3), dtype=np.uint8))


def test_pick_tube_inputs_rejects_unknown_image_slots() -> None:
    with pytest.raises(ValueError, match="subset"):
        PickTubeInputs(model_type=ModelType.PI05)(
            {"image": {"nonexistent_rgb": np.zeros((3, 4, 4), dtype=np.uint8)}, "state": np.zeros(20)}
        )


def test_pick_tube_outputs_strips_the_model_action_padding() -> None:
    padded = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)
    out = PickTubeOutputs(action_dim=20)({"actions": padded})
    assert out["actions"].shape == (50, 20)
    np.testing.assert_array_equal(out["actions"], padded[:, :20])


def test_rename_keys_only_touches_listed_keys() -> None:
    rename = _data_loader.RenameKeys(
        {"observation.images.camera0": "observation.images.camera1", "actions": "actions"}
    )
    out = rename({"observation.images.camera0": 1, "observation.state": 2, "actions": 3})
    assert out == {"observation.images.camera1": 1, "observation.state": 2, "actions": 3}


def test_prompt_from_task_requires_a_task_column() -> None:
    assert str(_data_loader.PromptFromTask()({"task": "pick it up"})["prompt"]) == "pick it up"
    with pytest.raises(ValueError, match="task"):
        _data_loader.PromptFromTask()({"state": np.zeros(3)})


def test_create_torch_dataset_rejects_a_config_without_sources() -> None:
    config = _config.get_config("pi05_pick_tube")
    data_config = _config.DataConfig(repo_id="org/data", sources=())
    with pytest.raises(ValueError, match="sources"):
        _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)


def test_checkpoint_weight_loader_fills_in_missing_lora_weights() -> None:
    """LoRA weights do not exist in pi05_base, so the loader must synthesize them from the freshly
    initialized params instead of failing -- otherwise every LoRA config would refuse to start."""
    loaded = {"PaliGemma": {"llm": {"w": np.zeros((2, 2), dtype=np.float32)}}}
    reference = {
        "PaliGemma": {
            "llm": {
                "w": np.ones((2, 2), dtype=np.float32),
                "w_lora_a": np.full((2, 2), 7.0, dtype=np.float32),
            }
        }
    }

    merged = weight_loaders._merge_params(loaded, reference, missing_regex=".*lora.*")

    np.testing.assert_array_equal(merged["PaliGemma"]["llm"]["w"], np.zeros((2, 2)))
    np.testing.assert_array_equal(merged["PaliGemma"]["llm"]["w_lora_a"], np.full((2, 2), 7.0))


def _observation_dict(image, mask, state):
    return {
        "image": dict.fromkeys(IMAGE_KEYS, image),
        "image_mask": dict.fromkeys(IMAGE_KEYS, mask),
        "state": state,
        "tokenized_prompt": np.zeros(200, dtype=np.int32),
        "tokenized_prompt_mask": np.ones(200, dtype=bool),
    }


def test_observation_requires_one_array_type_across_all_fields() -> None:
    """Pins why `Pi05SampleProcessor.prepare_sample` numpy-ifies before `Observation.from_dict`.

    `Observation` is `@at.typecheck`'d and its fields share one `ArrayT` TypeVar. The transform
    chain naturally produces a mix -- `ResizeImages` is jitted so it returns *jax* arrays, while
    state/tokens stay numpy, and `PickTubeInputs` emits `np.True_`, a numpy scalar that is not an
    `ndarray` at all. Without the normalization step that mix raises here rather than in training.
    """
    jax_image = jnp.zeros((224, 224, 3), dtype=jnp.float32)
    numpy_state = np.zeros(32, dtype=np.float32)

    with pytest.raises(Exception):  # noqa: B017  (beartype/jaxtyping raise their own types)
        model.Observation.from_dict(_observation_dict(jax_image, np.True_, numpy_state))

    # ...and the fix -- one np.asarray sweep, the single-sample analogue of _collate_fn -- works.
    normalized = jax.tree.map(np.asarray, _observation_dict(jax_image, np.True_, numpy_state))
    observation = model.Observation.from_dict(normalized)
    assert observation.state.shape == (32,)
    assert set(observation.images) == set(IMAGE_KEYS)


def test_loading_a_lora_checkpoint_without_lora_variants_is_refused(tmp_path) -> None:
    """The failure this guards against is silent, not loud.

    `BaseModelConfig.load` defaults to `remove_extra_params=True`, so a LoRA fine-tune loaded with
    a default `Pi0Config` would drop every LoRA weight, keep the frozen base weights, and return a
    model that is bit-for-bit the un-fine-tuned pi05_base -- with no error anywhere. That would
    silently invalidate a whole FRS action cache built on top of it.
    """
    lora_config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    base_config = dataclasses.replace(
        lora_config, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
    )
    lora_params = nnx.split(nnx.eval_shape(lora_config.create, jax.random.key(0)))[1].to_pure_dict()

    # Same config it was trained with: accepted.
    policy_config._reject_unused_params(lora_config, lora_params, tmp_path)

    # Default (non-LoRA) config: refused, and the message must name the actual cause.
    with pytest.raises(ValueError, match="LoRA"):
        policy_config._reject_unused_params(base_config, lora_params, tmp_path)


def test_cosine_schedule_warms_up_then_decays() -> None:
    schedule = _optimizer.CosineDecaySchedule(
        warmup_steps=100, peak_lr=5e-5, decay_steps=1_000, decay_lr=2.5e-6
    ).create()
    assert float(schedule(0)) < float(schedule(100))
    assert float(schedule(100)) == pytest.approx(5e-5, rel=1e-6)
    assert float(schedule(1_000)) == pytest.approx(2.5e-6, rel=1e-3)


def test_create_optimizer_clips_gradients() -> None:
    tx = _optimizer.create_optimizer(
        _optimizer.AdamW(clip_gradient_norm=1.0), _optimizer.CosineDecaySchedule(), weight_decay_mask=None
    )
    assert tx is not None
