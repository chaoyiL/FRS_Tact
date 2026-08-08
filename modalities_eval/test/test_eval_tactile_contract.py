from __future__ import annotations

import argparse
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modalities_eval import utils as eval_utils
from modalities_eval.action_error_evaluate import (
    _prediction_error,
    evaluate_modality_error_change,
)
from modalities_eval.utils import (
    EvalObservation,
    SmolVLAEvalModel,
    ablate_modality_observation,
    create_velocity_context,
)
from lerobot.policies.smolvla_jax.modeling import PrefixContext


def _observation(*, tactile: bool = True) -> EvalObservation:
    return EvalObservation(
        images=jnp.ones((2, 3, 4, 4), dtype=jnp.float32),
        image_masks=jnp.ones((2,), dtype=jnp.bool_),
        language_tokens=jnp.ones((3,), dtype=jnp.int32),
        language_masks=jnp.ones((3,), dtype=jnp.bool_),
        state=jnp.ones((2,), dtype=jnp.float32),
        state_mask=jnp.asarray(True),
        tactile_images=None,
        tactile_embeddings=(
            jnp.ones((2, 3), dtype=jnp.float32) if tactile else None
        ),
        tactile_masks=jnp.ones((2,), dtype=jnp.bool_) if tactile else None,
        image_keys=("camera1", "camera2"),
        tactile_keys=("touch_left", "touch_right") if tactile else (),
    )


def test_evaluator_prepares_loaded_master_params_for_compute(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace()
    master_params = {"model.action_in_proj.weight": jnp.ones((1, 1), dtype=jnp.float32)}
    prepared_params = {"prepared": jnp.ones((1,), dtype=jnp.bfloat16)}
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(eval_utils, "resolve_checkpoint", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        eval_utils,
        "JaxSmolVLAConfig",
        SimpleNamespace(from_pretrained=lambda checkpoint: config),
    )
    monkeypatch.setattr(eval_utils, "load_params", lambda checkpoint: master_params)
    monkeypatch.setattr(eval_utils, "JaxSmolVLA", lambda cfg: SimpleNamespace(config=cfg))
    monkeypatch.setattr(
        eval_utils,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(
            root=tmp_path,
            revision="revision",
            features={"action": {}},
            stats={},
        ),
    )
    monkeypatch.setattr(eval_utils, "resolve_action_key", lambda *args, **kwargs: "action")
    monkeypatch.setattr(
        eval_utils,
        "JaxSmolVLAPreprocessor",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def record_prepare(params: object, cfg: object) -> object:
        calls.append((params, cfg))
        return prepared_params

    monkeypatch.setattr(eval_utils, "prepare_params_for_compute", record_prepare)

    evaluator = SmolVLAEvalModel(
        tmp_path,
        dataset_repo_id="owner/data",
        normalization_source="checkpoint",
    )

    assert calls == [(master_params, config)]
    assert evaluator.params is prepared_params


def test_evaluation_normalization_defaults_to_checkpoint_everywhere() -> None:
    assert inspect.signature(SmolVLAEvalModel).parameters["normalization_source"].default == (
        "checkpoint"
    )
    assert inspect.signature(eval_utils.load_model).parameters["normalization_source"].default == (
        "checkpoint"
    )
    parser = argparse.ArgumentParser()
    eval_utils.add_eval_data_arguments(parser, required=False)
    args = parser.parse_args([])
    assert args.normalization_source == "checkpoint"
    assert args.unsafe_legacy_dataset_normalization is False


def test_protocol_checkpoint_rejects_dataset_stats_without_explicit_unsafe_override(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "normalization_manifest.json").write_text("{}\n", encoding="utf-8")
    stats_reads: list[str] = []

    class Metadata:
        root = tmp_path
        revision = "revision"
        features = {"action": {}}

        @property
        def stats(self):
            stats_reads.append("dataset_stats")
            return {}

    monkeypatch.setattr(eval_utils, "resolve_checkpoint", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        eval_utils,
        "JaxSmolVLAConfig",
        SimpleNamespace(from_pretrained=lambda checkpoint: SimpleNamespace()),
    )
    monkeypatch.setattr(eval_utils, "load_params", lambda checkpoint: {})
    monkeypatch.setattr(eval_utils, "prepare_params_for_compute", lambda params, config: params)
    monkeypatch.setattr(eval_utils, "JaxSmolVLA", lambda config: SimpleNamespace())
    monkeypatch.setattr(eval_utils, "LeRobotDatasetMetadata", lambda *args, **kwargs: Metadata())
    monkeypatch.setattr(eval_utils, "resolve_action_key", lambda *args, **kwargs: "action")
    monkeypatch.setattr(
        eval_utils,
        "JaxSmolVLAPreprocessor",
        lambda *args, **kwargs: pytest.fail("unsafe dataset normalization must fail first"),
    )

    with pytest.raises(ValueError, match="unsafe|train-only|protocol"):
        SmolVLAEvalModel(
            tmp_path,
            dataset_repo_id="owner/data",
            normalization_source="dataset",
        )
    assert stats_reads == [], "protocol checkpoints must fail before global dataset stats are read"


def test_protocol_checkpoint_requires_complete_checkpoint_assets_before_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "normalization_manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(eval_utils, "resolve_checkpoint", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        eval_utils,
        "JaxSmolVLAConfig",
        SimpleNamespace(from_pretrained=lambda checkpoint: SimpleNamespace()),
    )
    monkeypatch.setattr(eval_utils, "load_params", lambda checkpoint: {})
    monkeypatch.setattr(eval_utils, "prepare_params_for_compute", lambda params, config: params)
    monkeypatch.setattr(eval_utils, "JaxSmolVLA", lambda config: SimpleNamespace())
    monkeypatch.setattr(
        eval_utils,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: pytest.fail("protocol integrity must fail before dataset metadata"),
    )

    with pytest.raises(ValueError, match="normalization protocol|data_split|manifest|asset"):
        SmolVLAEvalModel(tmp_path, dataset_repo_id="owner/data")


def test_unsafe_dataset_override_cannot_bypass_protocol_integrity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "normalization_manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(eval_utils, "resolve_checkpoint", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        eval_utils,
        "JaxSmolVLAConfig",
        SimpleNamespace(from_pretrained=lambda checkpoint: SimpleNamespace()),
    )
    monkeypatch.setattr(eval_utils, "load_params", lambda checkpoint: {})
    monkeypatch.setattr(eval_utils, "prepare_params_for_compute", lambda params, config: params)
    monkeypatch.setattr(eval_utils, "JaxSmolVLA", lambda config: SimpleNamespace())
    monkeypatch.setattr(
        eval_utils,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: pytest.fail("unsafe override must not bypass protocol integrity"),
    )

    with pytest.raises(ValueError, match="normalization protocol|data_split|manifest|asset"):
        SmolVLAEvalModel(
            tmp_path,
            dataset_repo_id="owner/data",
            normalization_source="dataset",
            unsafe_legacy_dataset_normalization=True,
        )


def test_legacy_dataset_normalization_requires_named_unsafe_override(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    stats = {"action": {"mean": np.zeros(1), "std": np.ones(1)}}
    metadata = SimpleNamespace(
        root=tmp_path,
        revision="revision",
        features={"action": {}},
        stats=stats,
    )
    monkeypatch.setattr(eval_utils, "resolve_checkpoint", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        eval_utils,
        "JaxSmolVLAConfig",
        SimpleNamespace(from_pretrained=lambda checkpoint: SimpleNamespace()),
    )
    monkeypatch.setattr(eval_utils, "load_params", lambda checkpoint: {})
    monkeypatch.setattr(eval_utils, "prepare_params_for_compute", lambda params, config: params)
    monkeypatch.setattr(eval_utils, "JaxSmolVLA", lambda config: SimpleNamespace())
    monkeypatch.setattr(eval_utils, "LeRobotDatasetMetadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(eval_utils, "resolve_action_key", lambda *args, **kwargs: "action")
    monkeypatch.setattr(eval_utils, "canonicalize_dataset_stats", lambda value, key: value)

    def preprocessor(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(eval_utils, "JaxSmolVLAPreprocessor", preprocessor)

    evaluator = SmolVLAEvalModel(
        tmp_path,
        dataset_repo_id="owner/data",
        normalization_source="dataset",
        unsafe_legacy_dataset_normalization=True,
    )
    assert evaluator.normalization_source == "dataset"
    assert captured["stats"] is stats


def test_prepare_sample_preserves_independent_tactile_and_action_padding() -> None:
    evaluator = object.__new__(SmolVLAEvalModel)
    evaluator.action_key = "actions"
    evaluator.config = SimpleNamespace(tactile_keys=("touch_left", "touch_right"))
    evaluator.image_keys_for_sample = lambda sample: ("camera1", "camera2")
    evaluator.preprocessor = SimpleNamespace(
        prepare=lambda observation, prompt: {
            "images": jnp.ones((1, 2, 3, 4, 4)),
            "image_masks": jnp.ones((1, 2), dtype=jnp.bool_),
            "language_tokens": jnp.ones((1, 3), dtype=jnp.int32),
            "language_masks": jnp.ones((1, 3), dtype=jnp.bool_),
            "state": jnp.ones((1, 2)),
            "tactile_embeddings": jnp.ones((1, 2, 3)),
            "tactile_masks": jnp.ones((1, 2), dtype=jnp.bool_),
        },
        normalize_actions=lambda actions: actions,
    )
    sample = {
        "task": "pick",
        "actions": np.ones((3, 2), dtype=np.float32),
        "actions_is_pad": np.asarray([False, True, True]),
    }

    observation, actions, action_is_pad, prompt = evaluator.prepare_sample(sample)

    assert observation.tactile_images is None
    assert observation.tactile_embeddings.shape == (2, 3)
    assert observation.tactile_masks.tolist() == [True, True]
    assert observation.tactile_keys == ("touch_left", "touch_right")
    assert observation.state_mask.tolist() is True
    assert actions.shape == (3, 2)
    assert action_is_pad.tolist() == [False, True, True]
    assert prompt == "pick"


def test_prepare_sample_accepts_visual_config_without_tactile_keys() -> None:
    evaluator = object.__new__(SmolVLAEvalModel)
    evaluator.action_key = "action"
    evaluator.config = SimpleNamespace(tactile_keys=None)
    evaluator.image_keys_for_sample = lambda sample: ("camera1",)
    evaluator.preprocessor = SimpleNamespace(
        prepare=lambda observation, prompt: {
            "images": jnp.ones((1, 1, 3, 4, 4)),
            "image_masks": jnp.ones((1, 1), dtype=jnp.bool_),
            "language_tokens": jnp.ones((1, 3), dtype=jnp.int32),
            "language_masks": jnp.ones((1, 3), dtype=jnp.bool_),
            "state": jnp.ones((1, 2)),
        },
        normalize_actions=lambda actions: actions,
    )

    observation, _, _, _ = evaluator.prepare_sample(
        {"action": np.ones((2, 2), dtype=np.float32)}
    )

    assert observation.tactile_keys == ()


def test_prepare_and_sampling_forward_raw_tactile_images() -> None:
    evaluator = object.__new__(SmolVLAEvalModel)
    evaluator.action_key = "action"
    evaluator.config = SimpleNamespace(
        tactile_keys=("touch_left", "touch_right"),
        chunk_size=2,
        max_action_dim=2,
        action_dim=2,
    )
    evaluator.image_keys_for_sample = lambda sample: ("camera1",)
    evaluator.preprocessor = SimpleNamespace(
        prepare=lambda observation, prompt: {
            "images": jnp.ones((1, 1, 3, 4, 4)),
            "image_masks": jnp.ones((1, 1), dtype=jnp.bool_),
            "language_tokens": jnp.ones((1, 3), dtype=jnp.int32),
            "language_masks": jnp.ones((1, 3), dtype=jnp.bool_),
            "state": jnp.ones((1, 2)),
            "tactile_images": jnp.ones((1, 2, 3, 8, 8)),
            "tactile_masks": jnp.ones((1, 2), dtype=jnp.bool_),
        },
        normalize_actions=lambda actions: actions,
    )
    captured: dict[str, object] = {}

    class FakeFunctionalModel:
        def sample_actions(self, params, *args, **kwargs):
            captured.update(kwargs)
            return kwargs["noise"]

    evaluator.params = {}
    evaluator.model = FakeFunctionalModel()
    evaluator._sample_cache = {}
    observation, _, _, _ = evaluator.prepare_sample(
        {
            "action": np.ones((2, 2), dtype=np.float32),
            "observation.images.touch_left": np.ones((3, 8, 8), dtype=np.float32),
            "observation.images.touch_right": np.ones((3, 8, 8), dtype=np.float32),
        }
    )

    assert observation.tactile_images.shape == (2, 3, 8, 8)
    assert observation.tactile_embeddings is None
    batched = jax.tree.map(lambda value: value if value is None else value[None, ...], observation)
    evaluator.sample_actions(
        jax.random.key(0),
        batched,
        num_steps=1,
        noise=jnp.ones((1, 2, 2), dtype=jnp.float32),
    )
    assert captured["tactile_images"] is not None
    assert captured["tactile_embeddings"] is None
    assert captured["tactile_masks"] is not None


def test_tactile_and_state_ablation_only_change_their_masks() -> None:
    observation = _observation()

    without_tactile = ablate_modality_observation(observation, modality="tactile")
    without_state = ablate_modality_observation(observation, modality="state")

    np.testing.assert_array_equal(without_tactile.tactile_masks, [False, False])
    np.testing.assert_array_equal(without_tactile.image_masks, observation.image_masks)
    np.testing.assert_array_equal(without_tactile.tactile_embeddings, observation.tactile_embeddings)
    assert without_state.state_mask.tolist() is False
    np.testing.assert_array_equal(without_state.state, observation.state)


def test_tactile_ablation_is_explicitly_not_applicable_to_visual_model() -> None:
    with pytest.raises(ValueError, match="does not use tactile"):
        ablate_modality_observation(_observation(tactile=False), modality="tactile")


def test_context_and_sampling_forward_independent_tactile_and_state_masks() -> None:
    captured: dict[str, object] = {}

    class FakeFunctionalModel:
        def build_prefix_context(self, params, *args, **kwargs):
            captured["context"] = kwargs
            return PrefixContext(pad_mask=jnp.ones((1, 1), dtype=jnp.bool_), cache=())

        def sample_actions(self, params, *args, **kwargs):
            captured["sample"] = kwargs
            return kwargs["noise"][..., :2]

    evaluator = object.__new__(SmolVLAEvalModel)
    evaluator.params = {}
    evaluator.model = FakeFunctionalModel()
    evaluator.config = SimpleNamespace(chunk_size=2, max_action_dim=3, action_dim=2)
    evaluator._sample_cache = {}
    observation = _observation()
    batched = jax.tree.map(lambda value: value if value is None else value[None, ...], observation)

    create_velocity_context(evaluator, batched)
    evaluator.sample_actions(
        jax.random.key(0),
        batched,
        num_steps=1,
        noise=jnp.ones((1, 2, 3), dtype=jnp.float32),
    )

    for call in (captured["context"], captured["sample"]):
        assert call["state_mask"] is not None
        assert call["tactile_images"] is None
        assert call["tactile_embeddings"] is not None
        assert call["tactile_masks"] is not None


def test_prediction_error_ignores_padded_action_steps() -> None:
    predicted = jnp.asarray([[[1.0], [100.0], [3.0]]])
    reference = jnp.zeros_like(predicted)
    padding = jnp.asarray([[False, True, False]])

    result = _prediction_error(predicted, reference, action_is_pad=padding)

    np.testing.assert_allclose(result.mse, [5.0])
    np.testing.assert_allclose(result.mae, [2.0])


def test_action_error_reports_physical_metrics_at_common_horizon() -> None:
    class FakeEvalModel:
        config = SimpleNamespace(chunk_size=3, max_action_dim=2, action_dim=2)
        preprocessor = SimpleNamespace(unnormalize_actions=lambda actions: actions * 10.0)

        def sample_actions(self, rng, observation, *, num_steps, noise):
            del rng, observation, num_steps, noise
            return jnp.ones((1, 3, 2), dtype=jnp.float32)

    original, ablated, delta = evaluate_modality_error_change(
        FakeEvalModel(),
        _observation(tactile=False),
        jnp.zeros((3, 2), dtype=jnp.float32),
        action_is_pad=jnp.asarray([False, True, False]),
        evaluation_horizon=2,
        modality="language_prompt",
        num_steps=1,
        rng=jax.random.key(0),
    )

    np.testing.assert_allclose(original.mse, [1.0])
    np.testing.assert_allclose(original.physical_mse, [100.0])
    np.testing.assert_allclose(original.physical_rmse, [10.0])
    np.testing.assert_allclose(original.physical_mae, [10.0])
    assert original.actions.shape == (1, 2, 2)
    assert original.physical_actions.shape == (1, 2, 2)
    np.testing.assert_allclose(ablated.physical_mse, original.physical_mse)
    np.testing.assert_allclose(delta, [0.0])
