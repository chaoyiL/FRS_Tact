from __future__ import annotations

import pathlib
from types import ModuleType, SimpleNamespace

import jax.numpy as jnp
import pytest

from modalities_eval import action_error_evaluate, loglike_evaluate

_REQUIRED_ARGS = ["--checkpoint-dir", "checkpoint", "--dataset-repo-id", "dataset"]


class _StopAfterEpisodeLoad(RuntimeError):
    pass


def _episode_with_padding(*padding_rows: list[bool]) -> SimpleNamespace:
    count = len(padding_rows)
    return SimpleNamespace(
        indices=tuple(range(count)),
        frames=tuple(range(count)),
        observations=tuple(object() for _ in range(count)),
        actions=tuple(jnp.zeros((len(row), 2)) for row in padding_rows),
        action_is_pad=tuple(jnp.asarray(row) for row in padding_rows),
        prompts=tuple("task" for _ in range(count)),
    )


@pytest.mark.parametrize("module", (action_error_evaluate, loglike_evaluate))
def test_frames_manifest_is_forwarded_to_episode_loader(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(module, "load_model_from_args", lambda args: object())

    def fake_load_episode(model, episode_index, **kwargs):
        captured.update(kwargs)
        raise _StopAfterEpisodeLoad

    monkeypatch.setattr(module, "load_episode", fake_load_episode)

    with pytest.raises(_StopAfterEpisodeLoad):
        module.main([*_REQUIRED_ARGS, "--frames", "2", "7", "11"])

    assert captured["frame_indices"] == (2, 7, 11)
    assert "sample_interval" not in captured
    assert "start_frame" not in captured


@pytest.mark.parametrize("module", (action_error_evaluate, loglike_evaluate))
def test_frames_manifest_and_sample_interval_are_mutually_exclusive(
    module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        module.main(
            [*_REQUIRED_ARGS, "--frames", "2", "7", "--sample-interval", "3"]
        )

    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("module", (action_error_evaluate, loglike_evaluate))
def test_legacy_frame_selection_is_preserved(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(module, "load_model_from_args", lambda args: object())

    def fake_load_episode(model, episode_index, **kwargs):
        captured.update(kwargs)
        raise _StopAfterEpisodeLoad

    monkeypatch.setattr(module, "load_episode", fake_load_episode)

    with pytest.raises(_StopAfterEpisodeLoad):
        module.main([*_REQUIRED_ARGS, "--frame", "5"])

    assert captured["frame_indices"] == (5,)


@pytest.mark.parametrize("module", (action_error_evaluate, loglike_evaluate))
def test_legacy_interval_selection_starts_from_frame(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(module, "load_model_from_args", lambda args: object())

    def fake_load_episode(model, episode_index, **kwargs):
        captured.update(kwargs)
        raise _StopAfterEpisodeLoad

    monkeypatch.setattr(module, "load_episode", fake_load_episode)

    with pytest.raises(_StopAfterEpisodeLoad):
        module.main(
            [*_REQUIRED_ARGS, "--frame", "5", "--sample-interval", "3"]
        )

    assert captured["start_frame"] == 5
    assert captured["sample_interval"] == 3
    assert "frame_indices" not in captured


def test_loglike_rejects_any_padded_action_before_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    computed = False
    model = SimpleNamespace(params={"weight": jnp.ones(())})
    episode = _episode_with_padding([False, False], [False, True])
    monkeypatch.setattr(loglike_evaluate, "load_model_from_args", lambda args: model)
    monkeypatch.setattr(loglike_evaluate, "load_episode", lambda *args, **kwargs: episode)

    def fake_compute(*args, **kwargs):
        nonlocal computed
        computed = True
        return []

    monkeypatch.setattr(loglike_evaluate, "compute_episode_modality_contributions", fake_compute)

    with pytest.raises(ValueError, match="H_safe"):
        loglike_evaluate.main(_REQUIRED_ARGS)

    assert not computed


def test_loglike_all_unpadded_actions_reach_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    computed = False
    model = SimpleNamespace(params={"weight": jnp.ones(())})
    episode = _episode_with_padding([False, False], [False, False])
    monkeypatch.setattr(loglike_evaluate, "load_model_from_args", lambda args: model)
    monkeypatch.setattr(loglike_evaluate, "load_episode", lambda *args, **kwargs: episode)

    def fake_compute(*args, **kwargs):
        nonlocal computed
        computed = True
        return []

    monkeypatch.setattr(loglike_evaluate, "compute_episode_modality_contributions", fake_compute)
    monkeypatch.setattr(
        loglike_evaluate,
        "save_contribution_curve",
        lambda *args, **kwargs: (pathlib.Path("curve.csv"), None),
    )

    loglike_evaluate.main(_REQUIRED_ARGS)

    assert computed
