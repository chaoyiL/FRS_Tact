from __future__ import annotations

import pathlib
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from modalities_eval import plot_loglike_modalities


def test_evaluate_modalities_rejects_padded_actions_before_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    computed = False
    episode = SimpleNamespace(
        indices=(0,),
        frames=(0,),
        observations=(object(),),
        actions=(jnp.zeros((2, 2)),),
        action_is_pad=(jnp.asarray([False, True]),),
        prompts=("task",),
    )
    model = SimpleNamespace(params={"weight": jnp.ones(())})
    monkeypatch.setattr(plot_loglike_modalities, "load_episode", lambda *args, **kwargs: episode)

    def fake_compute(*args, **kwargs):
        nonlocal computed
        computed = True
        return []

    monkeypatch.setattr(
        plot_loglike_modalities,
        "compute_episode_modality_contributions",
        fake_compute,
    )
    monkeypatch.setattr(
        plot_loglike_modalities,
        "save_contribution_curve",
        lambda *args, **kwargs: (pathlib.Path("curve.csv"), None),
    )

    with pytest.raises(ValueError, match="H_safe"):
        plot_loglike_modalities.evaluate_modalities(
            model=model,
            episode_index=0,
            frame=0,
            max_frames=10,
            sample_interval=None,
            num_steps=1,
            ode_solver="euler",
            eval_batch_size=1,
            hutchinson_samples=1,
            hutchinson_seed=0,
            modalities=("vision",),
            output_dir=pathlib.Path("out"),
        )

    assert not computed


def test_default_arguments_run_requested_evaluation() -> None:
    args = plot_loglike_modalities._build_parser().parse_args([])

    assert args.checkpoint_dir == pathlib.Path("/home/typhon/models/tactile_test_05_1.5w")
    assert args.dataset_repo_id == "chaoyi/tactile_test_03"
    assert args.episode_index == 0
    assert args.sample_interval == 10
    assert args.num_steps == 50
    assert args.ode_solver == "fireflow"
    assert args.modalities == ["vision", "state", "language_prompt"]
    assert args.output_dir == pathlib.Path("eval_outputs/loglike")


def test_plot_only_cli_inputs_are_removed() -> None:
    parser = plot_loglike_modalities._build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--plot-only" not in option_strings
    assert "--input-dir" not in option_strings
    with pytest.raises(SystemExit):
        parser.parse_args(["existing.csv"])


def test_main_always_evaluates_before_plotting(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    model = object()
    generated_csv = pathlib.Path("eval_outputs/loglike/state_contribution_episode_0.csv")

    def fake_load_model(args):
        events.append("load")
        return model

    def fake_evaluate_modalities(**kwargs):
        events.append("evaluate")
        assert kwargs["model"] is model
        return [generated_csv]

    def fake_plot_modalities(csv_paths, *, y_field, output_path):
        events.append("plot")
        assert csv_paths == [generated_csv]
        assert y_field == "contribution"
        return output_path

    monkeypatch.setattr(plot_loglike_modalities, "load_model_from_args", fake_load_model)
    monkeypatch.setattr(
        plot_loglike_modalities,
        "evaluate_modalities",
        fake_evaluate_modalities,
    )
    monkeypatch.setattr(plot_loglike_modalities, "plot_modalities", fake_plot_modalities)

    plot_loglike_modalities.main([])

    assert events == ["load", "evaluate", "plot"]


def test_main_forwards_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    generated_csv = pathlib.Path("custom/state_contribution_episode_2.csv")

    def fake_load_model(args):
        captured["checkpoint_dir"] = args.checkpoint_dir
        captured["dataset_repo_id"] = args.dataset_repo_id
        return object()

    def fake_evaluate_modalities(**kwargs):
        captured.update(kwargs)
        return [generated_csv]

    def fake_plot_modalities(csv_paths, *, y_field, output_path):
        captured["csv_paths"] = csv_paths
        captured["y_field"] = y_field
        captured["output_path"] = output_path
        return output_path

    monkeypatch.setattr(plot_loglike_modalities, "load_model_from_args", fake_load_model)
    monkeypatch.setattr(
        plot_loglike_modalities,
        "evaluate_modalities",
        fake_evaluate_modalities,
    )
    monkeypatch.setattr(plot_loglike_modalities, "plot_modalities", fake_plot_modalities)

    plot_loglike_modalities.main(
        [
            "--checkpoint-dir",
            "custom-checkpoint",
            "--dataset-repo-id",
            "custom/dataset",
            "--episode-index",
            "2",
            "--single-frame",
            "--num-steps",
            "7",
            "--ode-solver",
            "euler",
            "--modalities",
            "state",
            "--output-dir",
            "custom",
            "--output-path",
            "custom/plot.png",
        ]
    )

    assert captured["checkpoint_dir"] == pathlib.Path("custom-checkpoint")
    assert captured["dataset_repo_id"] == "custom/dataset"
    assert captured["episode_index"] == 2
    assert captured["sample_interval"] is None
    assert captured["num_steps"] == 7
    assert captured["ode_solver"] == "euler"
    assert captured["modalities"] == ["state"]
    assert captured["output_dir"] == pathlib.Path("custom")
    assert captured["csv_paths"] == [generated_csv]
    assert captured["output_path"] == pathlib.Path("custom/plot.png")
