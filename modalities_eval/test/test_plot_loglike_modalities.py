from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = ROOT / "modalities_eval"
if str(EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPTS))

import pytest

import plot_loglike_config
import plot_loglike_modalities


def test_default_arguments_run_requested_evaluation() -> None:
    args = plot_loglike_config.parse_args_with_config(
        plot_loglike_modalities._build_parser,
        script="reverse",
        argv=[],
    )

    assert args.config == plot_loglike_config.DEFAULT_CONFIG
    assert args.checkpoint_dir == pathlib.Path("/home/typhon/models/tactile_test_05_1.5w")
    assert args.dataset_repo_id == "chaoyi/tactile_test_03"
    assert args.episode_index == 0
    assert args.sample_interval == 10
    assert args.num_steps == 15
    assert args.ode_solver == "fireflow"
    assert args.eval_batch_size == 4
    assert args.hutchinson_samples == 1
    assert args.modalities == ["vision", "state", "language_prompt"]
    assert args.output_dir == pathlib.Path("eval_outputs/loglike")


def test_plot_only_cli_inputs_are_removed() -> None:
    parser = plot_loglike_modalities._build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--config" in option_strings
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
