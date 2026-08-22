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
    cfg = plot_loglike_config.load_yaml_config(plot_loglike_config.DEFAULT_CONFIG)
    args = plot_loglike_config.parse_args_with_config(
        plot_loglike_modalities._build_parser,
        script="reverse",
        argv=[],
    )

    assert args.config == plot_loglike_config.DEFAULT_CONFIG
    assert args.checkpoint_dir == pathlib.Path(cfg["data"]["checkpoint_dir"])
    assert args.dataset_repo_id == cfg["data"]["dataset_repo_id"]
    assert args.episode_index == cfg["data"]["episode_index"]
    assert args.sample_interval == cfg["data"]["sample_interval"]
    assert args.num_steps == cfg["integration"]["num_steps"]
    assert args.ode_solver == cfg["integration"]["ode_solver"]
    assert args.eval_batch_size == cfg["integration"]["eval_batch_size"]
    assert args.hutchinson_samples == cfg["integration"]["hutchinson_samples"]
    assert args.modalities == cfg["reverse"]["modalities"]
    assert args.output_dir == pathlib.Path(cfg["reverse"]["output_dir"])


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
