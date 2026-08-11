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
import plot_loglike_modalities_forward


def test_default_arguments_include_noise_seed_and_forward_outdir() -> None:
    args = plot_loglike_config.parse_args_with_config(
        plot_loglike_modalities_forward._build_parser,
        script="forward",
        argv=[],
    )

    assert args.config == plot_loglike_config.DEFAULT_CONFIG
    assert args.checkpoint_dir == pathlib.Path("/home/typhon/models/tactile_test_05_1.5w")
    assert args.dataset_repo_id == "chaoyi/tactile_test_03"
    assert args.episode_index == 0
    assert args.sample_interval == 10
    assert args.num_steps == 15
    assert args.ode_solver == "fireflow"
    assert args.noise_seed == 0
    assert args.modalities == ["vision", "state", "language_prompt"]
    assert args.output_dir == pathlib.Path("eval_outputs/loglike_forward")
    assert args.compare_reverse_dir is None


def test_ode_solver_accepts_slerpflow() -> None:
    args = plot_loglike_config.parse_args_with_config(
        plot_loglike_modalities_forward._build_parser,
        script="forward",
        argv=["--ode-solver", "slerpflow"],
    )
    assert args.ode_solver == "slerpflow"


def test_shared_data_and_integration_match_reverse_defaults() -> None:
    reverse_args = plot_loglike_config.parse_args_with_config(
        plot_loglike_modalities._build_parser,
        script="reverse",
        argv=[],
    )
    forward_args = plot_loglike_config.parse_args_with_config(
        plot_loglike_modalities_forward._build_parser,
        script="forward",
        argv=[],
    )
    for key in (
        "checkpoint_dir",
        "dataset_repo_id",
        "episode_index",
        "frame",
        "max_frames",
        "sample_interval",
        "num_steps",
        "ode_solver",
        "eval_batch_size",
        "hutchinson_samples",
        "hutchinson_seed",
    ):
        assert getattr(forward_args, key) == getattr(reverse_args, key)


def test_main_forwards_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    generated_csv = pathlib.Path("custom/state_contribution_episode_2.csv")

    def fake_load_model(args):
        captured["checkpoint_dir"] = args.checkpoint_dir
        captured["dataset_repo_id"] = args.dataset_repo_id
        captured["noise_seed_arg"] = args.noise_seed
        return object()

    def fake_evaluate_modalities_from_noise(**kwargs):
        captured.update(kwargs)
        return [generated_csv]

    def fake_plot_modalities(csv_paths, *, y_field, output_path):
        captured["csv_paths"] = csv_paths
        captured["y_field"] = y_field
        captured["output_path"] = output_path
        return output_path

    def fake_compare(*args, **kwargs):
        captured["compare_called"] = True
        captured["compare_kwargs"] = kwargs
        return []

    monkeypatch.setattr(plot_loglike_modalities_forward, "load_model_from_args", fake_load_model)
    monkeypatch.setattr(
        plot_loglike_modalities_forward,
        "evaluate_modalities_from_noise",
        fake_evaluate_modalities_from_noise,
    )
    monkeypatch.setattr(plot_loglike_modalities_forward, "plot_modalities", fake_plot_modalities)
    monkeypatch.setattr(
        plot_loglike_modalities_forward,
        "compare_forward_to_reverse",
        fake_compare,
    )

    plot_loglike_modalities_forward.main(
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
            "slerpflow",
            "--noise-seed",
            "9",
            "--modalities",
            "state",
            "--output-dir",
            "custom",
            "--output-path",
            "custom/plot.png",
            "--compare-reverse-dir",
            "rev_dir",
        ]
    )

    assert captured["checkpoint_dir"] == pathlib.Path("custom-checkpoint")
    assert captured["dataset_repo_id"] == "custom/dataset"
    assert captured["episode_index"] == 2
    assert captured["sample_interval"] is None
    assert captured["num_steps"] == 7
    assert captured["ode_solver"] == "slerpflow"
    assert captured["noise_seed"] == 9
    assert captured["modalities"] == ["state"]
    assert captured["output_dir"] == pathlib.Path("custom")
    assert captured["csv_paths"] == [generated_csv]
    assert captured["output_path"] == pathlib.Path("custom/plot.png")
    assert captured["compare_called"] is True
    assert captured["compare_kwargs"]["reverse_dir"] == pathlib.Path("rev_dir")
    assert captured["compare_kwargs"]["episode_index"] == 2
