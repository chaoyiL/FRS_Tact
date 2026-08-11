from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = ROOT / "modalities_eval"
if str(EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPTS))

import plot_loglike_config


def test_flatten_uses_shared_sections_and_script_block() -> None:
    cfg = {
        "data": {
            "checkpoint_dir": "/ckpt",
            "dataset_repo_id": "org/ds",
            "episode_index": 3,
        },
        "integration": {
            "num_steps": 21,
            "ode_solver": "slerpflow",
            "eval_batch_size": 2,
            "hutchinson_samples": 4,
        },
        "reverse": {
            "output_dir": "out/rev",
            "modalities": ["state"],
        },
        "forward": {
            "output_dir": "out/fwd",
            "noise_seed": 8,
            "compare_reverse_dir": "out/rev",
        },
    }

    reverse = plot_loglike_config.flatten_plot_loglike_defaults(cfg, script="reverse")
    forward = plot_loglike_config.flatten_plot_loglike_defaults(cfg, script="forward")

    assert reverse["checkpoint_dir"] == pathlib.Path("/ckpt")
    assert reverse["num_steps"] == 21
    assert reverse["ode_solver"] == "slerpflow"
    assert reverse["output_dir"] == pathlib.Path("out/rev")
    assert reverse["modalities"] == ["state"]
    assert "noise_seed" not in reverse

    assert forward["checkpoint_dir"] == pathlib.Path("/ckpt")
    assert forward["num_steps"] == 21
    assert forward["noise_seed"] == 8
    assert forward["output_dir"] == pathlib.Path("out/fwd")
    assert forward["compare_reverse_dir"] == pathlib.Path("out/rev")


def test_default_config_file_exists() -> None:
    assert plot_loglike_config.DEFAULT_CONFIG.is_file()
    cfg = plot_loglike_config.load_yaml_config(plot_loglike_config.DEFAULT_CONFIG)
    assert "data" in cfg and "integration" in cfg
    assert "reverse" in cfg and "forward" in cfg
