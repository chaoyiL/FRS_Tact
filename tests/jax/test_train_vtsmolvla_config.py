from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tools.train_vtsmolvla_jax import _validate_vt_config


ROOT = Path(__file__).resolve().parents[2]


def _load_repo_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def _same_experiment_config(config: dict) -> dict:
    normalized = deepcopy(config)
    normalized.pop("output", None)
    wandb = normalized.get("wandb") or {}
    wandb.pop("name", None)
    wandb.pop("tags", None)
    normalized["wandb"] = wandb
    normalized["model"] = dict(normalized["model"])
    normalized["model"].pop("tactile_token_repeat_factor", None)
    return normalized


def _active_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _active_strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _active_strings(nested)]
    return [value] if isinstance(value, str) else []


def test_cache_experiment_configs_use_server_paths_and_only_change_identity() -> None:
    tactile16 = _load_repo_yaml("train_vtsmolvla_jax_tactile16.yaml")
    tactile32 = _load_repo_yaml("train_vtsmolvla_jax_tactile32.yaml")

    assert tactile16["model"]["tactile_token_repeat_factor"] == 8
    assert tactile32["model"]["tactile_token_repeat_factor"] == 21
    assert _same_experiment_config(tactile16) == _same_experiment_config(tactile32)
    assert tactile16["offline_training_cache"] == {
        "enabled": True,
        "root": "/DATA/ljl/substage/smolvla_training_cache",
        "dtype": "bfloat16",
        "precompute_batch_size": 64,
        "precompute_num_workers": 8,
        "loader_num_workers": 2,
        "host_prefetch_batches": 2,
    }
    assert tactile16["model"]["tactile_encoder_path"] == \
        "/DATA/ljl/substage/checkpoints/encoder_ckpt_05"
    assert tactile16["image_transforms"]["enable"] is False
    assert tactile16["num_workers"] == 2
    assert tactile16["output"] == "/DATA/ljl/substage/outputs/vtsmolvla_tactile_repeat16"
    assert tactile32["output"] == "/DATA/ljl/substage/outputs/vtsmolvla_tactile_repeat32"
    assert tactile16["normalization"]["protocol_dir"] not in {
        tactile16["output"],
        tactile32["output"],
    }
    assert not any(
        value.startswith("/workspace")
        for config in (tactile16, tactile32)
        for value in _active_strings(config)
    )


def test_cache_configs_use_identical_encoder_05_path() -> None:
    configs = [
        _load_repo_yaml("train_vtsmolvla_jax_tactile16.yaml"),
        _load_repo_yaml("train_vtsmolvla_jax_tactile32.yaml"),
    ]

    assert {config["model"]["tactile_encoder_path"] for config in configs} == {
        "/DATA/ljl/substage/checkpoints/encoder_ckpt_05"
    }


@pytest.mark.parametrize(
    ("config_name", "tmux_session"),
    [
        ("train_vtsmolvla_jax_tactile16.yaml", "vtsmolvla_tactile16"),
        ("train_vtsmolvla_jax_tactile32.yaml", "vtsmolvla_tactile32"),
    ],
)
def test_paper_ratio_config_header_selects_its_own_launcher_session(
    config_name: str, tmux_session: str
) -> None:
    header = (ROOT / "configs" / config_name).read_text().split("# ----", maxsplit=1)[0]

    assert (
        f"#   FRS_TMUX_SESSION={tmux_session} bash scripts/start_vtsmolvla_train.sh \\\n"
        f"#     --config configs/{config_name}"
    ) in header


def _valid_config() -> dict:
    return {
        "model": {
            "use_tactile_encoder": True,
            "tactile_encoder_path": "encoder",
            "tactile_keys": ["t0", "t1", "t2", "t3"],
            "tactile_embedding_dim": 512,
            "tactile_num_tokens": 4,
            "tactile_token_repeat_factor": 8,
            "image_keys": ["camera1", "camera2"],
            "module_modes": {"vision": "frozen", "connector": "frozen"},
        },
        "output": "/runs/k8",
        "normalization": {"protocol_dir": "/shared/protocol"},
        "tactile_embedding_cache": {"enabled": False},
        "image_transforms": {"enable": False},
        "offline_training_cache": {
            "enabled": True,
            "root": "/cache",
            "dtype": "bfloat16",
            "precompute_batch_size": 64,
            "precompute_num_workers": 8,
            "loader_num_workers": 2,
            "host_prefetch_batches": 2,
        },
    }


def _write(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_vt_launcher_accepts_legacy_default_and_paper_factors(tmp_path: Path) -> None:
    config = _valid_config()
    for factor in (1, 8, 21):
        config["model"]["tactile_token_repeat_factor"] = factor
        _validate_vt_config(_write(tmp_path / f"k{factor}.yaml", config))

    del config["model"]["tactile_token_repeat_factor"]
    config.pop("normalization")
    _validate_vt_config(_write(tmp_path / "legacy.yaml", config))


def test_paper_factors_require_explicit_normalization_protocol(tmp_path: Path) -> None:
    config = _valid_config()
    config.pop("normalization")

    with pytest.raises(ValueError, match="normalization.*protocol_dir"):
        _validate_vt_config(_write(tmp_path / "missing-protocol.yaml", config))


@pytest.mark.parametrize(
    ("protocol_dir", "output"),
    [
        ("/runs/k8", "/runs/k8"),
        ("/runs/k8/normalization", "/runs/k8"),
        ("/runs", "/runs/k8"),
    ],
)
def test_paper_factors_reject_protocol_output_ancestor_overlap(
    tmp_path: Path,
    protocol_dir: str,
    output: str,
) -> None:
    config = _valid_config()
    config["normalization"]["protocol_dir"] = protocol_dir
    config["output"] = output

    with pytest.raises(ValueError, match="独立|output"):
        _validate_vt_config(_write(tmp_path / "overlap.yaml", config))


def test_paper_factors_reject_symlinked_protocol_output_overlap(tmp_path: Path) -> None:
    output = tmp_path / "runs" / "k8"
    output.mkdir(parents=True)
    alias = tmp_path / "output-alias"
    alias.symlink_to(output, target_is_directory=True)
    config = _valid_config()
    config["normalization"]["protocol_dir"] = str(alias / "normalization")
    config["output"] = str(output)

    with pytest.raises(ValueError, match="独立|output"):
        _validate_vt_config(_write(tmp_path / "symlink-overlap.yaml", config))


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "8"])
def test_vt_launcher_rejects_invalid_repeat_factor(tmp_path: Path, invalid: object) -> None:
    config = deepcopy(_valid_config())
    config["model"]["tactile_token_repeat_factor"] = invalid
    with pytest.raises(ValueError, match="tactile_token_repeat_factor"):
        _validate_vt_config(_write(tmp_path / "invalid.yaml", config))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["model"]["module_modes"].update(vision="full"), "frozen vision"),
        (lambda config: config["model"]["module_modes"].update(connector="lora"), "frozen vision"),
        (lambda config: config["image_transforms"].update(enable=True), "image_transforms"),
        (lambda config: config["offline_training_cache"].update(dtype="float16"), "bfloat16"),
        (lambda config: config["offline_training_cache"].update(root=""), "root"),
        (lambda config: config["offline_training_cache"].update(precompute_batch_size=0), "precompute_batch_size"),
        (lambda config: config["offline_training_cache"].update(precompute_num_workers=0), "precompute_num_workers"),
        (lambda config: config["offline_training_cache"].update(loader_num_workers=0), "loader_num_workers"),
        (lambda config: config["offline_training_cache"].update(host_prefetch_batches=0), "host_prefetch_batches"),
    ],
)
def test_offline_cache_rejects_incompatible_configuration(
    tmp_path: Path, mutate: object, message: str
) -> None:
    config = _valid_config()
    mutate(config)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        _validate_vt_config(_write(tmp_path / "offline-invalid.yaml", config))
