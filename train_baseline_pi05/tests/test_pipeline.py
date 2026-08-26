"""Contract tests for the standalone Pi0.5 baseline training handoff."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_module_is_present() -> None:
    assert (ROOT / "train_baseline_pi05" / "pipeline.py").is_file()


def _config(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    norm_stats = tmp_path / "norm_stats"
    encoder = tmp_path / "encoder"
    for path in (dataset, checkpoint, norm_stats, encoder):
        path.mkdir()
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""dataset:
  repo_id: local/demo
  root: {dataset}
  revision: null
  action_key: actions
  train_fraction: 0.8
  validation_fraction: 0.1
  test_fraction: 0.1
  split_seed: 0
source:
  checkpoint: {checkpoint}
  norm_stats_dir: {norm_stats}
  norm_stats_asset_id: demo
  seed: 0
  sample_steps: 10
  action_horizon: 50
  model_action_dim: 20
  paligemma_variant: demo
  action_expert_variant: demo
  use_quantile_norm: true
  allow_download: false
tactile:
  encoder_checkpoint: {encoder}
  embedding_dim: 512
  freeze_encoder: true
cache:
  action_root: {tmp_path / 'action-cache'}
  tactile_root: {tmp_path / 'tactile-cache'}
decoder:
  output: {tmp_path / 'decoder'}
  action_horizon: 50
  action_dim: 20
  tactile_dim: 512
  d_model: 128
  nhead: 4
  num_layers: 2
  dim_feedforward: 256
  dropout: 0.1
  batch_size: 2
  epochs: 1
  learning_rate: 0.001
  weight_decay: 0.0
  seed: 0
  workers: 0
  pin_memory: false
  device: cpu
  resume: false
""",
        encoding="utf-8",
    )
    return path


def test_pipeline_runs_jax_producers_then_torch_training_with_bounded_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from train_baseline_pi05 import pipeline

    config = _config(tmp_path)
    commands: list[list[str]] = []
    metadata: list[dict[str, object]] = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, check: commands.append(command))
    monkeypatch.setattr(pipeline, "_write_run_metadata", lambda _config, value: metadata.append(value))

    pipeline.run_pipeline(config, max_samples=7, max_steps=3)

    assert commands == [
        [sys.executable, "-m", "train_baseline_pi05.tactile_cache", "--config", str(config.resolve()), "--max-samples", "7"],
        [sys.executable, "-m", "train_baseline_pi05.prepare_action_cache", "--config", str(config.resolve()), "--max-samples", "7"],
        [sys.executable, "-m", "train_baseline_pi05.train", "--config", str(config.resolve()), "--max-steps", "3"],
    ]
    assert metadata == [{"config": str(config.resolve()), "max_samples": 7, "max_steps": 3, "stages": ["tactile_cache", "prepare_action_cache", "train"]}]


def test_pipeline_propagates_stage_failure_without_running_later_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from train_baseline_pi05 import pipeline

    commands: list[list[str]] = []

    def fail(command: list[str], *, check: bool) -> None:
        commands.append(command)
        raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    monkeypatch.setattr(pipeline, "_write_run_metadata", lambda *_args: pytest.fail("must not write metadata"))

    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(_config(tmp_path))
    assert len(commands) == 1


def test_check_mode_is_read_only_and_never_enters_subprocess_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from train_baseline_pi05 import pipeline

    config = _config(tmp_path)
    heavy_before = {name for name in sys.modules if name.startswith(("jax", "torch"))}
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("check must not start a process"))
    monkeypatch.setattr(pipeline, "_write_run_metadata", lambda *_args: pytest.fail("check must not write metadata"))

    report = pipeline.run_pipeline(config, check=True, max_samples=5, max_steps=1)

    assert report["mode"] == "check"
    assert report["inputs"]["dataset"]["readable"] is True
    assert report["destinations"]["decoder"] == str((tmp_path / "decoder").resolve())
    assert report["overrides"] == {"max_samples": 5, "max_steps": 1}
    assert json.loads(capsys.readouterr().out)["mode"] == "check"
    assert not (tmp_path / "decoder").exists()
    assert {name for name in sys.modules if name.startswith(("jax", "torch"))} == heavy_before


def test_check_mode_rejects_missing_local_reference_input(tmp_path: Path) -> None:
    from train_baseline_pi05 import pipeline

    config = _config(tmp_path)
    (tmp_path / "checkpoint").rmdir()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        pipeline.run_pipeline(config, check=True)


def test_tactile_cache_cap_covers_the_last_strided_action_record() -> None:
    from train_baseline_pi05.tactile_cache import _required_tactile_frames

    class Metadata:
        total_episodes = 1
        episodes = [{"dataset_from_index": 0, "dataset_to_index": 20}]

    assert _required_tactile_frames(Metadata(), frame_stride=5, max_samples=2) == 6


def test_scripts_readme_and_lock_keep_standalone_contract() -> None:
    setup = ROOT / "train_baseline_pi05/scripts/setup_env.sh"
    start = ROOT / "train_baseline_pi05/scripts/start_train.sh"
    readme = ROOT / "train_baseline_pi05/README.md"
    lock = ROOT / "train_baseline_pi05/uv.lock"
    assert setup.is_file() and start.is_file() and readme.is_file() and lock.is_file()
    assert subprocess.run(["bash", "-n", str(setup)], check=False).returncode == 0
