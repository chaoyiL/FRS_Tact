from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax import traverse_util

from train_smolvla_frs.utils.checkpoint import load_checkpoint, save_checkpoint
from train_smolvla_frs.utils.model import DecoderConfig, TactileConditionedFlowDecoder
from train_smolvla_frs.verify_gt_fm import (
    RUN_NAMES,
    format_run_line,
    plot_relative_reduction,
    relative_reduction,
    summarize_comparison,
    write_relative_reduction_plots,
)


def _no_gru_config(**overrides) -> DecoderConfig:
    values = dict(
        action_dim=3,
        action_horizon=4,
        tactile_window=1,
        gru_hidden_dim=8,
        resnet_embedding_dim=6,
        model_dim=16,
        depth=1,
        num_heads=4,
        num_tactile_tokens=4,
        use_gru=False,
        state_conditioning=False,
    )
    values.update(overrides)
    return DecoderConfig(**values)


def test_default_decoder_keeps_gru_and_nonzero_tokens() -> None:
    config = DecoderConfig(
        action_dim=3,
        action_horizon=4,
        tactile_window=3,
        gru_hidden_dim=8,
        resnet_embedding_dim=6,
        model_dim=16,
        depth=1,
        num_heads=4,
        num_tactile_tokens=2,
    )
    assert config.use_gru is True
    assert config.zero_tactile_tokens is False
    assert config.use_flow_matching is True
    model = TactileConditionedFlowDecoder(config, rngs=nnx.Rngs(0))
    tactile = jax.random.normal(jax.random.key(1), (2, 3, 2, 6))
    tokens = model.encode_tactile_tokens(tactile)
    assert tokens.shape == (2, 2, config.gru_hidden_dim)
    assert hasattr(model, "tactile_gru")


def test_no_gru_projects_current_frame_resnet_tokens() -> None:
    model = TactileConditionedFlowDecoder(_no_gru_config(), rngs=nnx.Rngs(2))
    first = jax.random.normal(jax.random.key(3), (2, 1, 4, 6))
    second = first + 1.0
    first_tokens = model.encode_tactile_tokens(first)
    second_tokens = model.encode_tactile_tokens(second)
    condition = model.encode_tactile_condition(first)
    assert first_tokens.shape == (2, 4, 6)
    assert condition.shape == (2, 4, 16)
    assert float(jnp.max(jnp.abs(first_tokens - first[:, -1]))) == 0.0
    assert float(jnp.max(jnp.abs(second_tokens - first_tokens))) > 1e-6


def test_no_gru_has_no_gru_parameters() -> None:
    model = TactileConditionedFlowDecoder(_no_gru_config(), rngs=nnx.Rngs(4))
    paths = ["/".join(str(part) for part in path) for path in traverse_util.flatten_dict(nnx.state(model).to_pure_dict())]
    assert all("tactile_gru" not in path for path in paths)
    assert not hasattr(model, "tactile_gru")


def test_zero_tactile_tokens_are_all_zeros_after_projection() -> None:
    model = TactileConditionedFlowDecoder(
        _no_gru_config(zero_tactile_tokens=True),
        rngs=nnx.Rngs(5),
    )
    tactile = jax.random.normal(jax.random.key(6), (3, 1, 4, 6))
    condition = model.encode_condition(tactile)
    np.testing.assert_allclose(np.asarray(condition), 0.0, atol=0.0)


def test_zero_tactile_tokens_velocity_ignores_embeddings() -> None:
    model = TactileConditionedFlowDecoder(
        _no_gru_config(zero_tactile_tokens=True),
        rngs=nnx.Rngs(7),
    )
    x_t = jax.random.normal(jax.random.key(8), (2, 4, 3))
    t = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    tactile_a = jax.random.normal(jax.random.key(9), (2, 1, 4, 6))
    tactile_b = jax.random.normal(jax.random.key(10), (2, 1, 4, 6))
    velocity_a = model(x_t, t, tactile_a)
    velocity_b = model(x_t, t, tactile_b)
    np.testing.assert_allclose(np.asarray(velocity_a), np.asarray(velocity_b), rtol=1e-5, atol=1e-6)


def test_direct_action_head_ignores_time_and_predicts_action() -> None:
    model = TactileConditionedFlowDecoder(
        _no_gru_config(use_flow_matching=False),
        rngs=nnx.Rngs(13),
    )
    vla = jax.random.normal(jax.random.key(14), (2, 4, 3))
    tactile = jax.random.normal(jax.random.key(15), (2, 1, 4, 6))
    t_a = jnp.asarray([0.1, 0.2], dtype=jnp.float32)
    t_b = jnp.asarray([0.8, 0.9], dtype=jnp.float32)
    pred_a = model(vla, t_a, tactile)
    pred_b = model(vla, t_b, tactile)
    assert pred_a.shape == vla.shape
    np.testing.assert_allclose(np.asarray(pred_a), np.asarray(pred_b), rtol=1e-5, atol=1e-6)
    assert not hasattr(model, "time_mlp")


def test_vla_direct_ignores_tactile_embeddings() -> None:
    model = TactileConditionedFlowDecoder(
        _no_gru_config(use_flow_matching=False, zero_tactile_tokens=True),
        rngs=nnx.Rngs(16),
    )
    vla = jax.random.normal(jax.random.key(17), (2, 4, 3))
    t = jnp.zeros((2,), dtype=jnp.float32)
    tactile_a = jax.random.normal(jax.random.key(18), (2, 1, 4, 6))
    tactile_b = jax.random.normal(jax.random.key(19), (2, 1, 4, 6))
    pred_a = model(vla, t, tactile_a)
    pred_b = model(vla, t, tactile_b)
    np.testing.assert_allclose(np.asarray(pred_a), np.asarray(pred_b), rtol=1e-5, atol=1e-6)


def test_vla_tactile_direct_uses_tactile_embeddings() -> None:
    model = TactileConditionedFlowDecoder(
        _no_gru_config(use_flow_matching=False, zero_tactile_tokens=False),
        rngs=nnx.Rngs(20),
    )
    vla = jax.random.normal(jax.random.key(21), (2, 4, 3))
    t = jnp.zeros((2,), dtype=jnp.float32)
    tactile_a = jax.random.normal(jax.random.key(22), (2, 1, 4, 6))
    tactile_b = tactile_a + 1.0
    pred_a = model(vla, t, tactile_a)
    pred_b = model(vla, t, tactile_b)
    assert float(jnp.max(jnp.abs(pred_a - pred_b))) > 1e-6


def test_relative_reduction_is_vla_minus_frs_over_vla() -> None:
    assert relative_reduction(0.25, 1.0) == 0.75
    assert relative_reduction(1.0, 1.0) == 0.0
    assert relative_reduction(0.0, 0.0) == 0.0


def test_summarize_comparison_reports_reduction_gap() -> None:
    tactile = {"mse_frs_gt": 0.25, "mse_vla_gt": 1.0, "relative_reduction": 0.75}
    zero = {"mse_frs_gt": 0.9, "mse_vla_gt": 1.0, "relative_reduction": 0.1}
    vla_direct = {"mse_frs_gt": 0.8, "mse_vla_gt": 1.0, "relative_reduction": 0.2}
    vla_tactile = {"mse_frs_gt": 0.4, "mse_vla_gt": 1.0, "relative_reduction": 0.6}
    summary = summarize_comparison(
        {
            "tactile": tactile,
            "zero_tactile_tokens": zero,
            "vla_direct": vla_direct,
            "vla_tactile_direct": vla_tactile,
        }
    )
    assert summary["reduction_gap"] == 0.65
    assert summary["reduction_gap_direct"] == pytest.approx(0.4)
    assert summary["tactile"]["mse_frs_gt"] == 0.25
    assert summary["vla_tactile_direct"]["relative_reduction"] == 0.6
    assert "23.40%" in format_run_line("tactile", {"mse_frs_gt": 0.766, "mse_vla_gt": 1.0, "relative_reduction": 0.234})
    assert RUN_NAMES == ("tactile", "zero_tactile_tokens", "vla_direct", "vla_tactile_direct")


def test_legacy_decoder_config_defaults_keep_gru() -> None:
    config = DecoderConfig(
        action_dim=3,
        action_horizon=4,
        tactile_window=3,
        gru_hidden_dim=8,
        resnet_embedding_dim=6,
        model_dim=16,
        depth=1,
        num_heads=4,
        num_tactile_tokens=2,
    )
    assert config.use_gru is True
    assert config.zero_tactile_tokens is False


def test_no_gru_checkpoint_round_trip() -> None:
    model = TactileConditionedFlowDecoder(
        _no_gru_config(zero_tactile_tokens=True),
        rngs=nnx.Rngs(11),
    )
    x_t = jnp.ones((2, 4, 3), dtype=jnp.float32)
    t = jnp.asarray([0.2, 0.8], dtype=jnp.float32)
    tactile = jax.random.normal(jax.random.key(12), (2, 1, 4, 6))
    expected = model(x_t, t, tactile)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_dir = Path(directory)
        save_checkpoint(checkpoint_dir, model, epoch=2, metrics={"val_mse_frs_gt": 0.1})
        restored, metadata = load_checkpoint(checkpoint_dir)
        np.testing.assert_allclose(np.asarray(restored(x_t, t, tactile)), np.asarray(expected))
        assert metadata["decoder_config"]["use_gru"] is False
        assert metadata["decoder_config"]["zero_tactile_tokens"] is True


def test_plot_relative_reduction_writes_png(tmp_path: Path) -> None:
    history = tmp_path / "tactile" / "history.csv"
    history.parent.mkdir()
    history.write_text(
        "epoch,train_loss,val_mse_frs_gt,val_mse_vla_gt,val_relative_reduction\n"
        "1,0.4,0.9,1.0,0.1\n"
        "2,0.3,0.7,1.0,0.3\n",
        encoding="utf-8",
    )
    output = tmp_path / "relative_reduction.png"
    assert plot_relative_reduction({"tactile": history}, output) == output
    assert output.is_file()
    assert output.stat().st_size > 0


def test_write_relative_reduction_plots_adds_combined_curve(tmp_path: Path) -> None:
    header = "epoch,train_loss,val_mse_frs_gt,val_mse_vla_gt,val_relative_reduction\n"
    for name, row in (
        ("tactile", "1,0.4,0.8,1.0,0.2\n"),
        ("zero_tactile_tokens", "1,0.5,1.1,1.0,-0.1\n"),
        ("vla_direct", "1,0.6,0.9,1.0,0.1\n"),
        ("vla_tactile_direct", "1,0.3,0.5,1.0,0.5\n"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "history.csv").write_text(header + row, encoding="utf-8")
    written = write_relative_reduction_plots(tmp_path / "zero_tactile_tokens", "zero_tactile_tokens")
    assert tmp_path / "tactile" / "relative_reduction.png" not in written
    assert tmp_path / "zero_tactile_tokens" / "relative_reduction.png" in written
    assert tmp_path / "relative_reduction.png" in written
    assert all(path.is_file() for path in written)


def test_verify_gt_fm_module_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "train_smolvla_frs.verify_gt_fm", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--run" in completed.stdout
    assert "vla_direct" in completed.stdout
    assert "vla_tactile_direct" in completed.stdout


def test_prepare_tactile_embeddings_skips_when_disabled() -> None:
    from train_smolvla_frs.prepare_frs_caches import prepare_tactile_embeddings_from_config

    assert prepare_tactile_embeddings_from_config({}) is None
    assert (
        prepare_tactile_embeddings_from_config(
            {"tactile_embedding_cache": {"enabled": False, "root": "unused"}}
        )
        is None
    )


def test_prepare_tactile_embeddings_invokes_shared_precompute(monkeypatch, tmp_path: Path) -> None:
    from train_smolvla_frs.prepare_frs_caches import prepare_tactile_embeddings_from_config

    seen: dict[str, object] = {}

    def fake_precompute(config, **kwargs):
        seen["config"] = config
        seen["kwargs"] = kwargs
        return tmp_path

    monkeypatch.setattr("train_vtsmolvla.precompute.precompute_from_config", fake_precompute)
    config = {"tactile_embedding_cache": {"root": str(tmp_path)}}
    assert prepare_tactile_embeddings_from_config(config) == tmp_path
    assert seen["config"] is config
    assert seen["kwargs"]["require_use_tactile_encoder"] is False
