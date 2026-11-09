from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from flax import nnx

from train_pi05_frs.pi05_cache.cache import SampleRecord, build_records, create_cache_arrays, records_digest
from train_pi05_frs.shared_data import SharedTaskData, canonical_action, canonical_tactile
from train_pi05_frs.shared_model import (
    decode_shared,
    make_shared_train_step,
    masked_mse,
    repair_native_actions,
    shared_flow_loss,
)
from train_pi05_frs.tools.train_shared_frs import DEFAULT_CONFIG, load_shared_config
from train_pi05_frs.utils.model import DecoderConfig, TactileConditionedFlowDecoder, make_optimizer


def small_model():
    return TactileConditionedFlowDecoder(
        DecoderConfig(
            action_dim=20,
            action_horizon=2,
            tactile_window=2,
            resnet_embedding_dim=4,
            gru_hidden_dim=4,
            model_dim=8,
            depth=1,
            num_heads=2,
            num_tactile_tokens=4,
            tactile_token_identity=True,
        ),
        rngs=nnx.Rngs(0),
    )


def small_batch():
    mask = jnp.array([[0] * 10 + [1] * 10, [1] * 20], dtype=jnp.float32)
    return {
        "x_base": jnp.ones((2, 2, 20)) * mask[:, None, :],
        "gt": jnp.ones((2, 2, 20)) * 0.5 * mask[:, None, :],
        "predicted": jnp.ones((2, 2, 20)) * 0.8 * mask[:, None, :],
        "tactile": jnp.ones((2, 2, 4, 4)),
        "action_mask": mask,
        "token_mask": jnp.array([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=jnp.float32),
        "gates": jnp.array([[0, 0.9], [0.1, 0.9]]),
        "source": jnp.array([0, 1]),
    }


def test_native_action_and_sensor_mapping():
    action = np.arange(10, dtype=np.float32)[None, :]
    canonical = canonical_action(action, ["right"])
    np.testing.assert_array_equal(canonical[..., :10], 0)
    np.testing.assert_array_equal(canonical[..., 10:], action)
    padded = canonical_action(np.arange(32)[None, :], ["left", "right"])
    np.testing.assert_array_equal(padded, np.arange(20)[None, :])
    sensors = np.arange(8).reshape(1, 2, 4)
    tactile = canonical_tactile(
        sensors, ["observation.images.tactile_right_1", "observation.images.tactile_left_1"], ["right"]
    )
    np.testing.assert_array_equal(tactile[:, :2], 0)
    np.testing.assert_array_equal(tactile[:, 2], sensors[:, 1])
    np.testing.assert_array_equal(tactile[:, 3], sensors[:, 0])
    with pytest.raises(ValueError, match="physical right"):
        canonical_action(action, ["left", "right"])


def test_idle_actions_and_tokens_cannot_change_active_decode_or_loss():
    model, batch = small_model(), small_batch()
    first = decode_shared(model, batch, num_steps=2)
    changed = dict(batch)
    changed["x_base"] = batch["x_base"].at[0, :, :10].set(10000)
    changed["tactile"] = batch["tactile"].at[0, :, :2].set(-9999)
    second = decode_shared(model, changed, num_steps=2)
    np.testing.assert_array_equal(np.asarray(first)[0, :, :10], 0)
    np.testing.assert_allclose(first[0, :, 10:], second[0, :, 10:], atol=1e-5)
    for mode in ("gt", "bimanual_gated"):
        loss, _ = shared_flow_loss(
            model,
            batch,
            jnp.array([0.2, 0.7]),
            loss_mode=mode,
            low_threshold=0.3,
            high_threshold=0.7,
            aux_decode_weight=1,
            aux_decode_steps=2,
        )
        assert jnp.isfinite(loss)
        perturbed = {
            **changed,
            "gt": batch["gt"].at[0, :, :10].set(2000),
            "predicted": batch["predicted"].at[0, :, :10].set(-1000),
        }
        other, _ = shared_flow_loss(
            model,
            perturbed,
            jnp.array([0.2, 0.7]),
            loss_mode=mode,
            low_threshold=0.3,
            high_threshold=0.7,
            aux_decode_weight=1,
            aux_decode_steps=2,
        )
        np.testing.assert_allclose(loss, other, atol=1e-5)
    errors = masked_mse(jnp.ones((2, 2, 20)), jnp.zeros((2, 2, 20)), batch["action_mask"])
    np.testing.assert_allclose(errors, 1)


def test_mixed_batch_updates_one_model_and_has_finite_auxiliaries():
    model, batch = small_model(), small_batch()
    optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0, total_steps=4)
    before = np.asarray(model.action_in.kernel[...]).copy()
    step = make_shared_train_step(
        loss_mode="bimanual_gated",
        low_threshold=0.3,
        high_threshold=0.7,
        aux_decode_weight=1,
        aux_decode_steps=2,
        low_gate_safety_weight=0.5,
        rank_weight=1,
        repair_weight=1,
    )
    loss, metrics = step(model, optimizer, batch, jax.random.key(1))
    assert np.isfinite(float(loss))
    assert all(np.isfinite(float(v)) for v in metrics.values())
    assert not np.array_equal(before, np.asarray(model.action_in.kernel[...]))
    repaired = repair_native_actions(
        model,
        np.ones((1, 2, 10), dtype=np.float32),
        np.ones((1, 2, 2, 4), dtype=np.float32),
        active_wrists=["right"],
        num_steps=2,
    )
    assert repaired.shape == (1, 2, 10)
    with pytest.raises(ValueError, match="predicted_action"):
        repair_native_actions(
            model, np.ones((1, 2, 32)), np.ones((1, 2, 4, 4)), active_wrists=["left", "right"], num_steps=1
        )


def test_exact_300_train_35_val_record_budget():
    metadata = SimpleNamespace(
        total_episodes=500,
        episodes=[{"dataset_from_index": i * 80, "dataset_to_index": (i + 1) * 80} for i in range(500)],
    )
    kwargs = {
        "val_fraction": 0.1,
        "split_seed": 42,
        "frame_stride": 5,
        "max_episodes": None,
        "max_samples": None,
        "action_horizon": 50,
        "num_train_episodes": 300,
        "num_val_episodes": 35,
    }
    records, train, val = build_records(metadata, **kwargs)
    assert (len(train), len(val)) == (300, 35)
    assert not set(train) & set(val)
    assert records == build_records(metadata, **kwargs)[0]
    assert {*train, *val} != set(range(335))
    with pytest.raises(ValueError, match="cannot be combined"):
        build_records(metadata, **{**kwargs, "max_episodes": 335})


def create_source(tmp_path, name, action_dim, active, source_i):
    root, action, tactile = (
        tmp_path / name / "dataset",
        tmp_path / name / "actions",
        tmp_path / name / "tactile",
    )
    meta_root = root / "meta/episodes/chunk-000"
    meta_root.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0, 1, 2], "dataset_from_index": [0, 4, 8]}),
        meta_root / "file-000.parquet",
    )
    records = [SampleRecord(ep * 4 + 1, ep, "val" if ep == 2 else "train") for ep in range(3)]
    arrays = create_cache_arrays(
        action, records, action_horizon=2, action_dim=action_dim, state_dim=7 if action_dim == 10 else 20
    )
    for key in ("x_base", "target", "gt_action"):
        arrays[key][...] = source_i + 0.25
    for array in arrays.values():
        array.flush()
    manifest = {
        "version": 3,
        "status": "complete",
        "completed_samples": 3,
        "sample_count": 3,
        "action_horizon": 2,
        "action_dim": action_dim,
        "state_dim": 7 if action_dim == 10 else 20,
        "train_sample_count": 2,
        "val_sample_count": 1,
        "train_episodes": [0, 1],
        "val_episodes": [2],
        "records_sha256": records_digest(records),
        "configuration": {
            "base_model": "pi0.5",
            "dataset_repo_id": f"org/{name}",
            "dataset_root": str(root),
            "dataset_revision": None,
        },
    }
    (action / "manifest.json").write_text(json.dumps(manifest))
    tactile.mkdir()
    keys = (
        ["tactile_left_1", "tactile_right_1"]
        if action_dim == 10
        else ["tactile_left_0", "tactile_right_0", "tactile_left_1", "tactile_right_1"]
    )
    embeddings = np.random.default_rng(source_i).normal(size=(12, len(keys), 4)).astype(np.float16)
    np.save(tactile / "embeddings.npy", embeddings)
    (tactile / "metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "repo_id": f"org/{name}",
                "encoder_sha256": "test",
                "total_frames": 12,
                "completed_frames": 12,
                "num_tactile_tokens": len(keys),
                "embedding_dim": 4,
                "tactile_keys": keys,
                "dataset_root": str(root),
                "revision": None,
            }
        )
    )
    return {
        "name": name,
        "repo_id": f"org/{name}",
        "dataset_root": str(root),
        "action_cache": str(action),
        "tactile_cache": str(tactile),
        "active_wrists": active,
    }


def test_four_task_cache_batching_and_native_split_isolation(tmp_path):
    sources = [
        create_source(tmp_path, name, width, wrists, i)
        for i, (name, width, wrists) in enumerate(
            (
                ("pick", 20, ["right"]),
                ("insert", 10, ["right"]),
                ("bread", 20, ["left", "right"]),
                ("two", 32, ["left", "right"]),
            )
        )
    ]
    data = SharedTaskData(
        sources,
        encoder_sha256="test",
        tactile_window=2,
        history_stride=3,
        num_train_episodes=2,
        num_val_episodes=1,
        seed=42,
    )
    batch = next(data.batches("train", batch_size=8, seed=1, tau=0.65, temperature=0.15))
    assert batch["x_base"].shape == (8, 2, 20)
    assert batch["tactile"].shape == (8, 2, 4, 4)
    assert set(batch["source"]) == {0, 1, 2, 3}
    for i, source_i in enumerate(batch["source"]):
        if source_i < 2:
            np.testing.assert_array_equal(batch["x_base"][i, :, :10], 0)
            np.testing.assert_array_equal(batch["tactile"][i, :, :2], 0)
        np.testing.assert_allclose(batch["gt"][i, :, 10:], source_i + 0.25)
    assert all(
        selection["train"] == [0, 1] and selection["val"] == [2] for selection in data.selection.values()
    )
    with pytest.raises(ValueError, match="provenance"):
        SharedTaskData(
            sources,
            encoder_sha256="wrong",
            tactile_window=2,
            history_stride=3,
            num_train_episodes=2,
            num_val_episodes=1,
            seed=42,
        )


def test_shared_config_uses_all_four_sources_and_keeps_native_insert_10d():
    config = load_shared_config(DEFAULT_CONFIG)
    assert config["tactile_encoder_repo"] == "KaiyueChen/encoder_ckpt_0824"
    assert config["tactile_encoder_revision"] == "1a432a8b9c1a38f75cb40c58a4924a401486d832"
    assert config["tactile_encoder"] == "/workspace/FRS_Tact/checkpoints/encoder_ckpt_0824"
    assert [s["name"] for s in config["sources"]] == ["insert", "press", "bread", "pick"]
    assert config["sources"][1]["native_action_dim"] == 10
    assert config["num_train_episodes"] == 300
    assert config["num_val_episodes"] == 35


def test_shared_training_checkpoint_resume_and_insert_evaluation(tmp_path):
    from train_pi05_frs.shared_data import encoder_fingerprint
    from train_pi05_frs.shared_training import train_shared
    from train_pi05_frs.utils.checkpoint import load_checkpoint

    sources = [
        create_source(tmp_path, name, width, wrists, i)
        for i, (name, width, wrists) in enumerate(
            (
                ("pick", 20, ["right"]),
                ("insert", 10, ["right"]),
                ("bread", 20, ["left", "right"]),
                ("two", 32, ["left", "right"]),
            )
        )
    ]
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    (encoder / "checkpoint.json").write_text('{"params_file": "params.npz"}')
    np.savez(encoder / "params.npz", p00000=np.ones((1,)))
    fingerprint = encoder_fingerprint(encoder)
    for source in sources:
        path = Path(source["tactile_cache"]) / "metadata.json"
        metadata = json.loads(path.read_text())
        path.write_text(json.dumps({**metadata, "encoder_sha256": fingerprint}))
    config = load_shared_config(DEFAULT_CONFIG)
    config.update(
        tactile_encoder=str(encoder),
        num_train_episodes=2,
        num_val_episodes=1,
        action_horizon=2,
        output=str(tmp_path / "run"),
    )
    config["training"].update(
        model_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2,
        epochs=2,
        eval_every=1,
        batch_size=4,
        aux_decode_steps=1,
        validation_steps=1,
        tactile_window=2,
    )
    train_shared(config, sources)
    _, last = load_checkpoint(tmp_path / "run/last")
    assert last["epoch"] == 2
    assert last["decoder_config"]["action_dim"] == 20
    assert last["decoder_config"]["tactile_token_identity"] is True
    assert last["extra_metadata"]["shared_frs_interface_version"] == 1
    assert len(last["extra_metadata"]["episode_selection"]) == 4
    train_shared(config, sources, resume=True)
    train_shared(config, sources, evaluate_only=True, evaluate_task="insert")
    report = json.loads((tmp_path / "run/evaluation_insert.json").read_text())
    assert report and all(key.startswith("val/insert/") for key in report)
    with pytest.raises(FileExistsError, match="not empty"):
        train_shared(config, sources)
