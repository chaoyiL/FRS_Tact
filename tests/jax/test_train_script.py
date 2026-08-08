from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file as save_safetensors_file

from lerobot.policies.smolvla_jax.data import DatasetSource
from lerobot.policies.smolvla_jax.provenance import write_tactile_encoder_provenance
from lerobot.policies.smolvla_jax.validation import CheckpointContract

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "train_smolvla_jax.py"
SPEC = importlib.util.spec_from_file_location("train_smolvla_jax_test_module", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_SCRIPT)
_sources_from_split_manifest = TRAIN_SCRIPT._sources_from_split_manifest
_split_manifest = TRAIN_SCRIPT._split_manifest
run_validation = TRAIN_SCRIPT.run_validation
load_yaml_config = TRAIN_SCRIPT.load_yaml_config

VT_IMAGE_KEYS = ("observation.images.camera1", "observation.images.camera2")
VT_TACTILE_KEYS = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)
VT_CONTRACT = CheckpointContract(
    state_dim=20,
    action_dim=20,
    chunk_size=20,
    image_keys=VT_IMAGE_KEYS,
    tactile_keys=VT_TACTILE_KEYS,
    tactile_embedding_dim=512,
    tactile_num_tokens=4,
    lora_rank=16,
    vlm_lora_target_modules=("q_proj", "v_proj"),
)


def _vt_config():
    return TRAIN_SCRIPT.JaxSmolVLAConfig(
        state_dim=20,
        action_dim=20,
        chunk_size=20,
        image_keys=VT_IMAGE_KEYS,
        use_tactile_encoder=True,
        tactile_keys=VT_TACTILE_KEYS,
        tactile_embedding_dim=512,
        tactile_num_tokens=4,
        lora_rank=16,
        vlm_lora_target_modules=("q_proj", "v_proj"),
        tactile_encoder_repo_id="liuchaoyi/encoder_ckpt_05",
        tactile_encoder_revision="a" * 40,
        tactile_encoder_sha256="b" * 64,
    )


def test_persisted_split_reconstructs_episode_lists() -> None:
    sources = [DatasetSource(repo_id="org/a"), DatasetSource(repo_id="org/b")]
    train = [
        DatasetSource(repo_id="org/a", episodes=[0, 2]),
        DatasetSource(repo_id="org/b", episodes=[1, 3]),
    ]
    val = [
        DatasetSource(repo_id="org/a", episodes=[1]),
        DatasetSource(repo_id="org/b", episodes=[0]),
    ]
    manifest = _split_manifest(
        sources,
        train,
        val,
        val_fraction=0.25,
        split_seed=7,
        eval_seed=8,
        sample_seed=9,
    )

    restored_train, restored_val = _sources_from_split_manifest(sources, manifest, val_fraction=0.25)

    assert [source.episodes for source in restored_train] == [[0, 2], [1, 3]]
    assert [source.episodes for source in restored_val] == [[1], [0]]


def test_validation_reuses_fixed_seed_at_every_training_step() -> None:
    class FakeTrainer:
        def __init__(self):
            self.seeds = []

        def evaluate(self, batches, *, seed, **kwargs):
            del batches, kwargs
            self.seeds.append(seed)
            return {"loss": 1.0, "action_mse": 2.0, "n_samples": 64.0}

    class FakeData:
        def batches(self):
            return iter(())

    trainer = FakeTrainer()
    for step in (500, 1000):
        run_validation(
            trainer,
            FakeData(),
            step=step,
            eval_count=0,
            seed=1234,
            val_cfg={"max_batches": 1, "rollout": True},
            wandb_run=None,
        )

    assert trainer.seeds == [1234, 1234]


def test_unknown_top_level_training_key_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("checkpoint: model\nsteps: 10\neval_frqe: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="eval_frqe"):
        load_yaml_config(config)


def test_normalization_protocol_is_an_allowed_training_config(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(
        "checkpoint: model\nsteps: 10\nnormalization:\n  protocol_dir: /shared/protocol\n",
        encoding="utf-8",
    )

    assert load_yaml_config(config)["normalization"]["protocol_dir"] == "/shared/protocol"


def test_resume_provenance_is_validated_before_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    resume = tmp_path / "checkpoint"
    resume.mkdir()
    split_path = resume / TRAIN_SCRIPT.DATA_SPLIT_FILENAME
    split_path.write_text('{"version": 1}\n', encoding="utf-8")
    (resume / TRAIN_SCRIPT.NORMALIZATION_MANIFEST_FILENAME).write_text(
        '{"algorithm_version": 1}\n', encoding="utf-8"
    )
    (resume / TRAIN_SCRIPT.VALIDATION_PROVENANCE_FILENAME).write_text(
        '{"version": 1}\n', encoding="utf-8"
    )
    (resume / TRAIN_SCRIPT.TACTILE_ENCODER_PROVENANCE_FILENAME).write_text(
        '{"version": 1}\n', encoding="utf-8"
    )
    (resume / "resume_metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "experiment_provenance": {
                    "data_split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                    "normalization_manifest_sha256": hashlib.sha256(
                        (resume / TRAIN_SCRIPT.NORMALIZATION_MANIFEST_FILENAME).read_bytes()
                    ).hexdigest(),
                    "validation_provenance_sha256": hashlib.sha256(
                        (resume / TRAIN_SCRIPT.VALIDATION_PROVENANCE_FILENAME).read_bytes()
                    ).hexdigest(),
                    "tactile_encoder_provenance_sha256": hashlib.sha256(
                        (resume / TRAIN_SCRIPT.TACTILE_ENCODER_PROVENANCE_FILENAME).read_bytes()
                    ).hexdigest(),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = SimpleNamespace(
        stats={},
        split_path=split_path,
        manifest_path=resume / TRAIN_SCRIPT.NORMALIZATION_MANIFEST_FILENAME,
    )

    def build_or_validate(protocol_dir, **kwargs):
        events.append("validate_protocol")
        assert protocol_dir == resume
        assert kwargs["allow_create"] is False
        return result

    class FakePreprocessor:
        def __init__(self, checkpoint, *args, **kwargs):
            events.append("load_authoritative_assets")
            assert checkpoint == resume
            assert kwargs["stats"] is None

    class FakeTrainer:
        def restore(self, path):
            events.append("restore")
            assert path == resume

    monkeypatch.setattr(TRAIN_SCRIPT, "build_or_validate_normalization_protocol", build_or_validate)
    monkeypatch.setattr(TRAIN_SCRIPT, "JaxSmolVLAPreprocessor", FakePreprocessor)

    protocol_dir = TRAIN_SCRIPT._resolve_normalization_protocol_dir(
        None,
        resume=resume,
        output=tmp_path / "new-output",
    )
    assert protocol_dir == resume, "checkpoint protocol must be detected when YAML omits normalization"

    actual, _ = TRAIN_SCRIPT._prepare_normalization_and_resume(
        trainer=FakeTrainer(),
        resume=resume,
        protocol_dir=protocol_dir,
        split_path=split_path,
        train_sources=[DatasetSource(repo_id="org/a", episodes=[0])],
        checkpoint=tmp_path / "base",
        config=_vt_config(),
        local_files_only=True,
    )

    assert actual is result
    assert events == ["validate_protocol", "load_authoritative_assets", "restore"]


def test_protocol_resume_missing_checkpoint_assets_fails_before_dataset_fallback(tmp_path: Path) -> None:
    resume = tmp_path / "checkpoint"
    resume.mkdir()
    (resume / TRAIN_SCRIPT.NORMALIZATION_MANIFEST_FILENAME).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="data_split|protocol.*missing"):
        TRAIN_SCRIPT._resolve_normalization_protocol_dir(
            None,
            resume=resume,
            output=tmp_path / "output",
        )


def test_resume_provenance_tamper_fails_before_protocol_build_assets_or_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = tmp_path / "checkpoint"
    resume.mkdir()
    for filename, content in (
        (TRAIN_SCRIPT.DATA_SPLIT_FILENAME, '{"version": 1}\n'),
        (TRAIN_SCRIPT.NORMALIZATION_MANIFEST_FILENAME, '{"algorithm_version": 1}\n'),
        (TRAIN_SCRIPT.VALIDATION_PROVENANCE_FILENAME, '{"version": 1}\n'),
        (TRAIN_SCRIPT.TACTILE_ENCODER_PROVENANCE_FILENAME, '{"version": 1}\n'),
    ):
        (resume / filename).write_text(content, encoding="utf-8")
    (resume / "resume_metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "experiment_provenance": {
                    "data_split_sha256": "0" * 64,
                    "normalization_manifest_sha256": hashlib.sha256(
                        (resume / TRAIN_SCRIPT.NORMALIZATION_MANIFEST_FILENAME).read_bytes()
                    ).hexdigest(),
                    "validation_provenance_sha256": hashlib.sha256(
                        (resume / TRAIN_SCRIPT.VALIDATION_PROVENANCE_FILENAME).read_bytes()
                    ).hexdigest(),
                    "tactile_encoder_provenance_sha256": hashlib.sha256(
                        (resume / TRAIN_SCRIPT.TACTILE_ENCODER_PROVENANCE_FILENAME).read_bytes()
                    ).hexdigest(),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        TRAIN_SCRIPT,
        "build_or_validate_normalization_protocol",
        lambda *args, **kwargs: pytest.fail("tamper reached dataset/protocol metadata"),
    )
    monkeypatch.setattr(
        TRAIN_SCRIPT,
        "JaxSmolVLAPreprocessor",
        lambda *args, **kwargs: pytest.fail("tamper reached normalization assets"),
    )

    class Trainer:
        def restore(self, path):
            pytest.fail(f"tamper reached restore: {path}")

    with pytest.raises(ValueError, match="digest mismatch.*data_split"):
        TRAIN_SCRIPT._prepare_normalization_and_resume(
            trainer=Trainer(),
            resume=resume,
            protocol_dir=resume,
            split_path=resume / TRAIN_SCRIPT.DATA_SPLIT_FILENAME,
            train_sources=[DatasetSource(repo_id="org/a", episodes=[0])],
            checkpoint=tmp_path / "base",
            config=_vt_config(),
            local_files_only=True,
        )


def test_validation_provenance_is_immutable_and_separate_from_sealed_split(tmp_path: Path) -> None:
    path = tmp_path / TRAIN_SCRIPT.VALIDATION_PROVENANCE_FILENAME
    payload = {
        "version": 1,
        "validation_protocol": {"batch_size": 64, "max_batches": 5},
        "validation_dataset_frames": {"org/data": 100},
        "validation_sample_indices": [1, 3, 7],
        "eval_seed": 11,
        "sample_seed": 12,
    }
    TRAIN_SCRIPT._write_or_validate_validation_provenance(path, payload)
    before = path.read_bytes()
    TRAIN_SCRIPT._write_or_validate_validation_provenance(path, payload)
    assert path.read_bytes() == before

    drifted = dict(payload)
    drifted["validation_sample_indices"] = [2, 4, 8]
    with pytest.raises(ValueError, match="validation provenance|changed|mismatch"):
        TRAIN_SCRIPT._write_or_validate_validation_provenance(path, drifted)


def test_protocol_requires_at_least_one_held_out_episode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="normalization protocol.*held-out"):
        TRAIN_SCRIPT._require_protocol_validation_sources(tmp_path / "protocol", [])

    TRAIN_SCRIPT._require_protocol_validation_sources(None, [])
    TRAIN_SCRIPT._require_protocol_validation_sources(
        tmp_path / "protocol",
        [DatasetSource(repo_id="org/data", episodes=[1])],
    )


def test_training_encoder_preflight_binds_approved_repo_sha_and_content(tmp_path: Path) -> None:
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    (encoder / "checkpoint.json").write_text('{}\n', encoding="utf-8")
    (encoder / "params.npz").write_bytes(b"encoder-weights")
    write_tactile_encoder_provenance(
        encoder,
        repo_id="liuchaoyi/encoder_ckpt_05",
        requested_revision="main",
        resolved_revision="a" * 40,
    )
    config = replace(
        _vt_config(),
        tactile_encoder_path=str(encoder),
        tactile_encoder_revision=None,
        tactile_encoder_sha256=None,
    )

    resolved, identity, provenance_path = TRAIN_SCRIPT._validate_tactile_encoder_for_training(
        config
    )

    assert resolved.tactile_encoder_repo_id == "liuchaoyi/encoder_ckpt_05"
    assert resolved.tactile_encoder_revision == "a" * 40
    assert resolved.tactile_encoder_sha256 == identity["checkpoint_sha256"]
    assert provenance_path == encoder / TRAIN_SCRIPT.TACTILE_ENCODER_PROVENANCE_FILENAME

    with (encoder / "params.npz").open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError, match="digest|sha256|provenance"):
        TRAIN_SCRIPT._validate_tactile_encoder_for_training(config)


def test_validation_without_rollout_does_not_log_nan(capsys) -> None:
    class FakeTrainer:
        def evaluate(self, batches, *, seed, **kwargs):
            del batches, seed, kwargs
            return {"loss": 1.0, "action_mse": float("nan"), "n_samples": 8.0}

    class FakeData:
        def batches(self):
            return iter(())

    run_validation(
        FakeTrainer(),
        FakeData(),
        step=10,
        eval_count=0,
        seed=0,
        val_cfg={"rollout": False},
        wandb_run=None,
    )

    assert "action_mse" not in capsys.readouterr().out


def test_training_checkpoint_writes_all_assets_before_shared_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "checkpoint-00000020"
    source = tmp_path / "source"
    source.mkdir()
    split_path = tmp_path / TRAIN_SCRIPT.DATA_SPLIT_FILENAME
    split_path.write_text('{"version": 1}\n', encoding="utf-8")
    normalization_manifest_path = tmp_path / TRAIN_SCRIPT.NORMALIZATION_MANIFEST_FILENAME
    normalization_manifest_path.write_text('{"algorithm_version": 1}\n', encoding="utf-8")
    validation_provenance_path = tmp_path / TRAIN_SCRIPT.VALIDATION_PROVENANCE_FILENAME
    validation_provenance_path.write_text('{"version": 1}\n', encoding="utf-8")
    encoder_provenance_path = tmp_path / TRAIN_SCRIPT.TACTILE_ENCODER_PROVENANCE_FILENAME
    encoder_provenance_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_id": "liuchaoyi/encoder_ckpt_05",
                "resolved_revision": "a" * 40,
                "checkpoint_sha256": "b" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events: list[str] = []

    class FakeTrainer:
        config = _vt_config()

        def save(self, destination: Path, *, source_dir: Path) -> None:
            events.append("trainer.save")
            assert source_dir == source
            assert destination.name.endswith(".incomplete")
            assert not final.exists()
            (destination / "model-marker").write_text("weights", encoding="utf-8")
            (destination / "resume_metadata.json").write_text(
                '{"version": 1}\n', encoding="utf-8"
            )

    class FakePreprocessor:
        def save_normalization_assets(self, destination: Path) -> None:
            events.append("save_normalization_assets")
            assert (destination / "model-marker").is_file()
            assert not final.exists()
            (destination / "normalization-marker").write_text("stats", encoding="utf-8")

    class PassingReport:
        def require_valid(self) -> None:
            events.append("require_valid")

    def validate_checkpoint(
        staging: Path,
        *,
        expected: CheckpointContract,
        base_sidecars: Path,
    ) -> PassingReport:
        events.append("validate_checkpoint")
        assert expected == VT_CONTRACT
        assert base_sidecars == source
        assert not final.exists()
        assert (staging / "model-marker").is_file()
        assert (staging / "normalization-marker").is_file()
        assert (staging / TRAIN_SCRIPT.DATA_SPLIT_FILENAME).read_text(encoding="utf-8") == (
            split_path.read_text(encoding="utf-8")
        )
        assert (
            staging / TRAIN_SCRIPT.NORMALIZATION_MANIFEST_FILENAME
        ).read_text(encoding="utf-8") == normalization_manifest_path.read_text(encoding="utf-8")
        assert (staging / TRAIN_SCRIPT.VALIDATION_PROVENANCE_FILENAME).read_bytes() == (
            validation_provenance_path.read_bytes()
        )
        assert (staging / TRAIN_SCRIPT.TACTILE_ENCODER_PROVENANCE_FILENAME).read_bytes() == (
            encoder_provenance_path.read_bytes()
        )
        resume_metadata = json.loads((staging / "resume_metadata.json").read_text())
        provenance = resume_metadata["experiment_provenance"]
        assert provenance["normalization_manifest_sha256"] == hashlib.sha256(
            normalization_manifest_path.read_bytes()
        ).hexdigest()
        assert provenance["data_split_sha256"] == hashlib.sha256(split_path.read_bytes()).hexdigest()
        assert provenance["validation_provenance_sha256"] == hashlib.sha256(
            validation_provenance_path.read_bytes()
        ).hexdigest()
        assert provenance["tactile_encoder_provenance_sha256"] == hashlib.sha256(
            encoder_provenance_path.read_bytes()
        ).hexdigest()
        return PassingReport()

    monkeypatch.setattr(TRAIN_SCRIPT, "validate_checkpoint", validate_checkpoint)

    result = TRAIN_SCRIPT._save_training_checkpoint_atomically(
        final,
        trainer=FakeTrainer(),
        preprocessor=FakePreprocessor(),
        source_dir=source,
        data_split_path=split_path,
        normalization_manifest_path=normalization_manifest_path,
        validation_provenance_path=validation_provenance_path,
        tactile_encoder_provenance_path=encoder_provenance_path,
    )

    assert result == final
    assert events == [
        "trainer.save",
        "save_normalization_assets",
        "validate_checkpoint",
        "require_valid",
    ]
    assert (final / TRAIN_SCRIPT.DATA_SPLIT_FILENAME).is_file()
    assert (final / TRAIN_SCRIPT.NORMALIZATION_MANIFEST_FILENAME).is_file()


def test_training_checkpoint_rejects_base_sidecars_for_vt_weights(tmp_path: Path) -> None:
    source = tmp_path / "base"
    source.mkdir()
    input_features = {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.camera1": {"type": "VISUAL", "shape": [3, 512, 512]},
        "observation.images.camera2": {"type": "VISUAL", "shape": [3, 512, 512]},
        "observation.images.camera3": {"type": "VISUAL", "shape": [3, 512, 512]},
    }
    (source / "config.json").write_text(
        json.dumps(
            {
                "chunk_size": 50,
                "input_features": input_features,
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
                "use_tactile_encoder": False,
                "tactile_keys": [],
                "tactile_embedding_dim": 512,
                "tactile_num_tokens": 0,
                "lora_rank": 0,
                "vlm_lora_target_modules": [],
            }
        ),
        encoding="utf-8",
    )
    normalizer_features = {
        "observation.state": {"type": "STATE", "shape": [6]},
        "action": {"type": "ACTION", "shape": [6]},
        **{key: value for key, value in input_features.items() if key != "observation.state"},
    }
    (source / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {"features": normalizer_features},
                        "state_file": ("policy_preprocessor_step_5_normalizer_processor.safetensors"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (source / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {"features": {"action": {"type": "ACTION", "shape": [6]}}},
                        "state_file": ("policy_postprocessor_step_0_unnormalizer_processor.safetensors"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    save_safetensors_file(
        {
            "observation.state.mean": np.zeros(6, dtype=np.float32),
            "observation.state.std": np.ones(6, dtype=np.float32),
            "action.mean": np.zeros(6, dtype=np.float32),
            "action.std": np.ones(6, dtype=np.float32),
        },
        source / "policy_preprocessor_step_5_normalizer_processor.safetensors",
    )
    save_safetensors_file(
        {
            "action.mean": np.zeros(6, dtype=np.float32),
            "action.std": np.ones(6, dtype=np.float32),
        },
        source / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )

    class FakeTrainer:
        config = _vt_config()

        def save(self, destination: Path, *, source_dir: Path) -> None:
            for sidecar in (
                "config.json",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "policy_preprocessor_step_5_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            ):
                shutil.copy2(source_dir / sidecar, destination / sidecar)
            save_safetensors_file(
                {
                    "model.state_proj.weight": np.zeros((1, 1), dtype=np.float32),
                    "model.tactile_encoder.params/conv_init/kernel": np.zeros((1,), dtype=np.float32),
                    "model.tactile_proj.weight": np.zeros((1, 1), dtype=np.float32),
                },
                destination / "model.safetensors",
            )
            (destination / "resume_metadata.json").write_text(
                '{"version": 1}\n', encoding="utf-8"
            )

    class NoOpPreprocessor:
        def save_normalization_assets(self, destination: Path) -> None:
            del destination

    final = tmp_path / "checkpoint-00000020"
    with pytest.raises(ValueError, match="byte-identical to base"):
        TRAIN_SCRIPT._save_training_checkpoint_atomically(
            final,
            trainer=FakeTrainer(),
            preprocessor=NoOpPreprocessor(),
            source_dir=source,
            data_split_path=None,
            normalization_manifest_path=None,
        )

    assert not final.exists()
    assert final.with_name(final.name + ".incomplete").is_dir()
