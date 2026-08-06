from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file as save_safetensors_file

from lerobot.policies.smolvla_jax.data import DatasetSource
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
    events: list[str] = []

    class FakeTrainer:
        config = _vt_config()

        def save(self, destination: Path, *, source_dir: Path) -> None:
            events.append("trainer.save")
            assert source_dir == source
            assert destination.name.endswith(".incomplete")
            assert not final.exists()
            (destination / "model-marker").write_text("weights", encoding="utf-8")

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
        return PassingReport()

    monkeypatch.setattr(TRAIN_SCRIPT, "validate_checkpoint", validate_checkpoint)

    result = TRAIN_SCRIPT._save_training_checkpoint_atomically(
        final,
        trainer=FakeTrainer(),
        preprocessor=FakePreprocessor(),
        source_dir=source,
        data_split_path=split_path,
    )

    assert result == final
    assert events == [
        "trainer.save",
        "save_normalization_assets",
        "validate_checkpoint",
        "require_valid",
    ]
    assert (final / TRAIN_SCRIPT.DATA_SPLIT_FILENAME).is_file()


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
        )

    assert not final.exists()
    assert final.with_name(final.name + ".incomplete").is_dir()
