from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lerobot.policies.smolvla_jax.data import DatasetSource

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "train_smolvla_jax.py"
SPEC = importlib.util.spec_from_file_location("train_smolvla_jax_test_module", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_SCRIPT)
_sources_from_split_manifest = TRAIN_SCRIPT._sources_from_split_manifest
_split_manifest = TRAIN_SCRIPT._split_manifest
run_validation = TRAIN_SCRIPT.run_validation
load_yaml_config = TRAIN_SCRIPT.load_yaml_config


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

    def validate_checkpoint(staging: Path) -> PassingReport:
        events.append("validate_checkpoint")
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
