from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from lerobot.policies.smolvla_jax import offline_cache_precompute as PRECOMPUTE_MODULE
from lerobot.policies.smolvla_jax.offline_cache_precompute import (
    InjectedStop,
    OfflineCachePrecomputer,
    read_progress,
)
from lerobot.policies.smolvla_jax.offline_training_cache import (
    OfflineCacheSpec,
    OfflineTrainingCache,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "precompute_smolvla_training_cache.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "precompute_smolvla_training_cache_test_module", SCRIPT_PATH
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
PRECOMPUTE_SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(PRECOMPUTE_SCRIPT)


def _spec() -> OfflineCacheSpec:
    return OfflineCacheSpec(
        repo_id="org/five-frames",
        total_frames=5,
        camera_keys=("left", "right"),
        vision_tokens_per_camera=2,
        vision_hidden_size=4,
        state_dim=3,
        action_dim=2,
        chunk_size=4,
        tokenizer_max_length=6,
        checkpoint_source="/checkpoints/smolvla",
        vision_mode="frozen",
        connector_mode="frozen",
    )


class FakeDataset:
    def __init__(self, *, non_finite_state: bool = False):
        self.accessed: list[int] = []
        self.non_finite_state = non_finite_state

    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> dict[str, object]:
        self.accessed.append(index)
        state = np.asarray([index, index + 0.25, index + 0.5], dtype=np.float32)
        if self.non_finite_state and index == 4:
            state[1] = np.nan
        return {
            "images": np.full((2, 3, 4, 4), index, dtype=np.float32),
            "state": state,
            "actions": np.full((4, 2), index + 0.75, dtype=np.float32),
            "action_is_pad": np.asarray([False, False, False, index == 4]),
            "task": f"task-{index}",
            "episode_index": index // 3,
            "frame_index": index,
        }


def _encode(images: np.ndarray) -> np.ndarray:
    values = images[:, :, 0, 0, 0]
    return np.broadcast_to(values[:, :, None, None], (len(images), 2, 2, 4)).astype(
        ml_dtypes.bfloat16
    )


def _tokenize(tasks: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((len(tasks), 6), dtype=np.int32)
    for row, task in enumerate(tasks):
        tokens[row, 0] = int(task.rsplit("-", 1)[1])
    return tokens, tokens >= 0


def _writer(cache_dir: Path, dataset: FakeDataset) -> OfflineCachePrecomputer:
    return OfflineCachePrecomputer(
        spec=_spec(),
        output_dir=cache_dir,
        dataset=dataset,
        encode_vision=_encode,
        tokenize=_tokenize,
        batch_size=2,
    )


def test_first_creation_validates_then_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    staging = cache_dir.with_name(cache_dir.name + ".incomplete")
    events: list[str] = []
    real_rename = PRECOMPUTE_MODULE._rename_noreplace

    def checked_rename(source: Path, destination: Path) -> None:
        events.append("publish")
        assert source == staging
        assert destination == cache_dir
        assert not destination.exists()
        assert json.loads((source / "metadata.json").read_text())["status"] == "complete"
        OfflineTrainingCache(source, _spec())
        real_rename(source, destination)

    monkeypatch.setattr(PRECOMPUTE_MODULE, "_rename_noreplace", checked_rename)

    result = _writer(cache_dir, FakeDataset()).run()

    assert result == cache_dir
    assert events == ["publish"]
    assert cache_dir.is_dir()
    assert not staging.exists()
    cache = OfflineTrainingCache(cache_dir, _spec())
    assert len(cache) == 5
    np.testing.assert_array_equal(cache[3]["state"], [3.0, 3.25, 3.5])
    np.testing.assert_array_equal(cache[4]["action_is_pad"], [False, False, False, True])


def test_interruption_resumes_at_next_index_without_rewriting_completed_rows(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    first_dataset = FakeDataset()
    writer = _writer(cache_dir, first_dataset)

    with pytest.raises(InjectedStop):
        writer.run(stop_after=2)

    staging = cache_dir.with_name(cache_dir.name + ".incomplete")
    assert read_progress(staging)["next_index"] == 2
    assert first_dataset.accessed == [0, 1]
    state_path = staging / "state.npy"
    before = np.load(state_path, mmap_mode="r", allow_pickle=False)[:2].copy()

    resumed_dataset = FakeDataset()
    _writer(cache_dir, resumed_dataset).run()

    assert resumed_dataset.accessed == [2, 3, 4]
    cache = OfflineTrainingCache(cache_dir, _spec())
    np.testing.assert_array_equal(cache._arrays["state"][:2], before)


def test_compatible_completed_cache_is_validated_and_skipped(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _writer(cache_dir, FakeDataset()).run()

    class FailOnRead(FakeDataset):
        def __getitem__(self, index: int) -> dict[str, object]:
            raise AssertionError(f"completed cache read dataset row {index}")

    result = OfflineCachePrecomputer(
        spec=_spec(),
        output_dir=cache_dir,
        dataset=FailOnRead(),
        encode_vision=lambda images: (_ for _ in ()).throw(AssertionError("encoded")),
        tokenize=lambda tasks: (_ for _ in ()).throw(AssertionError("tokenized")),
        batch_size=2,
    ).run()

    assert result == cache_dir


def test_resume_rejects_incorrect_progress(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    with pytest.raises(InjectedStop):
        _writer(cache_dir, FakeDataset()).run(stop_after=2)
    staging = cache_dir.with_name(cache_dir.name + ".incomplete")
    progress_path = staging / "progress.json"
    progress = read_progress(staging)
    progress["next_index"] = 6
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    with pytest.raises(ValueError, match="next_index"):
        _writer(cache_dir, FakeDataset()).run()


def test_non_finite_numeric_data_is_not_published(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    with pytest.raises(ValueError, match="state.*non-finite"):
        _writer(cache_dir, FakeDataset(non_finite_state=True)).run()

    assert not cache_dir.exists()
    assert cache_dir.with_name(cache_dir.name + ".incomplete").is_dir()


def test_cli_rejects_negative_dataset_index() -> None:
    with pytest.raises(SystemExit):
        PRECOMPUTE_SCRIPT.parse_args(["--config", "train.yaml", "--dataset-index", "-1"])


def test_dataset_index_five_selects_only_the_sixth_yaml_source() -> None:
    config = {"datasets": [{"repo_id": f"org/dataset-{index}"} for index in range(6)]}

    selected = PRECOMPUTE_SCRIPT.select_dataset_source(config, 5)

    assert selected.repo_id == "org/dataset-5"
