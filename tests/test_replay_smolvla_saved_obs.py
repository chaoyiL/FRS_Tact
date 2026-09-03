from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "replay_smolvla_saved_obs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("replay_smolvla_saved_obs", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTorch:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed(self, value: int) -> None:
        self.seeds.append(value)

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class _FakePolicy:
    def __init__(self) -> None:
        self.reset_count = 0
        self.frames: list[dict] = []

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(self, frame: dict) -> np.ndarray:
        self.frames.append(frame)
        return np.full((1, 20, 20), len(self.frames), dtype=np.float32)


def test_replay_sorts_and_maps_saved_observations_with_deterministic_counterfactual(tmp_path, monkeypatch):
    module = _load_module()
    saved = [
        SimpleNamespace(step=10, camera0_rgb=np.full((2, 2, 3), 10, dtype=np.uint8), camera1_rgb=np.full((2, 2, 3), 11, dtype=np.uint8)),
        SimpleNamespace(step=0, camera0_rgb=np.full((2, 2, 3), 0, dtype=np.uint8), camera1_rgb=np.full((2, 2, 3), 1, dtype=np.uint8)),
    ]
    chunks = [
        SimpleNamespace(obs_seq=2, raw_actions=np.full((20, 20), 20.0, dtype=np.float32)),
        SimpleNamespace(obs_seq=1, raw_actions=np.full((20, 20), 10.0, dtype=np.float32)),
    ]
    torch = _FakeTorch()
    policy = _FakePolicy()
    runtime = SimpleNamespace(
        policy=policy,
        torch=torch,
        horizon=20,
        action_dim=20,
        prepare_frame=lambda observation: observation,
        preprocess=lambda frame: frame,
        postprocess=lambda action: action,
    )
    references: list[tuple[int, int]] = []

    def reconstruct(observation, reference):
        references.append((observation.step, reference.step))
        return np.full(20, observation.step, dtype=np.float32)

    monkeypatch.setattr(module, "load_eval_runtime", lambda config, device: runtime)
    monkeypatch.setattr(module, "load_saved_observations", lambda path: saved)
    monkeypatch.setattr(module, "load_chunk_trace", lambda path: chunks)
    monkeypatch.setattr(module, "reconstruct_state", reconstruct)

    result = module.run_saved_obs_replay(
        config_path=tmp_path / "config.yaml",
        obs_dir=tmp_path / "obs",
        trace_dir=tmp_path / "trace",
        output_dir=tmp_path / "output",
        device="cpu",
        seed=7,
    )

    assert policy.reset_count == 1
    assert torch.seeds == [7, 17]
    assert references == [(0, 0), (10, 0)]
    assert [frame["observation.state"][0] for frame in policy.frames] == [0.0, 10.0]
    for frame in policy.frames:
        assert set(frame) == {"observation.state", "observation.images.camera0", "observation.images.camera1"}
    assert result["steps"] == [0, 10]
    assert result["obs_seq"] == [1, 2]
    assert result["counterfactual"] is True
    assert result["state_reference"] == "saved_step_0_approximation"
    assert "not runtime mismatch proof" in result["full_array_inequality_note"]
    assert result["per_step"][0]["metrics"]["dim_9"]["replay_le_0_09_count"] == 0
    assert result["per_step"][0]["metrics"]["dim_19"]["live_le_0_09_count"] == 0

    archive = np.load(tmp_path / "output" / "replay_predictions.npz")
    assert archive["replay_actions"].shape == (2, 20, 20)
    assert archive["live_actions"].shape == (2, 20, 20)
    assert archive["approximate_states"].shape == (2, 20)
    summary = json.loads((tmp_path / "output" / "replay_summary.json").read_text())
    assert summary["counterfactual"] is True
