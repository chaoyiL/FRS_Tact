from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest
import yaml
from deploy_baseline_state.runtime import StateDecoderRuntime
from deploy_baseline_state.deployment import (
    load_deployment_config,
    make_server_config,
    expected_source_contract,
)
from deploy_baseline_state.remote_client import run_schedule
from deploy_baseline_state.protocol import (
    ScheduleChunkStart,
    ScheduleSteerRequest,
    ScheduleSteerAck,
    ScheduleChunkEnd,
)

ROOT = Path(__file__).resolve().parents[2]


def runtime(width=20):
    calls = []

    class Policy:
        samples = 0

        def predict_and_reverse_action_chunk(self, obs, task, **kwargs):
            self.samples += 1
            return np.full((1, 50, width), 0.1, np.float32), np.full((1, 50, width), 0.7, np.float32)

        def normalize_state(self, state):
            return np.asarray(state, np.float32) * 2

        def unnormalize_actions(self, actions):
            return np.asarray(actions, np.float32)

    class Decoder:
        def decode(self, start, *, state, vla_action):
            calls.append((start.copy(), state.copy()))
            return np.full_like(start, 0.3)

    policy = Policy()
    r = StateDecoderRuntime(
        policy=policy,
        decoder=Decoder(),
        action_dim=width,
        max_normalized_action_abs=8,
        max_normalized_delta_rms=4,
    )
    return r, policy, calls


def observation(value):
    return {"observation.state": np.full(20, value, np.float32)}


@pytest.mark.parametrize("width", [10, 20])
def test_fixed_structured_origin_latest_state_and_duplicate_payload(width):
    r, policy, calls = runtime(width)
    ready = r.begin_chunk(1, observation(0.1), "task")
    np.testing.assert_allclose(ready.structured_noise, 0.7)
    first = r.steer_action(1, 2, observation(0.2), 0)
    duplicate = {**observation(0.2), "observation.images.tactile_right_0": object()}
    assert r.steer_action(1, 2, duplicate, 0) is first
    second = r.steer_action(1, 3, observation(0.4), 1)
    assert policy.samples == 1 and len(calls) == 2
    np.testing.assert_allclose(calls[0][0], 0.7)
    np.testing.assert_allclose(calls[1][0], 0.7)
    np.testing.assert_allclose(calls[0][1], 0.4)
    np.testing.assert_allclose(calls[1][1], 0.8)
    np.testing.assert_allclose(second.decoded_normalized[..., 9::10], 0.1)
    with pytest.raises(ValueError, match="duplicate"):
        r.steer_action(1, 2, observation(0.3), 0)
    with pytest.raises(ValueError, match="increasing"):
        r.steer_action(1, 4, observation(0.3), 0)
    r.end_chunk(1)
    r.begin_chunk(2, observation(0.1), "task")
    assert policy.samples == 2


@pytest.mark.parametrize("failure", ["reverse", "decode", "nonfinite", "delta"])
def test_fail_stop_does_not_return_coarse_fallback(failure):
    r, policy, calls = runtime()
    if failure == "reverse":
        policy.predict_and_reverse_action_chunk = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("reverse failed")
        )
        with pytest.raises(RuntimeError, match="reverse"):
            r.begin_chunk(1, observation(0.1), "task")
        assert r._active_chunk_id is None
        return
    r.begin_chunk(1, observation(0.1), "task")
    if failure == "decode":
        r.decoder.decode = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("decode failed"))
    else:
        r.decoder.decode = lambda *a, **k: np.full(
            (1, 50, 20), np.nan if failure == "nonfinite" else 100, np.float32
        )
    with pytest.raises((RuntimeError, ValueError)):
        r.steer_action(1, 1, observation(0.2), 0)
    assert r.cached_result(1) is None


def test_deployment_config_has_no_encoder_assets_and_empty_sensor_wire(tmp_path):
    for path in (ROOT / "deploy_baseline_state/configs").glob("*.yaml"):
        config = load_deployment_config(path)
        wire = make_server_config(config)
        assert wire["data_type"] == "vision" and wire["frs_tactile_keys"] == []
        assert wire["execution_protocol"] == "frs_steering_v1"
        assert not hasattr(config, "tactile_encoder")
        assert expected_source_contract(config)["flow"]["origin"] == "structured_noise"
    raw = yaml.safe_load(path.read_text())
    raw["tactile_encoder"] = {}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="tactile"):
        load_deployment_config(bad)


def test_schedule_runs_one_chunk_and_ack_cycle_without_sensor_images():
    r, policy, calls = runtime()
    messages = iter(
        [
            ScheduleChunkStart(1, 1, observation(0.1), 0.0, 0.1, 50, "block", None, None),
            ScheduleSteerRequest(1, 1, 0, None, False, observation(0.2)),
            ScheduleSteerAck(1, 1, 0, "scheduled", 0.0),
            ScheduleChunkEnd(1, "exhausted", 1, 0),
        ]
    )
    sent = []
    bridge = SimpleNamespace(
        receive_schedule_message=lambda timeout: next(messages),
        send_frs_chunk_ready=lambda *a, **k: sent.append("ready"),
        send_frs_steer_action=lambda *a, **k: sent.append("action"),
    )
    run_schedule(
        bridge,
        r,
        task="task",
        observation_timeout_s=1,
        action_ack_timeout_s=1,
        seed=0,
        sample_steps=10,
        max_iterations=1,
    )
    assert sent == ["ready", "action"] and policy.samples == 1 and len(calls) == 1


def test_check_cli_needs_only_vla_norms_and_state_checkpoint(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "deploy_baseline_state/configs/deploy_baseline_state_task3.yaml").read_text()
    )
    source = tmp_path / "vla"
    (source / "params").mkdir(parents=True)
    stats = tmp_path / "assets" / raw["norm_stats"]["asset_id"]
    stats.mkdir(parents=True)
    (stats / "norm_stats.json").write_text("{}")
    decoder = tmp_path / "best.pt"
    decoder.write_bytes(b"asset-only-preflight")
    raw["source"]["checkpoint"] = str(source)
    raw["norm_stats"]["dir"] = str(stats.parent)
    raw["decoder"]["checkpoint"] = str(decoder)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    result = subprocess.run(
        [sys.executable, "-m", "deploy_baseline_state.remote_client", "--config", str(config), "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
