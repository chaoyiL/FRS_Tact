from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SHARED_LAUNCHER = ROOT / "deploy_smolvla" / "scripts" / "start_remote_client.sh"
SMOLVLA_LAUNCHER = ROOT / "deploy_smolvla" / "scripts" / "start_smolvla.sh"
FRS_CONFIG = ROOT / "deploy_smolvla" / "configs" / "deploy_frs.yaml"
DEFAULT_CONFIG = ROOT / "deploy_smolvla" / "configs" / "deploy_smolvla_pytorch.yaml"
DEFAULT_MODEL_CACHE = ROOT / "checkpoints" / "model"


def test_pi05_path_loader_delegates_raw_bytes_to_bytes_loader(monkeypatch, tmp_path: Path) -> None:
    from deploy_pi05 import deployment

    config_path = tmp_path / "deploy_pi05.yaml"
    raw_config = b"the bytes loader receives this exact payload\n"
    config_path.write_bytes(raw_config)
    seen: list[tuple[bytes, str]] = []
    expected = {"loaded": "from raw bytes"}

    monkeypatch.setattr(
        deployment,
        "load_deployment_config_bytes",
        lambda payload, mode: seen.append((payload, mode)) or expected,
        raising=False,
    )

    assert deployment.load_deployment_config(config_path, "pi05") is expected
    assert seen == [(raw_config, "pi05")]


def test_plain_pi05_client_starts_without_config_handshake(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from deploy_pi05 import pi05_client

    events: list[object] = []
    config_path = tmp_path / "deploy_pi05.yaml"
    raw_config = b"raw YAML bytes are hashed exactly\n"
    config_path.write_bytes(raw_config)
    config = {
        "connection": {
            "address": "robot.example",
            "port": 8000,
            "observation_timeout_s": 1.25,
        },
        "observation": {"language_prompt": "move the block"},
        "runtime": {"warmup_runs": 1, "auto_start": True},
        "seed": 7,
        "num_steps": 2,
        "logging": {},
    }

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            events.append(("connect", kwargs["address"], kwargs["port"]))
            self.received = 0

        def receive_observation(self, *, timeout: float) -> tuple[int, dict[str, object]]:
            self.received += 1
            events.append(("receive", timeout))
            return self.received * 10, {"source": self.received}

        def send_state(self, state: str) -> None:
            events.append(("state", state))

        def send_action(self, action: np.ndarray, obs_seq: int) -> None:
            events.append(("action", action.shape, obs_seq))

    class FakePolicy:
        config = SimpleNamespace(
            state_dim=2,
            action_horizon=2,
            action_dim=2,
            robot_action_dim=2,
        )
        robot_image_keys = ("camera",)

        def predict_action_chunk(self, observation, task: str, *, seed: int, num_steps: int):
            events.append(("predict", observation["source"], task, seed, num_steps))
            return np.zeros((1, 2, 2), dtype=np.float32)

        def unnormalize_actions(self, action: np.ndarray) -> np.ndarray:
            return action

    read_paths: list[Path] = []
    original_read_bytes = Path.read_bytes

    def record_read_bytes(path: Path) -> bytes:
        read_paths.append(path)
        return original_read_bytes(path)

    loaded_payloads: list[tuple[bytes, str]] = []

    def load_from_bytes(payload: bytes, mode: str) -> dict[str, object]:
        loaded_payloads.append((payload, mode))
        if payload != raw_config:
            raise AssertionError("pi0.5 config loader received bytes different from the hash input")
        return config

    monkeypatch.setattr(Path, "read_bytes", record_read_bytes)
    monkeypatch.setattr(
        pi05_client,
        "load_deployment_config",
        lambda *args, **kwargs: pytest.fail("pi0.5 client must not reload config by path"),
        raising=False,
    )
    monkeypatch.setattr(
        pi05_client,
        "load_deployment_config_bytes",
        load_from_bytes,
        raising=False,
    )
    monkeypatch.setattr(pi05_client, "make_policy_config", lambda *args: object())
    monkeypatch.setattr(
        pi05_client,
        "make_server_config",
        lambda *args, **kwargs: {},
        raising=False,
    )
    monkeypatch.setattr(pi05_client, "_jax_runtime", lambda: ("cpu", ()))
    monkeypatch.setattr(pi05_client, "print_startup_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(pi05_client, "_make_policy", lambda policy_config: FakePolicy())
    monkeypatch.setattr(pi05_client, "RobotBridgeClient", FakeBridge)
    monkeypatch.setattr(pi05_client, "prepare_observation", lambda observation, **kwargs: observation)
    monkeypatch.setattr(pi05_client, "start_observation_saver", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pi05_client,
        "cleanup_deployment_resources",
        lambda *args, **kwargs: events.append("cleanup"),
    )

    pi05_client.run(config_path, max_iterations_override=1)

    output = capsys.readouterr().out
    assert f"[startup] deploy config path: {config_path.resolve()}" in output
    assert f"[startup] deploy config sha256: {hashlib.sha256(raw_config).hexdigest()}" in output
    assert read_paths == [config_path.resolve()]
    assert loaded_payloads == [(raw_config, "pi05")]
    assert events == [
        ("connect", "robot.example", 8000),
        ("receive", 1.25),
        ("predict", 1, "move the block", 7, 2),
        ("state", "start"),
        ("receive", 1.25),
        ("predict", 2, "move the block", 7, 2),
        ("action", (2, 2), 20),
        ("receive", 1.25),
        "cleanup",
    ]


def test_frs_pi05_client_starts_without_config_handshake(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    policy_module = ModuleType("deploy_pi05.policy")
    policy_module.Pi05RemotePolicy = object
    monkeypatch.setitem(sys.modules, "deploy_pi05.policy", policy_module)
    from deploy_pi05 import remote_client
    from deploy_pi05.frs_protocol import FRSChunkEnd, FRSChunkStart
    from deploy_pi05.frs_runtime import FRSChunkReady

    events: list[object] = []
    config_path = tmp_path / "deploy_pi05_frs.yaml"
    raw_config = b"FRS YAML bytes are hashed and parsed exactly once\n"
    config_path.write_bytes(raw_config)
    config = {
        "connection": {
            "address": "robot.example",
            "port": 8000,
            "observation_timeout_s": 1.25,
            "action_ack_timeout_s": 0.5,
        },
        "observation": {"language_prompt": "move the block"},
        "control": {"control_frequency": 10.0, "controller_frequency": 80.0},
        "runtime": {"warmup_runs": 1, "auto_start": False},
        "seed": 7,
        "num_steps": 2,
        "frs": {},
        "gripper": {
            "left_close_threshold": 0.08,
            "left_reopen_threshold": 0.10,
            "left_closed_command": 0.01,
            "right_close_threshold": 0.09,
            "right_reopen_threshold": 0.10,
            "right_closed_command": 0.01,
        },
        "logging": {},
    }
    observation = {"source": "robot"}

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            events.append(("connect", kwargs["address"], kwargs["port"]))
            self.frs_messages = 0

        def receive_observation(self, *, timeout: float) -> tuple[int, dict[str, object]]:
            events.append(("receive_warmup", timeout))
            return 10, observation

        def send_state(self, state: str, obs_seq: int | None = None) -> None:
            events.append(("state", state, obs_seq))

        def receive_frs_message(self, timeout: float) -> object:
            self.frs_messages += 1
            events.append(("receive_frs", self.frs_messages, timeout))
            if self.frs_messages == 1:
                return FRSChunkStart(
                    obs_seq=11,
                    chunk_id=5,
                    observation=observation,
                    observation_timestamp=100.0,
                    control_dt=0.1,
                    action_horizon=2,
                    execution_mode="block",
                    action_timestamps=None,
                    nominal_chunk_end=None,
                )
            return FRSChunkEnd(5, "exhausted", 0, 0)

        def send_frs_chunk_ready(
            self, obs_seq: int, chunk_id: int, prediction_trace: object
        ) -> None:
            del prediction_trace
            events.append(("ready", obs_seq, chunk_id))

    class FakePolicy:
        config = SimpleNamespace(state_dim=2, action_horizon=2, robot_action_dim=2)
        robot_image_keys = ("camera",)

        def __init__(self, policy_config: object) -> None:
            del policy_config

    class FakeFRS:
        tactile_keys = ("tactile",)

        def __init__(
            self,
            _config,
            *,
            config_path,
            policy,
            source_sample_steps,
            gripper_hysteresis,
            task,
            task1_motion_gain,
        ) -> None:
            del config_path, source_sample_steps
            assert task == 0
            assert task1_motion_gain.translation_gain == pytest.approx(1.0)
            assert task1_motion_gain.rotation_gain == pytest.approx(1.0)
            assert gripper_hysteresis.left_close_threshold == pytest.approx(0.08)
            assert gripper_hysteresis.right_close_threshold == pytest.approx(0.09)
            self.policy = policy

        @staticmethod
        def reset_episode(warmup: object) -> None:
            del warmup
            events.append("reset_episode")

        @staticmethod
        def warmup(warmup: object, task: str, *, seed: int, sample_steps: int) -> None:
            del warmup
            events.append(("warmup", task, seed, sample_steps))

        @staticmethod
        def begin_chunk(
            chunk_id: int, initial_observation: object, task: str, *, seed: int, num_steps: int
        ) -> FRSChunkReady:
            del initial_observation, task, seed, num_steps
            events.append(("begin_chunk", chunk_id))
            chunk = np.zeros((1, 2, 2), dtype=np.float32)
            return FRSChunkReady(chunk_id, chunk, chunk, chunk, 1.0, 2.0)

        @staticmethod
        def end_chunk(chunk_id: int) -> None:
            events.append(("end_chunk", chunk_id))

    read_paths: list[Path] = []
    original_read_bytes = Path.read_bytes

    def record_read_bytes(path: Path) -> bytes:
        read_paths.append(path)
        return original_read_bytes(path)

    loaded_payloads: list[tuple[bytes, str]] = []

    def load_from_bytes(payload: bytes, mode: str) -> dict[str, object]:
        loaded_payloads.append((payload, mode))
        if payload != raw_config:
            raise AssertionError("FRS config loader received bytes different from the hash input")
        return config

    server_config_calls: list[object] = []

    def record_server_config(*args: object, **kwargs: object) -> dict[str, object]:
        server_config_calls.append((args, kwargs))
        return {}

    monkeypatch.setattr(Path, "read_bytes", record_read_bytes)
    monkeypatch.setattr(remote_client, "load_config", lambda _path: config, raising=False)
    monkeypatch.setattr(
        remote_client,
        "load_deployment_config_bytes",
        load_from_bytes,
        raising=False,
    )
    monkeypatch.setattr(
        remote_client, "make_policy_config", lambda *args: SimpleNamespace(state_dim=2)
    )
    monkeypatch.setattr(remote_client, "make_server_config", record_server_config, raising=False)
    monkeypatch.setattr(remote_client, "_jax_runtime", lambda: ("cpu", ()))
    monkeypatch.setattr(remote_client, "print_startup_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(remote_client, "Pi05RemotePolicy", FakePolicy)
    monkeypatch.setattr(remote_client, "FRSRuntime", FakeFRS)
    monkeypatch.setattr(remote_client, "RobotBridgeClient", FakeBridge)
    monkeypatch.setattr(remote_client, "prepare_observation", lambda value, **kwargs: value)
    monkeypatch.setattr(remote_client, "start_observation_saver", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: events.append(("confirm_start", prompt)) or "",
    )
    monkeypatch.setattr(
        remote_client,
        "cleanup_deployment_resources",
        lambda *args, **kwargs: events.append("cleanup"),
    )

    remote_client.run(config_path, max_iterations_override=1)

    output = capsys.readouterr().out
    assert f"[startup] deploy config path: {config_path.resolve()}" in output
    assert f"[startup] deploy config sha256: {hashlib.sha256(raw_config).hexdigest()}" in output
    assert read_paths == [config_path.resolve()]
    assert loaded_payloads == [(raw_config, "frs")]
    assert server_config_calls == []
    assert events == [
        ("connect", "robot.example", 8000),
        ("receive_warmup", 1.25),
        "reset_episode",
        ("warmup", "move the block", 7, 2),
        ("confirm_start", "[client] Ready. Press Enter to send START to the robot server... "),
        ("state", "start", 10),
        ("receive_frs", 1, 1.25),
        ("begin_chunk", 5),
        ("ready", 11, 5),
        ("receive_frs", 2, 1.25),
        ("end_chunk", 5),
        "cleanup",
    ]


def test_frs_deployment_rejects_auto_start_true() -> None:
    from deploy_pi05.deployment import load_deployment_config_bytes

    config_path = ROOT / "deploy_pi05" / "configs" / "deploy_pi05_frs.yaml"
    config = yaml.safe_load(config_path.read_bytes())
    config["runtime"]["auto_start"] = True

    with pytest.raises(ValueError, match="FRS.*auto_start.*false"):
        load_deployment_config_bytes(yaml.safe_dump(config).encode(), "frs")


def test_pi05_bridge_state_obs_seq_extension_preserves_legacy_payload() -> None:
    from deploy_pi05.bridge_client import RobotBridgeClient

    bridge = object.__new__(RobotBridgeClient)
    sent: list[dict[str, object]] = []
    bridge._send = sent.append

    bridge.send_state("start", obs_seq=7)
    bridge.send_state("stop")

    assert sent == [
        {"type": "state", "state": "start", "obs_seq": 7},
        {"type": "state", "state": "stop"},
    ]


def test_visual_contract_loads_visual_policy(
    monkeypatch, tmp_path
) -> None:
    from deploy_smolvla import remote_client

    config = {
        "checkpoint_contract": {
            "state_dim": 20,
            "action_dim": 20,
            "chunk_size": 20,
            "image_keys": ["rgb"],
            "lora_rank": 0,
            "vlm_lora_target_modules": [],
        }
    }
    expected = remote_client._checkpoint_contract(config, {"action_horizon": 20})
    selected = []
    validated = []
    monkeypatch.setattr(remote_client, "resolve_checkpoint", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        remote_client,
        "validate_checkpoint",
        lambda *args, **kwargs: validated.append(kwargs["expected"])
        or type("Report", (), {"require_valid": lambda self: self})(),
    )
    monkeypatch.setattr(
        remote_client.JaxSmolVLAPolicy,
        "from_pretrained",
        lambda *args, **kwargs: selected.append("visual") or object(),
    )

    remote_client._load_validated_policy(
        "checkpoint",
        revision=None,
        allow_download=False,
        expected=expected,
        rename_map=None,
    )

    assert validated == [expected]
    assert selected == ["visual"]


def _copy_deploy_entry_points(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    deploy_dir = project / "deploy_smolvla"
    scripts_dir = deploy_dir / "scripts"
    config_dir = deploy_dir / "configs"
    scripts_dir.mkdir(parents=True)
    config_dir.mkdir()
    shutil.copy2(SHARED_LAUNCHER, scripts_dir / SHARED_LAUNCHER.name)
    (scripts_dir / SHARED_LAUNCHER.name).chmod(0o644)
    shutil.copy2(DEFAULT_CONFIG, config_dir / DEFAULT_CONFIG.name)
    return project


def _fake_python(tmp_path: Path) -> Path:
    python = tmp_path / "fake-python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ -n "${FAKE_PYTHON_MARKER:-}" ]]; then\n'
        '    : >"${FAKE_PYTHON_MARKER}"\n'
        "fi\n"
        "if [[ -v TRANSFORMERS_CACHE || -v PYTORCH_TRANSFORMERS_CACHE || "
        "-v PYTORCH_PRETRAINED_BERT_CACHE ]]; then\n"
        "    echo 'legacy model cache variable reached Python' >&2\n"
        "    exit 47\n"
        "fi\n"
        "printf '%s\\n' \"${HF_HUB_CACHE}\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return python


def _run_launcher(
    project: Path,
    python: Path,
    *,
    hub_cache: Path | None = None,
    marker: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["FRS_PYTHON"] = str(python)
    env["VB_ROBOT_TOKEN"] = "test-token"
    env["TRANSFORMERS_CACHE"] = str(project / "decoy/transformers")
    env["PYTORCH_TRANSFORMERS_CACHE"] = str(project / "decoy/pytorch-transformers")
    env["PYTORCH_PRETRAINED_BERT_CACHE"] = str(project / "decoy/pytorch-bert")
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    if hub_cache is not None:
        env["HF_HUB_CACHE"] = str(hub_cache)
    if marker is not None:
        env["FAKE_PYTHON_MARKER"] = str(marker)
    return subprocess.run(
        [
            "bash",
            str(project / "deploy_smolvla" / "scripts" / SHARED_LAUNCHER.name),
            "--config",
            str(project / "deploy_smolvla" / "configs" / DEFAULT_CONFIG.name),
        ],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_check(
    *, token_file: Path, token: str | None = None, hub_cache: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VB3_TOKEN_FILE"] = str(token_file)
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    if token is None:
        env.pop("VB_ROBOT_TOKEN", None)
    else:
        env["VB_ROBOT_TOKEN"] = token
    if hub_cache is not None:
        env["HF_HUB_CACHE"] = str(hub_cache)
    return subprocess.run(
        ["bash", str(SHARED_LAUNCHER), "--config", str(DEFAULT_CONFIG), "--check"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_wrapper_check(
    wrapper: Path,
    *,
    extra_args: tuple[str, ...] = (),
    config_override: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VB_ROBOT_TOKEN"] = "test-token"
    env["HF_HUB_CACHE"] = str(ROOT / "checkpoints" / "model")
    if config_override is not None:
        env["SMOLVLA_VISION_CONFIG"] = str(config_override)
        env["SMOLVLA_FRS_CONFIG"] = str(config_override)
    return subprocess.run(
        ["bash", str(wrapper), "--check", *extra_args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_smolvla_frs_mode_uses_frs_config_without_executable_nested_script() -> None:
    assert SHARED_LAUNCHER.stat().st_mode & 0o111 == 0

    result = _run_wrapper_check(SMOLVLA_LAUNCHER, extra_args=("--mode", "frs"))

    assert result.returncode == 0, result.stderr
    assert f"config={FRS_CONFIG}" in result.stdout


def test_smolvla_wrapper_uses_visual_config_without_executable_nested_script() -> None:
    assert SHARED_LAUNCHER.stat().st_mode & 0o111 == 0

    result = _run_wrapper_check(SMOLVLA_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert f"config={DEFAULT_CONFIG}" in result.stdout


@pytest.mark.parametrize(
    ("wrapper", "explicit_config"),
    (
        (SMOLVLA_LAUNCHER, DEFAULT_CONFIG),
        (SMOLVLA_LAUNCHER, FRS_CONFIG),
    ),
)
def test_wrapper_allows_later_explicit_config_override(
    wrapper: Path, explicit_config: Path
) -> None:
    result = _run_wrapper_check(
        wrapper,
        extra_args=("--config", str(explicit_config)),
    )

    assert result.returncode == 0, result.stderr
    assert f"config={explicit_config}" in result.stdout


@pytest.mark.parametrize(
    ("wrapper", "config_override"),
    (
        (SMOLVLA_LAUNCHER, DEFAULT_CONFIG),
        (SMOLVLA_LAUNCHER, FRS_CONFIG),
    ),
)
def test_public_wrapper_preserves_frs_deploy_config_override(
    wrapper: Path, config_override: Path
) -> None:
    result = _run_wrapper_check(wrapper, config_override=config_override)

    assert result.returncode == 0, result.stderr
    assert f"config={config_override}" in result.stdout


def test_shared_launcher_requires_explicit_config(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["VB_ROBOT_TOKEN"] = "test-token"
    env["HF_HUB_CACHE"] = str(tmp_path / "hub")

    result = subprocess.run(
        ["bash", str(SHARED_LAUNCHER), "--check"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--config is required" in result.stderr


def test_launcher_reads_token_file_without_printing_secret(tmp_path: Path) -> None:
    token_file = tmp_path / "token_list.txt"
    token_file.write_text("one-click-secret\n", encoding="utf-8")

    result = _run_check(token_file=token_file)

    assert result.returncode == 0, result.stderr
    assert f"config={DEFAULT_CONFIG}" in result.stdout
    assert f"token_source=file:{token_file}" in result.stdout
    assert "one-click-secret" not in result.stdout + result.stderr


def test_launcher_accepts_environment_token_without_token_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing-token-list.txt"

    result = _run_check(token_file=missing, token="environment-secret")

    assert result.returncode == 0, result.stderr
    assert "token_source=environment:VB_ROBOT_TOKEN" in result.stdout
    assert "environment-secret" not in result.stdout + result.stderr


def test_launcher_fails_before_start_when_token_is_unavailable(tmp_path: Path) -> None:
    missing = tmp_path / "missing-token-list.txt"

    result = _run_check(token_file=missing)

    assert result.returncode == 2
    assert "VB_ROBOT_TOKEN" in result.stderr
    assert str(missing) in result.stderr


def test_launcher_uses_project_model_cache_by_default(tmp_path: Path) -> None:
    result = _run_check(token_file=tmp_path / "missing", token="secret")

    assert result.returncode == 0, result.stderr
    assert f"model_cache={DEFAULT_MODEL_CACHE}" in result.stdout


def test_launcher_preserves_explicit_hf_hub_cache(tmp_path: Path) -> None:
    cache = tmp_path / "hub"

    result = _run_check(token_file=tmp_path / "missing", token="secret", hub_cache=cache)

    assert result.returncode == 0, result.stderr
    assert f"model_cache={cache}" in result.stdout
    assert cache.is_dir()


def test_launcher_exec_uses_project_model_cache_by_default(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    result = _run_launcher(project, _fake_python(tmp_path))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{project / 'checkpoints' / 'model'}\n"
    assert (project / "checkpoints" / "model").is_dir()
    assert (project / "checkpoints" / "encoder").is_dir()


def test_launcher_exec_drops_legacy_cache_overrides(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    cache = tmp_path / "selected-hub-cache"

    result = _run_launcher(
        project,
        _fake_python(tmp_path),
        hub_cache=cache,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{cache}\n"


def test_launcher_creates_checkpoint_directories_in_fresh_project(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    env = os.environ.copy()
    env["VB_ROBOT_TOKEN"] = "secret"
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)

    result = subprocess.run(
        [
            "bash",
            str(project / "deploy_smolvla" / "scripts" / SHARED_LAUNCHER.name),
            "--config",
            str(project / "deploy_smolvla" / "configs" / DEFAULT_CONFIG.name),
            "--check",
        ],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (project / "checkpoints" / "model").is_dir()
    assert (project / "checkpoints" / "encoder").is_dir()


def test_launcher_keeps_existing_checkpoint_files_unchanged(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    model_sentinel = project / "checkpoints" / "model" / "model.sentinel"
    encoder_sentinel = project / "checkpoints" / "encoder" / "encoder.sentinel"
    model_sentinel.parent.mkdir(parents=True)
    encoder_sentinel.parent.mkdir(parents=True)
    model_sentinel.write_text("existing model cache\n", encoding="utf-8")
    encoder_sentinel.write_text("existing encoder checkpoint\n", encoding="utf-8")
    env = os.environ.copy()
    env["VB_ROBOT_TOKEN"] = "secret"
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)

    result = subprocess.run(
        [
            "bash",
            str(project / "deploy_smolvla" / "scripts" / SHARED_LAUNCHER.name),
            "--config",
            str(project / "deploy_smolvla" / "configs" / DEFAULT_CONFIG.name),
            "--check",
        ],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert model_sentinel.read_text(encoding="utf-8") == "existing model cache\n"
    assert encoder_sentinel.read_text(encoding="utf-8") == "existing encoder checkpoint\n"


def test_launcher_exec_preserves_explicit_hf_hub_cache(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    cache = tmp_path / "custom-cache" / "hub"

    result = _run_launcher(project, _fake_python(tmp_path), hub_cache=cache)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{cache}\n"
    assert cache.is_dir()


def test_launcher_fails_before_python_when_hf_hub_cache_is_a_file(
    tmp_path: Path,
) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    cache = tmp_path / "hub-cache-file"
    marker = tmp_path / "python-started"
    cache.write_text("not a directory\n", encoding="utf-8")

    result = _run_launcher(
        project,
        _fake_python(tmp_path),
        hub_cache=cache,
        marker=marker,
    )

    assert result.returncode != 0
    assert not marker.exists()


def test_launcher_help_documents_project_local_hf_hub_cache(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    env = os.environ.copy()
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)

    result = subprocess.run(
        [
            "bash",
            str(project / "deploy_smolvla" / "scripts" / SHARED_LAUNCHER.name),
            "--help",
        ],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "HF_HUB_CACHE" in result.stdout
    assert "<project>/checkpoints/model" in result.stdout
    assert "FRS_DEPLOY_CONFIG" not in result.stdout


def test_checkpoint_gitignore_is_root_anchored() -> None:
    for path in ("checkpoints/model", "checkpoints/encoder"):
        result = subprocess.run(
            ["git", "check-ignore", "--verbose", "--no-index", "--", path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, path
        assert "/checkpoints/" in result.stdout

    nested = subprocess.run(
        [
            "git",
            "check-ignore",
            "--verbose",
            "--no-index",
            "--",
            "somewhere/checkpoints/model",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert nested.returncode == 1
    assert nested.stdout == ""
