# pi0.5 / pi0.5+FRS Unified Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a plain pi0.5 deployment entrypoint that shares one model configuration and one `vb3_robot_server` bridge with the existing pi0.5+FRS deployment.

**Architecture:** A new common deployment module owns YAML profile expansion, policy configuration, observation validation, token resolution, and background observation saving. Separate plain and FRS Python clients reuse that module and the existing `Pi05RemotePolicy`/`RobotBridgeClient`; two thin shell wrappers select the mode through one launcher.

**Tech Stack:** Bash, Python 3.12, PyYAML, NumPy, JAX, Orbax, msgpack, synchronous websockets, pytest.

## Global Constraints

- The migrated plain policy is the current pure-vision `pi05_bi` contract: bimanual 20D state, 20D robot action, 32D model action, two RGB cameras, and a 50-step horizon.
- Plain pi0.5 and pi0.5+FRS must read the same checkpoint, norm stats, model variants, dimensions, camera map, prompt, and control configuration from one YAML file.
- Plain mode uses `data_type: vision` and legacy chunks; FRS mode uses `data_type: vitac` and `frs_steering_v1`.
- Plain mode sends a finite float32 `[50, 20]` chunk with its source `obs_seq` and waits for the matching `action_ack` before receiving the next observation.
- Do not modify `/home/typhon/vb3_robot_server` or migrate the old 15-step AnyTouch/pi05_bi_vitac path.
- Preserve `start_pi05_frs.sh` as a working user command and retain direct-launcher compatibility by defaulting a missing `--mode` to `frs`.
- Follow test-driven development: each production behavior is preceded by a test that fails for the expected missing behavior.

---

## File Structure

- Create `deploy_pi05_frs/deployment.py`: common profile-aware config, policy config construction, observation validation, token/path helpers, server config construction, and `ObservationSaver`.
- Create `deploy_pi05_frs/pi05_client.py`: plain pi0.5 warmup and acknowledged legacy-chunk loop.
- Create `deploy_pi05_frs/configs/deploy_pi05.yaml`: the single shared deployment config.
- Create `deploy_pi05_frs/scripts/start_pi05.sh`: plain-mode wrapper.
- Modify `deploy_pi05_frs/remote_client.py`: consume shared deployment helpers without changing the FRS loop.
- Modify `deploy_pi05_frs/scripts/start_pi05_frs.sh`: select `frs` and the shared config.
- Modify `deploy_pi05_frs/scripts/start_remote_client.sh`: select mode/entrypoint/Python and forward bounded-run arguments.
- Delete `deploy_pi05_frs/configs/deploy_pi05_frs.yaml`: remove the second model-asset configuration source.
- Modify `deploy_pi05_frs/README.md` and `README.md`: document both client modes and current deployment status.
- Create `tests/deploy_pi05_frs/test_deployment.py`: shared config/profile/helper tests.
- Create `tests/deploy_pi05_frs/test_pi05_client.py`: plain loop and failure-path tests.
- Create `tests/deploy_pi05_frs/test_bridge_client.py`: legacy ACK validation characterization tests.
- Create `tests/deploy_pi05_frs/test_start_scripts.py`: shell wrapper and `--check` tests.

---

### Task 1: Shared profile-aware deployment configuration

**Files:**
- Create: `deploy_pi05_frs/deployment.py`
- Create: `deploy_pi05_frs/configs/deploy_pi05.yaml`
- Test: `tests/deploy_pi05_frs/test_deployment.py`

**Interfaces:**
- Produces: `load_deployment_config(path: Path, mode: Literal["pi05", "frs"]) -> dict[str, Any]`.
- Produces: `make_policy_config(config: Mapping[str, Any], config_path: Path) -> Pi05DeploymentConfig`.
- Produces: `prepare_observation(observation, *, state_dim, image_keys) -> dict[str, Any]`.
- Produces: `make_server_config(config, *, mode: Literal["pi05", "frs"], frs_runtime=None) -> dict[str, Any]`.
- Produces: `resolve_token`, `optional_bool`, `section`, and `ObservationSaver` for both clients.

- [ ] **Step 1: Write failing profile/config tests**

Create `tests/deploy_pi05_frs/test_deployment.py` with focused tests that load a minimal temporary YAML and assert profile expansion:

```python
from pathlib import Path

import pytest
import yaml

from deploy_pi05_frs.deployment import load_deployment_config, make_server_config


def _config() -> dict:
    return {
        "checkpoint": "/models/pi05/10000",
        "seed": 0,
        "num_steps": 10,
        "model": {
            "action_dim": 32,
            "action_horizon": 50,
            "state_dim": 20,
            "robot_action_dim": 20,
            "camera_map": {
                "left_wrist_0_rgb": "observation.images.camera0",
                "right_wrist_0_rgb": "observation.images.camera1",
            },
            "empty_cameras": ["base_0_rgb"],
        },
        "norm_stats": {"dir": "/models/pi05/10000/assets", "asset_id": "pick_tube", "use_quantile_norm": True},
        "profiles": {
            "pi05": {"data_type": "vision", "observation_output_dir": "outputs/pi05"},
            "frs": {"data_type": "vitac", "observation_output_dir": "outputs/frs"},
        },
        "connection": {"address": "127.0.0.1", "port": 26421, "action_ack_timeout_s": 30.0},
        "observation": {"language_prompt": "pick", "single_arm_mode": False, "no_state_obs_mode": False},
        "control": {"control_frequency": 20.0, "controller_frequency": 80.0, "action_horizon": 50, "steps_per_inference": 50},
        "runtime": {"warmup_runs": 1},
        "logging": {"save_observations": False, "save_every": 1, "queue_size": 2},
        "frs": {"enabled": True},
    }


def _write(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(("mode", "data_type", "output"), [("pi05", "vision", "outputs/pi05"), ("frs", "vitac", "outputs/frs")])
def test_load_deployment_config_expands_profile(tmp_path, mode, data_type, output):
    loaded = load_deployment_config(_write(tmp_path, _config()), mode)
    assert loaded["observation"]["data_type"] == data_type
    assert loaded["logging"]["output_dir"] == output
    assert loaded["checkpoint"] == "/models/pi05/10000"


def test_plain_server_config_uses_legacy_protocol(tmp_path):
    loaded = load_deployment_config(_write(tmp_path, _config()), "pi05")
    server = make_server_config(loaded, mode="pi05")
    assert server["data_type"] == "vision"
    assert server["action_horizon"] == 50
    assert "execution_protocol" not in server


def test_rejects_horizon_drift_before_model_load(tmp_path):
    config = _config()
    config["control"]["action_horizon"] = 49
    with pytest.raises(ValueError, match="action_horizon"):
        load_deployment_config(_write(tmp_path, config), "pi05")
```

Also add tests for an unknown mode, missing profile, `pi05` profile not using `vision`, `frs` profile not using `vitac`, `steps_per_inference` outside `[1, 50]`, invalid observation shapes, and plain mode succeeding without an `frs` section.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run --no-sync pytest -q tests/deploy_pi05_frs/test_deployment.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'deploy_pi05_frs.deployment'`.

- [ ] **Step 3: Implement the common deployment module**

Move the existing config/token/path/observation/saver behavior out of `remote_client.py`. Use an explicit profile expander rather than a generic recursive merge:

```python
DeploymentMode = Literal["pi05", "frs"]


def load_deployment_config(path: Path, mode: DeploymentMode) -> dict[str, Any]:
    if mode not in ("pi05", "frs"):
        raise ValueError(f"unsupported deployment mode: {mode!r}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    config = copy.deepcopy(payload)
    profiles = section(config, "profiles")
    profile = section(profiles, mode)
    expected_data_type = "vision" if mode == "pi05" else "vitac"
    if profile.get("data_type") != expected_data_type:
        raise ValueError(f"profiles.{mode}.data_type must be {expected_data_type!r}")
    observation = dict(section(config, "observation"))
    observation["data_type"] = expected_data_type
    config["observation"] = observation
    logging_config = dict(config.get("logging", {}) or {})
    logging_config["output_dir"] = required(profile, "observation_output_dir", f"profiles.{mode}")
    config["logging"] = logging_config
    validate_common_config(config)
    if mode == "frs":
        section(config, "frs")
        validate_frs_config_section(config)
    return config
```

Implement `make_server_config` so common keys are identical and only FRS adds protocol fields:

```python
def make_server_config(config, *, mode, frs_runtime=None):
    observation = section(config, "observation")
    control = section(config, "control")
    result = {
        "data_type": observation["data_type"],
        "language_prompt": observation["language_prompt"],
        "control_frequency": float(control["control_frequency"]),
        "controller_frequency": float(control["controller_frequency"]),
        "single_arm_mode": bool(observation["single_arm_mode"]),
        "no_state_obs_mode": bool(observation["no_state_obs_mode"]),
        "steps_per_inference": int(control["steps_per_inference"]),
        "action_horizon": int(control["action_horizon"]),
    }
    if mode == "frs":
        if frs_runtime is None:
            raise ValueError("frs_runtime is required for FRS server config")
        result.update(
            execution_protocol="frs_steering_v1",
            steering_protection_interval_s=frs_runtime.config.steering_protection_interval_s,
            frs_tactile_keys=list(frs_runtime.tactile_keys),
        )
    return result
```

Create `deploy_pi05_frs/configs/deploy_pi05.yaml` by moving all existing model/FRS values into the shared schema, removing `observation.data_type` and `logging.output_dir`, and adding the exact `profiles` block from the design.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
uv run --no-sync pytest -q tests/deploy_pi05_frs/test_deployment.py
```

Expected: all shared deployment tests pass.

- [ ] **Step 5: Commit the shared configuration unit**

```bash
git add deploy_pi05_frs/deployment.py deploy_pi05_frs/configs/deploy_pi05.yaml tests/deploy_pi05_frs/test_deployment.py
git commit -m "feat: add shared pi05 deployment profiles"
```

---

### Task 2: Plain pi0.5 acknowledged legacy-chunk client

**Files:**
- Create: `deploy_pi05_frs/pi05_client.py`
- Test: `tests/deploy_pi05_frs/test_pi05_client.py`
- Test: `tests/deploy_pi05_frs/test_bridge_client.py`

**Interfaces:**
- Consumes: `load_deployment_config`, `make_policy_config`, `make_server_config`, `prepare_observation`, `ObservationSaver` from Task 1.
- Consumes: `Pi05RemotePolicy.predict_action_chunk(...) -> jax.Array` and `.unnormalize_actions(...) -> np.ndarray`.
- Produces: `predict_robot_action_chunk(policy, observation, task, *, seed, num_steps) -> np.ndarray` with exact shape `[action_horizon, robot_action_dim]` and dtype float32.
- Produces: `run(config_path: Path, max_iterations_override: int | None = None) -> None`.

- [ ] **Step 1: Write failing action conversion and control-order tests**

Create fakes that append events and test only observable protocol behavior:

```python
class FakePolicy:
    config = SimpleNamespace(action_horizon=2, action_dim=3, robot_action_dim=2, state_dim=2)
    robot_image_keys = ("observation.images.camera0",)

    def predict_action_chunk(self, observation, task, *, seed, num_steps):
        return np.arange(6, dtype=np.float32).reshape(1, 2, 3)

    def unnormalize_actions(self, actions):
        return np.asarray(actions[..., :2], dtype=np.float32)


def test_predict_robot_action_chunk_returns_full_float32_robot_chunk():
    action = predict_robot_action_chunk(FakePolicy(), _observation(), "pick", seed=0, num_steps=10)
    assert action.shape == (2, 2)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()


def test_legacy_loop_waits_for_matching_ack_before_next_observation():
    bridge = FakeBridge(observations=[(7, _observation()), (8, _observation())])
    run_legacy_loop(
        bridge,
        FakePolicy(),
        task="pick",
        image_keys=FakePolicy.robot_image_keys,
        observation_timeout_s=1.0,
        action_ack_timeout_s=2.0,
        seed=0,
        sample_steps=10,
        max_iterations=2,
        saver=FakeSaver(),
    )
    assert bridge.events == [
        ("receive", 7), ("send_action", 7, (2, 2)), ("ack", 7),
        ("receive", 8), ("send_action", 8, (2, 2)), ("ack", 8),
    ]
```

Add tests that reject wrong output shape/NaN, stop at `max_iterations`, include no `execution_protocol` in the sent config, and ensure `run` attempts STOP and closes the bridge when inference or ACK raises.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --no-sync pytest -q tests/deploy_pi05_frs/test_pi05_client.py
```

Expected: collection fails because `deploy_pi05_frs.pi05_client` does not exist.

- [ ] **Step 3: Implement the minimal plain client**

Use shared helpers and keep the control loop independent from model construction:

```python
def predict_robot_action_chunk(policy, observation, task, *, seed, num_steps):
    normalized = policy.predict_action_chunk(observation, task, seed=seed, num_steps=num_steps)
    expected_model = (1, policy.config.action_horizon, policy.config.action_dim)
    if normalized.shape != expected_model:
        raise ValueError(f"pi0.5 action must have shape {expected_model}, got {normalized.shape}")
    action = np.asarray(policy.unnormalize_actions(normalized[0]), dtype=np.float32)
    expected_robot = (policy.config.action_horizon, policy.config.robot_action_dim)
    if action.shape != expected_robot or not np.isfinite(action).all():
        raise ValueError(f"robot action must be finite with shape {expected_robot}, got {action.shape}")
    return np.ascontiguousarray(action)
```

`run_legacy_loop` must call `receive_observation`, validate/copy the observation, infer, `send_action`, then `receive_action_ack` in that exact order. `run` performs one real-observation warmup phase, sends START only after confirmation, owns the saver, and wraps all exits in:

```python
finally:
    saver.close()
    try:
        bridge.send_state("stop")
    except Exception as error:
        LOGGER.warning("Could not send STOP: %s", error)
    finally:
        bridge.close()
```

Keep CLI parity with the FRS client: `--config` and `--max-iterations`.

- [ ] **Step 4: Add direct bridge ACK characterization tests**

Create `tests/deploy_pi05_frs/test_bridge_client.py` without opening a socket:

```python
def _client_with_message(message):
    client = object.__new__(RobotBridgeClient)
    client._receive = lambda timeout=None: message
    return client


def test_receive_action_ack_accepts_matching_sequence():
    _client_with_message({"type": "action_ack", "obs_seq": 7}).receive_action_ack(7, 1.0)


@pytest.mark.parametrize(
    "message",
    [
        {"type": "obs", "obs_seq": 7},
        {"type": "action_ack", "obs_seq": 6},
        {"type": "action_ack", "obs_seq": True},
    ],
)
def test_receive_action_ack_rejects_wrong_message_or_sequence(message):
    with pytest.raises(RuntimeError):
        _client_with_message(message).receive_action_ack(7, 1.0)
```

These tests lock down the already implemented `vb3_robot_server` ACK contract; do not alter FRS methods.

- [ ] **Step 5: Run plain client and bridge tests**

Run:

```bash
uv run --no-sync pytest -q tests/deploy_pi05_frs/test_pi05_client.py tests/deploy_pi05_frs/test_bridge_client.py tests/deploy_pi05_frs/test_frs_protocol.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit the plain client unit**

```bash
git add deploy_pi05_frs/pi05_client.py tests/deploy_pi05_frs/test_pi05_client.py tests/deploy_pi05_frs/test_bridge_client.py
git commit -m "feat: deploy plain pi05 through vb3 legacy chunks"
```

---

### Task 3: Dual wrappers and generic launcher

**Files:**
- Create: `deploy_pi05_frs/scripts/start_pi05.sh`
- Modify: `deploy_pi05_frs/scripts/start_pi05_frs.sh`
- Modify: `deploy_pi05_frs/scripts/start_remote_client.sh`
- Test: `tests/deploy_pi05_frs/test_start_scripts.py`

**Interfaces:**
- Consumes: `deploy_pi05_frs.pi05_client` and `deploy_pi05_frs.remote_client` module entrypoints.
- Produces: `start_pi05.sh [--check] [--max-iterations N]`.
- Preserves: `start_pi05_frs.sh [--check] [--max-iterations N]`.

- [ ] **Step 1: Write failing shell entrypoint tests**

Use subprocess with a redacted environment token and assert exact check output:

```python
def _check(script: str, *args: str) -> str:
    env = {**os.environ, "VB_ROBOT_TOKEN": "redacted"}
    result = subprocess.run(
        ["bash", str(ROOT / script), "--check", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_plain_wrapper_selects_plain_entrypoint():
    output = _check("deploy_pi05_frs/scripts/start_pi05.sh")
    assert "mode=pi05" in output
    assert "entrypoint=deploy_pi05_frs.pi05_client" in output
    assert "configs/deploy_pi05.yaml" in output


def test_frs_wrapper_selects_frs_entrypoint():
    output = _check("deploy_pi05_frs/scripts/start_pi05_frs.sh")
    assert "mode=frs" in output
    assert "entrypoint=deploy_pi05_frs.remote_client" in output
```

Add tests for default-direct mode `frs`, `PI05_DEPLOY_CONFIG` precedence, legacy `PI05_FRS_DEPLOY_CONFIG` fallback, mode-specific Python precedence, invalid mode, and a missing `--max-iterations` value. Verify forwarding with this executable fixture:

```python
fake_python = tmp_path / "fake-python"
fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >"${PI05_ARGS_FILE}"\n', encoding="utf-8")
fake_python.chmod(0o755)
args_file = tmp_path / "args.txt"
env = {
    **os.environ,
    "VB_ROBOT_TOKEN": "redacted",
    "PI05_PYTHON": str(fake_python),
    "PI05_ARGS_FILE": str(args_file),
}
subprocess.run(
    ["bash", str(ROOT / "deploy_pi05_frs/scripts/start_pi05.sh"), "--max-iterations", "2"],
    cwd=ROOT,
    env=env,
    check=True,
)
assert args_file.read_text(encoding="utf-8").splitlines()[-2:] == ["--max-iterations", "2"]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --no-sync pytest -q tests/deploy_pi05_frs/test_start_scripts.py
```

Expected: failure because `start_pi05.sh` is absent and the launcher has no mode support.

- [ ] **Step 3: Implement the wrappers and launcher**

The plain wrapper must be exactly the same shape as the FRS wrapper:

```bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${PI05_DEPLOY_CONFIG:-${ROOT}/deploy_pi05_frs/configs/deploy_pi05.yaml}"
exec bash "${ROOT}/deploy_pi05_frs/scripts/start_remote_client.sh" \
    --mode pi05 --config "${CONFIG}" "$@"
```

The FRS wrapper uses `PI05_DEPLOY_CONFIG`, then `PI05_FRS_DEPLOY_CONFIG`, then the same default, and passes `--mode frs`.

In the launcher, parse mode/config/check/max-iterations, select:

```bash
case "${MODE}" in
    pi05) ENTRYPOINT="deploy_pi05_frs.pi05_client"; MODE_PYTHON="${PI05_PYTHON:-}" ;;
    frs) ENTRYPOINT="deploy_pi05_frs.remote_client"; MODE_PYTHON="${PI05_FRS_PYTHON:-}" ;;
    *) echo "Unsupported mode: ${MODE}" >&2; exit 2 ;;
esac
```

Build a Bash array for Python arguments and append `--max-iterations "$MAX_ITERATIONS"` only when supplied. `--check` prints metadata and exits before module execution.

- [ ] **Step 4: Run shell tests and syntax checks**

Run:

```bash
bash -n deploy_pi05_frs/scripts/start_pi05.sh
bash -n deploy_pi05_frs/scripts/start_pi05_frs.sh
bash -n deploy_pi05_frs/scripts/start_remote_client.sh
uv run --no-sync pytest -q tests/deploy_pi05_frs/test_start_scripts.py
```

Expected: all commands pass.

- [ ] **Step 5: Commit the shell entrypoints**

```bash
git add deploy_pi05_frs/scripts/start_pi05.sh deploy_pi05_frs/scripts/start_pi05_frs.sh deploy_pi05_frs/scripts/start_remote_client.sh tests/deploy_pi05_frs/test_start_scripts.py
git commit -m "feat: add dual pi05 deployment launchers"
```

---

### Task 4: Move FRS onto shared configuration without behavior drift

**Files:**
- Modify: `deploy_pi05_frs/remote_client.py`
- Delete: `deploy_pi05_frs/configs/deploy_pi05_frs.yaml`
- Modify: `tests/deploy_pi05_frs/test_deployment.py`
- Modify: `tests/deploy_pi05_frs/test_frs_protocol.py`

**Interfaces:**
- Consumes: all common helper interfaces from Task 1.
- Preserves: `_run_frs` message ordering, `frs_steering_v1`, trace payloads, runtime validation, warmup, and STOP behavior.
- Preserves: `remote_client.load_config(path)` as a compatibility wrapper that selects the `frs` profile.

- [ ] **Step 1: Add a failing regression test for the default FRS config**

Add:

```python
def test_default_frs_profile_builds_frs_server_config():
    path = Path("deploy_pi05_frs/configs/deploy_pi05.yaml")
    config = load_deployment_config(path, "frs")
    runtime = SimpleNamespace(
        config=SimpleNamespace(steering_protection_interval_s=None),
        tactile_keys=("t0", "t1", "t2", "t3"),
    )
    server = make_server_config(config, mode="frs", frs_runtime=runtime)
    assert server["data_type"] == "vitac"
    assert server["execution_protocol"] == "frs_steering_v1"
    assert server["frs_tactile_keys"] == ["t0", "t1", "t2", "t3"]
```

Add an import-level test that `remote_client.load_config(path)` yields the FRS profile so external callers do not silently switch behavior.

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
uv run --no-sync pytest -q tests/deploy_pi05_frs/test_deployment.py
```

Expected: the compatibility wrapper/profile use is missing from `remote_client.py`.

- [ ] **Step 3: Refactor FRS client to shared helpers**

Delete duplicated `ObservationSaver`, section/required/config/token/path/observation helpers from `remote_client.py`; import them from `deployment.py`. Keep:

```python
def load_config(path: Path) -> dict[str, Any]:
    return load_deployment_config(path, "frs")
```

Replace the inline server config dictionary with:

```python
server_config = make_server_config(config, mode="frs", frs_runtime=frs)
```

Change `DEFAULT_CONFIG` to `configs/deploy_pi05.yaml`. Remove the old YAML only after both wrappers and both Python defaults reference the new shared YAML.

- [ ] **Step 4: Run focused and full deployment tests**

Run:

```bash
uv run --no-sync pytest -q tests/deploy_pi05_frs
```

Expected: all deployment tests pass and existing protocol tests remain unchanged in behavior.

- [ ] **Step 5: Commit the FRS migration**

```bash
git add deploy_pi05_frs/remote_client.py deploy_pi05_frs/configs tests/deploy_pi05_frs
git commit -m "refactor: share config across pi05 deployment modes"
```

---

### Task 5: Deployment documentation and final verification

**Files:**
- Modify: `deploy_pi05_frs/README.md`
- Modify: `README.md`

**Interfaces:**
- Documents: one shared config, `vb3_robot_server` launcher, both client commands, `--check`, bounded dry-run, token handling, and real-hardware safety sequence.

- [ ] **Step 1: Update documentation against executable commands**

Replace the stale deployment sections with these canonical commands:

```bash
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_smolvla.sh --dry-run

cd /home/typhon/FRS_Tact-pi05-frs-jax
export VB_ROBOT_TOKEN='...'
bash deploy_pi05_frs/scripts/start_pi05.sh --check
bash deploy_pi05_frs/scripts/start_pi05.sh --max-iterations 2
bash deploy_pi05_frs/scripts/start_pi05_frs.sh --check
bash deploy_pi05_frs/scripts/start_pi05_frs.sh --max-iterations 2
```

State that the plain model ignores tactile images and requests `vision`, FRS requests `vitac`, both use the same `checkpoint`/`norm_stats`, and a human must supervise hardware with emergency stop available.

- [ ] **Step 2: Run formatting, static, and focused tests**

Run:

```bash
git diff --check
bash -n deploy_pi05_frs/scripts/start_pi05.sh
bash -n deploy_pi05_frs/scripts/start_pi05_frs.sh
bash -n deploy_pi05_frs/scripts/start_remote_client.sh
VB_ROBOT_TOKEN=redacted bash deploy_pi05_frs/scripts/start_pi05.sh --check
VB_ROBOT_TOKEN=redacted bash deploy_pi05_frs/scripts/start_pi05_frs.sh --check
uv run --no-sync pytest -q tests/deploy_pi05_frs
```

Expected: no diff errors, both checks report the same config and different correct entrypoints, and all deployment tests pass.

- [ ] **Step 3: Run broader regression tests**

Run:

```bash
uv run --no-sync pytest -q tests/pi05 tests/deploy_pi05_frs
```

Expected: all pi0.5 model-contract and deployment tests pass. If this machine lacks the locked environment or a JAX GPU, report that environmental limitation separately; do not claim GPU or robot validation.

- [ ] **Step 4: Inspect the final change set**

Run:

```bash
git status --short
git diff --stat a97dde4..HEAD
git diff --check a97dde4..HEAD
```

Expected: only the planned deployment, tests, configuration, and documentation files changed; no checkpoint, token, observation output, or unrelated user file is present.

- [ ] **Step 5: Commit documentation**

```bash
git add deploy_pi05_frs/README.md README.md
git commit -m "docs: document pi05 vb3 deployment modes"
```

- [ ] **Step 6: Request final code review**

Use the `requesting-code-review` skill against the complete implementation diff. Address correctness findings before claiming completion, then rerun the focused verification commands.
