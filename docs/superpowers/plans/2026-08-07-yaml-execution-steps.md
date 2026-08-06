# YAML Execution Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `control.steps_per_inference` in the FRS deployment YAML the only value a user changes to select how many actions the client and robot server execute per inference.

**Architecture:** The remote client continues loading a 20-step action chunk, validates the YAML execution count against that chunk, and sends the requested count to the robot server. The robot server keeps a 20-step safety ceiling, so its existing negotiation path honors every valid YAML value from 1 through 20 without a hidden five-step cap.

**Tech Stack:** Python 3.12, Python 3.11, pytest, YAML, JAX SmolVLA, Click robot server

## Global Constraints

- `control.action_horizon` and checkpoint `chunk_size` remain 20.
- Checkpoint weights and checkpoint metadata remain unchanged.
- The user changes only `configs/deploy_smolvla_jax.yaml:control.steps_per_inference`.
- The client accepts only integer execution counts in the inclusive range `[1, action_horizon]`.
- Preserve the user's current language prompt and unrelated working-tree changes.

---

### Task 1: Let the deployment YAML override checkpoint `n_action_steps`

**Files:**
- Modify: `tests/jax/test_tactile_integration.py`
- Modify: `deploy_smolvla/remote_client.py`
- Modify: `configs/deploy_smolvla_jax.yaml`

**Interfaces:**
- Consumes: checkpoint `chunk_size=20`, checkpoint metadata `n_action_steps=5`, and YAML `control.steps_per_inference`.
- Produces: a validated `steps_per_inference` integer passed unchanged in `server_config` to `RobotBridgeClient.send_config()`.

- [ ] **Step 1: Change the deployment test fixture and default-config assertion to request 10 steps**

```python
"control": {
    "control_frequency": 30.0,
    "controller_frequency": 80.0,
    "steps_per_inference": 10,
    "action_horizon": 20,
},
```

and:

```python
assert config["control"]["steps_per_inference"] == 10
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/jax/test_tactile_integration.py::test_checkpoint_revision_is_resolved_and_validated_before_robot_connection tests/jax/test_tactile_integration.py::test_default_deployment_config_pins_the_bimanual_vt_contract -q
```

Expected: failures because checkpoint `n_action_steps=5` is rejected against requested 10 and the checked-in YAML still contains 5.

- [ ] **Step 3: Remove only the checkpoint-metadata equality check**

Delete this block from `deploy_smolvla/remote_client.py` while retaining the existing YAML range validation and the `chunk_size == action_horizon` check:

```python
if policy.config.n_action_steps != configured_steps:
    raise ValueError(
        f"Checkpoint n_action_steps={policy.config.n_action_steps} does not match "
        f"steps_per_inference={configured_steps}"
    )
```

- [ ] **Step 4: Set the checked-in deployment YAML execution count to 10**

```yaml
control:
  action_horizon: 20
  steps_per_inference: 10
```

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/jax/test_tactile_integration.py::test_checkpoint_revision_is_resolved_and_validated_before_robot_connection tests/jax/test_tactile_integration.py::test_default_deployment_config_pins_the_bimanual_vt_contract -q
```

Expected: `2 passed`.

### Task 2: Remove the robot server's hidden five-step default cap

**Files:**
- Modify: `/home/typhon/vb3_robot_server/tests/test_smolvla_runtime_contract.py`
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online_test.py`
- Modify: `/home/typhon/vb3_robot_server/configs/server_config.py`

**Interfaces:**
- Consumes: client-negotiated `steps_per_inference` in `[1, action_horizon]`.
- Produces: server defaults `steps_per_inference=20` and `max_executed_actions=20`, allowing the existing runtime calculation to execute the client-requested 5 or 10 actions unchanged.

- [ ] **Step 1: Change server default assertions to the full horizon**

```python
assert config.steps_per_inference == 20
assert config.max_executed_actions == 20
```

- [ ] **Step 2: Change the scheduling test to request 10 and expect the negotiated count by default**

Use:

```python
client = RuntimeClient(config=valid_config(steps_per_inference=10))
```

and update its cases to:

```python
[
    pytest.param({}, 10, id="default-cli-limit"),
    pytest.param({"max_executed_actions": 1}, 1, id="tighter-cli-limit"),
    pytest.param(
        {"max_executed_actions": smolvla.SMOLVLA_ACTION_HORIZON},
        10,
        id="explicit-cli-cannot-loosen",
    ),
]
```

- [ ] **Step 3: Run the focused server tests and verify RED**

Run from `/home/typhon/vb3_robot_server`:

```bash
.venv/bin/python -m pytest tests/test_smolvla_runtime_contract.py::test_server_config_collects_runtime_defaults deploy_scripts/bimanual_smolvla_online_test.py::test_main_limits_scheduled_actions_to_negotiated_steps_per_inference -q
```

Expected: failures showing the current defaults and scheduled count remain 5.

- [ ] **Step 4: Raise both server defaults to 20**

```python
max_executed_actions: int = 20
steps_per_inference: int = 20
```

Keep the server's existing validation that negotiated steps are in `[1, action_horizon]` and the explicit CLI safety ceiling.

- [ ] **Step 5: Run the focused server tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_smolvla_runtime_contract.py::test_server_config_collects_runtime_defaults deploy_scripts/bimanual_smolvla_online_test.py::test_main_limits_scheduled_actions_to_negotiated_steps_per_inference -q
```

Expected: `4 passed` because the scheduling test has three parameter cases.

### Task 3: Verify the combined deployment contract

**Files:**
- Verify: `deploy_smolvla/remote_client.py`
- Verify: `configs/deploy_smolvla_jax.yaml`
- Verify: `/home/typhon/vb3_robot_server/configs/server_config.py`
- Verify: client and server test suites listed below

**Interfaces:**
- Consumes: completed Task 1 and Task 2 changes.
- Produces: evidence that YAML 10 is accepted by a checkpoint carrying `n_action_steps=5`, forwarded through the bridge contract, and not capped by the server defaults.

- [ ] **Step 1: Run the relevant FRS test suites**

```bash
.venv/bin/python -m pytest tests/jax/test_tactile_integration.py tests/jax/test_deploy_launcher.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the relevant robot-server test suites**

From `/home/typhon/vb3_robot_server`:

```bash
.venv/bin/python -m pytest deploy_scripts/bimanual_smolvla_online_test.py tests/test_smolvla_runtime_contract.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Check formatting and the exact effective values**

```bash
git diff --check
rg -n "steps_per_inference|max_executed_actions" configs/deploy_smolvla_jax.yaml deploy_smolvla/remote_client.py /home/typhon/vb3_robot_server/configs/server_config.py
```

Expected: no whitespace errors; deployment YAML requests 10; server defaults allow 20; no client equality check remains.
