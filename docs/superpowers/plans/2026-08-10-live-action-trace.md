# Live Action Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate continuously updated headless PNGs comparing VLA, FRS, and measured robot motion for the full deployment run.

**Architecture:** The client adds an optional diagnostic trace to the existing action message. The server validates and logs trace/action/controller data in append-only JSONL, while an isolated plotter process tails the logs and atomically renders PNGs.

**Tech Stack:** Python 3.12 client, Python 3.11 server, NumPy, msgpack, Matplotlib Agg, pytest.

## Global Constraints

- Do not change the existing robot `action` payload or action safety path.
- Plotting and trace failures must never delay or prevent robot execution, ACK, or STOP.
- Preserve backward compatibility with action messages that omit `trace`.
- Produce six panels: left/right arm by position delta, rotation delta, and gripper.
- Produce both `full_prediction.png` and `executed_vs_actual.png` for the full run.
- Keep full-resolution data in JSONL; display-only downsampling is allowed.
- Do not start cameras or robot hardware during verification.
- Preserve all pre-existing uncommitted configuration changes in both repositories.

---

### Task 1: Versioned client/server action trace envelope

**Files:**
- Modify: `deploy_smolvla/frs_runtime.py`
- Modify: `deploy_smolvla/remote_client.py`
- Modify: `deploy_smolvla/bridge_client.py`
- Modify: `/home/typhon/vb3_robot_server/client/robot_client.py`
- Test: `tests/jax/test_frs_deployment.py`
- Test: `/home/typhon/vb3_robot_server/tests/test_vbvla_dry_run.py`

**Interfaces:**
- Produce client trace version 1 with VLA/FRS normalized and unnormalized arrays plus FRS diagnostics.
- Preserve `RobotBridgeClient.send_action(action, obs_seq)` compatibility by adding an optional keyword-only trace.
- Add a server action-packet API while retaining legacy `wait_for_action()` behavior.

- [ ] Write tests that fail because the original VLA chunk and optional trace are unavailable.
- [ ] Run focused client/server tests and confirm the intended assertion failures.
- [ ] Implement the smallest versioned trace envelope and compatibility API.
- [ ] Run focused tests and confirm they pass.

### Task 2: Server trace persistence and controller wall-clock feedback

**Files:**
- Modify: `/home/typhon/vb3_robot_server/real_world/robot_api/arm/Controller.py`
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online.py`
- Create: `/home/typhon/vb3_robot_server/deploy_scripts/action_trace.py`
- Test: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online_test.py`
- Test: `/home/typhon/vb3_robot_server/tests/test_action_trace.py`

**Interfaces:**
- `ActionTraceLogger.log_chunk(...)` appends one complete chunk record.
- `ActionTraceLogger.log_controller_samples(...)` appends controller feedback records.
- Controller debug messages include an epoch timestamp without removing existing relative time.

- [ ] Write failing schema, timestamp, finiteness, and failure-isolation tests.
- [ ] Confirm the tests fail for missing trace persistence.
- [ ] Implement append-only line-buffered JSONL and integrate it after scheduling.
- [ ] Drain and persist controller feedback without blocking control.
- [ ] Run focused tests and confirm they pass.

### Task 3: Headless full-run PNG renderer

**Files:**
- Create: `/home/typhon/vb3_robot_server/deploy_scripts/action_trace_plotter.py`
- Modify: `/home/typhon/vb3_robot_server/pyproject.toml`
- Modify: `/home/typhon/vb3_robot_server/uv.lock`
- Test: `/home/typhon/vb3_robot_server/tests/test_action_trace_plotter.py`

**Interfaces:**
- Parse both JSONL streams and compute SE(3) delta metrics.
- Render both required 2x3 figures with Matplotlib Agg using atomic replacement.
- Provide a bounded polling CLI that the server can launch as an isolated subprocess.

- [ ] Write failing tests for metrics, executed masks, feedback alignment, and two PNG files.
- [ ] Confirm failures are caused by the missing plotter.
- [ ] Implement pure metric/alignment functions and renderer.
- [ ] Implement plotter subprocess lifecycle with at-most-once-per-second refresh.
- [ ] Run focused tests and confirm they pass without a display or Chrome.

### Task 4: Integration, dry-run, and documentation

**Files:**
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online.py`
- Modify: `/home/typhon/vb3_robot_server/README.md`
- Modify: `tests/jax/test_tactile_integration.py`

**Interfaces:**
- The plotter starts after the session directory exists, receives no robot handles, and is finalized on every shutdown path.
- Dry-run accepts trace-bearing action messages but never initializes hardware.

- [ ] Add failing integration tests for traced dry-run and plotter failure isolation.
- [ ] Implement lifecycle integration and concise output-path logging.
- [ ] Run all relevant client and server deployment suites.
- [ ] Run shell/static checks and inspect both repository diffs for unrelated changes.
