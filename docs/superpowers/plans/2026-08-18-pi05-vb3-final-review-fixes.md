# pi0.5 / FRS Final Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every finding from the final review by enforcing strict preflight configuration types, making robot cleanup unconditional and best-effort, testing the real FRS state machine on CPU fakes, and correcting the deployment documentation.

**Architecture:** `deployment.py` remains the single common deployment boundary and gains exact boolean/numeric validation plus shared saver/cleanup helpers. The pure FRS section validator moves to a dependency-light `frs_config.py`, while `frs_runtime.py` imports that one implementation. Both client entrypoints establish the bridge inside a total cleanup guard and preserve the existing legacy and `frs_steering_v1` wire behavior.

**Tech Stack:** Python 3.12, PyYAML, NumPy, pytest, Bash.

## Global Constraints

- Do not modify `/home/typhon/vb3_robot_server`.
- Do not install dependencies or use the network.
- Preserve the existing pi0.5 and FRS protocol schemas and numerical behavior.
- Use test-driven development: every production behavior must first fail for the expected reason.
- Preserve optional `connection.add_port: null` while rejecting every non-boolean non-null value.
- Once a bridge connection exists, cleanup order is STOP, saver close, bridge close; each operation is independent and best-effort.
- Observation saving is optional and must never prevent robot control or connection cleanup.

---

### Task 1: Strict preflight configuration types

**Files:**
- Create: `deploy_pi05_frs/frs_config.py`
- Modify: `deploy_pi05_frs/deployment.py`
- Modify: `deploy_pi05_frs/frs_runtime.py`
- Modify: `tests/deploy_pi05_frs/test_deployment.py`
- Modify: `tests/deploy_pi05_frs/test_pi05_client.py`

**Interfaces:**
- Produces: exact boolean validation for common boolean fields.
- Produces: `validate_frs_config_section(config: Mapping[str, Any]) -> None` in a dependency-light module.
- Preserves: `optional_bool(None) is None`; otherwise `optional_bool` accepts only built-in booleans.

- [ ] Add parameterized tests that reject `runtime.auto_start` values `"false"` and `1`, `control.control_frequency=True`, non-boolean `norm_stats.use_quantile_norm`, observation flags, `connection.add_port`, `connection.require_token`, and `logging.save_observations`.
- [ ] Add tests that optional `connection.add_port=None` remains valid and that the real FRS validator rejects pseudo-booleans.
- [ ] Add a client-level test proving invalid `auto_start` fails before bridge construction and never sends START.
- [ ] Run the focused tests and confirm failures identify missing strict validation.
- [ ] Implement `_as_bool`, make `_as_float` reject booleans, validate all common values before model construction, and use the typed helpers at conversion sites.
- [ ] Move the pure FRS section validator into `frs_config.py`; import it from both deployment and runtime so there is exactly one implementation.
- [ ] Re-run focused tests and the complete deployment suite.

### Task 2: Total bridge and saver lifecycle cleanup

**Files:**
- Modify: `deploy_pi05_frs/bridge_client.py`
- Modify: `deploy_pi05_frs/deployment.py`
- Modify: `deploy_pi05_frs/pi05_client.py`
- Modify: `deploy_pi05_frs/remote_client.py`
- Modify: `tests/deploy_pi05_frs/test_bridge_client.py`
- Modify: `tests/deploy_pi05_frs/test_pi05_client.py`
- Create: `tests/deploy_pi05_frs/test_remote_client.py`

**Interfaces:**
- Produces: `start_observation_saver(...) -> ObservationSaver | None` and safe submission/cleanup helpers.
- Produces: cleanup order `bridge.send_state("stop")`, `saver.close()`, `bridge.close()` with independent exception handling.

- [ ] Add a bridge-constructor test proving a connected WebSocket closes when greeting validation fails.
- [ ] Add plain-client tests for saver constructor/start/close failures, STOP failure, cleanup order, and send-config failure.
- [ ] Add equivalent FRS run lifecycle and failure-injection tests.
- [ ] Run focused tests and confirm the current implementation leaks or blocks cleanup.
- [ ] Guard the bridge greeting in its constructor and close the socket before re-raising.
- [ ] Implement shared best-effort saver start/submit and runtime cleanup helpers.
- [ ] Move both clients' post-connect work under `try/finally`, parse negative iteration limits before model/connection construction, and use the shared helpers.
- [ ] Re-run the focused lifecycle tests.

### Task 3: CPU-only real FRS state-machine coverage

**Files:**
- Create: `tests/deploy_pi05_frs/test_remote_client.py`

**Interfaces:**
- Exercises: real `remote_client._run_frs` and `remote_client.run` functions with fake policy/runtime/bridge/saver dependencies.

- [ ] Build typed `FRSChunkStart`, `FRSSteerRequest`, `FRSSteerAck`, and `FRSChunkEnd` sequences using the production protocol dataclasses.
- [ ] Add an `_run_frs` test asserting chunk-ready, steering action, matching ACK, chunk end, and event order.
- [ ] Add `run` tests for both manual and automatic START after warmup, then STOP/saver/close cleanup.
- [ ] Add mismatched ACK and runtime exception cases proving STOP and close still occur.
- [ ] Run the new file red, implement only lifecycle seams needed by the tests, then run it green.

### Task 4: Accurate dry-run documentation

**Files:**
- Modify: `README.md`
- Modify: `deploy_pi05_frs/README.md`

- [ ] Replace the shared-mode dry-run implication with an explicit statement that `bimanual_smolvla.sh --dry-run` validates only plain pi0.5 legacy obs/action/ACK.
- [ ] State that FRS must use a real server flow that already supports `frs_steering_v1`, with trained supervision and an immediately available emergency stop.
- [ ] Do not invent or document an unverified FRS server flag.
- [ ] Search both READMEs to confirm no FRS client command remains attached to the legacy-only dry-run flow.

### Task 5: Verification, report, and commit

**Files:**
- Create or update: `.superpowers/sdd/final-review-fixes-report.md`

- [ ] Run `tests/deploy_pi05_frs` with the existing `/home/typhon/FRS_Tact/.venv` because the repository `.venv` has no pytest executable.
- [ ] Run `bash -n` for all three launchers and both wrapper `--check` commands with a redacted token.
- [ ] Run `git diff --check` and `python -m compileall` for changed Python modules and tests.
- [ ] Request an independent code review against all final-review findings; fix every Critical and Important result.
- [ ] Re-run all verification commands after review changes.
- [ ] Record exact test counts, commands, limitations, and changed behavior in the report.
- [ ] Stage only intended files and create one intent-focused commit; do not push.
