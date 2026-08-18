# Final Review Fixes Report

## Scope

Closed all Critical and Important findings from the review of `76fbe38..b8be84d` without modifying `/home/typhon/vb3_robot_server`, installing dependencies, or using the network.

## 1. Strict deployment configuration

- Added exact built-in boolean validation for:
  - `runtime.auto_start`
  - `norm_stats.use_quantile_norm`
  - `observation.single_arm_mode`
  - `observation.no_state_obs_mode`
  - `connection.require_token`
  - `logging.save_observations`
  - optional `connection.add_port`, preserving `null` as `None`
- `_as_float` now rejects booleans; `_as_int` continues to reject booleans and is applied to every common integer consumed by the clients before model/connection creation.
- Common preflight now validates root seed/sample-step types, runtime limits, port/timeouts, and logging queue/sample values before model loading or connection side effects.
- `make_policy_config`, token resolution, server config construction, optional-bool conversion, and `ObservationSaver` retain strict validation at their own boundaries.
- Added an end-to-end plain-client test proving `auto_start: "false"` and `auto_start: 1` fail before START is sent.
- Moved `validate_frs_config_section` to dependency-light `deploy_pi05_frs/frs_config.py`; `deployment.py` and `frs_runtime.py` use that single implementation. Tests now call the real validator without copying or monkeypatching its rules.

TDD evidence:

- Common strict-type tests: RED `13 failed, 1 passed`; GREEN `14 passed`.
- Real FRS validator: RED collection error (`frs_config` absent); GREEN `4 passed`.
- Combined strict/config/client checkpoint: `59 passed`.

## 2. Total cleanup guard and best-effort saving

- `RobotBridgeClient` now closes a connected WebSocket if greeting receive or protocol validation fails during construction.
- Both plain and FRS clients establish `bridge`/`saver` variables before a total `try/finally`; every operation after a successful bridge construction, including `send_config` and saver setup, is inside the guard.
- Shared cleanup order is fixed to:
  1. best-effort `STOP`
  2. best-effort `saver.close()`
  3. best-effort `bridge.close()`
- Each cleanup failure is logged independently and cannot block later cleanup or replace an active body exception.
- Saver construction (including output `mkdir`), `start`, `submit`, and `close` failures are best-effort and cannot stop control or socket cleanup.
- FRS `max_iterations` validation moved before policy/runtime/bridge construction; plain mode already rejected negative values before connecting.

TDD evidence:

- Bridge greeting cleanup: RED `2 failed`; GREEN `6 passed` for the bridge file.
- Plain cleanup fault injection: RED `4 failed, 2 passed`; GREEN `20 passed` for the plain client file.
- FRS lifecycle/cleanup suite: RED `7 failed, 2 passed`; GREEN included in the combined `35 passed` bridge/plain/FRS lifecycle run.

## 3. FRS CPU-only state-machine coverage

Added `tests/deploy_pi05_frs/test_remote_client.py`, which stubs only heavyweight JAX/model/runtime construction and calls the real `remote_client._run_frs` and `remote_client.run` functions.

Coverage includes:

- warmup observation and episode reset;
- both manual-confirmation and automatic START paths;
- `FRSChunkStart` -> chunk-ready -> `FRSSteerRequest` -> selected action -> matching `FRSSteerAck` -> `FRSChunkEnd`;
- mismatched steering ACK rejection;
- negative direct-Python iteration limit before bridge construction;
- saver constructor/start/close failures;
- `send_config`, STOP, saver close, and bridge close failures;
- preservation of the original control-loop exception while all cleanup operations fail.

## 4. Deployment documentation

Updated both `README.md` files to state explicitly:

- `bimanual_smolvla.sh --dry-run` supports only the plain pi0.5 legacy `obs`/`action`/`action_ack` flow;
- it cannot validate `frs_steering_v1`;
- FRS must use a real server flow already verified to support `frs_steering_v1`;
- no unverified server flag is proposed;
- every FRS/hardware run requires trained human supervision and an immediately available emergency stop.

## Verification

Passed before final independent review:

```text
PYTHONPATH=.:src:tests /home/typhon/FRS_Tact/.venv/bin/python -m pytest -q tests/deploy_pi05_frs
105 passed in 0.36s

bash -n deploy_pi05_frs/scripts/start_pi05.sh \
  deploy_pi05_frs/scripts/start_pi05_frs.sh \
  deploy_pi05_frs/scripts/start_remote_client.sh
exit 0

VB_ROBOT_TOKEN=redacted bash deploy_pi05_frs/scripts/start_pi05.sh --check
mode=pi05; shared config; deploy_pi05_frs.pi05_client

VB_ROBOT_TOKEN=redacted bash deploy_pi05_frs/scripts/start_pi05_frs.sh --check
mode=frs; shared config; deploy_pi05_frs.remote_client

PYTHONPATH=.:src:tests /home/typhon/FRS_Tact/.venv/bin/python -m compileall -q \
  deploy_pi05_frs tests/deploy_pi05_frs
exit 0

git diff --check
exit 0
```

## Limitations

- The repository `.venv` does not contain pytest, so `uv run --no-sync pytest` fails to spawn `pytest`. Tests used the existing `/home/typhon/FRS_Tact/.venv` without installing or changing dependencies.
- No GPU checkpoint load, robot, WebSocket server, or hardware validation was run.
- The available VB3 dry-run is legacy-only; there is no claimed hardware-free FRS end-to-end server test.
- No push was performed.

## Independent final review

An independent agent reviewed the complete working-tree diff against all four final-review findings and reported no Critical, Important, or Minor findings. Its independent deployment-suite run also reported `105 passed`.
