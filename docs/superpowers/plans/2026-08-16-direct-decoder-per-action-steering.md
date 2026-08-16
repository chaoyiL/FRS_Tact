# Direct Decoder Per-Action Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make direct-decoder deployment use the same chunk-indexed, fresh-observation, one-action-at-a-time execution lifecycle as FRS.

**Architecture:** Keep `DirectDecoderRuntime` as the low-level frozen encoder/decoder and add `DirectDecoderSteeringRuntime` as its chunk-state adapter. Extract the existing FRS client loop into a backend-neutral per-action runner with FRS and direct trace callbacks, then negotiate `frs_steering_v1` for either backend. SmolVLA runs once at direct chunk start; every steer request refines the fixed coarse chunk with current tactile input and sends only the requested row.

**Tech Stack:** Python 3.11, JAX, NumPy, PyTorch, pytest, msgpack/WebSocket robot bridge.

## Global Constraints

- Direct decoder uses visual-only `JaxSmolVLAPolicy`, the fixed `[1,20,32]` noise asset, and exactly four tactile keys in training order.
- Direct `begin_chunk` runs SmolVLA exactly once and stores an immutable normalized `[1,20,20]` coarse chunk.
- Direct `steer_action` does not rerun SmolVLA; it refines the fixed chunk from the request's current observation and selects `decoded[0, action_index]`.
- Selected normalized actions are unnormalized exactly once and sent as finite rank-one `[20]` arrays.
- `steps_per_inference == action_horizon == 20` for direct per-action protocol negotiation.
- Direct and FRS model backends remain mutually exclusive; RTC remains disabled.
- No FRS checkpoint, reverse integration, tactile history, temporal ensemble, robot-server change, or silent fallback.
- Preserve the user's existing `deploy_smolvla/configs/deploy_frs.yaml` modification and unrelated untracked files.

---

## File structure

- Modify `deploy_smolvla/direct_decoder.py`: low-level refine remains; add direct chunk/steer result dataclasses and stateful per-action adapter.
- Modify `deploy_smolvla/remote_client.py`: backend-neutral per-action runner, direct trace adapters, direct protocol routing, and server-config negotiation.
- Modify `deploy_smolvla/configs/deploy_direct_decoder.yaml`: negotiate the full 20-step chunk.
- Modify `tests/jax/test_direct_decoder_deployment.py`: direct lifecycle, config, trace, and routing tests.
- Modify `tests/jax/test_frs_remote_protocol.py`: prove the refactored common runner preserves FRS ordering and errors.
- Modify `tests/jax/test_frs_deployment.py` only if an existing direct trace assertion must follow the new per-action trace entry point; preserve ordinary trace coverage.

---

### Task 1: Direct decoder chunk lifecycle

**Files:**
- Modify: `deploy_smolvla/direct_decoder.py`
- Test: `tests/jax/test_direct_decoder_deployment.py`

**Interfaces:**
- Consumes: low-level `DirectDecoderRuntime.refine(coarse_normalized, observation) -> np.ndarray` and a visual policy exposing `predict_action_chunk`, `preprocessor.unnormalize_actions`, and `config.chunk_size/action_dim`.
- Produces: `DirectChunkReady`, `DirectSteerDiagnostics`, `DirectSteerResult`, and `DirectDecoderSteeringRuntime` with `reset()`, `begin_chunk()`, `steer_action()`, and `end_chunk()`.

- [ ] **Step 1: Write failing lifecycle tests**

Add policy and decoder spies that return distinct rows and record calls. Test:

```python
def test_direct_steering_runs_vla_once_and_refines_current_observation_per_action():
    steering = DirectDecoderSteeringRuntime(policy=policy, decoder=decoder)
    ready = steering.begin_chunk(
        3, observation(10), "pick", seed=7, jit=False, num_steps=4
    )
    first = steering.steer_action(3, 11, observation(20), 0)
    second = steering.steer_action(3, 12, observation(30), 1)

    assert len(policy.predict_calls) == 1
    assert [call.observation_value for call in decoder.refine_calls] == [20, 30]
    np.testing.assert_array_equal(first.selected_normalized, first.decoded_normalized[0, 0])
    np.testing.assert_array_equal(second.selected_normalized, second.decoded_normalized[0, 1])
    assert ready.chunk_id == first.chunk_id == second.chunk_id == 3
```

Also test:

- `begin_chunk` rejects an already active chunk;
- `steer_action` rejects no active chunk, wrong chunk ID, bool/non-integer/out-of-range index, and non-increasing unique indices;
- identical `request_id/chunk_id/action_index/tactile payload` returns the cached object without a second refine;
- a conflicting duplicate request raises;
- `end_chunk` clears state and permits a new chunk;
- mutation of spy inputs or returned arrays cannot change stored chunk/result snapshots;
- selected robot action is finite, rank one, and unnormalized once.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/jax/test_direct_decoder_deployment.py -k 'steering or duplicate or chunk_lifecycle'
```

Expected: collection/import failure because the new direct steering types do not exist.

- [ ] **Step 3: Add immutable direct result types and adapter state**

Add frozen dataclasses:

```python
@dataclass(frozen=True)
class DirectChunkReady:
    chunk_id: int
    action_vla_normalized: np.ndarray
    action_vla: np.ndarray
    prediction_started_at: float
    prediction_finished_at: float

@dataclass(frozen=True)
class DirectSteerDiagnostics:
    delta_rms: float
    max_normalized_action_abs: float

@dataclass(frozen=True)
class DirectSteerResult:
    chunk_id: int
    request_id: int
    action_index: int
    action_vla_normalized: np.ndarray
    decoded_normalized: np.ndarray
    selected_normalized: np.ndarray
    selected_action: np.ndarray
    diagnostics: DirectSteerDiagnostics
    decode_started_at: float
    decode_finished_at: float
```

Implement `DirectDecoderSteeringRuntime(policy: Any, decoder: DirectDecoderRuntime)` with:

- `tactile_keys = decoder.tactile_keys`;
- immutable byte-backed array copies;
- active chunk ID, fixed coarse chunk, last unique action index, and request cache;
- a SHA-256 payload hash over each tactile key's name, dtype, shape, and contiguous bytes;
- transactional state changes only after successful prediction/refinement/validation.

- [ ] **Step 4: Implement `begin_chunk`**

Use the fixed noise and no RTC inputs:

```python
coarse = self.policy.predict_action_chunk(
    initial_observation,
    task,
    seed=seed,
    noise=self.decoder.fixed_noise_jax,
    jit=jit,
    normalized=True,
    num_steps=num_steps,
    previous_chunk=None,
    inference_delay=None,
    execution_horizon=None,
)
jax.block_until_ready(coarse)
```

Require finite shape `(1, chunk_size, action_dim)`, unnormalize the complete coarse chunk for diagnostics, activate it, and return `DirectChunkReady`.

- [ ] **Step 5: Implement `steer_action` and `end_chunk`**

For a unique request:

```python
decoded = self.decoder.refine(self._action_vla_normalized, observation)
selected_normalized = decoded[0, action_index]
selected_action = self.policy.preprocessor.unnormalize_actions(selected_normalized)
```

Validate shapes/finiteness, calculate full-chunk delta RMS and maximum normalized magnitude, cache an immutable `DirectSteerResult`, and advance the last index only after success. `end_chunk(chunk_id)` validates the ID and clears all chunk-local state.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
pytest -q tests/jax/test_direct_decoder_deployment.py -k 'steering or duplicate or chunk_lifecycle'
```

Expected: all selected tests pass with no warnings introduced by these changes.

- [ ] **Step 7: Commit the lifecycle**

```bash
git add deploy_smolvla/direct_decoder.py tests/jax/test_direct_decoder_deployment.py
git commit -m "feat: add direct decoder steering lifecycle"
```

---

### Task 2: Backend-neutral per-action client runner

**Files:**
- Modify: `deploy_smolvla/remote_client.py`
- Test: `tests/jax/test_frs_remote_protocol.py`
- Test: `tests/jax/test_direct_decoder_deployment.py`

**Interfaces:**
- Consumes: both FRS and direct runtimes exposing `tactile_keys`, `policy.config`, `begin_chunk`, `steer_action`, and `end_chunk`; trace builders passed as callbacks.
- Produces: `_run_per_action_protocol(...)`, compatibility wrapper `_run_frs_protocol(...)`, and `_run_direct_decoder_protocol(...)`.

- [ ] **Step 1: Write failing direct protocol ordering test**

Create a scripted bridge using the existing `FRSChunkStart`, `FRSSteerRequest`, `FRSSteerAck`, and `FRSChunkEnd` message types. Assert:

```python
assert [event[0] for event in events] == [
    "receive", "begin", "ready",
    "receive", "steer", "action", "receive",
    "receive", "steer", "action", "receive",
    "receive", "end",
]
```

Verify the two sent action vectors are selected rows 0 and 1, every request observation reaches the runtime, ACK identities are checked, and the direct wrapper rejects mismatched results before sending.

- [ ] **Step 2: Run protocol tests and verify RED**

Run:

```bash
pytest -q tests/jax/test_direct_decoder_deployment.py -k protocol
```

Expected: failure because `_run_direct_decoder_protocol` is absent.

- [ ] **Step 3: Extract the common runner**

Move the existing strict loop body into:

```python
def _run_per_action_protocol(
    bridge: RobotBridgeClient,
    steering_policy: Any,
    *,
    backend_label: str,
    build_chunk_trace: Any,
    build_steer_trace: Any,
    task: str,
    state_dim: int,
    image_keys: Sequence[str],
    empty_cameras: int,
    observation_timeout_s: float,
    action_ack_timeout_s: float,
    seed: int,
    jit: bool,
    num_steps: int | None,
    max_chunks: int,
    observation_saver: ObservationSaver,
) -> None:
```

Preserve the current receive → prepare → begin → ready → request → steer → rank-one send → ACK → end ordering and all identity/shape checks. Use `backend_label` only in messages; do not weaken FRS validation.

- [ ] **Step 4: Add FRS and direct wrappers**

`_run_frs_protocol` calls the common runner with existing FRS trace builders. `_run_direct_decoder_protocol` calls it with direct trace builders. Direct builders emit the existing server-compatible version-2 steering wire trace shape while sourcing values only from direct result fields:

- chunk `action_vla_normalized/action_vla` are the fixed coarse chunk;
- compatibility `x_base` is an immutable copy of the fixed coarse normalized chunk;
- steer `decoded_normalized`, selected arrays, IDs, timestamps, delta RMS, and max magnitude come from the direct result;
- compatibility `tactile_sequence_length` is `1` and `tactile_change` is `0.0` because direct decoding uses only the current frame.

Trace serialization remains non-fatal via `_build_trace_or_none`.

- [ ] **Step 5: Verify direct and FRS protocol GREEN**

Run:

```bash
pytest -q tests/jax/test_direct_decoder_deployment.py -k protocol
pytest -q tests/jax/test_frs_remote_protocol.py
```

Expected: direct protocol tests pass; all existing FRS ordering, timeout, rejection, and trace-failure tests remain green.

- [ ] **Step 6: Commit the common protocol runner**

```bash
git add deploy_smolvla/remote_client.py tests/jax/test_direct_decoder_deployment.py tests/jax/test_frs_remote_protocol.py
git commit -m "refactor: share per-action steering protocol"
```

---

### Task 3: Route direct deployment through per-action steering

**Files:**
- Modify: `deploy_smolvla/remote_client.py`
- Modify: `deploy_smolvla/configs/deploy_direct_decoder.yaml`
- Test: `tests/jax/test_direct_decoder_deployment.py`
- Test: `tests/jax/test_tactile_integration.py`

**Interfaces:**
- Consumes: `DirectDecoderSteeringRuntime` and `_run_direct_decoder_protocol`.
- Produces: direct config negotiation and live-session routing; ordinary visual sessions remain on `send_action/action_ack`.

- [ ] **Step 1: Write failing config and run-routing tests**

Update the direct YAML assertion to:

```python
assert config["control"]["steps_per_inference"] == 20
assert config["control"]["steps_per_inference"] == config["control"]["action_horizon"]
```

Add rejection coverage for direct `steps_per_inference != action_horizon`. In a run fixture, assert the server config includes:

```python
{
    "execution_protocol": "frs_steering_v1",
    "steering_protection_interval_s": None,
    "frs_tactile_keys": list(DIRECT_TACTILE_KEYS),
}
```

Assert live direct execution calls `_run_direct_decoder_protocol`, never enters the ordinary `send_action` loop, and interprets `runtime.max_iterations` as maximum chunks.

- [ ] **Step 2: Run routing tests and verify RED**

Run:

```bash
pytest -q tests/jax/test_direct_decoder_deployment.py -k 'config or route or server_config'
```

Expected: failure because direct currently negotiates legacy chunks and the YAML requests 10 steps.

- [ ] **Step 3: Enforce full-horizon direct configuration**

In `load_config`, after validating direct horizon 20, require:

```python
if int(control["steps_per_inference"]) != int(control["action_horizon"]):
    raise ValueError(
        "direct_tactile_decoder per-action execution requires "
        "steps_per_inference to equal action_horizon"
    )
```

Change `deploy_direct_decoder.yaml` to `steps_per_inference: 20` and explain that the value negotiates the complete indexed chunk; each request still executes one action.

- [ ] **Step 4: Negotiate steering protocol for direct**

Generalize `_build_server_config` to accept an optional per-action runtime or explicit steering fields. When either FRS or direct is active, add `execution_protocol="frs_steering_v1"`; use FRS's configured protection interval for FRS and `None` for direct; advertise the active runtime's tactile keys.

- [ ] **Step 5: Instantiate and route the direct adapter**

After loading `DirectDecoderRuntime.from_bundle`, construct:

```python
direct_steering = DirectDecoderSteeringRuntime(
    policy=policy,
    decoder=direct_decoder,
)
```

Keep existing direct warmup through `_predict_chunk`, then reset the steering adapter before START. Route FRS first, direct second, and ordinary sessions last:

```python
if frs_runtime is not None:
    _run_frs_protocol(...)
    return
if direct_steering is not None:
    _run_direct_decoder_protocol(...)
    return
```

- [ ] **Step 6: Run routing and ordinary regression tests**

Run:

```bash
pytest -q tests/jax/test_direct_decoder_deployment.py
pytest -q tests/jax/test_tactile_integration.py -k 'server_config or protocol or run'
```

Expected: all direct tests pass; ordinary visual/VT configurations still omit `execution_protocol` and use full-chunk `send_action/action_ack`.

- [ ] **Step 7: Commit deployment routing**

```bash
git add deploy_smolvla/remote_client.py deploy_smolvla/configs/deploy_direct_decoder.yaml tests/jax/test_direct_decoder_deployment.py tests/jax/test_tactile_integration.py
git commit -m "feat: steer direct decoder per action"
```

---

### Task 4: Full verification and focused review

**Files:**
- Verify: `deploy_smolvla/direct_decoder.py`
- Verify: `deploy_smolvla/remote_client.py`
- Verify: `deploy_smolvla/configs/deploy_direct_decoder.yaml`
- Verify: `tests/jax/test_direct_decoder_deployment.py`
- Verify: `tests/jax/test_frs_remote_protocol.py`
- Verify: `tests/jax/test_frs_deployment.py`

**Interfaces:**
- Consumes: completed direct per-action deployment.
- Produces: fresh evidence that behavior, configuration, compilation, and regressions match the spec.

- [ ] **Step 1: Run focused direct and protocol suites**

```bash
pytest -q +  tests/jax/test_direct_decoder_deployment.py +  tests/jax/test_frs_remote_protocol.py +  tests/jax/test_frs_protocol.py
```

Expected: zero failures.

- [ ] **Step 2: Run broader deployment regressions**

```bash
pytest -q tests/jax/test_frs_deployment.py tests/jax/test_tactile_integration.py
```

Expected: zero failures; CUDA-only released-assets test may be skipped when CUDA is unavailable.

- [ ] **Step 3: Compile modified Python modules**

```bash
python -m py_compile deploy_smolvla/direct_decoder.py deploy_smolvla/remote_client.py
```

Expected: exit status 0 and no output.

- [ ] **Step 4: Check config and diff hygiene**

```bash
bash deploy_smolvla/scripts/start_direct_decoder.sh --check
git diff --check
git status --short
```

Expected: config validation succeeds when required authentication environment is present; diff check emits no errors; status preserves the pre-existing `deploy_frs.yaml` and untracked guide without staging them.

- [ ] **Step 5: Review requirements against the diff**

Confirm from the final diff:

- one SmolVLA call per direct chunk;
- one current-observation direct refine per steer request;
- selected row equals request `action_index`;
- one rank-one send and matching ACK per request;
- chunk end clears direct state;
- FRS and ordinary behavior are unchanged;
- no robot-server files changed.

- [ ] **Step 6: Commit any test-only corrections**

If verification required scoped test corrections, commit only those files:

```bash
git add tests/jax/test_direct_decoder_deployment.py tests/jax/test_frs_remote_protocol.py tests/jax/test_frs_deployment.py tests/jax/test_tactile_integration.py
git commit -m "test: verify direct per-action steering"
```
