# Per-Action FRS Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in `frs_steering_v1` duplex protocol in which one visual VLA chunk is reverse-integrated once, fresh chunk-local tactile history re-decodes it per requested index, and the robot server schedules only the selected action while preserving every legacy client path.

**Architecture:** `JaxSmolVLAPolicy` owns source-flow reversal, while `FRSSteeringPolicy` owns episode/chunk/tactile/decode state. The remote client is a typed protocol orchestrator; the robot server negotiates the explicit protocol and runs a separate, generation-bound RTC/block state machine with server-authoritative clocks and single-action safety. Legacy full-chunk transport and execution remain unchanged when `execution_protocol` is absent.

**Tech Stack:** Python 3, JAX/Flax NNX, NumPy, msgpack over synchronous WebSocket, pytest, Ruff, Hydra/YAML.

## Global Constraints

- Implement the approved design in `docs/superpowers/specs/2026-08-12-frs-per-action-steering-design.md` across `/home/typhon/FRS_Tact` and `/home/typhon/vb3_robot_server`.
- `execution_protocol="frs_steering_v1"` is the only FRS opt-in; missing protocol keeps legacy payloads, action ACKs, RTC, and block behavior byte-for-byte compatible.
- Server `exec_mode` defines RTC; source-policy `rtc_config` and `previous_chunk` remain disabled in FRS mode.
- `action_vla` and normalized `x_base` are created exactly once per chunk; every unique steer appends one unpadded tactile token, decodes the full fixed base chunk, then selects one action row.
- Variable tactile length is `1 <= K <= H`; checkpoint `tactile_window` remains training metadata and lengths above it are logged as out-of-distribution.
- The server owns timestamps, target selection, stale decisions, controller conversion, and submission; client/server wall-clock synchronization is not required.
- RTC uses `t_i = observation_timestamp + i * control_dt`; null protection resolves to one control period and protection applies only to the first request.
- Block mode requests every index `0..H-1`; after scheduling index `i`, it waits until that target before capturing tactile input for `i+1`.
- FRS failures never fall back to an unsteered VLA action.
- Preserve all pre-existing dirty worktree changes. In FRS_Tact, never overwrite or accidentally stage `deploy_smolvla/configs/deploy_smolvla_jax.yaml`, `.hf/`, or `modalities_eval/frs/`. In vb3_robot_server, use patch-level staging for every overlapping modified file.
- Use `apply_patch` for source edits, `git add -p` for overlapping files, and run `git diff --check` before each commit.
- FRS_Tact test prefix: `PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync`.
- Robot-server tests must not initialize cameras, controllers, or hardware.

---

## File Structure

### `/home/typhon/FRS_Tact`

- `train_frs/utils/model.py`: variable-length tactile contract and one-time conditioning reuse during decode.
- `utils/source_model.py`: prepared-observation reverse-flow core shared by evaluation and policy deployment.
- `train_smolvla/policy.py`: public normalized `reverse_action_chunk` source-policy capability.
- `deploy_smolvla/frs_runtime.py`: `FRSSteeringPolicy` episode/chunk lifecycle, idempotent per-action steering, diagnostics, safety, and warmup.
- `deploy_smolvla/frs_protocol.py`: typed validation for server-to-client FRS messages.
- `deploy_smolvla/bridge_client.py`: FRS send/receive methods alongside untouched legacy methods.
- `deploy_smolvla/remote_client.py`: config negotiation, trace builders, warmup ordering, FRS orchestration, and legacy routing.
- `deploy_smolvla/configs/deploy_frs.yaml`: FRS protection configuration and horizon-sized execution contract.
- `tests/train_frs/test_model.py`: variable sequence and cached-conditioning model tests.
- `tests/jax/test_policy.py`: public reverse API tests.
- `tests/jax/test_frs_deployment.py`: FRS policy lifecycle, safety, config, trace, and compatibility tests.
- `tests/jax/test_frs_protocol.py`: client wire-schema and bridge tests.
- `tests/jax/test_frs_remote_protocol.py`: remote orchestration event-order tests.

### `/home/typhon/vb3_robot_server`

- `client/robot_client.py`: generation-bound FIFO transport for FRS messages while retaining legacy slots.
- `deploy_scripts/frs_protocol.py`: negotiation and exact wire message validation/builders.
- `deploy_scripts/frs_execution.py`: fake-clock-friendly RTC/block chunk state machine and selected-action safety.
- `deploy_scripts/bimanual_smolvla_online.py`: startup negotiation and explicit legacy/FRS session routing.
- `deploy_scripts/action_trace.py`: isolated trace-v2 validation and persistence.
- `tests/test_robot_client_frs_transport.py`: FIFO, generation, disconnect, STOP, and duplicate transport tests.
- `tests/test_frs_protocol.py`: negotiation and server wire validation tests.
- `tests/test_frs_execution.py`: RTC/block timing, safety, idempotence, and failure tests.
- `deploy_scripts/bimanual_smolvla_online_test.py`: startup/main routing and cleanup regression tests.
- `tests/test_action_trace.py`: trace-v2 validation/isolation tests.

---

### Task 1: Accept True Variable-Length Tactile Sequences

**Files:**
- Modify: `train_frs/utils/model.py:98-120,187-206`
- Modify: `tests/train_frs/test_model.py`

**Interfaces:**
- Consumes: tactile arrays shaped `[B, K, N, D]`.
- Produces: `TactileConditionedFlowDecoder.encode_tactile_tokens(tactile_seq) -> [B, N, gru_hidden_dim]` for every `K >= 1`.

- [ ] **Step 1: Add failing variable-length contract tests**

Add a parametrized test that constructs the existing small decoder fixture for `K in (1, config.tactile_window, config.action_horizon)`, calls `encode_tactile_tokens`, and asserts exact shape and finiteness. Add an empty-sequence test and retain rank/stream/embedding mismatch checks:

```python
@pytest.mark.parametrize("sequence_length", [1, 3, 6])
def test_tactile_encoder_accepts_sequence_lengths_one_training_window_and_horizon(
    decoder,
    sequence_length,
):
    tactile = jnp.ones((2, sequence_length, 2, 8), dtype=jnp.float32)
    encoded = decoder.encode_tactile_tokens(tactile)
    assert encoded.shape == (2, 2, decoder.config.gru_hidden_dim)
    assert bool(jnp.isfinite(encoded).all())


def test_tactile_encoder_rejects_empty_sequence(decoder):
    tactile = jnp.ones((2, 0, 2, 8), dtype=jnp.float32)
    with pytest.raises(ValueError, match="at least one time step"):
        decoder.encode_tactile_tokens(tactile)
```

- [ ] **Step 2: Run the focused tests and confirm the fixed-window failure**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/train_frs/test_model.py -k 'tactile_encoder_accepts_sequence_lengths or tactile_encoder_rejects_empty_sequence'
```

Expected: `K=1` and `K=6` fail because `encode_tactile_tokens` still requires `config.tactile_window`; the empty case does not yet raise the intended error.

- [ ] **Step 3: Replace only the time-window equality check**

Keep rank, stream-count, and embedding-dimension validation, but replace the training-window constraint with:

```python
batch_size, time_steps, num_streams, embedding_dim = tactile_seq.shape
if time_steps < 1:
    raise ValueError("tactile_seq must contain at least one time step")
if num_streams != self.config.num_tactile_streams:
    raise ValueError(
        f"Expected {self.config.num_tactile_streams} tactile streams, got {num_streams}"
    )
if embedding_dim != self.config.tactile_embedding_dim:
    raise ValueError(
        f"Expected tactile embedding dim {self.config.tactile_embedding_dim}, got {embedding_dim}"
    )
```

Do not change GRU zero-carry behavior or checkpoint fields.

- [ ] **Step 4: Run model regression tests**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/train_frs/test_model.py
```

Expected: all tests pass, including `K=1`, training-window `K`, horizon `K`, and invalid-shape tests.

- [ ] **Step 5: Commit the isolated model contract**

```bash
git add train_frs/utils/model.py tests/train_frs/test_model.py
git diff --cached --check
git commit -m "feat: support variable tactile sequence lengths"
```

### Task 2: Encode Tactile Conditioning Once Per Decode

**Files:**
- Modify: `train_frs/utils/model.py:163-231,637-687`
- Modify: `tests/train_frs/test_model.py`

**Interfaces:**
- Consumes: Task 1 variable `[B, K, N, D]` input.
- Produces:
  - `encode_tactile_condition(tactile_seq) -> [B, N, model_dim]`
  - `velocity_from_condition(x_t, t, tactile_condition, gate_weights=None) -> [B, H, A]`
  - unchanged public `decode_actions(model, x_base, tactile_seq, gate_weights=None, *, num_steps, solver)`.

- [ ] **Step 1: Add numerical-equivalence and call-count tests**

For Euler and FireFlow, compare `decode_actions` against explicit integration of `lambda x, t: model(x, t, tactile, gate)`. Wrap `encode_tactile_condition` with a counting callable before invoking multi-step decode:

```python
@pytest.mark.parametrize("solver", ["euler", "fireflow"])
def test_cached_condition_decode_matches_recomputed_condition(decoder, inputs, solver):
    x_base, tactile, gate = inputs
    expected = integrate_reference(
        lambda x_t, t: decoder(x_t, t, tactile, gate),
        x_base,
        num_steps=4,
        solver=solver,
    )
    actual = decode_actions(
        decoder,
        x_base,
        tactile,
        gate,
        num_steps=4,
        solver=solver,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
```

The counting test must assert exactly one call for each solver.

- [ ] **Step 2: Run the new tests and confirm repeated encoding**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/train_frs/test_model.py -k 'cached_condition or encodes_tactile_condition_once'
```

Expected: failure because the split APIs do not exist and the current integrators call `model.__call__` at every velocity evaluation.

- [ ] **Step 3: Split conditioning from velocity and cache it outside integration**

Implement the training-compatible delegation:

```python
def encode_tactile_condition(self, tactile_seq: jax.Array) -> jax.Array:
    tactile_tokens = self.encode_tactile_tokens(tactile_seq)
    return self.tactile_projection(tactile_tokens)

def velocity_from_condition(
    self,
    x_t: jax.Array,
    t: jax.Array,
    tactile_condition: jax.Array,
    gate_weights: jax.Array | None = None,
) -> jax.Array:
    gated_condition = self._apply_gate(tactile_condition, gate_weights)
    return self._decode_velocity(x_t, t, gated_condition)

def __call__(self, x_t, t, tactile_seq, gate_weights=None):
    condition = self.encode_tactile_condition(tactile_seq)
    return self.velocity_from_condition(x_t, t, condition, gate_weights)
```

Move `condition = model.encode_tactile_condition(tactile_seq)` before solver dispatch in `decode_actions`; jitted Euler/FireFlow helpers accept condition and call only `velocity_from_condition` inside their loops.

- [ ] **Step 4: Run model and integration regressions**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/train_frs/test_model.py tests/flow_decoder/test_integration.py
```

Expected: all tests pass and call counts equal one.

- [ ] **Step 5: Commit conditioning reuse**

```bash
git add train_frs/utils/model.py tests/train_frs/test_model.py
git diff --cached --check
git commit -m "perf: reuse tactile conditioning during FRS decode"
```

### Task 3: Give the Source Policy a Public Reverse-Flow Capability

**Files:**
- Modify: `utils/source_model.py:73-82,117-207`
- Modify: `train_smolvla/policy.py:53-171`
- Modify: `tests/jax/test_policy.py`

**Interfaces:**
- Produces `reverse_integrate_prepared_actions(model, batch, normalized_actions, *, num_steps, solver="slerpflow") -> jax.Array`.
- Produces `JaxSmolVLAPolicy.reverse_action_chunk(observation, task, normalized_actions, *, num_steps, solver) -> jax.Array`.
- Preserves the full `reverse_integrate_actions(model, observation, actions, *, num_steps, solver)` signature for training-cache/evaluation callers.

- [ ] **Step 1: Add public-policy reverse tests**

Use a small mocked policy/model to assert preprocessing once, exact normalized shape, all three solvers, and rejection of shape/nonfinite inputs:

```python
@pytest.mark.parametrize("solver", ["euler", "fireflow", "slerpflow"])
def test_reverse_action_chunk_supports_all_solvers(policy, observation, solver):
    actions = jnp.zeros((1, policy.config.chunk_size, policy.config.action_dim))
    result = policy.reverse_action_chunk(
        observation,
        "pick the tube",
        actions,
        num_steps=4,
        solver=solver,
    )
    assert result.shape == actions.shape
    assert bool(jnp.isfinite(result).all())
```

Add an equivalence test showing the `EvalObservation` wrapper and prepared core return matching arrays.

- [ ] **Step 2: Run the focused policy tests**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_policy.py -k 'reverse_action_chunk or reverse_integrate_actions_eval_wrapper'
```

Expected: failure because the policy method and prepared core do not exist.

- [ ] **Step 3: Extract the prepared core and implement the policy method**

The policy method must preprocess once, validate exact shape before reverse integration, keep float32 normalized values, and reject nonfinite output:

```python
def reverse_action_chunk(
    self,
    observation: Mapping[str, Any],
    task: str,
    normalized_actions: jax.Array,
    *,
    num_steps: int,
    solver: ReverseSolver,
) -> jax.Array:
    actions = jnp.asarray(normalized_actions, dtype=jnp.float32)
    expected = (1, self.config.chunk_size, self.config.action_dim)
    if actions.shape != expected:
        raise ValueError(f"normalized_actions must have shape {expected}, got {actions.shape}")
    if not bool(jnp.isfinite(actions).all()):
        raise ValueError("normalized_actions must be finite")
    batch = self.preprocessor.prepare(observation, task)
    result = reverse_integrate_prepared_actions(
        self,
        batch,
        actions,
        num_steps=num_steps,
        solver=solver,
    )
    if result.shape != expected or not bool(jnp.isfinite(result).all()):
        raise RuntimeError("reverse integration returned an invalid normalized chunk")
    return result
```

Make the existing evaluation function a thin adapter that prepares `EvalObservation` then calls the new core.

- [ ] **Step 4: Run policy/source safety regressions**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_policy.py tests/flow_decoder/test_integration.py tests/flow_decoder/test_frs_safety.py
```

Expected: all tests pass; training cache callers continue importing `reverse_integrate_actions`.

- [ ] **Step 5: Commit source-policy reversal**

```bash
git add utils/source_model.py train_smolvla/policy.py tests/jax/test_policy.py
git diff --cached --check
git commit -m "feat: expose normalized source action reversal"
```

### Task 4: Introduce the FRS Episode and Chunk Lifecycle

**Files:**
- Modify: `deploy_smolvla/frs_runtime.py`
- Modify: `tests/jax/test_frs_deployment.py`

**Interfaces:**
- Consumes Task 3 `predict_action_chunk` with `normalized=True` and `reverse_action_chunk`.
- Produces immutable `FRSChunkReady` and `FRSSteeringPolicy.reset_episode`, `begin_chunk`, `end_chunk`.
- Produces compatibility alias `FRSRuntime = FRSSteeringPolicy`.

- [ ] **Step 1: Add lifecycle tests before refactoring**

Add exact-once spies and state assertions:

```python
def test_begin_chunk_predicts_and_reverses_exactly_once_without_decoding(runtime, source):
    runtime.reset_episode(initial_observation())
    ready = runtime.begin_chunk(
        7,
        initial_observation(),
        "pick the tube",
        seed=3,
        jit=True,
        num_steps=None,
    )
    assert source.predict_calls == 1
    assert source.reverse_calls == 1
    assert runtime.decode_calls == 0
    assert ready.chunk_id == 7
    assert ready.action_vla_normalized.shape == ready.x_base.shape
```

Cover missing baseline, nested chunks, wrong end ID, chunk-local clear, and episode-baseline preservation.

- [ ] **Step 2: Run lifecycle tests and confirm missing APIs**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_frs_deployment.py -k 'alias or reset_episode or begin_chunk or end_chunk or chunk_lifecycle'
```

Expected: failure because the old one-shot `steer` runtime has no chunk lifecycle.

- [ ] **Step 3: Refactor to `FRSSteeringPolicy` without duplicating implementation**

Define:

```python
@dataclass(frozen=True)
class FRSChunkReady:
    chunk_id: int
    action_vla_normalized: np.ndarray
    action_vla: np.ndarray
    x_base: np.ndarray
    prediction_started_at: float
    prediction_finished_at: float


class FRSSteeringPolicy:
    def reset_episode(self, initial_observation: Mapping[str, Any]) -> None:
        baseline = self._encode_tactile(initial_observation)
        self._episode_baseline = np.array(baseline, copy=True)
        self._clear_chunk_state()

    def begin_chunk(self, chunk_id, initial_observation, task, *, seed, jit, num_steps):
        self._require_episode_and_no_active_chunk()
        started = time.time()
        normalized = self.policy.predict_action_chunk(
            initial_observation, task, seed=seed, jit=jit, num_steps=num_steps, normalized=True
        )
        x_base = self.policy.reverse_action_chunk(
            initial_observation,
            task,
            normalized,
            num_steps=self.config.reverse_num_steps,
            solver=self.config.reverse_solver,
        )
        self._activate_chunk(chunk_id, normalized, x_base)
        return self._make_chunk_ready(started, time.time())

    def end_chunk(self, chunk_id: int) -> None:
        self._require_active_chunk(chunk_id)
        self._clear_chunk_state()


FRSRuntime = FRSSteeringPolicy
```

Use read-only NumPy copies for stored `action_vla` and `x_base`; `end_chunk` must not clear `_episode_baseline`.

- [ ] **Step 4: Run FRS deployment lifecycle and contract tests**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_frs_deployment.py -k 'alias or reset_episode or begin_chunk or end_chunk or chunk_lifecycle or contract'
```

Expected: all selected tests pass.

- [ ] **Step 5: Stage only task hunks and commit**

Because `frs_runtime.py` already contains user edits, use:

```bash
git add -p deploy_smolvla/frs_runtime.py tests/jax/test_frs_deployment.py
git diff --cached --check
git commit -m "refactor: model FRS episode and chunk lifecycle"
```

### Task 5: Add Idempotent Per-Action Steering, Safety, and Warmup

**Files:**
- Modify: `deploy_smolvla/frs_runtime.py`
- Modify: `tests/jax/test_frs_deployment.py`

**Interfaces:**
- Consumes Task 4 active chunk and Task 2 `decode_actions`.
- Produces `FRSSteerResult`, `steer_action(chunk_id, request_id, observation, action_index)`, and `warmup_all_tactile_lengths`.

- [ ] **Step 1: Add sequence, idempotency, selection, and warmup tests**

Define a test result contract and assert true lengths grow from `1` through `H` without padding:

```python
def test_unique_steer_requests_grow_true_unpadded_sequence_one_at_a_time(runtime):
    start_test_chunk(runtime, chunk_id=4)
    first = runtime.steer_action(4, 10, tactile_observation(1.0), 1)
    second = runtime.steer_action(4, 11, tactile_observation(2.0), 3)
    assert first.tactile_sequence_length == 1
    assert second.tactile_sequence_length == 2
    assert runtime.decode_input_shapes == [(1, 1, 4, 512), (1, 2, 4, 512)]


def test_duplicate_identical_request_is_cached_without_side_effects(runtime):
    start_test_chunk(runtime, chunk_id=4)
    observation = tactile_observation(1.0)
    first = runtime.steer_action(4, 10, observation, 1)
    second = runtime.steer_action(4, 10, observation, 1)
    assert second is first
    assert runtime.encode_calls == 1
    assert runtime.decode_calls == 1
```

Also cover conflicting duplicates, strictly increasing indices, full-chunk safety before selection, fixed `x_base`, chunk reset, warmup K=1..H, and state snapshot equality.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_frs_deployment.py -k 'steer_action or steer_requests or duplicate or chunk_boundary or warmup'
```

Expected: failure because steering is still one-shot and history is episode-window based.

- [ ] **Step 3: Implement chunk-local steer results and request cache**

Define exact result fields:

```python
@dataclass(frozen=True)
class FRSSteerResult:
    chunk_id: int
    request_id: int
    action_index: int
    action_vla_normalized: np.ndarray
    x_base: np.ndarray
    decoded_normalized: np.ndarray
    selected_normalized: np.ndarray
    selected_action: np.ndarray
    tactile_sequence_length: int
    diagnostics: FRSDiagnostics
    encode_started_at: float
    encode_finished_at: float
    decode_started_at: float
    decode_finished_at: float
```

For every new request, hash the ordered tactile payload keys/dtypes/shapes/bytes, append one `[N,D]`, reject `K>H`, decode full `[1,H,A]`, run full-chunk shape/finite/max/RMS checks, select `[A]`, then unnormalize only that row. Cache `(chunk_id, action_index, tactile_hash, result)` under `request_id`; identical replay returns the exact result object.

- [ ] **Step 4: Implement non-mutating warmup and run the full deployment test file**

Warmup repeats the encoded baseline token to each true shape and blocks on all results:

```python
snapshot = self._snapshot_live_state()
try:
    baseline = np.asarray(self._episode_baseline, dtype=np.float32)
    for length in range(1, self.policy.config.chunk_size + 1):
        tactile = jnp.expand_dims(jnp.asarray(np.stack([baseline] * length)), axis=0)
        warmed = decode_actions(
            self.model,
            synthetic_x_base,
            tactile,
            synthetic_gate,
            num_steps=self.config.decode_num_steps,
            solver=self.config.decode_solver,
        )
        jax.block_until_ready(warmed)
finally:
    self._restore_live_state(snapshot)
```

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_frs_deployment.py
```

Expected: all tests pass; a warning is logged when `H > checkpoint tactile_window`.

- [ ] **Step 5: Commit per-action steering**

```bash
git add -p deploy_smolvla/frs_runtime.py tests/jax/test_frs_deployment.py
git diff --cached --check
git commit -m "feat: steer individual actions from chunk-local tactile history"
```

### Task 6: Add Typed Client-Side FRS Wire Messages

**Files:**
- Create: `deploy_smolvla/frs_protocol.py`
- Modify: `deploy_smolvla/bridge_client.py:171-226`
- Create: `tests/jax/test_frs_protocol.py`
- Modify: `tests/jax/test_frs_deployment.py`

**Interfaces:**
- Produces dataclasses `FRSChunkStart`, `FRSSteerRequest`, `FRSSteerAck`, `FRSChunkEnd` and union `FRSServerMessage`.
- Produces `parse_frs_server_message(message) -> FRSServerMessage`.
- Produces bridge methods `receive_frs_message`, `send_frs_chunk_ready`, and `send_frs_steer_action`.

- [ ] **Step 1: Write exact schema and legacy-payload tests**

Test valid RTC/block chunk starts, request observation preservation, ACK/end enums, malformed bool IDs, nonfinite timestamps, wrong timestamp shapes, and exact selected-action payload:

```python
def test_bridge_sends_only_rank_one_selected_action(bridge, socket):
    bridge.send_frs_steer_action(3, 8, 4, np.zeros((14,), dtype=np.float32))
    assert unpack(socket.sent[-1]) == {
        "type": "frs_steer_action",
        "chunk_id": 3,
        "request_id": 8,
        "action_index": 4,
        "action": np.zeros((14,), dtype=np.float32),
        "trace": None,
    }
```

Retain the existing assertion that legacy `send_action` has no FRS fields.

- [ ] **Step 2: Run the client protocol tests**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_frs_protocol.py tests/jax/test_frs_deployment.py -k 'protocol or bridge_send_action'
```

Expected: failure because typed parser and methods do not exist.

- [ ] **Step 3: Implement strict dataclasses/parser and bridge methods**

Parser dispatch must be closed over the four server message types:

```python
def parse_frs_server_message(message: Mapping[str, Any]) -> FRSServerMessage:
    message_type = message.get("type")
    if message_type == "frs_chunk_start":
        return _parse_chunk_start(message)
    if message_type == "frs_steer_request":
        return _parse_steer_request(message)
    if message_type == "frs_steer_ack":
        return _parse_steer_ack(message)
    if message_type == "frs_chunk_end":
        return _parse_chunk_end(message)
    raise FRSProtocolError(f"unsupported FRS server message type: {message_type!r}")
```

Validate identifiers as nonnegative integers excluding bool, action timestamps as finite exact `[H]` only in RTC, block null invariants, and selected action as rank-one floating finite float32-representable.

- [ ] **Step 4: Run protocol and legacy bridge regressions**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_frs_protocol.py tests/jax/test_frs_deployment.py
```

Expected: all tests pass and legacy payload assertions remain exact.

- [ ] **Step 5: Commit typed client protocol**

```bash
git add deploy_smolvla/frs_protocol.py deploy_smolvla/bridge_client.py tests/jax/test_frs_protocol.py
git add -p tests/jax/test_frs_deployment.py
git diff --cached --check
git commit -m "feat: add typed FRS steering bridge protocol"
```

### Task 7: Negotiate FRS Explicitly on the Remote Client

**Files:**
- Modify: `deploy_smolvla/frs_runtime.py:37-129,132-177`
- Modify: `deploy_smolvla/remote_client.py:149-208,575-685`
- Modify: `deploy_smolvla/configs/deploy_frs.yaml`
- Modify: `tests/jax/test_frs_deployment.py`
- Modify: `tests/jax/test_tactile_integration.py`

**Interfaces:**
- Adds `FRSConfig.steering_protection_interval_s: float | None`.
- Produces `_build_server_config(observation_config, control, *, frs_policy) -> dict[str, Any]`.
- Produces trace helpers `_build_frs_chunk_trace`, `_build_frs_steer_trace`, and fail-open diagnostics wrapper `_build_trace_or_none`.

- [ ] **Step 1: Add config-negotiation and trace-v2 tests**

Assert FRS adds exactly the versioned fields and legacy omits all of them:

```python
def test_frs_server_config_advertises_explicit_v1_fields(frs_policy):
    config = _build_server_config(OBS_CONFIG, CONTROL, frs_policy=frs_policy)
    assert config["execution_protocol"] == "frs_steering_v1"
    assert config["steering_protection_interval_s"] is None
    assert config["frs_tactile_keys"] == list(frs_policy.tactile_keys)


def test_legacy_server_config_omits_all_frs_protocol_fields():
    config = _build_server_config(OBS_CONFIG, CONTROL, frs_policy=None)
    assert "execution_protocol" not in config
    assert "steering_protection_interval_s" not in config
    assert "frs_tactile_keys" not in config
```

Test null/nonnegative finite protection, reject negative/NaN/infinity/bool, require FRS `steps_per_inference == action_horizon`, and make trace-builder exceptions return `None`.

- [ ] **Step 2: Run focused config tests**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_frs_deployment.py -k 'config or server_config or trace_v2' tests/jax/test_tactile_integration.py -k 'load_config or default_deployment_config'
```

Expected: failure because protection and explicit protocol fields are absent.

- [ ] **Step 3: Add validated config and trace builders**

In FRS config parsing, reject bool and nonfinite values before storing seconds. In `_build_server_config`, start from the legacy dictionary and update only when `frs_policy is not None`:

```python
if frs_policy is not None:
    server_config.update(
        execution_protocol="frs_steering_v1",
        steering_protection_interval_s=frs_policy.config.steering_protection_interval_s,
        frs_tactile_keys=list(frs_policy.tactile_keys),
    )
return server_config
```

Trace v2 includes immutable source/base arrays at chunk start and full decoded chunk, selected rows, K, target metadata, timing, diagnostics per steer. Wrap all trace building with:

```python
def _build_trace_or_none(builder, *args):
    try:
        return builder(*args)
    except Exception as exc:
        LOGGER.warning("Omitting FRS trace after serialization failure: %s", exc)
        return None
```

- [ ] **Step 4: Update only `deploy_frs.yaml` and run launcher regressions**

Set:

```yaml
frs:
  enabled: true
  steering_protection_interval_s: null
```

Set `steps_per_inference` equal to the configured action horizon and correct the stale comment. Do not modify `deploy_smolvla_jax.yaml`.

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_frs_deployment.py tests/jax/test_tactile_integration.py tests/jax/test_deploy_launcher.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit negotiated client config**

```bash
git add deploy_smolvla/configs/deploy_frs.yaml tests/jax/test_tactile_integration.py
git add -p deploy_smolvla/frs_runtime.py deploy_smolvla/remote_client.py tests/jax/test_frs_deployment.py
git diff --cached --check
git commit -m "feat: negotiate FRS steering protocol explicitly"
```

### Task 8: Add Generation-Bound FIFO Transport on the Robot Server

**Files:**
- Modify: `/home/typhon/vb3_robot_server/client/robot_client.py`
- Create: `/home/typhon/vb3_robot_server/tests/test_robot_client_frs_transport.py`

**Interfaces:**
- Produces `BridgeEnvelope(generation, message)`.
- Produces `active_connection_generation`, `generation_is_connected`, `publish_message`, and `wait_for_message`.
- Preserves all legacy `publish_obs`, `wait_for_action`, `publish_action_ack`, hello, obs, action, and ACK payloads.

- [ ] **Step 1: Add FIFO, connection-generation, disconnect, and STOP tests**

Use the existing fake WebSocket/msgpack pattern to prove ordered NumPy round trips and isolation:

```python
def test_frs_messages_round_trip_numpy_payloads_in_fifo_order(robot_client, generation):
    first = {"type": "frs_chunk_start", "chunk_id": 1}
    second = {"type": "frs_steer_request", "chunk_id": 1, "request_id": 2}
    robot_client.publish_message(first, generation=generation)
    robot_client.publish_message(second, generation=generation)
    assert drain_outbound(robot_client, generation) == [first, second]


def test_replacement_connection_cannot_receive_old_generation_ack(robot_client):
    old = connect(robot_client)
    robot_client.publish_message({"type": "frs_steer_ack"}, generation=old)
    disconnect(robot_client, old)
    new = connect(robot_client)
    assert drain_outbound(robot_client, new) == []
```

Also test ACK-before-next-request FIFO order, conflicting duplicate failure, and STOP interrupt.

- [ ] **Step 2: Run transport tests and confirm missing API**

Run from `/home/typhon/vb3_robot_server`:

```bash
uv run --no-sync pytest -q tests/test_robot_client_frs_transport.py
```

Expected: failure because the generic generation-bound transport does not exist.

- [ ] **Step 3: Implement per-generation queues without changing legacy slots**

Add:

```python
@dataclass(frozen=True)
class BridgeEnvelope:
    generation: int
    message: Mapping[str, Any]
```

Store `_frs_outbound: dict[int, deque[dict[str, Any]]]` and `_frs_inbound: dict[int, deque[dict[str, Any]]]`. The connection loop drains only its own outbound FIFO. `_handle_message` routes only `frs_chunk_ready` and `frs_steer_action` to inbound; all current types retain current assignments. Disconnect removes both queues and all FRS replay state for that generation.

- [ ] **Step 4: Run transport plus legacy wire regressions**

Run:

```bash
uv run --no-sync pytest -q tests/test_robot_client_frs_transport.py tests/test_vbvla_dry_run.py tests/test_msgpack_numpy.py
```

Expected: all tests pass, including exact legacy hello/obs/action/ack sequences.

- [ ] **Step 5: Commit server transport in its repository**

```bash
git add -p client/robot_client.py tests/test_robot_client_frs_transport.py
git diff --cached --check
git commit -m "feat: add generation-bound FRS message transport"
```

### Task 9: Validate Robot-Server FRS Negotiation and Wire Identity

**Files:**
- Create: `/home/typhon/vb3_robot_server/deploy_scripts/frs_protocol.py`
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online.py:276-390,986-1044`
- Create: `/home/typhon/vb3_robot_server/tests/test_frs_protocol.py`

**Interfaces:**
- Produces `NegotiatedExecution` with protocol, generation, mode, horizon, frequency, protection, and tactile keys.
- Produces `validate_execution_config(config_dict, *, server_action_horizon, server_control_frequency, execution_mode, available_observation_keys, connection_generation) -> NegotiatedExecution`.
- Produces exact server message builders and client response validators.

- [ ] **Step 1: Add legacy/FRS negotiation and identity tests**

Cover absence vs explicit version, exact horizon/frequency, `math.isclose` tolerances, protection, tactile keys, and exact response IDs:

```python
def test_missing_execution_protocol_selects_legacy_chunk_protocol():
    negotiated = validate_execution_config(
        valid_config(),
        server_action_horizon=10,
        server_control_frequency=30.0,
        execution_mode="rtc",
        available_observation_keys={"camera0_rgb", "robot0_eef_pos"},
        connection_generation=4,
    )
    assert negotiated.protocol == "legacy_chunk"


def test_data_type_vitac_does_not_implicitly_enable_frs():
    negotiated = validate_execution_config(
        valid_config(data_type="vitac"),
        server_action_horizon=10,
        server_control_frequency=30.0,
        execution_mode="rtc",
        available_observation_keys={"camera0_rgb", "camera2_rgb"},
        connection_generation=4,
    )
    assert negotiated.protocol == "legacy_chunk"
```

- [ ] **Step 2: Run server protocol tests**

Run:

```bash
uv run --no-sync pytest -q tests/test_frs_protocol.py
```

Expected: failure because negotiation and wire validators do not exist.

- [ ] **Step 3: Implement strict negotiation and builders**

Use:

```python
@dataclass(frozen=True)
class NegotiatedExecution:
    protocol: Literal["legacy_chunk", "frs_steering_v1"]
    connection_generation: int
    execution_mode: Literal["rtc", "block"]
    action_horizon: int
    control_frequency: float
    steering_protection_interval_s: float | None
    frs_tactile_keys: Sequence[str]
```

Call existing `validate_smolvla_config` first. For explicit FRS, require server horizon equality, `math.isclose(client_frequency, server_frequency, rel_tol=1e-9, abs_tol=1e-12)`, `steps_per_inference == H`, null or finite nonnegative non-bool protection, and nonempty unique string tactile keys included in available observation keys. Unknown non-null protocols fail before START.

- [ ] **Step 4: Run protocol and startup validation regressions**

Run:

```bash
uv run --no-sync pytest -q tests/test_frs_protocol.py deploy_scripts/bimanual_smolvla_online_test.py -k 'config or startup or protocol'
```

Expected: all tests pass; invalid FRS config fails before hardware construction.

- [ ] **Step 5: Commit negotiation and protocol schema**

```bash
git add deploy_scripts/frs_protocol.py tests/test_frs_protocol.py
git add -p deploy_scripts/bimanual_smolvla_online.py
git diff --cached --check
git commit -m "feat: validate FRS steering negotiation"
```

### Task 10: Implement Selected-Action Safety and the RTC/Block State Machine

**Files:**
- Create: `/home/typhon/vb3_robot_server/deploy_scripts/frs_execution.py`
- Create: `/home/typhon/vb3_robot_server/tests/test_frs_execution.py`

**Interfaces:**
- Consumes Task 8 transport and Task 9 protocol.
- Produces `FRSChunkState`, `CachedResponse`, `FRSChunkResult`, `Clock`, `validate_single_action`, and `run_frs_chunk`.

- [ ] **Step 1: Add strict single-action and fake-clock tests**

Use a deterministic clock:

```python
class FakeClock:
    def __init__(self, wall_time: float):
        self.wall = wall_time
        self.monotonic_value = 0.0
        self.waits = []

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.monotonic_value

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.monotonic_value += seconds

    def wait(self, timeout_s: float) -> None:
        self.waits.append(timeout_s)
        self.advance(timeout_s)
```

Test shape `[A]`, floating dtype, finite float32 range, translation, 6D rotation, gripper bounds, and ensure failures precede converter/controller calls.

- [ ] **Step 2: Add RTC and block state-machine tests, then run RED**

RTC tests assert strict first cutoff, null-to-dt protection, server observation anchor, immediate next capture after ACK, final stale recheck, no reapplied protection, skipped stale indices, interruptible nominal end, and expired-chunk end. Block tests assert every index, target spacing, wait-before-capture, and slow steer shifting.

Run:

```bash
uv run --no-sync pytest -q tests/test_frs_execution.py
```

Expected: failure because `frs_execution.py` does not exist.

- [ ] **Step 3: Implement exact chunk state and RTC selection**

Define:

```python
@dataclass
class FRSChunkState:
    connection_generation: int
    obs_seq: int
    chunk_id: int
    observation_timestamp: float
    control_dt: float
    action_horizon: int
    execution_mode: Literal["rtc", "block"]
    action_timestamps: np.ndarray | None
    nominal_chunk_end: float | None
    next_action_index: int
    next_request_id: int
    previous_block_target: float | None
    protection_pending: bool
    scheduled_count: int
    stale_count: int
    responses: dict[int, CachedResponse]
```

RTC computes timestamps once from observation capture time. First selection uses `timestamp > clock.time() + effective_protection`; every later selection uses `timestamp > clock.time()`. Immediately before conversion/submission, `clock.time() >= target` returns stale without controller calls. Publish ACK before capturing the next observation.

- [ ] **Step 4: Implement block pacing, idempotence, and interrupt checks**

Block target calculation is:

```python
dispatch_target = clock.time() + 0.01
if state.previous_block_target is not None:
    dispatch_target = max(
        dispatch_target,
        state.previous_block_target + state.control_dt,
    )
```

After ACK, poll until target while checking STOP, exact connection generation, controller health, and deadline per `interrupt_poll_s`. Cache response identity/content/result; identical duplicates replay ACK and never submit twice, conflicts reject the session.

Run:

```bash
uv run --no-sync pytest -q tests/test_frs_execution.py tests/test_vbvla_safety.py tests/test_relative_action_stale_prefix.py tests/test_bimanual_action_scheduling.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit state machine and safety**

```bash
git add deploy_scripts/frs_execution.py tests/test_frs_execution.py
git diff --cached --check
git commit -m "feat: schedule safe per-action FRS steering"
```

### Task 11: Orchestrate FRS Messages in the Remote Client

**Files:**
- Modify: `deploy_smolvla/remote_client.py:436-486,575-823`
- Create: `tests/jax/test_frs_remote_protocol.py`
- Modify: `tests/jax/test_tactile_integration.py`

**Interfaces:**
- Consumes Tasks 4-7 policy/protocol/bridge/config APIs.
- Produces `_run_frs_protocol(bridge, steering_policy, *, task, state_dim, image_keys, empty_cameras, observation_timeout_s, action_ack_timeout_s, seed, jit, num_steps, max_chunks, observation_saver) -> None` and explicit routing in `run()`.
- Preserves the legacy full-chunk `_predict_chunk`, `send_action`, `receive_action_ack` path when FRS is disabled.

- [ ] **Step 1: Add event-order, ACK, selected-shape, STOP, and legacy tests**

Use a scripted bridge and policy spy:

```python
def test_frs_loop_orders_ready_request_action_ack_and_chunk_end(scripted_bridge, policy):
    scripted_bridge.inbound.extend(
        [chunk_start(1), steer_request(1, 4, 2), steer_ack(1, 4, 2, "scheduled"), chunk_end(1)]
    )
    _run_frs_protocol(scripted_bridge, policy, **run_args(max_chunks=1))
    assert [message["type"] for message in scripted_bridge.sent] == [
        "frs_chunk_ready",
        "frs_steer_action",
    ]
    assert scripted_bridge.sent[1]["action"].shape == (policy.action_dim,)
```

Cover stale as normal, rejected as terminal, mismatched ACK, next chunk without baseline reset, trace failures, timeout/model failure STOP, and exact legacy full chunk.

- [ ] **Step 2: Run remote-protocol tests and confirm missing loop**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_frs_remote_protocol.py
```

Expected: failure because `_run_frs_protocol` does not exist.

- [ ] **Step 3: Implement the typed client state machine**

The loop accepts only this ordering:

```python
chunk_start = _expect(bridge.receive_frs_message(observation_timeout_s), FRSChunkStart)
ready = steering_policy.begin_chunk(
    chunk_start.chunk_id,
    prepare_observation(chunk_start.observation),
    task,
    seed=seed,
    jit=jit,
    num_steps=num_steps,
)
bridge.send_frs_chunk_ready(
    chunk_start.obs_seq,
    chunk_start.chunk_id,
    _build_trace_or_none(_build_frs_chunk_trace, ready),
)
```

Then repeat request -> `steer_action` -> send exact `[A]` -> matching ACK. `scheduled` and `stale` continue; `rejected` raises. Only `FRSChunkEnd` permits `end_chunk`. Remote code never selects indices or timestamps.

- [ ] **Step 4: Route startup/warmup and retain legacy branch**

After the existing warmup observation: call `reset_episode`, compile every `K=1..H`, verify warmup returns, then send START. Remove only the one-shot FRS branch from `_predict_chunk`; leave legacy behavior intact.

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/jax/test_frs_remote_protocol.py tests/jax/test_tactile_integration.py tests/jax/test_deploy_launcher.py
```

Expected: all tests pass, including STOP in `finally` on model/protocol errors.

- [ ] **Step 5: Commit remote orchestration with patch staging**

```bash
git add tests/jax/test_frs_remote_protocol.py tests/jax/test_tactile_integration.py
git add -p deploy_smolvla/remote_client.py
git diff --cached --check
git commit -m "feat: orchestrate per-action FRS steering remotely"
```

### Task 12: Route and Trace the FRS Session on the Robot Server

**Files:**
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online.py:986-1044,1162-1378`
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/action_trace.py`
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online_test.py`
- Modify: `/home/typhon/vb3_robot_server/tests/test_action_trace.py`

**Interfaces:**
- Consumes Tasks 8-10 negotiated session, transport, and `run_frs_chunk`.
- Produces `run_frs_session(client, env, negotiation, observation_builder, converter, action_timeout_s, trace_logger)` and explicit `legacy_chunk`/`frs_steering_v1` routing.
- Extends diagnostics to trace version 2 without coupling trace success to control.

- [ ] **Step 1: Add main routing, cleanup, and trace-isolation tests**

Use injected fake hardware factories to prove routing and pre-hardware rejection:

```python
def test_main_routes_missing_protocol_to_unchanged_legacy_loop(monkeypatch):
    calls = install_session_spies(monkeypatch)
    invoke_main(valid_config())
    assert calls == ["legacy"]


def test_main_routes_frs_v1_to_separate_frs_session(monkeypatch):
    calls = install_session_spies(monkeypatch)
    invoke_main(valid_frs_config())
    assert calls == ["frs"]
```

Assert invalid contracts construct no env; STOP/disconnect joins controller before client cleanup; steps-per-inference does not cap FRS. Trace tests make validation/persistence throw and assert scheduled control result is unchanged.

- [ ] **Step 2: Run main/trace tests and confirm missing route**

Run:

```bash
uv run --no-sync pytest -q deploy_scripts/bimanual_smolvla_online_test.py tests/test_action_trace.py -k 'frs or trace_v2 or routes_missing_protocol'
```

Expected: failure because FRS routing and trace-v2 records do not exist.

- [ ] **Step 3: Extract explicit session routing while preserving legacy code**

After negotiation and warmup START:

```python
if negotiation.protocol == "frs_steering_v1":
    run_frs_session(
        client=client,
        env=env,
        negotiation=negotiation,
        observation_builder=observation_builder,
        converter=get_real_umi_action,
        action_timeout_s=action_timeout_s,
        trace_logger=action_debug_logger,
    )
else:
    run_legacy_session(
        client=client,
        env=env,
        config_dict=config_dict,
        action_timeout_s=action_timeout_s,
        trace_logger=action_debug_logger,
    )
```

Move existing loop statements into `run_legacy_session` without semantic edits. Only legacy calls `enable_action_ack` and enforces `max_executed_actions >= steps_per_inference`.

- [ ] **Step 4: Persist trace v2 without affecting control and run server regressions**

Trace-v2 chunk header stores action VLA/base once; each steer entry stores full decode, selected row, target, K, timing, diagnostics, and final ACK status. Validation exceptions become record-local `trace_error` strings; logger queue/persistence exceptions are logged and swallowed.

Run:

```bash
uv run --no-sync pytest -q tests/test_frs_protocol.py tests/test_frs_execution.py tests/test_robot_client_frs_transport.py deploy_scripts/bimanual_smolvla_online_test.py tests/test_action_trace.py tests/test_vbvla_dry_run.py tests/test_vbvla_safety.py tests/test_relative_action_stale_prefix.py
```

Expected: all tests pass without initializing hardware.

- [ ] **Step 5: Commit server integration with patch staging**

```bash
git add -p deploy_scripts/bimanual_smolvla_online.py deploy_scripts/action_trace.py deploy_scripts/bimanual_smolvla_online_test.py tests/test_action_trace.py
git diff --cached --check
git commit -m "feat: run and trace negotiated FRS steering sessions"
```

### Task 13: Cross-Repository Verification and Compatibility Audit

**Files:**
- Verify all files listed above.
- Do not change unrelated dirty files during this task.

**Interfaces:**
- Consumes every prior task.
- Produces verified client/server protocol agreement and an evidence-backed handoff.

- [ ] **Step 1: Run the complete focused FRS_Tact suite**

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync pytest -q tests/train_frs/test_model.py tests/jax/test_policy.py tests/jax/test_frs_deployment.py tests/jax/test_frs_protocol.py tests/jax/test_frs_remote_protocol.py tests/jax/test_tactile_integration.py tests/jax/test_deploy_launcher.py tests/flow_decoder/test_integration.py tests/flow_decoder/test_frs_safety.py
```

Expected: all tests pass.

- [ ] **Step 2: Run FRS_Tact static checks**

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync ruff check train_frs/utils/model.py train_smolvla/policy.py utils/source_model.py deploy_smolvla/frs_runtime.py deploy_smolvla/frs_protocol.py deploy_smolvla/bridge_client.py deploy_smolvla/remote_client.py tests/train_frs/test_model.py tests/jax/test_policy.py tests/jax/test_frs_deployment.py tests/jax/test_frs_protocol.py tests/jax/test_frs_remote_protocol.py
git diff --check
git status --short
```

Expected: Ruff and diff checks succeed; pre-existing user changes remain present and unstaged unless their exact task hunks were intentionally committed.

- [ ] **Step 3: Run the complete no-hardware robot-server suite**

From `/home/typhon/vb3_robot_server`:

```bash
uv run --no-sync pytest -q tests deploy_scripts
```

Expected: all tests pass without camera/controller initialization.

- [ ] **Step 4: Run server static/worktree checks and compare the wire contract**

```bash
uv run --no-sync ruff check client/robot_client.py deploy_scripts/frs_protocol.py deploy_scripts/frs_execution.py deploy_scripts/bimanual_smolvla_online.py deploy_scripts/action_trace.py tests/test_robot_client_frs_transport.py tests/test_frs_protocol.py tests/test_frs_execution.py
git diff --check
git status --short
```

Manually compare the exact `frs_chunk_start`, `frs_chunk_ready`, `frs_steer_request`, `frs_steer_action`, `frs_steer_ack`, and `frs_chunk_end` fields in both protocol modules. Confirm action control payload is rank-one `[A]`, full chunks appear only in trace, and ACK precedes every next request.

- [ ] **Step 5: Record final evidence without combining repositories into one commit**

If verification required no new code, do not create an empty commit. Report both repository commit lists, exact test counts, any unavailable hardware-only validation, and preserved dirty paths. If a test-only correction was required, stage only that repository's correction and commit with:

```bash
git commit -m "test: verify end-to-end FRS steering protocol"
```
