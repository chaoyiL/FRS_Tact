# Per-Action FRS Steering Protocol Design

**Status:** Approved for implementation on 2026-08-12.

## Goal

Refactor FRS deployment so one visual SmolVLA prediction produces a normalized
action chunk and a fixed reverse-integrated base noise, after which fresh tactile
observations repeatedly re-steer that same base noise. Each re-steer decodes the
complete action chunk but sends only the requested action index to the robot
server.

The robot server remains compatible with every existing non-FRS client. The new
behavior is enabled only by an explicit, versioned handshake protocol.

## Scope

This design covers both repositories involved in live deployment:

- `/home/typhon/FRS_Tact`: source policy inversion, variable-length tactile
  conditioning, FRS chunk state, and the remote protocol client.
- `/home/typhon/vb3_robot_server`: protocol negotiation, fresh tactile requests,
  authoritative timestamp selection, single-action validation, and controller
  scheduling.

It does not retrain the FRS checkpoint, enable the SmolVLA checkpoint's internal
RTC stitching, or change legacy chunk execution behavior.

## Confirmed Semantics

- “RTC” means the robot server's `exec_mode="rtc"`. The source SmolVLA
  checkpoint's `rtc_config` remains disabled in FRS mode.
- The server is authoritative for action timestamps, target selection, and the
  final stale check.
- RTC timestamps use the initial observation capture time:

  ```text
  action_timestamp[i] = initial_observation_timestamp + i * control_dt
  nominal_chunk_end = initial_observation_timestamp + action_horizon * control_dt
  ```

- The first RTC target is the earliest action strictly beyond
  `server_now + steering_protection_interval_s`. A missing/null protection value
  resolves to one control period. Protection is applied only to the first target
  in a chunk.
- After an RTC action is successfully scheduled, the server immediately requests
  a new tactile observation and the next future action.
- If a decoded RTC action expires before controller submission, it is not sent to
  the controller. The server immediately requests the next future action without
  reapplying protection.
- An RTC outer iteration occupies the chunk's nominal duration. If steering ends
  early, it waits interruptibly until the nominal end; if computation already
  passed the end, the next chunk begins immediately.
- In non-RTC/block mode, steering begins at action index zero, every action is
  retained, and slow inference lengthens the chunk instead of dropping actions.
  After action `i` is scheduled, the server waits until its target timestamp
  before capturing tactile input for action `i+1`.
- A new tactile sequence is created for every action chunk. Each unique steer
  request appends one tactile embedding. The sequence is not padded and has a
  maximum length equal to the action horizon by default.
- `action_vla` and `x_base` are generated exactly once per chunk. Every steer
  uses the same `x_base`, decodes the full `[1, H, A]` chunk, and selects only the
  requested index afterward.

## Compatibility and Negotiation

The existing client-to-server config message gains an optional field:

```python
{
    "execution_protocol": "frs_steering_v1",
    "steering_protection_interval_s": None | float,
    "frs_tactile_keys": list[str],
}
```

The remote client adds these fields only when `frs.enabled` is true. Absence of
`execution_protocol` selects the existing legacy chunk protocol. The server must
not infer FRS mode from `data_type="vitac"`, action shape, checkpoint metadata,
or any other heuristic.

The server accepts `frs_steering_v1` only when:

- execution mode is `rtc` or `block`;
- client and server action horizons match;
- client and server control frequencies match under
  `math.isclose(rel_tol=1e-9, abs_tol=1e-12)`;
- the protection interval is null or a finite non-negative number; and
- `frs_tactile_keys` is a non-empty unique list and the negotiated observation
  mode supplies every named stream.

Unknown protocol versions fail before robot START. Legacy sessions retain their
current config validation, message schema, `steps_per_inference`,
`max_executed_actions`, RTC, and block behavior.

For FRS v1, `steps_per_inference` does not cap execution. The FRS deployment
configuration sets it equal to `action_horizon` to keep the shared legacy config
contract unambiguous.

## Component Boundaries

### Source SmolVLA policy

`JaxSmolVLAPolicy` exposes a public source-flow capability:

```python
reverse_action_chunk(
    observation,
    task,
    normalized_actions,
    *,
    num_steps,
    solver,
) -> jax.Array
```

The policy owns observation preprocessing, prefix construction, padding to
`max_action_dim`, source parameters, and `denoise_step`. Numerical integration
primitives remain in the shared integration utilities, but deployment code no
longer reaches into source-model internals.

The method accepts and returns normalized `[B, H, A]` actions. It integrates the
source velocity field from data time `t=0` to base-noise time `t=1`. Its solver,
step count, normalization source, and reverse integration version continue to be
validated against FRS checkpoint metadata.

### FRS steering policy

Introduce `FRSSteeringPolicy` as the stateful composition of:

- the visual source policy;
- the frozen tactile encoder;
- the tactile-conditioned FRS decoder; and
- episode/chunk safety and diagnostic state.

The class lives with the existing FRS deployment support. `FRSRuntime` remains
importable as a compatibility alias of `FRSSteeringPolicy`; there is only one
implementation and the remote client uses the new lifecycle.

Its public lifecycle is:

```python
reset_episode(initial_observation)
begin_chunk(chunk_id, initial_observation, task, *, seed, jit, num_steps)
steer_action(chunk_id, request_id, observation, action_index)
end_chunk(chunk_id)
```

`begin_chunk` predicts normalized `action_vla`, reverse-integrates it once, and
then clears chunk-local tactile/request state. It does not run the FRS decoder.

`steer_action` is idempotent by `(chunk_id, request_id)`. A new request:

1. validates the chunk and action index;
2. encodes the current tactile images into `[N, D]`;
3. appends that embedding to the chunk-local sequence;
4. computes the current gate relative to the episode baseline;
5. decodes the entire fixed `x_base` into `[1, H, A]`;
6. validates the complete normalized output; and
7. returns the full diagnostic result plus only the selected normalized and
   robot-space action for control.

A duplicate request returns the cached result without encoding tactile again.
A duplicate request whose chunk/index/content disagrees with the cached request
is a protocol error.

`end_chunk` releases `action_vla`, `x_base`, decoded chunks, tactile sequence,
and request results. It does not reset the episode tactile baseline.

### Remote client

`remote_client.py` is a protocol orchestrator. It loads the policy, negotiates
the execution protocol, performs warmup, and dispatches typed bridge messages.
It does not own tactile history, call low-level reverse/decode functions, select
RTC indices, or construct server timestamps.

### Robot server

The server owns:

- chunk/request identifiers;
- observation and action timestamps in its clock domain;
- RTC target selection and stale rechecks;
- non-RTC pacing;
- fresh observation acquisition before every steer request;
- single-action safety validation and relative-action conversion; and
- controller submission.

The new FRS execution state machine is separate from the legacy
`execute_action_chunk` path.

## Variable-Length Tactile Conditioning

`SharedTactileGRU` already scans over its input length. The hard constraint is in
`TactileConditionedFlowDecoder.encode_tactile_tokens`, which currently requires
`time_steps == config.tactile_window`. Remove that equality check.

The decoder continues to require:

- rank-four `[B, K, N, D]` tactile input;
- `K >= 1`;
- the configured number of tactile streams; and
- the configured ResNet embedding dimension.

The deployment policy separately enforces `K <= action_horizon`. It supplies the true sequence shapes
`[1, 1, N, D]` through `[1, H, N, D]`; it does not pad or mask shorter
sequences.

Checkpoint field `tactile_window` and metadata `history_stride` continue to
describe the training data distribution. They do not impose a deployment-time
sequence length. Startup logs clearly report that inference lengths above the
training window are out-of-distribution. The checkpoint remains loadable without
parameter conversion.

The gate remains current tactile change relative to the episode baseline. It is
not reset at chunk boundaries.

### Decode conditioning reuse

Currently every Euler/FireFlow velocity evaluation re-runs the GRU over an
unchanged tactile sequence. Split the decoder inference path so one steer:

1. encodes the tactile sequence once;
2. projects the resulting tactile tokens once; and
3. reuses that conditioning throughout all flow integration steps.

The existing model call remains available for training and other callers. Tests
must show that the cached-conditioning inference path is numerically equivalent
to the existing computation and invokes tactile sequence encoding once per
decode.

JAX compiles a separate executable for each concrete `K`. Before sending robot
START, warmup compiles all lengths `1..H` using synthetic/repeated warmup tokens
without mutating real episode or chunk state.

## Wire Protocol

All new messages carry `chunk_id`. Per-steer messages also carry a unique
`request_id` and `action_index`. Array payloads use the existing msgpack NumPy
encoding.

### Server to client: chunk start

```python
{
    "type": "frs_chunk_start",
    "obs_seq": int,
    "chunk_id": int,
    "observation": dict,
    "observation_timestamp": float,
    "control_dt": float,
    "action_horizon": int,
    "execution_mode": "rtc" | "block",
    "action_timestamps": np.ndarray | None,
    "nominal_chunk_end": float | None,
}
```

For RTC, `action_timestamps` has shape `[H]` and `nominal_chunk_end` is finite.
For block mode they are null because targets are assigned as actions become
ready.

### Client to server: chunk ready

```python
{
    "type": "frs_chunk_ready",
    "obs_seq": int,
    "chunk_id": int,
    "prediction_trace": dict | None,
}
```

This message means `action_vla` and `x_base` are ready and immutable. It does not
contain a control action.

### Server to client: steer request

```python
{
    "type": "frs_steer_request",
    "chunk_id": int,
    "request_id": int,
    "action_index": int,
    "target_timestamp": float | None,
    "protection_applied": bool,
    "observation": dict,
}
```

The server captures the observation immediately before publishing this request.
The full normal observation envelope may be reused initially; only the tactile
keys are consumed by `steer_action`.

### Client to server: selected action

```python
{
    "type": "frs_steer_action",
    "chunk_id": int,
    "request_id": int,
    "action_index": int,
    "action": np.ndarray,  # exact shape [A]
    "trace": dict | None,
}
```

Only the selected robot-space action crosses the control boundary. The complete
decoded chunk is diagnostic data and may appear in the optional trace, never in
the control `action` field.

### Server to client: steer acknowledgement

```python
{
    "type": "frs_steer_ack",
    "chunk_id": int,
    "request_id": int,
    "action_index": int,
    "status": "scheduled" | "stale" | "rejected",
    "scheduled_timestamp": float | None,
}
```

Every accepted `frs_steer_action` receives exactly one acknowledgement before a
new request is sent. `stale` is a normal RTC outcome. `rejected` is terminal for
the FRS session.

### Server to client: chunk end

```python
{
    "type": "frs_chunk_end",
    "chunk_id": int,
    "reason": "exhausted" | "deadline" | "no_future_action" | "stopped",
    "scheduled_count": int,
    "stale_count": int,
}
```

After receiving it, the client calls `end_chunk` and waits for the next chunk
start. Existing `state="stop"` remains the session stop mechanism.

## RTC State Machine

At chunk start the server computes:

```python
timestamps = observation_timestamp + np.arange(H) * control_dt
deadline = observation_timestamp + H * control_dt
```

After `frs_chunk_ready`, the first cutoff is:

```python
server_now + effective_protection_interval
```

The first request targets the lowest index whose timestamp is strictly greater
than that cutoff. Earlier indices are recorded as skipped by protection and are
never requested.

After every response, the server validates message identity and the selected
action. Immediately before controller submission it samples server time again:

- `now < target_timestamp`: convert from the freshest robot state, schedule the
  action at its original timestamp, and acknowledge `scheduled`.
- `now >= target_timestamp`: do not call the controller and acknowledge `stale`.

In both cases that index is complete. The server immediately chooses the lowest
later, not-yet-attempted index with `timestamp > now`. It does not add protection
again. Indices already stale before a request can be issued are skipped without
performing pointless inference.

When no future index remains, the server waits interruptibly until `deadline` and
then emits `frs_chunk_end`. If the deadline has already passed, it emits the end
message immediately. STOP, disconnect, or controller failure interrupts every
wait.

## Non-RTC/Block State Machine

After `frs_chunk_ready`, the server captures a fresh observation and requests
index zero. There is no protection interval and no stale-action dropping.

When action `i` arrives, the server validates it, converts it relative to the
fresh robot state, and assigns:

```python
target_i = max(
    server_now + controller_dispatch_lead,
    previous_target + control_dt,  # omitted for i == 0
)
```

`controller_dispatch_lead` is the server's existing 0.01 second scheduling lead;
it is separate from the FRS protection interval. After scheduling and
acknowledging the action, the server waits interruptibly until `target_i`. Only
then does it capture a new tactile observation and request `i+1`.

This continues through `H-1`. A slow steer moves later targets forward rather
than dropping actions. The chunk ends only after every index has been scheduled.
The wait is for the target timestamp, not an additional controller-completion
signal.

## Action Representation and Safety

Source sampling, reverse integration, FRS decoding, delta checks, and cached
chunks operate in the source checkpoint's normalized action space.
Unnormalization occurs only after selecting the target row.

FRS performs its existing full decoded-chunk checks on every steer:

- output shape exactly equals `action_vla.shape`;
- all values are finite;
- maximum normalized action magnitude is within its configured limit; and
- full-chunk RMS delta from `action_vla` is within its configured limit.

The server independently validates the selected robot-space action:

- exact shape `[action_dim]` and floating dtype;
- finite float32-representable values;
- per-arm translation delta;
- valid and bounded 6D rotation;
- gripper bounds; and
- matching chunk/request/index identity.

The single relative action is converted from the current robot state immediately
before scheduling. Legacy full-chunk validation remains unchanged.

FRS failures never fall back to an unsteered VLA action.

## Failure and Concurrency Rules

- `stale` is expected control flow in RTC, not a model or protocol error.
- Non-finite data, unsafe actions, invalid ordering, mismatched identifiers,
  impossible indices, and duplicate requests/responses with different content
  fail closed.
- On client model/protocol failure, the client reports the error when possible
  and sends STOP. The server cancels the current FRS session and stops the
  controller.
- On disconnect, the server invalidates all requests and refuses subsequent
  controller submission from that connection generation.
- A duplicate identical request or response is idempotent. It never grows the
  tactile sequence or schedules twice.
- The server sends a steer acknowledgement before publishing another request, so
  the client never mistakes a fresh observation for an acknowledgement.
- Deadline/target waits poll state or use an interruptible condition; they do not
  use a long uninterruptible sleep.

## Diagnostics

The existing optional action trace advances to version 2. It records one chunk
header plus per-steer entries:

- normalized and robot-space `action_vla`;
- the complete fixed normalized `x_base`;
- the complete normalized decoded chunk for each request;
- target index and timestamp;
- tactile sequence length;
- whether protection was applied;
- encode/decode start and finish timestamps;
- selected normalized and robot-space action;
- tactile change, gate, full-chunk delta RMS, and maximum magnitude; and
- final `scheduled`, `stale`, or `rejected` status.

Trace validation and persistence remain isolated from control. Diagnostic
serialization, logging, or plotting failures cannot change whether an otherwise
valid action is scheduled.

## Configuration Changes

The FRS deployment YAML gains:

```yaml
frs:
  enabled: true
  steering_protection_interval_s: null
```

Null means one server control period. Explicit values are seconds. The FRS
configuration uses `steps_per_inference == action_horizon`.

Non-FRS deployment YAML files do not opt into `frs_steering_v1`. Existing local
checkpoint, camera, frequency, and other uncommitted user configuration changes
must be preserved.

## Verification

All automated verification avoids cameras and robot hardware.

### FRS_Tact tests

- The GRU/decoder accepts `K=1`, the training window, and `K=H`.
- Invalid rank, empty sequences, tactile stream count, and embedding dimensions
  still fail.
- Cached-conditioning decode is numerically equivalent to the existing path and
  encodes tactile input once per decode.
- Source policy reverse integration preserves normalized `[B, H, A]` shape,
  finiteness, solver selection, and metadata contracts.
- `begin_chunk` produces `action_vla`/`x_base` exactly once and clears tactile
  state.
- Unique steer requests grow sequence length one at a time; duplicate IDs do
  not.
- Chunk boundaries clear tactile state while episode baseline remains.
- Every steer decodes the full chunk and returns only the requested row.
- Warmup compiles all `K=1..H` shapes without mutating live state.
- The bridge preserves legacy payloads when no FRS protocol is negotiated.
- FRS message schemas, acknowledgement ordering, STOP, timeout, and malformed
  message behavior are covered.

### Robot server tests

- Missing protocol fields preserve the current full-chunk RTC and block paths.
- FRS config negotiation rejects unsupported versions and contract mismatches
  before START.
- Fake-clock RTC tests cover the first protection cutoff, immediate continuation,
  final stale recheck, no-protection retry, skipped stale indices, nominal chunk
  end, and already-expired chunks.
- Fake-clock block tests cover indices `0..H-1`, minimum timestamp spacing, waits
  before fresh tactile capture, and slow-steer target shifting.
- Single-action safety validation rejects every unsafe class before conversion or
  controller submission.
- Duplicate messages are idempotent; out-of-order messages fail closed.
- STOP, disconnect, controller failure, and timeouts interrupt waits and prevent
  late scheduling.
- Trace version 2 failures remain isolated from control.

The focused suites are followed by both repositories' complete relevant
deployment/safety suites. No verification command initializes hardware.

## Out of Scope

- Retraining or modifying checkpoint weights.
- Claiming that sequence lengths above the training window are in-distribution.
- Enabling source-policy `rtc_config` or `previous_chunk` stitching.
- Changing legacy clients to use single-action messages.
- Moving JAX, the tactile encoder, or the FRS decoder onto the robot server.
- Requiring synchronized client/server wall clocks.
