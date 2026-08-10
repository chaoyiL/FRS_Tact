# Live VLA/FRS/Robot Action Trace Design

## Goal

Continuously update two headless PNG files for each robot run, comparing the
raw VLA prediction, the FRS-refined prediction, and measured robot feedback.
The plots cover the full run and never participate in the robot-control path.

## Outputs

The robot server creates one session directory under
`action_debug_logs/YYYYMMDD_HHMMSS/` containing:

- `chunk_trace.jsonl`: one record per observation/action chunk, including the
  complete VLA and FRS chunks, FRS diagnostics, action timestamps, freshness
  mask, converted waypoints, and controller scheduling records.
- `controller_trace.jsonl`: measured and interpolated controller state sampled
  by the existing controller debug stream.
- `full_prediction.png`: all ten predicted waypoints for every chunk; scheduled
  points are filled and rejected/stale points are hollow.
- `executed_vs_actual.png`: only scheduled VLA/FRS waypoints aligned with
  measured robot feedback.

Each PNG has six panels: left/right arm by position-delta magnitude in metres,
rotation-delta magnitude in radians, and gripper width.

## Wire protocol

The existing `action` and `obs_seq` fields remain unchanged. The client adds an
optional versioned `trace` mapping containing normalized and unnormalized VLA
and FRS chunks, inference timestamps, and FRS scalar diagnostics. An old server
continues to use `action` and ignore `trace`; a new server accepts action-only
legacy clients and records an explicit missing-trace warning.

Trace validation is diagnostic and independent of action safety validation. A
malformed trace is omitted from the logs and plots without changing whether a
separately valid robot action is executed.

## Alignment and units

The server is the source of truth. It attaches the same `obs_seq` to prediction,
freshness selection, scheduling, and feedback records. Both VLA and FRS chunks
are interpreted at the same observation and conversion origin.

Prediction position/rotation values are per-step SE(3) deltas. Measured
position and rotation are computed from consecutive feedback poses resampled at
the scheduled waypoint timestamps; rotations use relative matrices rather than
axis-angle subtraction. Measured gripper readings are converted with the same
width slope and offset as observations. The x-axis is elapsed wall-clock time
since START.

## Runtime isolation

The control process only appends flushed JSONL records. A separate headless
plotter process tails those files and atomically replaces PNG files no more than
once per second. It may downsample only in-memory display series; JSONL always
retains full data. Plotter startup, rendering, or shutdown failures emit a
warning and never delay ACKs, scheduling, controller execution, or STOP.

## Verification

Tests cover optional trace protocol compatibility, shape/finiteness rejection,
preservation of the actual action payload, JSONL schemas, SE(3) metrics,
feedback resampling, executed masks, atomic PNG rendering, plotter failure
isolation, and dry-run behavior. Verification does not initialize cameras or
robot hardware.
