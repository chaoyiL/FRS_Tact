# FRS Live Action Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render live and final FRS plots containing the full VLA chunk, only controller-scheduled steer actions, and measured robot feedback.

**Architecture:** Keep the public FRS wire protocol unchanged. Enrich server-local trace callbacks with authoritative timing/reference/converted-waypoint data, persist those records, and incrementally fold `frs_chunk` plus `frs_steer` JSONL records into the legacy plotting view. The existing Matplotlib plot core and legacy JSONL path remain available without schema changes.

**Tech Stack:** Python, NumPy, Matplotlib Agg, SciPy Rotation/Slerp, pytest, JSONL.

## Global Constraints

- Only `status == "scheduled"` steer actions may appear in the orange curve; `stale` and `rejected` remain diagnostics only.
- VLA is a full robot-space chunk plotted from the chunk-start reference waypoint.
- Scheduled steer waypoints come from the exact converted array submitted to `env.exec_actions`, never from re-decoding client-relative actions in the plotter.
- RTC uses server-authoritative chunk timestamps; block mode uses actual scheduled timestamps for known actions and `control_dt` extrapolation for the remaining suffix.
- Do not change `frs_steering_v1` client/server wire schemas, scheduling, safety, ACK, protection, stale, or block pacing behavior.
- Preserve legacy trace schema, rendering behavior, PNG names, output directory, isolated live plot process, and fail-open diagnostics.
- Existing uncommitted server work belongs to the user; edit with focused patches and commit only feature hunks.

---

### Task 1: Persist authoritative FRS plotting fields

**Files:**
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/frs_execution.py`
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online.py`
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/action_trace.py`
- Modify: `/home/typhon/vb3_robot_server/tests/test_frs_execution.py`
- Modify: `/home/typhon/vb3_robot_server/tests/test_action_trace.py`

**Interfaces:**
- `run_frs_chunk(..., reference_waypoint: np.ndarray | None = None)` includes execution metadata in the `chunk_ready` local event.
- Successful `steer_ack` local events include `absolute_waypoint` equal to the exact `[1, 14]` converted action row submitted to the environment.
- `ActionTraceLogger.log_frs_event` persists exact validated fields without changing network messages.

- [ ] Add failing tests proving RTC/block chunk events carry their timing mode and reference, scheduled events carry the exact converted waypoint, and stale/rejected events do not claim an executed waypoint.
- [ ] Run focused tests and verify failures are caused by missing trace fields.
- [ ] Add strict local validation and propagate the server-side fields through `run_frs_session`; keep control payloads and execution ordering unchanged.
- [ ] Make FRS execution drain controller feedback during the session so the independent plotter receives measured samples before shutdown, using the existing logger callback without retaining an unbounded duplicate history.
- [ ] Run focused execution/action-trace tests and the FRS scheduling regressions.
- [ ] Commit only Task 1 hunks.

### Task 2: Fold FRS JSONL events into live plot chunks

**Files:**
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/action_trace_plotter.py`
- Modify: `/home/typhon/vb3_robot_server/tests/test_action_trace_plotter.py`

**Interfaces:**
- A focused stateful `FRSLiveTrace`/equivalent fold consumes append-only records and yields legacy-shaped plot chunks.
- It tolerates steer-before-header/malformed records without raising into the control process.
- Existing `_plot_mode` remains the sole renderer for VLA/FRS/feedback series.

- [ ] Add failing tests for RTC aggregation, block suffix extrapolation and re-anchoring, scheduled-only filtering, exact absolute waypoint placement, incremental record arrival, duplicate replay idempotence, and chunk separation.
- [ ] Run focused tests and verify expected aggregation failures.
- [ ] Implement bounded per-chunk fold state and adapt live/offline JSONL ingestion before `_LiveChunkHistory`.
- [ ] Ensure incomplete FRS chunks render as soon as a valid header arrives and update when scheduled steer records arrive.
- [ ] Run the full plotter test file and legacy rendering tests.
- [ ] Commit only Task 2 hunks.

### Task 3: End-to-end live/final rendering and compatibility

**Files:**
- Modify: `/home/typhon/vb3_robot_server/tests/test_action_trace.py`
- Modify: `/home/typhon/vb3_robot_server/tests/test_action_trace_plotter.py`
- Modify only if a test exposes a gap: the Task 1/2 production files.

**Interfaces:**
- `render_trace_pngs` accepts mixed legacy and FRS JSONL records.
- The plotter subprocess refreshes on both FRS header and scheduled steer arrivals and performs one clean final render.

- [ ] Add an end-to-end JSONL-to-PNG test asserting both PNGs exist and plotted line labels/data include full VLA, scheduled steer only, and feedback.
- [ ] Add a live-tail test proving a header first produces a VLA view and a later scheduled record updates the orange series.
- [ ] Run RED, apply only minimal integration fixes, then run GREEN.
- [ ] Run server FRS protocol/execution/trace/plotter/legacy regression suite, `py_compile`, and `git diff --check`.
- [ ] Request independent code review and fix all Critical/Important findings.
- [ ] Commit only final feature/test hunks and record preserved dirty user files.
