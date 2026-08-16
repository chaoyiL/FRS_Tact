# Final Review Fixes Report

Base reviewed: `b06114d`.

## Scope

- `deploy_smolvla/direct_decoder.py`
- `deploy_smolvla/remote_client.py`
- `tests/jax/test_direct_decoder_deployment.py`

The user-owned `deploy_smolvla/configs/deploy_frs.yaml` edit and
`DIRECT_DECODER_DEPLOYMENT_MODIFICATION_GUIDE.md` were left untouched.

## RED

Added two focused tests before changing production code, then ran:

```console
.venv/bin/pytest -q tests/jax/test_direct_decoder_deployment.py -k \
  'reset_and_end_chunk_reset_decoder_snapshots or trace_builder_exception_is_fail_open'
```

Both failed as expected:

1. The decoder reset spy remained at `0` after `end_chunk`, proving the
   steering adapter did not reset the underlying decoder snapshots.
2. Direct-protocol trace serialization logged `Omitting FRS trace...` instead
   of a direct-decoder-steering label, while the action was still sent.

## GREEN

- `DirectDecoderSteeringRuntime.reset()` now clears its adapter state and calls
  `decoder.reset()`. Because `end_chunk()` delegates to `reset()`, both lifecycle
  paths clear `last_vla_normalized` and `last_direct_normalized`.
- `_build_trace_or_none()` accepts a trace label. The FRS wrapper explicitly
  passes `FRS`, preserving the exact established warning text; the direct wrapper
  passes `direct decoder steering` for both chunk and action traces.
- The new decoder spy verifies snapshot clearing after `end_chunk()` and an
  explicit adapter reset, then successfully starts a subsequent chunk.
- The new protocol test forces direct steer trace construction to fail, verifies
  the selected action is still delivered with `trace=None`, and checks the
  direct-decoder warning label.

Fresh serial verification:

```console
.venv/bin/pytest -q tests/jax/test_direct_decoder_deployment.py
# 22 passed

.venv/bin/pytest -q tests/jax/test_frs_deployment.py::test_trace_v2_builder_exception_is_fail_open
# 1 passed

.venv/bin/pytest -q tests/jax/test_frs_remote_protocol.py tests/jax/test_frs_protocol.py
# 46 passed

.venv/bin/python -m py_compile deploy_smolvla/direct_decoder.py deploy_smolvla/remote_client.py
git diff --check
# both exit 0
```

The same suites initially ran in parallel and hit CUDA import-time OOM. Running
them serially removed the resource contention and produced the results above.

## Commit

Implementation and tests: `fe03763` (`fix: clear direct decoder state and label traces`).

## Self-review

- `end_chunk()` uses the existing `reset()` path, so it makes exactly one
  low-level reset and retains lifecycle validation.
- The default trace label is `FRS`, and the FRS call site passes it explicitly;
  the existing exact-text regression test passes unchanged.
- The direct label is applied at the common per-action protocol layer, covering
  direct chunk and steer trace serialization failures.
- No unrelated production or user-owned files were staged in the implementation
  commit.

## Concern

The virtual environment does not provide Ruff (`.venv/bin/python -m ruff` reports
`No module named ruff`), so lint/format checks were unavailable. Pytest,
`py_compile`, and whitespace validation completed successfully.
