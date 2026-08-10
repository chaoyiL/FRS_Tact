# Deployment Launcher Boundary Design

## Goal

Make the FRS and VT-SmolVLA deployment entrypoints visibly independent while
sharing only the environment, authentication, argument parsing, and Python
launch mechanics that are common to both deployments. Fix the current
`Permission denied` failure without relying on executable file modes for nested
shell scripts.

## Current problem

`deploy_smolvla/scripts/start_frs.sh` directly executes
`start_vtsmolvla.sh`. The target is tracked with mode `0644`, so invoking the
FRS wrapper through Bash fails when the wrapper reaches `exec`. The name also
makes the runtime boundary misleading: FRS is a separate deployment mode, but
its wrapper appears to launch VT-SmolVLA.

The two modes do share the same Python entrypoint,
`deploy_smolvla.remote_client`. The selected YAML controls model construction;
`deploy_frs.yaml` enables the FRS decoder and tactile encoder through
`frs.enabled: true`.

## Selected design

Introduce `deploy_smolvla/scripts/start_remote_client.sh` as the shared,
model-neutral launcher.

- `start_remote_client.sh` owns project-root discovery, Python selection,
  `PYTHONPATH`, Hugging Face cache configuration, token loading, `--check`, and
  execution of `deploy_smolvla.remote_client`.
- `start_frs.sh` is the FRS-specific entrypoint. It invokes the shared launcher
  through `bash` and supplies `deploy_frs.yaml`.
- `start_vtsmolvla.sh` is the VT-SmolVLA-specific entrypoint. It invokes the
  shared launcher through `bash` and supplies `deploy_smolvla_jax.yaml`.
- Both wrappers forward all user arguments after their selected default
  configuration.
- The shared launcher continues to accept `--config PATH` so an explicit user
  override remains possible. Argument order means a later user-provided
  `--config` overrides the wrapper default, matching the existing behavior.

Calling nested scripts with `bash` makes the launch chain independent of their
executable bits. Users may continue to invoke either public wrapper with
`bash`.

## Cache behavior

The shared launcher retains an `HF_HUB_CACHE` environment override. The current
machine's complete SmolVLM tokenizer is in
`/home/typhon/.cache/huggingface/hub`, while the project checkpoint directory
does not contain it. This refactor will not silently move or download tokenizer
assets. Deployment instructions must continue to set the existing cache path
when `allow_download: false` is used.

## Error handling

- Missing or unreadable deployment configuration remains an exit-code 2 error.
- Missing token environment/file remains an exit-code 2 error without printing
  token contents.
- Missing Python remains surfaced by the final execution attempt, preserving
  the current launcher contract.
- `--check` remains non-invasive: it reports resolved config, token source,
  cache, Python, and Python entrypoint without connecting to the robot.

## Tests

Add black-box launcher coverage that first reproduces the current failure, then
protects the new boundary:

1. `bash start_frs.sh --check` succeeds even when the shared launcher is not
   executable and reports `deploy_frs.yaml`.
2. `bash start_vtsmolvla.sh --check` succeeds under the same permission
   condition and reports `deploy_smolvla_jax.yaml`.
3. Both wrappers forward a later explicit `--config` override.
4. Existing token, cache, Python-selection, and argument-validation tests run
   against the shared launcher without behavioral regression.

## Non-goals

- Changing FRS model parameters or checkpoint paths.
- Changing robot-server configuration or safety limits.
- Downloading tokenizer assets automatically.
- Changing the Python remote-client protocol or model-loading implementation.
