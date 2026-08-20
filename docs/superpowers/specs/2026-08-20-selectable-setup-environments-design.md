# Selectable Setup Environments Design

## Goal

Allow callers of `scripts/setup_env.sh` to select which existing uv environment is installed and
verified, while preserving the current no-argument behavior and environment-directory overrides.

## Command-line interface

The script accepts exactly one of these optional selectors:

```bash
# Backward-compatible default: root plus Pi0.5 deployment.
bash scripts/setup_env.sh

# Root SmolVLA/VT-SmolVLA/SmolVLA-FRS environment only.
bash scripts/setup_env.sh --root

# Pi0.5 deployment environment only.
bash scripts/setup_env.sh --pi05_deploy

# Print usage without side effects.
bash scripts/setup_env.sh --help
```

`-h` is an alias for `--help`. Unknown arguments, positional arguments, missing values, and the
combination `--root --pi05_deploy` are usage errors. Usage errors must be detected before package
installation, directory creation, `.bashrc` updates, environment synchronization, or verification.
They print a concise error and usage text to standard error and exit nonzero.

The two selectors are intentionally mutually exclusive. Installing both is already represented by
the no-argument command, which remains compatible with existing documentation and automation.

## Environment paths and precedence

Selection determines which project is synchronized and verified; it does not replace the existing
path configuration:

- `FRS_VENV_DIR` continues to override the root environment directory.
- `PI05_VENV_DIR` continues to override the Pi0.5 deployment environment directory.
- Existing platform defaults remain unchanged.
- Relative overrides continue to resolve from the repository root.
- Root and Pi0.5 deployment environments must remain distinct.

The generated `.env.frs` retains its current complete contract and records both configured paths,
including `UV_PROJECT_ENVIRONMENT`, `PI05_PYTHON`, and `PI05_FRS_PYTHON`. A selector guarantees only
that its selected environment was synchronized and verified; it does not claim that the unselected
environment exists.

## Execution flow

Argument parsing runs first and produces one internal mode: `all`, `root`, or `pi05_deploy`.

Common setup remains shared for all installation modes:

1. Validate arguments and configured environment targets.
2. Install/check common system dependencies and uv.
3. Configure uv storage and runtime cache directories.
4. Ensure Python 3.12 is available through uv.

Environment-specific work is split into independently callable operations:

- Root sync uses the repository `uv.lock` and `UV_PROJECT_ENVIRONMENT=$VENV_DIR`.
- Pi0.5 deployment sync uses `deploy_pi05/uv.lock`, `--project deploy_pi05`, and
  `UV_PROJECT_ENVIRONMENT=$PI05_VENV_DIR`.
- Root verification imports the root training dependencies and checks root PyTorch/JAX GPU access.
- Pi0.5 verification imports its private deployment dependencies and checks Pi0.5 JAX GPU access.

Mode `all` executes root then Pi0.5 work, matching the existing order. Mode `root` skips every
Pi0.5 sync and import/GPU verification. Mode `pi05_deploy` skips every root sync and import/GPU
verification. `.env.frs` is written after the selected synchronization succeeds.

The final summary reports the selected mode and only claims successful installation for selected
environments. It may still show the configured path for the unselected environment as configuration
information, clearly labeled as not installed by this invocation.

## Error handling

- Argument errors fail before side effects.
- A selected project must have both `pyproject.toml` and `uv.lock` before synchronization begins.
- Environment-path collision is rejected before any uv command.
- Failures retain the shell's nonzero status through `set -Eeuo pipefail`.
- `--help` exits zero and performs no installation, file creation, or environment verification.

## Testing

Tests extend `tests/test_setup_env_dual_environment.py` and source the script with fake functions and
a fake uv executable, so no real packages or system resources are changed.

Coverage includes:

1. No arguments preserve the existing two-lock synchronization behavior.
2. `--root` invokes only root synchronization and root verification.
3. `--pi05_deploy` invokes only Pi0.5 synchronization and Pi0.5 verification.
4. Existing `FRS_VENV_DIR` and `PI05_VENV_DIR` overrides still determine target directories.
5. `--help` returns success with zero side effects.
6. Unknown arguments and conflicting selectors fail before any side-effect function is called.
7. Existing distinct-directory and relative-path tests continue to pass.

Implementation follows test-driven development: each new entry-point behavior is first captured by
a failing test, then the minimum shell refactoring is added to make it pass.
