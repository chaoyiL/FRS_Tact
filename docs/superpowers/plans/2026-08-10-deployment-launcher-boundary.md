# Deployment Launcher Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the misleading FRS-to-VT launcher chain with one model-neutral shared launcher and two explicit model entrypoints that work regardless of nested script executable bits.

**Architecture:** `start_remote_client.sh` contains the common environment, token, cache, argument, check, and Python-launch logic. `start_frs.sh` and `start_vtsmolvla.sh` are thin Bash wrappers that choose their own default YAML, honor `FRS_DEPLOY_CONFIG`, and forward later command-line overrides.

**Tech Stack:** Bash 4+, Python 3.12, pytest, subprocess black-box tests.

## Global Constraints

- Do not change either deployment YAML or any checkpoint path/model parameter.
- Nested launcher calls must use `bash` and must not depend on executable mode bits.
- `start_frs.sh` must default to `deploy_frs.yaml`; `start_vtsmolvla.sh` must default to `deploy_smolvla_jax.yaml`.
- Preserve `VB_ROBOT_TOKEN`, `VB3_TOKEN_FILE`, `FRS_DEPLOY_CONFIG`, `FRS_PYTHON`, `VB3_PYTHON`, `HF_HUB_CACHE`, `--check`, `--config`, and current exit-code behavior.
- Never print token contents.

---

## File structure

- Create `deploy_smolvla/scripts/start_remote_client.sh`: model-neutral shared launcher.
- Modify `deploy_smolvla/scripts/start_frs.sh`: FRS-specific configuration wrapper only.
- Modify `deploy_smolvla/scripts/start_vtsmolvla.sh`: VT-SmolVLA-specific configuration wrapper only.
- Modify `tests/jax/test_deploy_launcher.py`: shared-launcher regression coverage and black-box wrapper tests.

### Task 1: Extract the shared launcher and protect both public entrypoints

**Files:**
- Create: `deploy_smolvla/scripts/start_remote_client.sh`
- Modify: `deploy_smolvla/scripts/start_frs.sh:1-6`
- Modify: `deploy_smolvla/scripts/start_vtsmolvla.sh:1-98`
- Test: `tests/jax/test_deploy_launcher.py`

**Interfaces:**
- Consumes: environment variables `VB_ROBOT_TOKEN`, `VB3_TOKEN_FILE`, `FRS_DEPLOY_CONFIG`, `FRS_PYTHON`, `VB3_PYTHON`, and `HF_HUB_CACHE`; CLI options `--check` and `--config PATH`.
- Produces: `bash deploy_smolvla/scripts/start_frs.sh [--check] [--config PATH]` and `bash deploy_smolvla/scripts/start_vtsmolvla.sh [--check] [--config PATH]`, both eventually executing `python -m deploy_smolvla.remote_client --config PATH`.

- [ ] **Step 1: Write failing black-box tests for the wrapper boundary**

Update launcher constants and the isolated-project copy helper so existing common behavior tests target the new shared filename, then add wrapper tests. The relevant structure must be:

```python
SHARED_LAUNCHER = ROOT / "deploy_smolvla" / "scripts" / "start_remote_client.sh"
FRS_LAUNCHER = ROOT / "deploy_smolvla" / "scripts" / "start_frs.sh"
VT_LAUNCHER = ROOT / "deploy_smolvla" / "scripts" / "start_vtsmolvla.sh"
FRS_CONFIG = ROOT / "deploy_smolvla" / "configs" / "deploy_frs.yaml"
DEFAULT_CONFIG = ROOT / "deploy_smolvla" / "configs" / "deploy_smolvla_jax.yaml"


def _run_wrapper_check(wrapper: Path, *, extra_args: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VB_ROBOT_TOKEN"] = "test-token"
    env["HF_HUB_CACHE"] = str(ROOT / "checkpoints" / "model")
    return subprocess.run(
        ["bash", str(wrapper), "--check", *extra_args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_frs_wrapper_uses_frs_config_without_executable_nested_script() -> None:
    result = _run_wrapper_check(FRS_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert f"config={FRS_CONFIG}" in result.stdout


def test_vt_wrapper_uses_vt_config_without_executable_nested_script() -> None:
    result = _run_wrapper_check(VT_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert f"config={DEFAULT_CONFIG}" in result.stdout


def test_wrapper_allows_later_explicit_config_override() -> None:
    result = _run_wrapper_check(
        FRS_LAUNCHER,
        extra_args=("--config", str(DEFAULT_CONFIG)),
    )

    assert result.returncode == 0, result.stderr
    assert f"config={DEFAULT_CONFIG}" in result.stdout
```

For existing shared-behavior helpers, invoke:

```python
["bash", str(SHARED_LAUNCHER), "--config", str(DEFAULT_CONFIG)]
```

Copy `start_remote_client.sh` and `deploy_smolvla_jax.yaml` into temporary test projects. Explicitly set the copied shared script to mode `0644` before running wrapper coverage so the regression cannot be hidden by a local executable bit.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/jax/test_deploy_launcher.py
```

Expected: the new tests fail because `start_remote_client.sh` does not exist and the FRS wrapper still directly executes a non-executable VT launcher.

- [ ] **Step 3: Implement the model-neutral shared launcher**

Move the common body from `start_vtsmolvla.sh` into `start_remote_client.sh`. Initialize configuration without a model-specific default:

```bash
CONFIG=""
```

Keep the existing option parser. After parsing, reject an omitted config explicitly:

```bash
if [[ -z "${CONFIG}" ]]; then
    echo "--config is required" >&2
    exit 2
fi
```

Use a model-neutral usage line:

```text
Usage: bash ./deploy_smolvla/scripts/start_remote_client.sh --config PATH [--check]
```

Retain the existing root, checkpoint-directory, cache, token, Python selection,
`--check`, `PYTHONPATH`, and final `exec ... remote_client` logic verbatim except
for the model-neutral usage/config initialization.

- [ ] **Step 4: Replace both public scripts with explicit wrappers**

`start_frs.sh` must be:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${FRS_DEPLOY_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_frs.yaml}"
exec bash "${ROOT}/deploy_smolvla/scripts/start_remote_client.sh" \
    --config "${CONFIG}" "$@"
```

`start_vtsmolvla.sh` must be:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${FRS_DEPLOY_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_smolvla_jax.yaml}"
exec bash "${ROOT}/deploy_smolvla/scripts/start_remote_client.sh" \
    --config "${CONFIG}" "$@"
```

Do not chmod the shared script as part of the fix; explicit Bash invocation is
the portability guarantee.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
bash -n deploy_smolvla/scripts/start_remote_client.sh
bash -n deploy_smolvla/scripts/start_frs.sh
bash -n deploy_smolvla/scripts/start_vtsmolvla.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/jax/test_deploy_launcher.py
```

Expected: all syntax checks pass and `tests/jax/test_deploy_launcher.py` is green.

- [ ] **Step 6: Run deployment regression tests and the real `--check` command**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/jax/test_deploy_launcher.py \
  tests/jax/test_frs_deployment.py \
  tests/jax/test_tactile_integration.py

HF_HUB_CACHE=/home/typhon/.cache/huggingface/hub \
  bash deploy_smolvla/scripts/start_frs.sh --check
```

Expected: all tests pass; the command exits 0 and reports
`deploy_smolvla/configs/deploy_frs.yaml`, the configured token source, the user
Hugging Face cache, the selected Python, and
`entrypoint=deploy_smolvla.remote_client` without printing a token.

- [ ] **Step 7: Commit only the launcher implementation and tests**

```bash
git add \
  deploy_smolvla/scripts/start_remote_client.sh \
  deploy_smolvla/scripts/start_frs.sh \
  deploy_smolvla/scripts/start_vtsmolvla.sh \
  tests/jax/test_deploy_launcher.py
git diff --cached --check
git commit -m "fix: separate deployment launcher entrypoints"
```

Verify that `deploy_smolvla/configs/deploy_frs.yaml` and
`deploy_smolvla/configs/deploy_smolvla_jax.yaml` remain unstaged.
