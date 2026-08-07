# Local Checkpoints Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put all future deployment-time VLA downloads under `checkpoints/model`, encoder downloads under `checkpoints/encoder`, and ignore the complete local checkpoint tree in Git.

**Architecture:** The two deployment shell entry points set the same default `HF_HUB_CACHE` before Python starts while preserving an explicit caller override. The encoder downloader uses a repository-relative default output path; Git ignores the parent checkpoint tree.

**Tech Stack:** Bash, Python 3.12, Hugging Face Hub cache environment variables, pytest, Git ignore rules.

## Global Constraints

- Do not move or delete existing files under `/home/typhon/.cache/huggingface`.
- Preserve an explicitly supplied `HF_HUB_CACHE`.
- Preserve the existing `--output-dir` encoder override.
- Do not modify the user's existing `configs/deploy_smolvla_jax.yaml` changes or unrelated deleted documentation.
- Use `<project>/checkpoints/model` for future deployment VLA downloads.
- Use `<project>/checkpoints/encoder/encoder_ckpt_06` for the default encoder download.

---

### Task 1: Project-local VLA cache

**Files:**
- Modify: `.gitignore`
- Modify: `deploy_smolvla/start_vtsmolvla.sh`
- Modify: `deploy_smolvla/run_client.sh`
- Modify: `tests/jax/test_deploy_launcher.py`

**Interfaces:**
- Consumes: the standard `HF_HUB_CACHE` environment variable.
- Produces: default `HF_HUB_CACHE=<project>/checkpoints/model`, local `checkpoints/model` and `checkpoints/encoder` directories, and `model_cache=...` in `--check` output.

- [ ] **Step 1: Write failing launcher tests**

Extend the check helper with an optional cache override, clear inherited cache variables for the default case, and add assertions equivalent to:

```python
DEFAULT_MODEL_CACHE = ROOT / "checkpoints" / "model"

def test_launcher_uses_project_model_cache_by_default(tmp_path: Path) -> None:
    result = _run_check(token_file=tmp_path / "missing", token="secret")
    assert result.returncode == 0, result.stderr
    assert f"model_cache={DEFAULT_MODEL_CACHE}" in result.stdout

def test_launcher_preserves_explicit_hf_hub_cache(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    result = _run_check(token_file=tmp_path / "missing", token="secret", hub_cache=cache)
    assert result.returncode == 0, result.stderr
    assert f"model_cache={cache}" in result.stdout
    assert cache.is_dir()
```

Add a direct `run_client.sh` test using a temporary executable through `FRS_PYTHON`; the executable prints `HF_HUB_CACHE`, proving the direct entry point applies the same default without loading a model.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
./.venv/bin/pytest -q tests/jax/test_deploy_launcher.py
```

Expected: new assertions fail because neither launcher currently sets or reports `HF_HUB_CACHE`.

- [ ] **Step 3: Implement the minimal launcher behavior**

In both shell entry points, after resolving `ROOT`, add behavior equivalent to:

```bash
CHECKPOINTS_DIR="${ROOT}/checkpoints"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CHECKPOINTS_DIR}/model}"
mkdir -p "${HF_HUB_CACHE}" "${CHECKPOINTS_DIR}/encoder"
```

In the check-only output, add:

```bash
echo "model_cache=${HF_HUB_CACHE}"
```

Append this anchored ignore rule to `.gitignore`:

```gitignore
/checkpoints/
```

- [ ] **Step 4: Run focused tests and Git-ignore verification**

Run:

```bash
./.venv/bin/pytest -q tests/jax/test_deploy_launcher.py
git check-ignore -v checkpoints/model checkpoints/encoder
```

Expected: launcher tests pass and both directories match `.gitignore`'s `/checkpoints/` rule.

- [ ] **Step 5: Commit Task 1**

```bash
git add .gitignore deploy_smolvla/start_vtsmolvla.sh deploy_smolvla/run_client.sh tests/jax/test_deploy_launcher.py
git commit -m "feat: store VLA downloads in project checkpoints"
```

### Task 2: Project-local encoder downloads

**Files:**
- Modify: `download_ckpt.py`
- Modify: `scripts/download_ckpt.sh`
- Create: `tests/test_download_ckpt.py`

**Interfaces:**
- Consumes: `download_ckpt.py --output-dir PATH` when explicitly provided.
- Produces: `DEFAULT_OUTPUT_DIR=<project>/checkpoints/encoder/encoder_ckpt_06` when no override is provided.

- [ ] **Step 1: Write the failing encoder path tests**

Create `tests/test_download_ckpt.py` with assertions equivalent to:

```python
from pathlib import Path

import download_ckpt

ROOT = Path(__file__).resolve().parents[1]

def test_default_encoder_output_is_project_local() -> None:
    assert download_ckpt.DEFAULT_OUTPUT_DIR == ROOT / "checkpoints" / "encoder" / "encoder_ckpt_06"

def test_output_dir_override_is_preserved(tmp_path: Path) -> None:
    args = download_ckpt.parse_args(["--output-dir", str(tmp_path / "custom")])
    assert args.output_dir == tmp_path / "custom"
```

Update `parse_args` to accept `argv: Sequence[str] | None = None` and call `parser.parse_args(argv)` so the override can be tested without modifying process arguments.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
./.venv/bin/pytest -q tests/test_download_ckpt.py
```

Expected: default-path and `parse_args(argv)` tests fail against the current `/workspace` constant and zero-argument parser function.

- [ ] **Step 3: Implement the encoder default and wrapper text**

In `download_ckpt.py`, add:

```python
from collections.abc import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "encoder" / "encoder_ckpt_06"
```

Use `parse_args(argv: Sequence[str] | None = None)` and `parser.parse_args(argv)`. In `scripts/download_ckpt.sh`, report repository `liuchaoyi/encoder_ckpt_06` and `${PROJECT_ROOT}/checkpoints/encoder/encoder_ckpt_06` so the wrapper matches the Python defaults.

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
./.venv/bin/pytest -q tests/test_download_ckpt.py tests/jax/test_deploy_launcher.py
./.venv/bin/pytest -q tests/jax/test_tactile_integration.py
```

Expected: new focused tests pass. The tactile integration suite may retain the one pre-existing failure caused solely by the user's uncommitted `data_type: vision` change; no additional failures are allowed.

- [ ] **Step 5: Commit Task 2**

```bash
git add download_ckpt.py scripts/download_ckpt.sh tests/test_download_ckpt.py
git commit -m "feat: store encoder downloads in project checkpoints"
```

