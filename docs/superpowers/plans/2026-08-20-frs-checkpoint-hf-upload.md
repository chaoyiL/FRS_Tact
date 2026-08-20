# FRS Checkpoint Hugging Face Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable Bash command that uploads an explicitly selected FRS checkpoint and its training PNGs to a public Hugging Face model repository.

**Architecture:** `scripts/upload_frs_ckpt.sh` performs local validation, then delegates authentication, repository creation, and two uploads to the existing `uv run --no-sync hf` CLI. `tests/test_upload_frs_ckpt_script.py` executes the real Bash script against a fake `uv` binary so command construction and failure boundaries are verified without network access.

**Tech Stack:** Bash 4+, `uv`, Hugging Face `hf` CLI, Python 3.12, pytest.

## Global Constraints

- Interface: `bash scripts/upload_frs_ckpt.sh OWNER/REPO /path/to/checkpoint [--figures-dir PATH]`.
- The repository is public by default; never pass `--private`.
- Checkpoint files upload to repository root; top-level PNGs upload to `figures/`.
- The script must reject symlink checkpoint/figure directories and unsafe `params_file` references.
- Do not modify or delete local checkpoint files.

---

### Task 1: Successful upload interface and HF commands

**Files:**
- Create: `scripts/upload_frs_ckpt.sh`
- Create: `tests/test_upload_frs_ckpt_script.py`

**Interfaces:**
- Consumes: positional `repo_id`, positional `checkpoint_dir`, optional `--figures-dir`.
- Produces: public model repository with checkpoint at `.` and PNGs at `figures/`; exits `0` on success.

- [ ] **Step 1: Write failing success-path tests**

Create helpers that copy `scripts/upload_frs_ckpt.sh` into a temporary project, create a checkpoint with `checkpoint.json` referencing `params-test.npz`, create PNGs, install a fake `uv` that appends each argument to `$UV_CALL_LOG`, and run the script. Assert explicit `--figures-dir` is honored, the default is the checkpoint parent, the repo-create invocation omits `--private`, and upload destinations are `.` and `figures`.

```python
def test_uploads_selected_checkpoint_and_explicit_figures(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path / "run" / "best")
    figures = tmp_path / "plots"
    figures.mkdir()
    (figures / "training_overview.png").write_bytes(b"png")
    result, calls = run_script(
        tmp_path,
        "KaiyueChen/frs-best",
        checkpoint,
        "--figures-dir",
        figures,
    )
    assert result.returncode == 0, result.stderr
    assert ["hf", "repo", "create", "KaiyueChen/frs-best"] in calls[1]
    assert "--private" not in calls[1]
    assert calls[2][-3:] == [str(checkpoint.resolve()), ".", "--repo-type", "model"][-3:]
    assert str(figures.resolve()) in calls[3]
    assert "figures" in calls[3]
    assert "*.png" in calls[3]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run --no-sync pytest -q tests/test_upload_frs_ckpt_script.py`

Expected: FAIL because `scripts/upload_frs_ckpt.sh` does not exist.

- [ ] **Step 3: Implement argument parsing and successful upload flow**

Create a strict Bash script with `usage`, `log`, and `fail`; validate `OWNER/REPO`; resolve directories using `cd -- "$path" && pwd -P`; find `uv`; call:

```bash
"${UV_BIN}" run --no-sync hf auth whoami
"${UV_BIN}" run --no-sync hf repo create "${repo_id}" --repo-type model --exist-ok
"${UV_BIN}" run --no-sync hf upload "${repo_id}" "${checkpoint_dir}" . \
    --repo-type model --commit-message "Upload FRS checkpoint"
"${UV_BIN}" run --no-sync hf upload "${repo_id}" "${figures_dir}" figures \
    --repo-type model --include "*.png" --commit-message "Upload FRS training figures"
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `uv run --no-sync pytest -q tests/test_upload_frs_ckpt_script.py`

Expected: all success-path tests PASS.

### Task 2: Local safety validation and error behavior

**Files:**
- Modify: `tests/test_upload_frs_ckpt_script.py`
- Modify: `scripts/upload_frs_ckpt.sh`

**Interfaces:**
- Consumes: checkpoint metadata field `params_file: str`.
- Produces: nonzero exit before any HF command for invalid local input.

- [ ] **Step 1: Write failing validation tests**

Add parameterized tests for invalid repository IDs, missing `checkpoint.json`, missing parameter file, no top-level PNG, directory symlinks, absolute `params_file`, and `../` traversal. Assert the fake call log is absent or empty. Add a help test that exits `0` without requiring `uv`.

```python
@pytest.mark.parametrize("params_file", ["../outside.npz", "/tmp/outside.npz"])
def test_rejects_unsafe_params_file(tmp_path: Path, params_file: str) -> None:
    checkpoint = make_checkpoint(tmp_path / "run" / "best", params_file=params_file)
    result, calls = run_script(tmp_path, "KaiyueChen/frs-best", checkpoint)
    assert result.returncode != 0
    assert "params_file" in result.stderr
    assert calls == []
```

- [ ] **Step 2: Run validation tests and verify RED**

Run: `uv run --no-sync pytest -q tests/test_upload_frs_ckpt_script.py`

Expected: new validation cases FAIL because the minimal script accepts invalid inputs.

- [ ] **Step 3: Implement minimal local validation**

Use `test -L` before resolving paths. Parse `params_file` with `uv run --no-sync python - checkpoint.json checkpoint_dir`, require a nonempty relative filename, resolve it, require `resolved.parent == checkpoint_dir` and `resolved.is_file()`, then print the validated path. Use `find "$figures_dir" -maxdepth 1 -type f -name '*.png' -print -quit` and fail when empty.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `uv run --no-sync pytest -q tests/test_upload_frs_ckpt_script.py`

Expected: all tests PASS.

- [ ] **Step 5: Run regression and static checks**

Run:

```bash
uv run --no-sync pytest -q tests/test_upload_frs_ckpt_script.py tests/test_download_ckpt.py
bash -n scripts/upload_frs_ckpt.sh
uv run --no-sync ruff check tests/test_upload_frs_ckpt_script.py
git diff --check
```

Expected: all commands exit `0`; no changes appear in `deploy_pi05/configs/deploy_pi05.yaml` beyond the user's pre-existing modification.

- [ ] **Step 6: Commit implementation**

```bash
git add scripts/upload_frs_ckpt.sh tests/test_upload_frs_ckpt_script.py
git commit -m "feat: add FRS checkpoint upload script"
```
