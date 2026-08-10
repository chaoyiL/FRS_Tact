# Project-Local Tokenizer Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the pinned SmolVLM tokenizer as a fourth resumable deployment asset so the default project-local Hugging Face cache supports fully offline FRS startup.

**Architecture:** `download.sh` downloads only eight tokenizer/config files into the standard Hugging Face cache below `${CHECKPOINT_ROOT}/model`, atomically points `refs/main` at the pinned commit, and skips only after a real offline `AutoTokenizer` load succeeds. The existing black-box fake downloader gains an isolated test-only `transformers` module so cache/ref behavior is exercised without network access; final production verification uses the installed Transformers package.

**Tech Stack:** Bash 4+, Hugging Face CLI/cache layout, Transformers `AutoTokenizer`, Python 3.12, pytest subprocess tests.

## Global Constraints

- Repository must be exactly `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Download only `config.json`, `tokenizer_config.json`, `tokenizer.json`, `special_tokens_map.json`, `added_tokens.json`, `chat_template.json`, `merges.txt`, and `vocab.json`; never download `model.safetensors` or wildcard repository content.
- Store the standard Hub cache only below `${CHECKPOINT_ROOT}/model`; never modify `~/.cache/huggingface`.
- Completeness requires exact `refs/main`, eight non-empty snapshot files, and a successful offline `AutoTokenizer.from_pretrained(..., local_files_only=True)`.
- Existing base, FRS, and encoder downloads remain independently skippable and retain their pinned repositories/revisions.
- Do not modify or stage either deployment YAML.

---

## File structure

- Modify `deploy_smolvla/scripts/download.sh`: tokenizer configuration, completeness validation, download/ref update, skip/failure output, and summary.
- Modify `tests/test_deploy_download_script.py`: exact-command and resumability black-box coverage with an isolated fake Transformers implementation.

### Task 1: Add the pinned tokenizer cache asset

**Files:**
- Modify: `deploy_smolvla/scripts/download.sh`
- Modify: `tests/test_deploy_download_script.py`

**Interfaces:**
- Consumes: `FRS_CHECKPOINT_ROOT`, `FRS_DOWNLOAD_UV`, `FRS_DOWNLOAD_PYTHON`, Hugging Face CLI `download --cache-dir`, and Transformers `AutoTokenizer.from_pretrained`.
- Produces: a standard cache at `${CHECKPOINT_ROOT}/model/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct`, an exact pinned `refs/main`, explicit tokenizer skip/failure output, and final `HF_HUB_CACHE` summary.

- [ ] **Step 1: Extend black-box fixtures and write failing tokenizer tests**

Add constants:

```python
TOKENIZER_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TOKENIZER_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
TOKENIZER_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "merges.txt",
    "vocab.json",
)
TOKENIZER_CACHE_ROOT = Path("checkpoints/model")
TOKENIZER_REPO_CACHE = TOKENIZER_CACHE_ROOT / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
```

In `make_project`, create `project/transformers/__init__.py` with this isolated
test double:

```python
from __future__ import annotations

import os
from pathlib import Path


class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, repo_id: str, *, local_files_only: bool):
        if not local_files_only:
            raise OSError("tokenizer must be offline")
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise OSError("HF_HUB_OFFLINE must be 1")
        if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise OSError("TRANSFORMERS_OFFLINE must be 1")
        cache_root = Path(os.environ["HF_HUB_CACHE"])
        repo_cache = cache_root / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
        revision = (repo_cache / "refs/main").read_text(encoding="utf-8").strip()
        tokenizer_json = repo_cache / "snapshots" / revision / "tokenizer.json"
        if tokenizer_json.read_text(encoding="utf-8") == "broken":
            raise OSError("invalid tokenizer")
        return cls()
```

Extend fake `uv` detection with a `tokenizer` asset for the exact repository.
Parse `--cache-dir`, create
`models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/<revision>`, and
write non-empty content for exactly `TOKENIZER_FILES`. Do not write `refs/main`;
the production script must own that step.

Insert this exact tokenizer call between base and FRS in `expected_calls`:

```python
[
    "run",
    "--no-sync",
    "hf",
    "download",
    TOKENIZER_REPO,
    "--revision",
    TOKENIZER_REVISION,
    "--include",
    *TOKENIZER_FILES,
    "--cache-dir",
    str(project / TOKENIZER_CACHE_ROOT),
]
```

Add `write_complete_tokenizer(project, revision=TOKENIZER_REVISION,
broken=False)` that creates the eight files and exact `refs/main`. Ensure every
pre-existing test that is not about tokenizer downloading calls this helper so
its original single-asset expectation remains meaningful.

Add focused tests with these assertions:

```python
def test_wrong_tokenizer_ref_refreshes_only_tokenizer(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_tokenizer(project, revision="wrong-revision")
    write_complete_frs(project)
    write_complete_encoder(project)
    result = run_download(project, env)
    assert result.returncode == 0, result.stderr
    assert read_calls(log_path) == [expected_calls(project)[1]]


def test_missing_tokenizer_file_refreshes_only_tokenizer(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_tokenizer(project)
    (project / TOKENIZER_REPO_CACHE / "snapshots" / TOKENIZER_REVISION / "vocab.json").unlink()
    write_complete_frs(project)
    write_complete_encoder(project)
    result = run_download(project, env)
    assert result.returncode == 0, result.stderr
    assert read_calls(log_path) == [expected_calls(project)[1]]


def test_broken_tokenizer_refreshes_only_tokenizer(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_tokenizer(project, broken=True)
    write_complete_frs(project)
    write_complete_encoder(project)
    result = run_download(project, env)
    assert result.returncode == 0, result.stderr
    assert read_calls(log_path) == [expected_calls(project)[1]]
```

Extend the all-complete test to require the tokenizer `skip:` line and exact
`HF_HUB_CACHE` summary. Extend delegated failure parametrization with asset
`tokenizer`, label `tokenizer download`, and directory
`TOKENIZER_CACHE_ROOT`; the error must also contain `TOKENIZER_REPO`.

- [ ] **Step 2: Run the downloader suite and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_deploy_download_script.py
```

Expected: failures show the fourth exact call, project-local cache/ref,
tokenizer skip/refresh, failure diagnostic, and summary behavior are absent.

- [ ] **Step 3: Add pinned tokenizer configuration and completeness validation**

At the top of `download.sh`, add:

```bash
TOKENIZER_REPO="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TOKENIZER_REVISION="7b375e1b73b11138ff12fe22c8f2822d8fe03467"
TOKENIZER_FILES=(
    config.json
    tokenizer_config.json
    tokenizer.json
    special_tokens_map.json
    added_tokens.json
    chat_template.json
    merges.txt
    vocab.json
)
TOKENIZER_CACHE_ROOT="${CHECKPOINT_ROOT}/model"
TOKENIZER_REPO_CACHE="${TOKENIZER_CACHE_ROOT}/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
```

Define `tokenizer_complete` using the selected Python command. Pass the cache,
repo cache, repo ID, revision, and file list as arguments. Its Python body must:

```python
cache_root = Path(sys.argv[1]).resolve()
repo_cache = Path(sys.argv[2]).resolve()
repo_id = sys.argv[3]
revision = sys.argv[4]
required = tuple(sys.argv[5:])
try:
    if (repo_cache / "refs/main").read_text(encoding="utf-8").strip() != revision:
        raise ValueError("tokenizer revision mismatch")
    snapshot = repo_cache / "snapshots" / revision
    if any(not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0 for name in required):
        raise ValueError("tokenizer cache is incomplete")
    os.environ["HF_HUB_CACHE"] = str(cache_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
except (ImportError, OSError, ValueError, TypeError):
    raise SystemExit(1)
```

Define `write_tokenizer_ref` with Python `Path.mkdir`, a sibling temporary
file, and `Path.replace` so `refs/main` changes atomically only after the
pinned download succeeds.

- [ ] **Step 4: Add tokenizer download, skip, validation, and summary**

Insert this independent asset between base and FRS:

```bash
if tokenizer_complete; then
    echo "skip: tokenizer cache: ${TOKENIZER_CACHE_ROOT}"
else
    if ! "${UV_BIN}" run --no-sync hf download "${TOKENIZER_REPO}" \
        --revision "${TOKENIZER_REVISION}" --include "${TOKENIZER_FILES[@]}" \
        --cache-dir "${TOKENIZER_CACHE_ROOT}"; then
        echo "tokenizer download failed: ${TOKENIZER_REPO} -> ${TOKENIZER_CACHE_ROOT}" >&2
        exit 1
    fi
    write_tokenizer_ref
    tokenizer_complete || {
        echo "tokenizer cache failed validation after download: ${TOKENIZER_CACHE_ROOT}" >&2
        exit 1
    }
fi
```

Append this exact summary entry:

```bash
echo "  HF_HUB_CACHE: ${TOKENIZER_CACHE_ROOT}"
```

- [ ] **Step 5: Run focused and deployment regressions and verify GREEN**

Run:

```bash
bash -n deploy_smolvla/scripts/download.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_deploy_download_script.py \
  tests/test_download_ckpt.py \
  tests/jax/test_deploy_launcher.py \
  tests/jax/test_frs_deployment.py \
  tests/jax/test_tactile_integration.py
```

Expected: shell syntax passes and all selected tests pass with no warnings.

- [ ] **Step 6: Commit only downloader implementation and tests**

```bash
git add deploy_smolvla/scripts/download.sh tests/test_deploy_download_script.py
git diff --cached --check
git commit -m "fix: cache deployment tokenizer locally"
```

Confirm both deployment YAML files remain unstaged.

- [ ] **Step 7: Execute the real downloader and prove offline startup resolution**

Run with network approval:

```bash
bash deploy_smolvla/scripts/download.sh
```

Expected: the three existing checkpoint assets print `skip:` and the tokenizer
downloads only the eight allowlisted files into the project cache.

Run immediately again. Expected: all four assets print `skip:`.

Then run the production offline smoke test with no user-cache override:

```bash
HF_HUB_CACHE=/home/typhon/FRS_Tact/checkpoints/model \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -c \
"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', local_files_only=True); print('project-tokenizer-cache-ok')"

bash deploy_smolvla/scripts/start_frs.sh --check
```

Expected: tokenizer smoke test prints `project-tokenizer-cache-ok`; launcher
check reports `model_cache=/home/typhon/FRS_Tact/checkpoints/model` and exits 0.
