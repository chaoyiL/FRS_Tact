# FRS Package Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every FRS-owned training file into `train_frs/`, preserve every existing basename and runtime behavior, delete old paths without forwarding modules, and update every repository consumer.

**Architecture:** `train_frs` becomes the owner of the tactile flow steering model, trainer, FRS cache preparation CLIs, evaluation helpers, YAML, and launcher. Shared SmolVLA, tactile encoder, LeRobot, cache primitives, PEFT merge, and tactile embedding precomputation remain in their existing shared locations and are imported explicitly by the new package.

**Tech Stack:** Python 3.12, JAX, Flax NNX, Optax, PyTorch DataLoader, PyYAML, Bash, pytest, uv.

## Global Constraints

- Preserve the basename of every migrated source file; only its owning directory and import path may change.
- Delete old FRS paths and do not add compatibility forwarding modules.
- Do not change training algorithms, YAML fields/defaults, cache formats, checkpoint formats, resume behavior, or external resource paths.
- Do not copy `tools/precompute_tactile_embeddings.py`, `tools/merge_smolvla_peft_to_jax.py`, `tactile_encoder/`, `src/lerobot/`, `modalities_eval/`, or shared `utils/` modules into `train_frs`.
- Update static imports, dynamic imports, Shell commands, YAML examples, active documentation, packaging, and tests across the whole repository.
- Preserve unrelated pre-existing worktree changes; stage and commit only files belonging to each task.
- Use `apply_patch` move directives and patches for tracked-file relocation and edits.

---

### Task 1: Move the core tactile FRS package

**Files:**
- Create: `tests/train_frs/__init__.py`
- Create: `tests/train_frs/test_package_layout.py`
- Move: `tactile_flow_steering/__init__.py` → `train_frs/__init__.py`
- Move: `tactile_flow_steering/train.py` → `train_frs/train.py`
- Move: `tactile_flow_steering/evaluate.py` → `train_frs/evaluate.py`
- Move: `tactile_flow_steering/plot_history.py` → `train_frs/plot_history.py`
- Move: `tactile_flow_steering/utils/*.py` → `train_frs/utils/*.py`
- Move: `tactile_flow_steering/tests/test_data.py` → `tests/train_frs/test_data.py`
- Move: `tactile_flow_steering/tests/test_model.py` → `tests/train_frs/test_model.py`
- Delete: `tactile_flow_steering/tests/__init__.py`

**Interfaces:**
- Consumes: shared `utils.cache`, `lerobot`, and `tactile_encoder` imports exactly as before.
- Produces: `train_frs.train.train_decoder`, `train_frs.utils.model.TactileConditionedFlowDecoder`, `train_frs.utils.data.CachedTactileEmbeddingBatches`, and the same checkpoint/evaluation APIs under `train_frs.utils`.

- [ ] **Step 1: Write the failing package-layout test**

Create `tests/train_frs/test_package_layout.py` with:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_core_frs_files_live_in_train_frs() -> None:
    expected = {
        "train.py",
        "evaluate.py",
        "plot_history.py",
        "utils/checkpoint.py",
        "utils/data.py",
        "utils/history_plot.py",
        "utils/integration.py",
        "utils/metrics.py",
        "utils/model.py",
        "utils/mp_batches.py",
        "utils/visualize.py",
        "utils/window_io.py",
    }
    missing = sorted(path for path in expected if not (ROOT / "train_frs" / path).is_file())
    assert missing == []


def test_new_core_modules_import_and_old_package_is_gone() -> None:
    assert importlib.util.find_spec("train_frs.train") is not None
    assert importlib.util.find_spec("train_frs.utils.model") is not None
    assert importlib.util.find_spec("tactile_flow_steering") is None
```

Add an empty `tests/train_frs/__init__.py`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run --no-sync pytest tests/train_frs/test_package_layout.py -q
```

Expected: failure lists missing `train_frs/train.py` and/or still finds `tactile_flow_steering`.

- [ ] **Step 3: Move the core files without renaming them**

Use `apply_patch` move directives for every production and test file listed above. Keep file contents unchanged except for these exact import/docstring replacements throughout the moved files and tests:

```text
tactile_flow_steering.train              -> train_frs.train
tactile_flow_steering.utils.checkpoint   -> train_frs.utils.checkpoint
tactile_flow_steering.utils.data         -> train_frs.utils.data
tactile_flow_steering.utils.history_plot -> train_frs.utils.history_plot
tactile_flow_steering.utils.integration  -> train_frs.utils.integration
tactile_flow_steering.utils.metrics      -> train_frs.utils.metrics
tactile_flow_steering.utils.model        -> train_frs.utils.model
tactile_flow_steering.utils.mp_batches   -> train_frs.utils.mp_batches
tactile_flow_steering.utils.visualize    -> train_frs.utils.visualize
tactile_flow_steering.utils.window_io    -> train_frs.utils.window_io
```

The dynamic worker import in `mp_batches.py` must be exactly:

```python
window_io = importlib.import_module("train_frs.utils.window_io")
```

Update the package docstring example to `train_frs.utils.model` and the plot-history help text to say `train_frs.train`.

- [ ] **Step 4: Run focused core tests and verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/train_frs/test_package_layout.py tests/train_frs/test_data.py tests/train_frs/test_model.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the core package move**

```bash
git add train_frs tests/train_frs tactile_flow_steering
git commit -m "refactor: move FRS core into train_frs"
```

---

### Task 2: Move FRS entrypoints and action-cache preparation

**Files:**
- Modify: `tests/train_frs/test_package_layout.py`
- Move: `tools/train_frs.py` → `train_frs/train_frs.py`
- Move: `prepare.py` → `train_frs/prepare.py`
- Move: `tools/prepare_frs_caches.py` → `train_frs/prepare_frs_caches.py`
- Move: `tools/compare_frs_reverse_solvers.py` → `train_frs/compare_frs_reverse_solvers.py`
- Modify: `tests/flow_decoder/test_frs_safety.py`
- Modify: `utils/cache.py`

**Interfaces:**
- Consumes: Task 1's `train_frs.train.train_decoder`; shared `utils.cache`, `utils.integration`, `utils.source_model`, `modalities_eval.utils`, and `lerobot` modules.
- Produces: `python -m train_frs.train_frs`, `python -m train_frs.prepare_frs_caches`, `python -m train_frs.compare_frs_reverse_solvers`, and `train_frs.prepare.prepare_cache`.

- [ ] **Step 1: Extend the failing path-contract test**

Append these tests to `tests/train_frs/test_package_layout.py`:

```python
import subprocess
import sys


def test_frs_entrypoints_live_in_package() -> None:
    expected = {
        "train_frs.py",
        "prepare.py",
        "prepare_frs_caches.py",
        "compare_frs_reverse_solvers.py",
    }
    missing = sorted(path for path in expected if not (ROOT / "train_frs" / path).is_file())
    assert missing == []
    assert not (ROOT / "tools" / "train_frs.py").exists()
    assert not (ROOT / "tools" / "prepare_frs_caches.py").exists()
    assert not (ROOT / "tools" / "compare_frs_reverse_solvers.py").exists()
    assert not (ROOT / "prepare.py").exists()


def test_train_frs_module_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "train_frs.train_frs", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
uv run --no-sync pytest tests/train_frs/test_package_layout.py -q
```

Expected: `test_frs_entrypoints_live_in_package` fails because the entrypoints are still in old locations.

- [ ] **Step 3: Move entrypoints and update their exact imports/default paths**

Use `apply_patch` move directives and make these replacements:

```python
# train_frs/train_frs.py
ROOT = Path(__file__).resolve().parents[1]
from train_frs.train import train_decoder
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_frs.yaml"

# train_frs/prepare_frs_caches.py
ROOT = Path(__file__).resolve().parents[1]
from train_frs.prepare import prepare_cache
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_frs.yaml"

# train_frs/compare_frs_reverse_solvers.py
ROOT = Path(__file__).resolve().parents[1]
from train_frs.prepare import prepare_cache
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_frs.yaml"
```

In `tests/flow_decoder/test_frs_safety.py`, replace imports with:

```python
from train_frs.compare_frs_reverse_solvers import mean_ratio, summarize_inversion_mse
from train_frs.prepare import (
    _ActionCacheRecordDataset,
    _create_batch_loader,
    _prepare_observation_batch,
    _require_finite_cache_batch,
)
from train_frs.train import _existing_run_artifacts, _validate_resume_cache
from train_frs.train_frs import resolve_resume_mode
```

In `utils/cache.py`, change the recovery hint from `Resume prepare.py first.` to `Resume train_frs/prepare.py first.` No cache logic changes.

- [ ] **Step 4: Run entrypoint and safety tests and verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/train_frs/test_package_layout.py tests/flow_decoder/test_frs_safety.py -q
uv run --no-sync python -m train_frs.prepare_frs_caches --help
uv run --no-sync python -m train_frs.compare_frs_reverse_solvers --help
```

Expected: tests pass and both CLI commands exit 0 with help text.

- [ ] **Step 5: Commit the entrypoint migration**

```bash
git add train_frs tools prepare.py tests/flow_decoder/test_frs_safety.py utils/cache.py tests/train_frs/test_package_layout.py
git commit -m "refactor: move FRS entrypoints into package"
```

---

### Task 3: Move the YAML and launcher and update repository consumers

**Files:**
- Modify: `tests/train_frs/test_package_layout.py`
- Move: `configs/train_frs.yaml` → `train_frs/configs/train_frs.yaml`
- Move: `scripts/start_frs_train.sh` → `train_frs/scripts/start_frs_train.sh`
- Create: `train_frs/README.md`
- Modify: `pyproject.toml`
- Verify unchanged shared call: `configs/train_vtsmolvla_jax.yaml`
- Verify unchanged shared call: `scripts/start_vtsmolvla_train.sh`
- Modify: `docs/superpowers/specs/2026-08-08-split-smolvla-training-packages-design.md`
- Modify: `docs/superpowers/plans/2026-08-08-split-smolvla-training-packages.md`

**Interfaces:**
- Consumes: Task 2's module CLIs and shared tools `tools/merge_smolvla_peft_to_jax.py` and `tools/precompute_tactile_embeddings.py`.
- Produces: `bash train_frs/scripts/start_frs_train.sh [CONFIG]`, defaulting to `train_frs/configs/train_frs.yaml`, plus an installable `train_frs*` package.

- [ ] **Step 1: Add failing launcher/config/package-data assertions**

Append to `tests/train_frs/test_package_layout.py`:

```python
import tomllib


def test_config_and_launcher_live_in_train_frs() -> None:
    assert (ROOT / "train_frs" / "configs" / "train_frs.yaml").is_file()
    assert (ROOT / "train_frs" / "scripts" / "start_frs_train.sh").is_file()
    assert not (ROOT / "configs" / "train_frs.yaml").exists()
    assert not (ROOT / "scripts" / "start_frs_train.sh").exists()


def test_train_frs_is_discovered_by_setuptools() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)
    includes = project["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "train_frs*" in includes


def test_launcher_uses_new_module_paths() -> None:
    launcher = (ROOT / "train_frs" / "scripts" / "start_frs_train.sh").read_text()
    assert "python -m train_frs.compare_frs_reverse_solvers" in launcher
    assert "python -m train_frs.prepare_frs_caches" in launcher
    assert "python -m train_frs.train_frs" in launcher
    assert "tools/precompute_tactile_embeddings.py" in launcher
    assert "tools/merge_smolvla_peft_to_jax.py" in launcher
```

- [ ] **Step 2: Run the assertions and verify RED**

Run:

```bash
uv run --no-sync pytest tests/train_frs/test_package_layout.py -q
```

Expected: config/launcher location and setuptools include assertions fail.

- [ ] **Step 3: Move YAML/Shell and patch path resolution**

Move both files with `apply_patch`. In `train_frs/scripts/start_frs_train.sh`, use:

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/train_frs/configs/train_frs.yaml}"
```

Keep shared-tool invocations unchanged and replace FRS-owned invocations with exactly:

```bash
"${UV_BIN}" run --no-sync python -m train_frs.compare_frs_reverse_solvers --config "${CONFIG_PATH}"
"${UV_BIN}" run --no-sync python tools/precompute_tactile_embeddings.py --config "${CONFIG_PATH}"
XLA_FLAGS="${PREPARE_XLA_FLAGS}" \
    "${UV_BIN}" run --no-sync python -m train_frs.prepare_frs_caches --config "${CONFIG_PATH}"
"${UV_BIN}" run --no-sync python -m train_frs.train_frs --config "${CONFIG_PATH}"
```

Update the five-step command examples in the moved YAML to the same new commands. Do not modify any YAML values.

- [ ] **Step 4: Package and document the new commands**

Add `"train_frs*"` to `[tool.setuptools.packages.find].include` in `pyproject.toml`.

Create `train_frs/README.md` with these exact operational facts:

````markdown
# FRS Training

FRS training code, cache preparation, evaluation, configuration, and launcher live in this package.

## One-command pipeline

```bash
bash train_frs/scripts/start_frs_train.sh
```

Pass another YAML as the first argument when needed. The default is
`train_frs/configs/train_frs.yaml`.

The launcher keeps shared SmolVLA checkpoint merge and tactile embedding precomputation in
`tools/`, then runs the FRS-owned reverse-solver check, action-cache preparation, and trainer
through `python -m train_frs.<module>`.

## Direct training

```bash
uv run --no-sync python -m train_frs.train_frs \
  --config train_frs/configs/train_frs.yaml
```

Training outputs, datasets, action caches, tactile embedding caches, encoder checkpoints, and
merged SmolVLA checkpoints remain external resources configured by the YAML.
````

Update active repository docs and examples to use the new FRS paths. Keep
`scripts/start_vtsmolvla_train.sh` and `configs/train_vtsmolvla_jax.yaml` pointed at the shared
`tools/precompute_tactile_embeddings.py`; VT must not import `train_frs`.

- [ ] **Step 5: Verify launcher, config, and package tests**

Run:

```bash
bash -n train_frs/scripts/start_frs_train.sh
uv run --no-sync pytest tests/train_frs/test_package_layout.py -q
uv run --no-sync python -m train_frs.train_frs --help
```

Expected: Shell syntax passes, tests pass, and CLI help exits 0.

- [ ] **Step 6: Commit launcher/config/consumer migration**

```bash
git add train_frs/configs/train_frs.yaml train_frs/scripts/start_frs_train.sh train_frs/README.md
git add configs/train_frs.yaml scripts/start_frs_train.sh pyproject.toml tests/train_frs/test_package_layout.py
git add docs/superpowers/specs/2026-08-08-split-smolvla-training-packages-design.md
git add docs/superpowers/plans/2026-08-08-split-smolvla-training-packages.md
git commit -m "refactor: move FRS config and launcher"
```

---

### Task 4: Eliminate old-path references and run migration regressions

**Files:**
- Verify: `train_frs/`, `tests/train_frs/`, `tests/flow_decoder/`, `tests/jax/`
- Verify: all repository files except the two migration documents explicitly excluded from the scan

**Interfaces:**
- Consumes: Tasks 1–3 complete package and CLIs.
- Produces: a repository with no live old FRS package, import, command, or default-config reference.

- [ ] **Step 1: Run the old-path migration contract**

Run:

```bash
rg -n "tactile_flow_steering|tools/(train_frs|prepare_frs_caches|compare_frs_reverse_solvers)\.py|tools\.(train_frs|prepare_frs_caches|compare_frs_reverse_solvers)|configs/train_frs\.yaml|scripts/start_frs_train\.sh|from prepare import|import_module\(\"tactile_flow_steering" \
  --glob '!docs/superpowers/specs/2026-08-08-frs-package-migration-design.md' \
  --glob '!docs/superpowers/plans/2026-08-08-frs-package-migration.md' \
  --glob '!.git/**' .
```

Expected: exit status 1 with no output. Any result means the owning Task 1, 2, or 3 is incomplete; return it to that task before continuing.

- [ ] **Step 2: Verify the exact replacement contract**

Confirm Tasks 1–3 used these exact replacements according to context:

```text
tactile_flow_steering                 -> train_frs
tools/train_frs.py                    -> -m train_frs.train_frs
tools/prepare_frs_caches.py           -> -m train_frs.prepare_frs_caches
tools/compare_frs_reverse_solvers.py  -> -m train_frs.compare_frs_reverse_solvers
configs/train_frs.yaml                -> train_frs/configs/train_frs.yaml
scripts/start_frs_train.sh            -> train_frs/scripts/start_frs_train.sh
from prepare import                   -> from train_frs.prepare import
```

Do not rewrite historical migration tables in the two excluded design/plan documents. If Step 1 found a match, patch that owning task with `apply_patch`, re-run its focused test, amend its task commit, and repeat Step 1 until it returns no output.

- [ ] **Step 3: Re-run old-path and reverse-dependency scans**

Run the Step 1 scan again; expected exit status is 1 with no output. Then run:

```bash
rg -n "train_frs" tools/precompute_tactile_embeddings.py tools/merge_smolvla_peft_to_jax.py tactile_encoder src/lerobot modalities_eval scripts/start_vtsmolvla_train.sh configs/train_vtsmolvla_jax.yaml
```

Expected: no non-comment import or command makes VT, encoder, LeRobot, shared merge, or shared embedding code depend on `train_frs`.

- [ ] **Step 4: Run focused and broad regressions**

Run:

```bash
uv run --no-sync pytest tests/train_frs tests/flow_decoder/test_cache.py tests/flow_decoder/test_multi_cache.py tests/flow_decoder/test_frs_safety.py tests/flow_decoder/test_integration.py -q
uv run --no-sync pytest tests/jax/test_tactile_cache.py tests/jax/test_tactile_integration.py tests/jax/test_peft_merge.py -q
uv run --no-sync python -m train_frs.train_frs --help
uv run --no-sync python -m train_frs.prepare_frs_caches --help
uv run --no-sync python -m train_frs.compare_frs_reverse_solvers --help
bash -n train_frs/scripts/start_frs_train.sh
git diff --check
```

Expected: all tests and help/Shell checks pass; `git diff --check` produces no output.

- [ ] **Step 5: Record verification evidence**

No production edit or extra commit is expected in this task. Record the exact commands, pass counts, CLI exit codes, Shell result, and both `rg` scan results in the task report for final review.
