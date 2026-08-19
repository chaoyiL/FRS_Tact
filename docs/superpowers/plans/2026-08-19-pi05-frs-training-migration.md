# Pi0.5 FRS Complete Training Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained `/home/typhon/FRS_Tact/train_pi05_frs` project that runs tactile embedding precomputation, Pi0.5 action-cache generation, FRS decoder training/evaluation, and produces deployment-compatible checkpoints.

**Architecture:** The training project owns a third isolated JAX 0.5.3/Flax 0.10.2 environment and a private minimal `src/lerobot` containing only the Pi0.5 and LeRobot dataset closure. It reuses target-root `train_encoder` through a narrow runtime API, keeps all Pi0.5 cache-producer additions under `train_pi05_frs`, and integrates with `deploy_pi05` only through the checkpoint format.

**Tech Stack:** Python 3.12, uv, JAX/JAXlib 0.5.3 CUDA 12, Flax 0.10.2 NNX, Optax, Orbax 0.11.13, PyTorch DataLoader, LeRobot v3 datasets, NumPy NPZ/memmap, Bash, PyYAML, pytest.

## Global Constraints

- The source of truth is the current clean source paths under `/home/typhon/FRS_Tact-pi05-frs-jax`; omit `__pycache__`, `.pyc`, `.venv`, checkpoints, caches, and generated outputs.
- Migrate only the Pi0.5 FRS complete training chain; do not copy encoder training, `modalities_eval` as a package, deployment clients, SmolVLA, SmolVLA FRS, or VT-SmolVLA code.
- Do not overwrite or behaviorally modify target-root `lerobot`, `train_encoder`, `utils`, `deploy_pi05`, root `pyproject.toml`, or root `uv.lock`.
- The training environment is `/home/typhon/FRS_Tact/train_pi05_frs/.venv` and must reject aliases of the root and deployment environment paths before any sync.
- Pin JAX/JAXlib 0.5.3, Flax 0.10.2, Orbax Checkpoint 0.11.13, Transformers 4.53.2, ml-dtypes 0.4.1, and Python 3.12.
- Replace source `nnx.List` only with the deployment-proven ordinary-list representation; do not change decoder layers, tensor shapes, loss, solver, or training hyperparameters.
- Preserve `decoder_input_version: 2`, parameter paths, metadata, and NPZ array semantics so training checkpoints load in `deploy_pi05`.
- Cache provenance mismatches, resume-config mismatches, missing local `params/`, invalid dataset v3 metadata, missing encoder/norm stats/camera map, and unavailable GPU for a real run must fail early.
- Never delete or silently overwrite user caches, checkpoints, datasets, or outputs.
- Use `apply_patch` for all repository file creation and edits; preserve unrelated target changes.

---

### Task 1: Standalone Project and Environment Boundary

**Files:**
- Create: `train_pi05_frs/pyproject.toml`
- Create: `train_pi05_frs/uv.lock`
- Create: `train_pi05_frs/scripts/setup_env.sh`
- Create: `tests/test_train_pi05_frs_project_boundary.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: target repository root resolved as two directories above `train_pi05_frs/scripts/setup_env.sh`.
- Produces: `TRAIN_PI05_FRS_PYTHON` override, default `train_pi05_frs/.venv/bin/python`, package-local `uv.lock`, and a sourceable setup script whose `main` only runs when executed directly.

- [ ] **Step 1: Write failing project-boundary tests**

Add tests that assert the exact three environment targets differ after `realpath -m`, a fake `uv` receives `--project <root>/train_pi05_frs` and `UV_PROJECT_ENVIRONMENT=<root>/train_pi05_frs/.venv`, and the standalone package metadata discovers only `train_pi05_frs*` plus private `lerobot*`. The same-target test must set a fake `uv` that raises if called:

```python
def test_setup_rejects_root_or_deploy_environment_before_uv(tmp_path: Path) -> None:
    for forbidden in (ROOT / ".venv", ROOT / "deploy_pi05/.venv"):
        result = run_sourced_setup(
            tmp_path,
            train_venv=forbidden,
            function="validate_environment_targets",
        )
        assert result.returncode != 0
        assert "独立虚拟环境" in result.stderr
        assert not (tmp_path / "uv.called").exists()
```

Also assert `.gitignore` contains `/train_pi05_frs/.venv/`, `/train_pi05_frs/.cache/`, and `/train_pi05_frs/outputs/` without ignoring source/config/tests.

- [ ] **Step 2: Run the boundary tests and confirm RED**

Run:

```bash
/home/typhon/FRS_Tact/.venv/bin/python -m pytest -q \
  tests/test_train_pi05_frs_project_boundary.py
```

Expected: failures for missing `pyproject.toml`, `setup_env.sh`, lock, and ignore rules.

- [ ] **Step 3: Add the minimal standalone package and setup script**

Create `train_pi05_frs/pyproject.toml` with project name `pi05-frs-training`, Python `>=3.12,<3.13`, package roots `src` and the repository parent, and the source Pi0.5 dependency set restricted to actual pipeline imports. Use exact pins from Global Constraints and retain PyTorch/torchvision/torchcodec, NumPy `<2.3`, OpenCV, Pillow, datasets, pandas, pyarrow, av, PyYAML, matplotlib, pytest, OpenPI typing/tokenizer/download dependencies, and the CUDA 12 local JAX extra.

Implement these shell contracts in `setup_env.sh`:

```bash
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TRAIN_ROOT}/.." && pwd)"
TRAIN_VENV="${TRAIN_PI05_FRS_VENV:-${TRAIN_ROOT}/.venv}"

validate_environment_targets() {
    TRAIN_VENV="$(realpath -m -- "${TRAIN_VENV}")"
    local root_venv deploy_venv
    root_venv="$(realpath -m -- "${REPO_ROOT}/.venv")"
    deploy_venv="$(realpath -m -- "${REPO_ROOT}/deploy_pi05/.venv")"
    [[ "${TRAIN_VENV}" != "${root_venv}" && "${TRAIN_VENV}" != "${deploy_venv}" ]] \
        || fail "Pi0.5 FRS 训练必须使用独立虚拟环境：${TRAIN_VENV}"
}

sync_environment() {
    validate_environment_targets
    UV_PROJECT_ENVIRONMENT="${TRAIN_VENV}" "${UV_BIN}" sync \
        --frozen --python 3.12 --project "${TRAIN_ROOT}"
}
```

Support `--check` as a dependency-free preflight that prints project, environment, Python selection, and entrypoints without syncing. Guard `main` with `[[ "${BASH_SOURCE[0]}" == "$0" ]]`.

- [ ] **Step 4: Generate and verify the isolated lock**

Run from `train_pi05_frs`:

```bash
UV_CACHE_DIR=/home/typhon/.cache/uv uv lock --offline
UV_CACHE_DIR=/home/typhon/.cache/uv uv lock --check --offline
UV_CACHE_DIR=/home/typhon/.cache/uv uv sync --frozen --python 3.12 --dry-run
bash scripts/setup_env.sh
```

Expected: the lock resolves the exact pinned Pi0.5 stack, the dry-run targets
`train_pi05_frs/.venv`, and the final command creates that environment without modifying either existing
environment.

- [ ] **Step 5: Run GREEN verification and commit**

```bash
bash -n train_pi05_frs/scripts/setup_env.sh
bash train_pi05_frs/scripts/setup_env.sh --check
/home/typhon/FRS_Tact/.venv/bin/python -m pytest -q \
  tests/test_train_pi05_frs_project_boundary.py
git diff --check
git add .gitignore train_pi05_frs/pyproject.toml train_pi05_frs/uv.lock \
  train_pi05_frs/scripts/setup_env.sh tests/test_train_pi05_frs_project_boundary.py
git commit -m "build: isolate pi05 frs training environment"
```

Expected: shell checks and tests pass; commit contains no virtualenv or cache files.

---

### Task 2: Private Pi0.5 Dataset and Action-Cache Producer

**Files:**
- Create: `train_pi05_frs/src/lerobot/**` for the exact minimal source closure
- Create: `train_pi05_frs/pi05_cache/__init__.py`
- Create: `train_pi05_frs/pi05_cache/cache.py`
- Create: `train_pi05_frs/pi05_cache/prepare.py`
- Create: `train_pi05_frs/pi05_cache/source_model.py`
- Create: `train_pi05_frs/pi05_cache/policy_inputs.py`
- Create: `train_pi05_frs/tests/test_pi05_cache.py`
- Modify: `tests/test_train_pi05_frs_project_boundary.py`

**Interfaces:**
- Consumes: source `prepare_pi05.py`, `modalities_eval/pi05_utils.py`, `utils/pi05_source_model.py`, `utils/{cache,integration}.py`, `src/lerobot/policies/pi05_jax/**`, and the LeRobot v3 dataset modules imported by those files.
- Produces: `train_pi05_frs.pi05_cache.prepare.prepare_cache` returning `pathlib.Path`, `CachedPairs`, `MultiCachedPairs`, and private `lerobot.policies.pi05_jax.{Pi0Config,load_pi0}`.

- [ ] **Step 1: Add RED tests for cache selection, transforms, and package isolation**

Port source action-cache unit cases so they assert episode-disjoint selection, tail trimming, frame stride, 20D robot action to configured model dimension padding, camera-map validation, norm-stat dimension validation, deterministic inference seed, manifest provenance, and resume behavior. Add this package-origin assertion:

```python
def test_training_lerobot_is_private() -> None:
    import lerobot
    path = Path(lerobot.__file__).resolve()
    assert TRAIN_ROOT / "src" in path.parents
```

The boundary test must compare a recorded source-to-target manifest and reject files outside the selected Pi0.5 policy/dataset closure.

- [ ] **Step 2: Run cache tests and confirm RED**

```bash
PYTHONPATH=train_pi05_frs/src:. \
  /home/typhon/FRS_Tact-pi05-frs-jax/.venv/bin/python -m pytest -q \
  train_pi05_frs/tests/test_pi05_cache.py \
  tests/test_train_pi05_frs_project_boundary.py
```

Expected: import failures for `train_pi05_frs.pi05_cache` and missing private `lerobot`.

- [ ] **Step 3: Migrate the exact private runtime closure with a provenance manifest**

Use `apply_patch` to add the source files required by `Pi0Config`, `load_pi0`, policy transforms,
normalization, tokenizer, checkpoint download/restore, LeRobot v3 metadata/reader, dataset-source mapping,
and tactile cache. Preserve source contents unless import roots must point inside this standalone project.
Write `train_pi05_frs/source_manifest.sha256` with paths relative to their source repository and SHA256 of
every unchanged file. Do not copy source training CLIs, deployment code, encoder package, or modalities package.

- [ ] **Step 4: Isolate cache-production logic under `pi05_cache`**

Move only the reachable source logic into the four files named above. Preserve the source
`prepare_cache` keyword-only interface and return type exactly: `checkpoint_dir: str`,
`cache_dir: pathlib.Path`, dataset repo/root/revision/action key, rename/camera maps, norm-stat
directory/asset/quantile flag, action dimension/horizon, both model variants, sample/reverse
steps and solver, batch/worker/seed/split/stride/limit/flush settings, optional preloaded model,
returning the completed `pathlib.Path`.

Replace imports of source-root `utils.cache` with `train_pi05_frs.pi05_cache.cache` and imports of
`modalities_eval.pi05_utils` with `train_pi05_frs.pi05_cache.policy_inputs`. Preserve manifest keys,
array names/dtypes, reverse solver behavior, and atomic writes.

- [ ] **Step 5: Run GREEN cache tests and source-boundary checks**

```bash
PYTHONPATH=train_pi05_frs/src:. \
  /home/typhon/FRS_Tact-pi05-frs-jax/.venv/bin/python -m pytest -q \
  train_pi05_frs/tests/test_pi05_cache.py \
  tests/test_train_pi05_frs_project_boundary.py
sha256sum -c train_pi05_frs/source_manifest.sha256
git diff --check
```

Expected: all tests and hashes pass; imports resolve inside `train_pi05_frs` or approved target-root
`train_encoder`, never source repository paths.

- [ ] **Step 6: Commit**

```bash
git add train_pi05_frs/src train_pi05_frs/pi05_cache \
  train_pi05_frs/tests/test_pi05_cache.py train_pi05_frs/source_manifest.sha256 \
  tests/test_train_pi05_frs_project_boundary.py
git commit -m "feat: migrate pi05 frs cache producer"
```

---

### Task 3: FRS Decoder Training, Evaluation, and Checkpoint Compatibility

**Files:**
- Create: `train_pi05_frs/{__init__,train,evaluate,plot_history}.py`
- Create: `train_pi05_frs/utils/{__init__,checkpoint,data,history_plot,integration,metrics,model,mp_batches,visualize,window_io}.py`
- Create: `train_pi05_frs/tests/{__init__,test_data,test_model}.py`
- Create: `train_pi05_frs/tests/test_deployment_checkpoint_compatibility.py`

**Interfaces:**
- Consumes: `train_pi05_frs.pi05_cache.cache.{CachedPairs,MultiCachedPairs}`, target-root `train_encoder.utils`, and action/tactile caches from Tasks 1–2.
- Produces: `train_decoder`, `DecoderConfig`, `TactileConditionedFlowDecoder`, `decode_actions`, `save_checkpoint`, and `load_checkpoint` with deployment wire compatibility.

- [ ] **Step 1: Port source tests unchanged where behavior is unchanged and add cross-runtime RED test**

Copy source `train_pi05_frs/tests/test_data.py` and `test_model.py` through `apply_patch`, changing only
cache/encoder import paths required by the new boundary. Add a cross-runtime test that writes a small
state-conditioned decoder checkpoint and invokes the deployment interpreter in a subprocess:

```python
subprocess.run(
    [str(DEPLOY_PYTHON), "-c", DEPLOY_LOAD_SCRIPT, str(checkpoint_dir)],
    cwd=REPO_ROOT,
    env={**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'deploy_pi05/src'}:{REPO_ROOT}"},
    check=True,
    text=True,
    capture_output=True,
)
```

The child script must compare metadata parameter paths and emit a SHA256 over restored numeric leaves;
the parent compares it with the training model digest and checks one finite forward output.

- [ ] **Step 2: Run tests and confirm RED**

```bash
PYTHONPATH=train_pi05_frs/src:. \
  /home/typhon/FRS_Tact-pi05-frs-jax/.venv/bin/python -m pytest -q \
  train_pi05_frs/tests/test_data.py \
  train_pi05_frs/tests/test_model.py \
  train_pi05_frs/tests/test_deployment_checkpoint_compatibility.py
```

Expected: missing trainer modules and/or `nnx.List` incompatibility.

- [ ] **Step 3: Migrate all 17 tracked source training files**

Add every tracked Python file under source `train_pi05_frs` through `apply_patch`, preserving model,
loss, optimizer, scheduler, gate metrics, visualization, resume, atomic checkpoint, multi-cache, worker,
window and evaluation behavior. Change only these integration points:

```python
from train_encoder.utils.checkpoint import load_tactile_encoder
from train_encoder.utils.image_dataset import create_image_dataset
from train_encoder.utils.model import encode_resnet18, tactile_clip_config_from_dict
from train_encoder.utils.prefetch import prefetch_iterator
from train_pi05_frs.pi05_cache.cache import CachedPairs, MultiCachedPairs, atomic_write_json
```

In `TactileConditionedFlowDecoder.__init__`, use the deployment-compatible ordinary list:

```python
self.blocks = [
    ConditionedTransformerBlock(
        config.model_dim, config.num_heads, config.mlp_ratio, rngs=rngs
    )
    for _ in range(config.depth)
]
```

Do not change any other architecture or hyperparameter default.

- [ ] **Step 4: Make checkpoint loading accept historical and filtered None-slot metadata**

Match `deploy_pi05/frs_inference/decoder_checkpoint.py` semantics: accept both the legacy full
`parameter_paths` list with object-None slots and the filtered numeric-only form; never call
`np.load(archive_path, allow_pickle=True)` on untrusted archives. Tests must restore non-default seeded parameters
and compare every numeric leaf.

- [ ] **Step 5: Run model/data/checkpoint GREEN tests**

```bash
PYTHONPATH=train_pi05_frs/src:. \
  /home/typhon/FRS_Tact-pi05-frs-jax/.venv/bin/python -m pytest -q \
  train_pi05_frs/tests
git diff --check
```

Expected: source data/model coverage plus cross-environment checkpoint test passes with Flax 0.10.2.

- [ ] **Step 6: Commit**

```bash
git add train_pi05_frs/__init__.py train_pi05_frs/train.py \
  train_pi05_frs/evaluate.py train_pi05_frs/plot_history.py \
  train_pi05_frs/utils train_pi05_frs/tests
git commit -m "feat: migrate pi05 frs decoder training"
```

---

### Task 4: Configuration and One-Command Three-Stage Pipeline

**Files:**
- Create: `train_pi05_frs/configs/train_pi05_frs.yaml`
- Create: `train_pi05_frs/tools/{precompute_tactile_embeddings,prepare_frs_pi05_cache,train_frs}.py`
- Create: `train_pi05_frs/scripts/start_frs_pi05_train.sh`
- Create: `train_pi05_frs/tests/test_pipeline.py`
- Create: `train_pi05_frs/README.md`

**Interfaces:**
- Consumes: Task 1 training interpreter, Task 2 `prepare_cache`, Task 3 `train_decoder`, target-root `train_encoder`, and the approved YAML schema.
- Produces: `load_config(path) -> dict[str, Any]`, `prepare_from_config(config) -> list[Path]`, `train_from_config(config) -> None`, and a shell launcher supporting `[CONFIG]`, `--check`, foreground, and tmux.

- [ ] **Step 1: Add RED config and pipeline-order tests**

Port the source configuration/tool tests and add a fake-Python launcher test whose event log must be exactly:

```python
assert events == [
    "validate",
    "checkpoint-smoke",
    "precompute-tactile",
    "prepare-pi05-cache",
    "train-frs",
]
```

Add failure injection at each stage and assert no later event occurs. `--check` must validate paths and
schema but produce no cache/output/checkpoint file and must not import JAX.

- [ ] **Step 2: Run pipeline tests and confirm RED**

```bash
PYTHONPATH=train_pi05_frs/src:. \
  /home/typhon/FRS_Tact/.venv/bin/python -m pytest -q \
  train_pi05_frs/tests/test_pipeline.py
```

Expected: missing config, tools, launcher and README.

- [ ] **Step 3: Migrate config and three Python entrypoints**

Copy source YAML values and the three tools using `apply_patch`; relocate defaults to
`train_pi05_frs/configs/train_pi05_frs.yaml` and imports to Tasks 2–3. Preserve URL checkpoint strings
without passing them through `Path`, preserve multi-dataset cache directory sanitization, and strictly
validate YAML mappings/lists and boolean types before loading models.

- [ ] **Step 4: Implement the one-command launcher**

Use the package-local interpreter directly rather than `uv run`:

```bash
TRAIN_PYTHON="${TRAIN_PI05_FRS_PYTHON:-${TRAIN_ROOT}/.venv/bin/python}"
export PYTHONPATH="${TRAIN_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${TRAIN_PYTHON}" -m train_pi05_frs.tools.precompute_tactile_embeddings --config "${CONFIG_PATH}"
"${TRAIN_PYTHON}" -m train_pi05_frs.tools.prepare_frs_pi05_cache --config "${CONFIG_PATH}"
"${TRAIN_PYTHON}" -m train_pi05_frs.tools.train_frs --config "${CONFIG_PATH}"
```

Before those commands, perform dependency-light YAML/path checks and then a Pi0.5 checkpoint shape smoke.
Keep tmux opt-in behavior compatible with source via `FRS_FOREGROUND=1`; write one timestamped pipeline log
under `frs_training.output`. Add `--check` before any `mkdir`, model import, GPU initialization, or tmux call.

- [ ] **Step 5: Write README for only this training project**

Document the third environment, exact setup/start/evaluate commands, YAML path fields, cache provenance,
resume, tmux logs, deployment handoff, and the fact that encoder training and modality analysis live elsewhere.
Do not document removed source-root paths.

- [ ] **Step 6: Run GREEN pipeline verification and commit**

```bash
bash -n train_pi05_frs/scripts/setup_env.sh
bash -n train_pi05_frs/scripts/start_frs_pi05_train.sh
bash train_pi05_frs/scripts/setup_env.sh --check
bash train_pi05_frs/scripts/start_frs_pi05_train.sh --check \
  train_pi05_frs/configs/train_pi05_frs.yaml
PYTHONPATH=train_pi05_frs/src:. \
  /home/typhon/FRS_Tact/.venv/bin/python -m pytest -q \
  train_pi05_frs/tests/test_pipeline.py
git diff --check
git add train_pi05_frs/configs train_pi05_frs/tools train_pi05_frs/scripts \
  train_pi05_frs/tests/test_pipeline.py train_pi05_frs/README.md
git commit -m "feat: add pi05 frs training pipeline"
```

Expected: preflights and tests pass without creating training outputs.

---

### Task 5: Integration, Boundary Audit, and Operator Handoff

**Files:**
- Modify: `tests/test_train_pi05_frs_project_boundary.py`
- Modify: `train_pi05_frs/tests/test_deployment_checkpoint_compatibility.py`
- Modify: `train_pi05_frs/tests/test_pipeline.py`
- Modify: `train_pi05_frs/README.md`

**Interfaces:**
- Consumes: the complete project from Tasks 1–4 and both existing target environments for read-only compatibility checks.
- Produces: a verified migration manifest, complete automated evidence, and explicit limitations for unavailable real GPU/data/checkpoint runs.

- [ ] **Step 1: Add final RED boundary assertions**

Assert all 17 source `train_pi05_frs` Python paths have a target mapping; all source three-stage entries and
configuration have a mapping; no target path contains `deploy_pi05_frs`, `tactile_encoder`,
`modalities_eval`, `train_smolvla`, `train_smolvla_frs`, or `train_vtsmolvla` package copies; no cache,
checkpoint, virtualenv or bytecode is tracked. Assert root protected paths have no branch diff:

```python
PROTECTED = (
    "pyproject.toml",
    "uv.lock",
    "lerobot",
    "train_encoder",
    "utils",
    "deploy_pi05",
    "train_smolvla",
    "train_smolvla_frs",
    "train_vtsmolvla",
)
```

- [ ] **Step 2: Run assertions and confirm any missing integration is RED**

```bash
/home/typhon/FRS_Tact/.venv/bin/python -m pytest -q \
  tests/test_train_pi05_frs_project_boundary.py
```

Expected: failure for any missing map, stale source path, forbidden copy, or protected-root diff.

- [ ] **Step 3: Fix only reported integration gaps and update README evidence**

Use `apply_patch` for the exact failing mappings/imports/docs. Add a README verification section that labels
mock/CPU checks separately from a real GPU run and never claims the long pipeline ran unless logs prove it.

- [ ] **Step 4: Run the complete fresh verification matrix**

```bash
bash -n train_pi05_frs/scripts/setup_env.sh
bash -n train_pi05_frs/scripts/start_frs_pi05_train.sh
UV_CACHE_DIR=/home/typhon/.cache/uv uv lock --check --offline \
  --project train_pi05_frs
UV_CACHE_DIR=/home/typhon/.cache/uv uv sync --frozen --python 3.12 \
  --project train_pi05_frs --dry-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=train_pi05_frs/src:. \
  train_pi05_frs/.venv/bin/python -m pytest -q -p no:cacheprovider \
  train_pi05_frs/tests tests/test_train_pi05_frs_project_boundary.py
bash train_pi05_frs/scripts/setup_env.sh --check
bash train_pi05_frs/scripts/start_frs_pi05_train.sh --check \
  train_pi05_frs/configs/train_pi05_frs.yaml
sha256sum -c train_pi05_frs/source_manifest.sha256
git diff --check
git status --short
```

Expected: all available checks pass; status lists only intentional source/docs/tests. If the configured real
assets exist and a GPU is visible, additionally run the checkpoint smoke and a bounded one-batch cache/training
integration. Otherwise record exactly which asset or device prevented that optional verification.

- [ ] **Step 5: Independent review and fix gate**

Generate a review package from design commit `9a321e6` to current HEAD. The reviewer must check scope,
environment isolation, source closure, cache provenance, model/checkpoint compatibility, launcher failure
ordering, protected target paths, tests, and truthful operator documentation. Fix every Critical/Important
finding with a focused test and repeat review until clean.

- [ ] **Step 6: Commit final integration**

```bash
git add tests/test_train_pi05_frs_project_boundary.py \
  train_pi05_frs/tests/test_deployment_checkpoint_compatibility.py \
  train_pi05_frs/tests/test_pipeline.py train_pi05_frs/README.md
git commit -m "test: verify pi05 frs training migration"
```

Expected: working tree clean and the branch ready for a fast-forward merge into `eric` after merged-result tests.
