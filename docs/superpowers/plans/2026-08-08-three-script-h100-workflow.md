# Three-Script H100 Training Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `setup_env.sh`, `download_data.sh`, and `start_vtsmolvla_train.sh` the only commands needed to prepare a two-H100 server and train K8 followed by K21.

**Architecture:** Reuse the existing uv lock, official LeRobot v3 converter, encoder downloader, tactile precompute tool, and JAX trainer. Persist one runtime layout in `.env.frs`; make download prepare data plus encoder; make one tmux launcher own cache creation and sequential experiment execution.

**Tech Stack:** Bash, Python 3.12, uv, JAX/Flax, PyTorch/TorchCodec, Hugging Face Hub, LeRobot v3, tmux, pytest.

## Global Constraints

- User-facing entrypoints remain exactly `scripts/setup_env.sh`, `scripts/download_data.sh`, and `scripts/start_vtsmolvla_train.sh`.
- Default persistent root is `/workspace`; virtual environment is `/workspace/.venvs/frs_tact`.
- Correct tactile encoder is `liuchaoyi/encoder_ckpt_05` at `/workspace/checkpoints/encoder_ckpt_05`.
- Dataset roots are `/workspace/lerobot_v30/KaiyueChen/pick_tube_01` through `pick_tube_04`.
- One JAX process sees exactly two H100 GPUs; do not use `torchrun`, `accelerate`, or one process per GPU.
- Default order is K8 then K21; K8 failure prevents K21 startup.
- Cache and train-only normalization protocol are shared; outputs and logs are separate.
- No Git-SHA gate, Hub publication, remote provenance lock, Docker, or Slurm orchestration.
- Preserve FP32 master checkpoints, BF16 compute behavior, tactile cache `[F,4,512]`, repeat factors K8/K21, masks and RoPE.

---

### Task 1: One authoritative H100 environment

**Files:**
- Modify: `scripts/setup_env.sh`
- Create: `tests/test_setup_env.py`
- Modify: `tests/test_start_vtsmolvla_train.py`

**Interfaces:**
- Produces: atomic `${PROJECT_ROOT}/.env.frs` defining `FRS_STORAGE_ROOT`, `FRS_VENV_DIR`, `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, all `HF_*` roots and `TMPDIR`.
- Consumed by: Tasks 2 and 3, which must source these values without overriding them.

- [ ] **Step 1: Write failing environment-contract tests**

Create subprocess tests with a temporary fake command directory. Tests assert:

```python
assert parsed_env["FRS_STORAGE_ROOT"] == "/workspace"
assert parsed_env["FRS_VENV_DIR"] == "/workspace/.venvs/frs_tact"
assert parsed_env["UV_PROJECT_ENVIRONMENT"] == parsed_env["FRS_VENV_DIR"]
assert parsed_env["HF_HOME"] == "/workspace/huggingface"
assert parsed_env["TMPDIR"] == "/workspace/tmp"
```

Also assert `.env.frs` is written through a same-directory temporary file,
setup does not choose `${PROJECT_ROOT}/.venv`, exactly two H100 device names are
required, driver `570.86` is accepted while an older driver fails, PyTorch and
JAX device counts must both equal two, and the verification program executes a
two-device sharded sum.

- [ ] **Step 2: Run RED**

Run:

```bash
/home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider tests/test_setup_env.py tests/test_start_vtsmolvla_train.py -q
```

Expected: failures show checkout-local `.venv`, non-atomic env writing, cache
overrides, and one-GPU acceptance.

- [ ] **Step 3: Implement the authoritative environment**

Use these concrete defaults and persist them verbatim:

```bash
STORAGE_ROOT="${FRS_STORAGE_ROOT:-/workspace}"
VENV_DIR="${FRS_VENV_DIR:-${STORAGE_ROOT}/.venvs/frs_tact}"
UV_CACHE_DIR_VALUE="${FRS_UV_CACHE_DIR:-${STORAGE_ROOT}/.cache/uv}"
```

Replace the global `ps | awk` uv-process scan with a project lock opened by:

```bash
exec 9>"${STORAGE_ROOT}/.locks/frs-setup.lock"
flock -n 9 || fail "另一个 FRS 环境安装正在运行"
```

Write `.env.frs` to `mktemp --tmpdir="${PROJECT_ROOT}" .env.frs.XXXXXX`,
`chmod 600`, then `mv` it atomically. Do not edit `~/.bashrc`.

Extend the verification program to require two H100 names from `nvidia-smi`,
driver `>=570.86`, CUDA/cuDNN/NCCL/libdevice availability, two PyTorch devices,
two JAX GPU devices, and a minimal sharded sum equal to its host reference.

- [ ] **Step 4: Run GREEN and syntax gates**

```bash
/home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider tests/test_setup_env.py tests/test_start_vtsmolvla_train.py -q
bash -n scripts/setup_env.sh
git diff --check
```

Expected: all tests pass and both static checks emit no errors.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/setup_env.sh tests/test_setup_env.py tests/test_start_vtsmolvla_train.py
git commit -m "fix: establish one H100 runtime environment"
```

### Task 2: Idempotent data plus encoder preparation

**Files:**
- Modify: `scripts/download_data.sh`
- Reuse: `scripts/download_ckpt.sh`
- Reuse: `deploy_smolvla/src/download_ckpt.py`
- Create: `tests/test_download_data.py`
- Modify: `tests/test_download_ckpt.py`

**Interfaces:**
- Consumes: Task 1 `.env.frs` without recomputing `HF_*` paths.
- Produces: four validated v3 dataset roots and validated minimal encoder_05.

- [ ] **Step 1: Write failing orchestration and safety tests**

Tests execute the script with fake `uv`, `hf`, converter and checkpoint wrapper
commands and assert:

```python
assert requested_datasets == [
    "KaiyueChen/pick_tube_01",
    "KaiyueChen/pick_tube_02",
    "KaiyueChen/pick_tube_03",
    "KaiyueChen/pick_tube_04",
]
assert encoder_call == ["scripts/download_ckpt.sh"]
assert no_precompute_call
```

Cover missing `.env.frs`, second-process lock rejection, valid-v3 skip,
v2.1 conversion, failed conversion preserving the source, schema/sample
validation failure, encoder download failure, rerun idempotency, unknown options,
and `--cleanup-source` as the only source-deletion path.

- [ ] **Step 2: Run RED**

```bash
/home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider tests/test_download_data.py tests/test_download_ckpt.py -q
```

Expected: failures show environment values are overwritten, encoder download is
never called, there is no lock/usage contract, and v3 schema is not validated.

- [ ] **Step 3: Implement the three-script download boundary**

Require `${PROJECT_ROOT}/.env.frs`, source it, and fail if its named environment
does not exist. Parse only:

```text
--cleanup-source
--help
```

Acquire `${FRS_STORAGE_ROOT}/.locks/frs-download-data.lock`. Keep the existing
official conversion functions. Preserve the original HF snapshots by default;
temporary conversion paths may be recreated only beneath the fixed
`lerobot_v30_work/KaiyueChen/<dataset>` roots. Removing source snapshots and
post-success conversion leftovers requires `--cleanup-source`. Never delete
outside the four derived repo roots.

After each final directory exists, run a project-Python validation probe that
loads `meta/info.json`, checks v3/action/state/RGB/tactile/stats/index contracts,
constructs `LeRobotDatasetMetadata`, constructs `LeRobotDataset` for one episode,
and reads sample zero.

After all datasets validate, execute:

```bash
bash "${PROJECT_ROOT}/scripts/download_ckpt.sh"
```

Then verify `checkpoint.json` and its declared params archive exist at the exact
YAML encoder path. Do not call `precompute_tactile_embeddings.py`.

- [ ] **Step 4: Run GREEN and syntax gates**

```bash
/home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider tests/test_download_data.py tests/test_download_ckpt.py -q
bash -n scripts/download_data.sh scripts/download_ckpt.sh
git diff --check
```

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/download_data.sh tests/test_download_data.py tests/test_download_ckpt.py
git commit -m "feat: prepare VT datasets and encoder together"
```

### Task 3: Sequential K8 then K21 launcher

**Files:**
- Modify: `scripts/start_vtsmolvla_train.sh`
- Modify: `tests/test_start_vtsmolvla_train.py`
- Modify: `tests/jax/test_train_vtsmolvla_config.py`

**Interfaces:**
- Consumes: Task 1 `.env.frs`; Task 2 datasets/encoder; existing K8/K21 YAMLs.
- Produces: one tmux-owned cache/K8/K21 pipeline with strict output/resume guards.

- [ ] **Step 1: Write failing CLI and orchestration tests**

Cover the exact interface:

```text
--experiment both|k8|k21
--gpus 0,1
--cache auto|skip|only
--resume none|auto|PATH
--smoke
--foreground
--session NAME
--config PATH
```

Assert default calls are:

```python
assert calls == [
    ("precompute", "configs/train_vtsmolvla_jax_tactile16.yaml"),
    ("train", "configs/train_vtsmolvla_jax_tactile16.yaml"),
    ("train", "configs/train_vtsmolvla_jax_tactile32.yaml"),
]
```

Tests also prove exact two-H100 preflight, `CUDA_VISIBLE_DEVICES` set before JAX,
cache once, K8 failure short-circuit, full tmux argument forwarding, single-config
legacy mode, mutual exclusions, output collision, incomplete checkpoint failure,
`resume auto` highest-complete selection, explicit resume validation, and smoke
temporary config cleanup without modifying tracked YAML.

- [ ] **Step 2: Run RED**

```bash
/home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider tests/test_start_vtsmolvla_train.py tests/jax/test_train_vtsmolvla_config.py -q
```

Expected: failures show missing experiment orchestration, one-GPU acceptance,
duplicate cache calls or unsupported resume/smoke arguments.

- [ ] **Step 3: Implement minimal sequential orchestration**

Map experiment names exactly:

```bash
K8_CONFIG="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile16.yaml"
K21_CONFIG="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile32.yaml"
```

Store the original argument vector before parsing and forward it with `%q` into
one tmux session. Export the selected `CUDA_VISIBLE_DEVICES` before any `uv run`
that imports JAX. Preflight requires two comma-separated GPU IDs and JAX device
kind containing `H100` for both.

For `both`, call cache precompute once with K8, run K8 synchronously, then K21.
Keep `set -Eeuo pipefail` so a nonzero K8 status aborts the shell.

Create smoke YAML with project Python and `tempfile.NamedTemporaryFile`, using a
deep copy and the exact overrides from the spec. Register trap cleanup. Use
separate timestamped smoke outputs so formal outputs are untouched.

Implement fresh/resume guards without deleting anything. `resume auto` accepts
only numeric `checkpoint-*` directories containing `training_state.msgpack` and
selects the largest step.

- [ ] **Step 4: Run GREEN and regression gates**

```bash
/home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider tests/test_start_vtsmolvla_train.py tests/jax/test_train_vtsmolvla_config.py -q
bash -n scripts/start_vtsmolvla_train.sh
git diff --check
```

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/start_vtsmolvla_train.sh tests/test_start_vtsmolvla_train.py tests/jax/test_train_vtsmolvla_config.py
git commit -m "feat: train K8 then K21 on two H100s"
```

### Task 4: Integrated verification and operator handoff

**Files:**
- Modify: `CODEBASE_MEMORY.md`
- Modify: `docs/reports/2026-08-08-h100-vt-training-consistency.md`
- Create: `docs/reports/2026-08-08-three-script-h100-workflow.md`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: exact three-command server handoff and evidence boundary.

- [ ] **Step 1: Run the complete focused suite**

```bash
/home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_setup_env.py tests/test_download_data.py tests/test_download_ckpt.py \
  tests/test_start_vtsmolvla_train.py tests/jax/test_train_vtsmolvla_config.py -q
```

- [ ] **Step 2: Run accepted VT/JAX regression**

Run the accepted VT scope used by the existing H100 report:

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/frs_three_script_matplotlib \
  XDG_CACHE_HOME=/tmp/frs_three_script_xdg \
  HF_DATASETS_CACHE=/tmp/frs_three_script_hf \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  modalities_eval/test tests/jax tests/test_setup_env.py tests/test_download_data.py \
  tests/test_start_vtsmolvla_train.py tests/test_download_ckpt.py \
  --ignore=tests/jax/test_functional.py \
  --ignore=tests/jax/test_lora.py \
  --ignore=tests/jax/test_training.py \
  --deselect=tests/jax/test_checkpoint.py::test_processor_configs_sync_rename_map_and_feature_shapes \
  --deselect=tests/jax/test_tactile_integration.py::test_default_deployment_config_pins_the_bimanual_vt_contract \
  -q
```

Record exact passes, skips, deselections and known out-of-scope pure-visual
collection blockers.

- [ ] **Step 3: Run static and contract gates**

```bash
bash -n scripts/setup_env.sh scripts/download_data.sh scripts/start_vtsmolvla_train.sh scripts/download_ckpt.sh
git diff --check
```

Also execute fake-command end-to-end tests proving the only operator sequence is:

```bash
bash scripts/setup_env.sh
bash scripts/download_data.sh
bash scripts/start_vtsmolvla_train.sh
```

- [ ] **Step 4: Perform independent whole-diff review**

Review for destructive path scope, shell quoting, lock lifetime, environment
override drift, exact H100 count, cache-once behavior, K8 failure short-circuit,
resume selection, and claims not backed by real H100 execution. Fix every
Critical/Important finding with a focused RED/GREEN test.

- [ ] **Step 5: Update operator documentation and folder memory**

Document the exact three commands, prerequisites, output/log locations, attach
commands, resume examples, smoke example, cleanup behavior, and the explicit
statement that local tests do not constitute a real two-H100 training run.

- [ ] **Step 6: Commit verification docs**

```bash
git add CODEBASE_MEMORY.md docs/reports/2026-08-08-h100-vt-training-consistency.md docs/reports/2026-08-08-three-script-h100-workflow.md
git commit -m "docs: hand off three-script H100 workflow"
```
