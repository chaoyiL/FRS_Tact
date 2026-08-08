# H100 VT Training Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the K8/K21 VT-SmolVLA baselines launch correctly and use identical BF16 compute and train-only normalization contracts on two H100 GPUs.

**Architecture:** Keep FP32 master checkpoints and the existing tactile cache/model schema. Add one shared compute-param conversion boundary and one immutable normalization-protocol artifact built from LeRobot v3 per-episode metadata, then make launcher, publish, deploy, policy and evaluation consumers validate the same contracts.

**Tech Stack:** Bash, Python 3.12, JAX/Flax/Optax, safetensors, PyArrow/LeRobot v3 metadata, pytest, YAML.

## Global Constraints

- Correct encoder: `liuchaoyi/encoder_ckpt_05` at `/workspace/checkpoints/encoder_ckpt_05`.
- Runtime trainable floating parameters use BF16; checkpoint optimizer masters remain FP32.
- K8/K21 differ only by repeat factor, output identity and W&B identity.
- Normalization uses training episodes only; no fallback to full-dataset stats.
- Do not change tactile cache `[F,4,512]`, projection parameters, repeat order, masks or RoPE.
- One Python process sees both H100 devices; do not use torchrun or two Slurm tasks.

---

### Task 1: Launcher and encoder_05 contract

**Files:**
- Modify: `scripts/start_vtsmolvla_train.sh`
- Modify: `scripts/download_ckpt.sh`
- Modify: `deploy_smolvla/src/download_ckpt.py`
- Modify: `configs/train_vtsmolvla_jax.yaml`
- Modify: `configs/train_vtsmolvla_jax_tactile16.yaml`
- Modify: `configs/train_vtsmolvla_jax_tactile32.yaml`
- Test: `tests/test_start_vtsmolvla_train.py`
- Test: `tests/test_download_ckpt.py`
- Test: `tests/jax/test_train_vtsmolvla_config.py`

**Interfaces:**
- Produces: launcher `--config PATH`; downloader defaults for encoder_05; identical YAML encoder path.

- [ ] **Step 1: Write failing tests** for default/explicit/equal-form config parsing, unknown and duplicate arguments, tmux forwarding including a path with spaces, encoder repository/output defaults, and exact YAML encoder identity.
- [ ] **Step 2: Run RED** with `python -m pytest -p no:cacheprovider tests/test_start_vtsmolvla_train.py tests/test_download_ckpt.py tests/jax/test_train_vtsmolvla_config.py -q`; expect assertions against the K1 hardcode and encoder_06 defaults.
- [ ] **Step 3: Implement the minimal parser and defaults** using shell argument parsing and quoted tmux forwarding; retain no-argument K1 behavior and fail before preflight on invalid input.
- [ ] **Step 4: Run GREEN** with the RED command plus `bash -n scripts/start_vtsmolvla_train.sh scripts/download_ckpt.sh`; expect all tests and syntax checks to pass.
- [ ] **Step 5: Commit** only Task 1 files with `git commit -m "fix: select VT config and encoder 05 explicitly"`.

### Task 2: Unified BF16 compute parameters

**Files:**
- Modify: `src/lerobot/policies/smolvla_jax/configuration.py`
- Modify: `src/lerobot/policies/smolvla_jax/training.py`
- Modify: `src/lerobot/policies/smolvla_jax/policy.py`
- Modify: `src/lerobot/policies/smolvla_jax/checkpoint.py`
- Modify: `modalities_eval/utils.py`
- Modify: `src/lerobot/policies/smolvla_jax/validation.py`
- Modify: `tools/publish_smolvla_checkpoint.py`
- Modify: `deploy_smolvla/remote_client.py`
- Modify: `deploy_smolvla/configs/deploy_smolvla_jax.yaml`
- Test: `tests/jax/test_training.py`
- Test: `tests/jax/test_tactile_training.py`
- Test: `tests/jax/test_tactile_checkpoint.py`
- Test: `tests/jax/test_checkpoint_validation.py`
- Test: `tests/jax/test_publish_checkpoint.py`
- Test: `tests/jax/test_remote_client.py`

**Interfaces:**
- Produces: `prepare_params_for_compute(params, config)` and serialized `trainable_compute_dtype="bfloat16"`.
- Consumes: existing `is_trainable_parameter(path, config)` classification.

- [ ] **Step 1: Write failing tests** proving only trainable floating leaves become BF16, frozen/integer leaves retain dtype, save-load-prepare matches trainer compute params, policy/evaluator apply the helper, FP32 masters remain in safetensors, missing legacy config resolves BF16, invalid values and deployment/publish mismatches fail closed.
- [ ] **Step 2: Run RED** on the focused JAX validation/publish/remote tests; expect missing field/helper and FP32 inference failures.
- [ ] **Step 3: Implement the helper and config/contract plumbing**; replace trainer's private cast, invoke once in policy/evaluator constructors, serialize effective config, and preserve full FP32 masters in save/resume.
- [ ] **Step 4: Run GREEN** on the focused tests; then run one fixed batch/rng/noise numerical parity test before/after save-load.
- [ ] **Step 5: Commit** only Task 2 files with `git commit -m "fix: unify VT BF16 compute parameters"`.

### Task 3: Immutable train-only normalization protocol

**Files:**
- Create: `src/lerobot/policies/smolvla_jax/normalization_protocol.py`
- Modify: `src/lerobot/policies/smolvla_jax/data.py`
- Modify: `tools/train_smolvla_jax.py`
- Modify: `tools/train_vtsmolvla_jax.py`
- Modify: `configs/train_vtsmolvla_jax_tactile16.yaml`
- Modify: `configs/train_vtsmolvla_jax_tactile32.yaml`
- Test: `tests/jax/test_normalization_protocol.py`
- Test: `tests/jax/test_data.py`
- Test: `tests/jax/test_train_script.py`
- Test: `tests/jax/test_train_vtsmolvla_config.py`

**Interfaces:**
- Produces: `build_or_validate_normalization_protocol(...)` returning canonical train-only stats, split path and manifest path.
- Consumes: persisted episode split, dataset sources/action key/rename map, `load_nested_dataset`, `cast_stats_to_numpy`, and `aggregate_stats`.

- [ ] **Step 1: Write failing synthetic tests** with one extreme validation episode, multiple train episode counts, four-source aggregation, missing/duplicate/non-finite/wrong-shape metadata, reuse, corruption, and source/split drift.
- [ ] **Step 2: Run RED** with `python -m pytest -p no:cacheprovider tests/jax/test_normalization_protocol.py tests/jax/test_data.py tests/jax/test_train_script.py tests/jax/test_train_vtsmolvla_config.py -q`; expect current full-dataset stats leakage and missing artifact errors.
- [ ] **Step 3: Implement metadata-only aggregation** using predicate-pushed episode parquet reads, action canonicalization, deterministic float32 hashing, staging/atomic rename, and strict existing-artifact validation.
- [ ] **Step 4: Integrate before loader construction** so train and val share the train-only preprocessor; copy manifest/split into checkpoints and verify provenance before resume step 1.
- [ ] **Step 5: Run GREEN** and assert K8/K21 resolve byte-identical normalization digests while their only model difference remains K.
- [ ] **Step 6: Commit** only Task 3 files with `git commit -m "fix: normalize VT training from train episodes only"`.

### Task 4: Regression and H100 handoff

**Files:**
- Modify: `CODEBASE_MEMORY.md`
- Create: `docs/reports/2026-08-08-h100-vt-training-consistency.md`

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: verified CPU evidence and exact two-H100 smoke commands.

- [ ] **Step 1: Run focused tests** from Tasks 1-3 and record exact counts/output.
- [ ] **Step 2: Run full CPU regression** with `JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider modalities_eval/test tests/jax tests/test_start_vtsmolvla_train.py tests/test_download_ckpt.py -q`.
- [ ] **Step 3: Run static gates** with `python -m py_compile` for changed Python files, `bash -n` for changed shell files, and `git diff --check`.
- [ ] **Step 4: Run checkpoint/schema smoke** proving tactile projection shape and cache metadata are unchanged and K8/K21 parse with the same protocol/encoder/dtype contracts.
- [ ] **Step 5: Review the complete diff** for Critical/Important findings and fix any in scope through RED/GREEN tests.
- [ ] **Step 6: Update memory and report** with limitations: CPU verification is not a real H100 training result.
- [ ] **Step 7: Commit** documentation with `git commit -m "docs: record VT H100 training consistency verification"`.

The server owner then runs K8 and K21 one-step two-H100 smoke configs, including save, followed by a same-total-steps strict resume smoke. Production runs remain blocked until both complete without OOM, non-finite loss, cache/contract error or resume mismatch.
