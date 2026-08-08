# Three-Script H100 Training Workflow Design

## Goal

Provide exactly three user-facing scripts under `scripts/` that prepare a fresh
two-H100 laboratory server and train the two tactile-token baselines in the fixed
order K8 then K21:

```text
setup_env.sh
  -> download_data.sh
  -> start_vtsmolvla_train.sh
       -> tactile cache once
       -> K8 training
       -> K21 training
```

This is a laboratory workflow. It does not add Git-SHA gates, Hub publication,
remote provenance locks, Docker, Slurm orchestration, `torchrun`, or
`accelerate`.

## Preconditions and fixed layout

- Server: Linux x86_64 with exactly two NVIDIA H100 GPUs visible to the job.
- The base image provides a compatible NVIDIA driver, CUDA 12 runtime/toolkit,
  cuDNN and NCCL. The environment script verifies these prerequisites; it does
  not install a kernel driver.
- Repository root may be anywhere, but persistent runtime storage defaults to
  `/workspace`.
- Python environment: `/workspace/.venvs/frs_tact`.
- Hugging Face and temporary caches derive only from `FRS_STORAGE_ROOT`, default
  `/workspace`, and are persisted in `.env.frs`.
- v3 datasets:
  `/workspace/lerobot_v30/KaiyueChen/pick_tube_01` through `pick_tube_04`.
- Tactile encoder: `/workspace/checkpoints/encoder_ckpt_05`, downloaded from
  `liuchaoyi/encoder_ckpt_05`.
- Tactile embedding cache: `/workspace/tactile_embeddings`.
- Shared train-only normalization protocol:
  `/workspace/normalization_protocols/pick_tube_vt_k8_k21`.
- K8 config: `configs/train_vtsmolvla_jax_tactile16.yaml`.
- K21 config: `configs/train_vtsmolvla_jax_tactile32.yaml`.

## Script 1: `scripts/setup_env.sh`

The script installs system CLI prerequisites when permitted, installs `uv`, and
runs `uv sync --frozen` against the repository lock. It writes `.env.frs`
atomically with the final values of:

- `FRS_STORAGE_ROOT`;
- `FRS_VENV_DIR` and `UV_PROJECT_ENVIRONMENT`;
- `UV_CACHE_DIR`;
- `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `HF_LEROBOT_HOME`;
- `TMPDIR`.

All later scripts require and source `.env.frs`; they do not recompute or
overwrite these values. The setup script does not use a checkout-local `.venv`
as an implicit fallback.

Setup fails unless:

- `nvidia-smi` reports exactly two GPUs and both names contain `H100`;
- the NVIDIA driver is at least `570.86`;
- the locked JAX local-CUDA build can find CUDA `>=12.1`, cuDNN `>=9.8`, NCCL
  `>=2.18`, and `libdevice10.bc`;
- PyTorch sees two CUDA devices;
- JAX sees exactly two GPU devices;
- a minimal JAX two-device sharding/collective computation returns the expected
  value.

Repeated setup runs reuse the same environment and caches. Project concurrency
is guarded by a scoped lock rather than scanning unrelated system `uv` jobs.

## Script 2: `scripts/download_data.sh`

The script requires `.env.frs`, obtains an exclusive download/conversion lock,
and prepares the following four repositories:

- `KaiyueChen/pick_tube_01`;
- `KaiyueChen/pick_tube_02`;
- `KaiyueChen/pick_tube_03`;
- `KaiyueChen/pick_tube_04`.

Existing valid v3 destinations are reused. A downloaded v2.1 snapshot is copied
to a bounded work directory and converted with LeRobot's existing official v3
converter; the Hub snapshot itself is not modified. By default the raw snapshot
is retained. `--cleanup-source` is the only mode authorized to remove the four
known source snapshots and conversion leftovers after successful validation.

For every final dataset, the script performs a loader-backed contract check:

- `codebase_version` is v3;
- two RGB keys resolve through the configured rename map;
- four tactile image keys exist;
- `observation.state` has width 20;
- `actions` has width 20;
- episode/global indices and required normalization stats are valid;
- at least one sample decodes successfully.

After all four datasets pass, the script calls the existing checkpoint downloader
in minimal mode. It downloads and validates `liuchaoyi/encoder_ckpt_05` at the
path used by both training configs. The download script supports bounded retry by
rerunning; completed data and encoder artifacts are skipped or revalidated.

The script does not precompute tactile embeddings. That GPU operation belongs to
the training launcher.

## Script 3: `scripts/start_vtsmolvla_train.sh`

With no arguments, the launcher performs the complete sequence:

```text
preflight -> cache once -> K8 -> K21
```

It sets `CUDA_VISIBLE_DEVICES=0,1` before any JAX process and fails unless JAX
sees exactly two H100 devices. Both trainings use one Python process with JAX data
parallelism; the launcher never starts one process per GPU.

The supported interface is:

```text
--experiment both|k8|k21   default: both
--gpus 0,1                 default: 0,1
--cache auto|skip|only     default: auto
--resume none|auto|PATH    default: none; PATH only for a single experiment
--smoke                    create an untracked temporary one-step YAML
--foreground               do not create tmux
--session NAME             override the tmux session name
--help
```

Legacy `--config PATH` remains available for one custom experiment and is
mutually exclusive with `--experiment`. Unknown or duplicate options fail before
environment or training work starts. The tmux relaunch forwards the complete,
properly quoted original argument vector.

For `both`, cache precomputation runs exactly once using the K8 config. Completed
caches are validated and skipped by the existing precompute tool. The launcher
then runs K8 synchronously. Only exit status zero permits K21 to start. Each
experiment has its own output and log files while sharing the cache and
normalization protocol.

Fresh mode rejects an output containing `checkpoint-*` or incomplete checkpoint
staging. `--resume auto` selects the highest complete checkpoint for that
experiment. Explicit resume verifies the checkpoint directory and training
state before invoking the trainer. The trainer's existing strict signature check
remains authoritative.

Smoke mode never edits tracked YAML. It creates a temporary config with:

- one training step;
- global batch size 2;
- zero data-loader workers;
- log/save frequency 1;
- validation enabled with one batch and rollout disabled;
- W&B disabled;
- a unique smoke output.

Smoke mode still requires a complete tactile cache when `--cache skip` is used;
`--cache auto` may need to build the full cache before the one-step smoke.

## Error handling and safety

- Every script uses `set -Eeuo pipefail`, clear stage-prefixed messages, and
  nonzero exit status on a failed prerequisite.
- K8 failure prevents K21 startup.
- Existing checkpoints are never silently overwritten.
- Dataset deletion is opt-in and limited to the four fixed repositories under
  derived workspace roots.
- Concurrent setup, download/conversion, and launcher operations use separate
  scoped locks.
- Cache integrity may continue using the existing encoder/data fingerprint; this
  is runtime cache correctness, not publication-grade provenance.

## Verification

Tests cover argument parsing, environment path consistency, atomic `.env.frs`,
exact two-H100 validation, official v3 conversion orchestration, schema/sample
validation, encoder download integration, lock behavior, K8-to-K21 sequencing,
one cache call, failure short-circuiting, tmux argument forwarding, fresh/resume
guards, and temporary smoke config isolation.

Static gates are `bash -n` for all three scripts, Python compilation for helper
probes, `git diff --check`, and existing focused JAX/config regressions. Actual
completion still requires running the three scripts on the target two-H100
server; local CPU/4090 tests are not represented as two-H100 evidence.
