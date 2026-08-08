# VT H100 training consistency handoff (2026-08-08)

## What was verified locally

All pytest commands used `/home/yunjing/FRS/FRS_Tact/.venv/bin/python`, `JAX_PLATFORMS=cpu`, `PYTHONPATH=src:.`, `PYTHONDONTWRITEBYTECODE=1`, and `-p no:cacheprovider`.

- Task 1 focused: `26 passed in 0.20s`.
- Task 2 focused: `151 passed in 5.35s`.
- Task 3 focused: `59 passed in 3.29s`.
- Accepted VT CPU regression: `307 passed, 1 skipped, 2 deselected in 8.03s`.

The literal unfiltered task-brief command stopped at collection with three pre-existing pure-visual `train_smolvla` import errors: missing `train_smolvla.tactile_cache` in `test_functional.py`, and missing `initialize_tactile_fusion_params` in `test_lora.py` and `test_training.py`. Those modules are outside the user-approved VT-only BF16 acceptance boundary; they were neither repaired nor counted as passing. The accepted command below explicitly excludes them and the two established non-VT assertions.

Static gates passed for all 28 Python and two shell files changed since the design base: `py_compile`, `bash -n`, `git diff --check b40a827..HEAD`, and the clean-worktree `git diff --check` all exited zero.

The exact static manifest command was:

```bash
mapfile -t frs_task4_py_files < <(git diff --name-only b40a827..HEAD -- '*.py')
printf 'changed_python_count=%s\n' "${#frs_task4_py_files[@]}"
printf '%s\n' "${frs_task4_py_files[@]}"
env PYTHONDONTWRITEBYTECODE=1 /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m py_compile "${frs_task4_py_files[@]}"
mapfile -t frs_task4_sh_files < <(git diff --name-only b40a827..HEAD -- '*.sh')
printf 'changed_shell_count=%s\n' "${#frs_task4_sh_files[@]}"
bash -n "${frs_task4_sh_files[@]}"
git diff --check b40a827..HEAD
git diff --check
```

It emitted 28 Python paths and two shell paths; the full emitted path list is retained in the Task 4 report.

The schema/config smoke opened the existing VT checkpoint at `/home/yunjing/FRS/KaiyueChen/vtsmolvla_01_3w`: projection weight/bias shapes remain `(960, 512)` / `(960,)`. A fresh runtime cache fixture validated cache metadata version `1` and `[F,4,512]`; neither K changes its four source tokens or 512-dimensional cache. K8 and K21 share `/workspace/checkpoints/encoder_ckpt_05`, the BF16 compute default, and `/workspace/normalization_protocols/pick_tube_vt_k8_k21`; their effective tactile token counts are 32 and 84. After removing output, normalization, W&B name/tags, and repeat factor, their YAML mappings are identical.

This is CPU and schema evidence only. It is not a two-H100 run, training result, or H100 memory/performance result.

## Exact server procedure

Run on the H100 server from `/workspace/FRS_Tact`. Use one JAX process across both GPUs; do not use `torchrun` or two Slurm tasks. This procedure deliberately does not fetch or pull: update the checkout only when the server owner has explicitly approved a target revision.

### 1. Preflight the exact two-device process

```bash
cd /workspace/FRS_Tact
set -Eeuo pipefail
git switch Lee
test -z "$(git status --porcelain)"
export PYTHONPATH=src:.
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p /workspace/vt_h100_smoke/logs
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee /workspace/vt_h100_smoke/logs/preflight-nvidia-smi.log
test "$(grep -ci 'H100' /workspace/vt_h100_smoke/logs/preflight-nvidia-smi.log)" -eq 2
uv run --no-sync python -c 'import jax; devices = jax.devices(); print(devices); assert len(devices) == 2 and all(d.platform == "gpu" and "H100" in d.device_kind.upper() for d in devices), devices' 2>&1 | tee /workspace/vt_h100_smoke/logs/preflight-jax.log
uv run --no-sync python - <<'PY' 2>&1 | tee /workspace/vt_h100_smoke/logs/preflight-schema.log
import json
import tempfile
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path("configs/train_vtsmolvla_jax_tactile16.yaml").read_text())
model = cfg["model"]
cache_root = Path(cfg["tactile_embedding_cache"]["root"])
cache_root.mkdir(parents=True, exist_ok=True)
assert cache_root.is_dir(), cache_root
probe = None
try:
    with tempfile.NamedTemporaryFile(dir=cache_root, prefix=".frs_write_probe-", delete=False) as file:
        probe = Path(file.name)
        file.write(b"ok")
finally:
    if probe is not None:
        probe.unlink(missing_ok=True)
encoder = Path(model["tactile_encoder_path"])
checkpoint_json = encoder / "checkpoint.json"
assert checkpoint_json.is_file(), checkpoint_json
params_file = encoder / json.loads(checkpoint_json.read_text()).get("params_file", "params.npz")
assert params_file.is_file(), params_file
for source in cfg["datasets"]:
    info_path = Path(source["root"]) / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    assert str(info.get("codebase_version", "")).startswith("v3."), (info_path, info.get("codebase_version"))
    features = info["features"]
    assert source["action_key"] in features and features[source["action_key"]]["shape"] == [model["action_dim"]], info_path
    assert features["observation.state"]["shape"] == [model["state_dim"]], info_path
    reverse_rename = {target: original for original, target in source.get("rename_map", {}).items()}
    rgb_keys = [reverse_rename.get(key, key) for key in model["image_keys"]]
    required_images = rgb_keys + list(model["tactile_keys"])
    for key in required_images:
        feature = features[key]
        assert feature["dtype"] == "image" and feature["shape"] == [224, 224, 3], (info_path, key, feature)
    print(f"schema_ok {source['repo_id']} {info_path}")
print(f"cache_root_writable {cache_root}")
print(f"encoder_files {checkpoint_json} {params_file}")
PY
```

The branch must be `Lee` and the tree clean. Proceed only when both the `nvidia-smi` names and the two JAX `device_kind` values contain `H100`, every v3 dataset passes the action/RGB/tactile/state schema check, the encoder files exist, and the cache root was created or verified writable.

### 2. Materialize the cache once

K8 and K21 have identical raw tactile-cache inputs. Reserve one H100 for this one-time cache operation, then reuse its four completed source caches for both runs. Do not pass `--overwrite` on a production-compatible cache.

```bash
cd /workspace/FRS_Tact
set -Eeuo pipefail
mkdir -p /workspace/vt_h100_smoke/logs
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python tools/precompute_tactile_embeddings.py \
  --config configs/train_vtsmolvla_jax_tactile16.yaml \
  2>&1 | tee /workspace/vt_h100_smoke/logs/cache-k8.log
```

Every source must report complete, or skip as already complete. A fingerprint, shape, encoder, metadata, or incomplete-cache error blocks the smoke.

### 3. One-step two-H100 save smoke, K8 and K21

Choose new, nonexistent outputs. These commands keep the YAML's `data_parallel: true`, batch size 64, BF16 contract, shared normalization protocol, and cached tactile inputs; only steps, frequencies, output, and disabled W&B logging are smoke-sized. Create non-overwriting, W&B-disabled copies first because the source YAMLs explicitly set W&B to online mode.

```bash
cd /workspace/FRS_Tact
set -Eeuo pipefail
mkdir -p /workspace/vt_h100_smoke
test ! -e /workspace/vt_h100_smoke/k8-smoke.yaml
test ! -e /workspace/vt_h100_smoke/k21-smoke.yaml
uv run --no-sync python -c 'from pathlib import Path; import yaml; root=Path("/workspace/vt_h100_smoke"); pairs=((Path("configs/train_vtsmolvla_jax_tactile16.yaml"), root/"k8-smoke.yaml"), (Path("configs/train_vtsmolvla_jax_tactile32.yaml"), root/"k21-smoke.yaml")); [(lambda c, d: (c.__setitem__("wandb", {**dict(c.get("wandb") or {}), "enabled": False, "mode": "offline"}), d.write_text(yaml.safe_dump(c, sort_keys=False)))(yaml.safe_load(s.read_text()), d) for s, d in pairs]'
test ! -e /workspace/vt_h100_smoke/k8-one-step
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python tools/train_vtsmolvla_jax.py \
  --config /workspace/vt_h100_smoke/k8-smoke.yaml \
  --steps 1 --save-freq 1 --eval-freq 1 --output /workspace/vt_h100_smoke/k8-one-step \
  2>&1 | tee /workspace/vt_h100_smoke/logs/k8-one-step.log

test ! -e /workspace/vt_h100_smoke/k21-one-step
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python tools/train_vtsmolvla_jax.py \
  --config /workspace/vt_h100_smoke/k21-smoke.yaml \
  --steps 1 --save-freq 1 --eval-freq 1 --output /workspace/vt_h100_smoke/k21-one-step \
  2>&1 | tee /workspace/vt_h100_smoke/logs/k21-one-step.log

test -d /workspace/vt_h100_smoke/k8-one-step/checkpoint-00000001
test -d /workspace/vt_h100_smoke/k21-one-step/checkpoint-00000001
```

Each run must print a finite step-1 loss and `saved checkpoint`. Preserve both logs and checkpoint directories.

### 4. Strict-resume smoke with the same total-step contract

For each K, create a two-step seed run and resume its step-1 checkpoint into a separate empty output, retaining `--steps 2` in both commands. This executes a resumed step while avoiding overwrite of the seed run's checkpoint-2 directory.

```bash
cd /workspace/FRS_Tact
set -Eeuo pipefail
test ! -e /workspace/vt_h100_smoke/k8-resume-seed
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python tools/train_vtsmolvla_jax.py \
  --config /workspace/vt_h100_smoke/k8-smoke.yaml \
  --steps 2 --save-freq 1 --eval-freq 1 --output /workspace/vt_h100_smoke/k8-resume-seed \
  2>&1 | tee /workspace/vt_h100_smoke/logs/k8-resume-seed.log
test ! -e /workspace/vt_h100_smoke/k8-resume-strict
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python tools/train_vtsmolvla_jax.py \
  --config /workspace/vt_h100_smoke/k8-smoke.yaml \
  --steps 2 --save-freq 1 --eval-freq 1 \
  --resume /workspace/vt_h100_smoke/k8-resume-seed/checkpoint-00000001 \
  --output /workspace/vt_h100_smoke/k8-resume-strict \
  2>&1 | tee /workspace/vt_h100_smoke/logs/k8-resume-strict.log

test ! -e /workspace/vt_h100_smoke/k21-resume-seed
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python tools/train_vtsmolvla_jax.py \
  --config /workspace/vt_h100_smoke/k21-smoke.yaml \
  --steps 2 --save-freq 1 --eval-freq 1 --output /workspace/vt_h100_smoke/k21-resume-seed \
  2>&1 | tee /workspace/vt_h100_smoke/logs/k21-resume-seed.log
test ! -e /workspace/vt_h100_smoke/k21-resume-strict
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python tools/train_vtsmolvla_jax.py \
  --config /workspace/vt_h100_smoke/k21-smoke.yaml \
  --steps 2 --save-freq 1 --eval-freq 1 \
  --resume /workspace/vt_h100_smoke/k21-resume-seed/checkpoint-00000001 \
  --output /workspace/vt_h100_smoke/k21-resume-strict \
  2>&1 | tee /workspace/vt_h100_smoke/logs/k21-resume-strict.log

test -d /workspace/vt_h100_smoke/k8-resume-strict/checkpoint-00000002
test -d /workspace/vt_h100_smoke/k21-resume-strict/checkpoint-00000002
```

The resumed process must validate checkpoint normalization provenance before restore, execute step 2, and write `checkpoint-00000002`.

## Production block criteria

The 40k K8/K21 runs remain blocked until both one-step and strict-resume runs complete on two H100s. Block immediately on fewer than two JAX GPUs, OOM, non-finite loss/gradient, cache metadata or `[F,4,512]` mismatch, encoder/BF16/normalization contract error, protocol provenance drift, resume-signature mismatch, absent checkpoint, or failure to create the resumed step-2 checkpoint. Keep K8 and K21 outputs separate: cross-K optimizer resume is intentionally invalid.

## Accepted CPU command

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/frs_task4_matplotlib XDG_CACHE_HOME=/tmp/frs_task4_xdg HF_DATASETS_CACHE=/tmp/frs_task4_full_accepted /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider modalities_eval/test tests/jax tests/test_start_vtsmolvla_train.py tests/test_download_ckpt.py --ignore=tests/jax/test_functional.py --ignore=tests/jax/test_lora.py --ignore=tests/jax/test_training.py --deselect=tests/jax/test_checkpoint.py::test_processor_configs_sync_rename_map_and_feature_shapes --deselect=tests/jax/test_tactile_integration.py::test_default_deployment_config_pins_the_bimanual_vt_contract -q
```

The three ignores are known pure-visual collection blockers outside the VT-only scope. The first deselection imports the same missing pure-visual cache boundary; the second retains the pre-existing default deployment identity expectation (`KaiyueChen/vtsmolvla_01_4w`) that conflicts with the checked-in non-VT default path. Neither is evidence that those pure-visual paths pass.
