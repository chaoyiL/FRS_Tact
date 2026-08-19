# pi05_jax

pi0.5 in JAX, vendored from [openpi](https://github.com/Physical-Intelligence/openpi)
(Physical Intelligence, Apache-2.0), commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`
(`main`, 2026-06-16). This is FRS's base model, and the only policy on this branch
(see [pi05_frs_plan.md](../../../../pi05_frs_plan.md) at the repo root for the decision history).

## Why vendored instead of a dependency

`pip install openpi` pulls in the official `lerobot` package (pinned to a specific commit),
which collides by package name with this repo's own `lerobot` package (`src/lerobot/`, this very
tree). Rather than renaming this repo's package to work around that, or running pi0.5 in a
separate environment and shuttling data across a process boundary, openpi's JAX code is copied
directly into this repo. That is also why the main `pyproject.toml` pins
jax/flax/transformers/orbax to openpi's exact versions on this branch.

## Layout: which openpi file each module came from

| here | openpi | verbatim? |
| --- | --- | --- |
| `array_typing.py` | `shared/array_typing.py` | yes |
| `download.py` | `shared/download.py` | yes |
| `image_tools.py` | `shared/image_tools.py` | yes |
| `nnx_utils.py` | `shared/nnx_utils.py` | yes |
| `normalize.py` | `shared/normalize.py` | yes |
| `transforms.py` | `transforms.py` | yes |
| `gemma.py` `lora.py` `siglip.py` `pi0.py` `pi0_config.py` `tokenizer.py` | `models/*` | yes |
| `utils/fsq_tokenizer.py` | `models/utils/fsq_tokenizer.py` | yes |
| `model.py` | `models/model.py` | trimmed (see below) |
| `training/sharding.py` `training/utils.py` `training/optimizer.py` `training/weight_loaders.py` `training/checkpoints.py` | `training/*` | yes |
| `training/data_loader.py` | `training/data_loader.py` | adapted (see below) |
| `training/config.py` | `training/config.py` | adapted (see below) |
| `policies/pick_tube_policy.py` | `policies/libero_policy.py` (as template) | new, same contract |
| `policy_config.py` | `policies/policy_config.py` | trimmed to weight loading |
| `frs.py` | -- | new, FRS-only |
| `../../../../tools/train_pi05_jax.py` | `scripts/train.py` | adapted (imports only) |
| `../../../../tools/compute_pi05_norm_stats.py` | `scripts/compute_norm_stats.py` | adapted |

"verbatim" means: byte-for-byte upstream apart from a provenance header comment and rewritten
import lines (`openpi.shared.x` -> `. x`, `openpi.training.x` -> `.training.x`, ...). That is
mechanically checked -- diffing any of those files against upstream should show only the header
and the import block.

### The deviations, and why

- **`model.py`** drops `BaseModelConfig.load_pytorch` and its `openpi.models_pytorch.pi0_pytorch`
  import. That path loads a *PyTorch* checkpoint via openpi's PyTorch mirror model, which itself
  vendors a large patch of transformers internals. This branch wants the native JAX/orbax
  checkpoint. Everything else in the file is upstream.
- **`training/data_loader.py`** replaces upstream's
  `import lerobot.common.datasets.lerobot_dataset` with this repo's
  `lerobot.datasets.LeRobotDataset` -- that is the whole reason openpi is vendored (see above).
  It also concatenates several datasets (`DataConfig.sources`), because the pick_tube capture is
  split across four LeRobot datasets while upstream assumes one `repo_id`; and it adds two small
  local transforms, `RenameKeys` and `PromptFromTask`, so the per-dataset camera renumbering
  happens without editing the verbatim `transforms.py`. Upstream's RLDS/DROID data path is not
  vendored (needs TensorFlow).
- **`training/config.py`** keeps upstream's `AssetsConfig` / `DataConfig` / `ModelTransformFactory`
  / `DataConfigFactory` / `TrainConfig` / `cli()` / `get_config()` and swaps upstream's robot
  configs for `LeRobotPickTubeDataConfig` plus the `_CONFIGS` entries this repo trains. It drops
  `TrainConfig.pytorch_weight_path` / `pytorch_training_precision` (no PyTorch mirror model) and
  the pi0-FAST branch of `ModelTransformFactory` (`models/pi0_fast.py` not vendored).
- **`policy_config.py`** keeps only the weight-loading half of upstream's `create_trained_policy`.
  FRS never serves a policy over the wire; it drives the model directly.
- **`frs.py`** is the only genuinely new model-level logic: `build_prefix_cache` / `denoise_step` /
  `Pi0PrefixCache`. FRS integrates the flow-matching velocity field *backwards* (t:0->1, see
  `../../../../utils/integration.py`), which needs v(x, t) at arbitrary (x, t), while upstream
  `sample_actions` only exposes a fixed t:1->0 loop with the per-step logic inlined in a closure.
  `frs.denoise_step` is that closure body, copied out and made independently callable. It lives
  in its own module precisely so `pi0.py` stays a verbatim copy -- so a bug in `frs.py` cannot
  affect ordinary `sample_actions` sampling.

Not vendored, not needed: `vit.py`/`gemma_fast.py`/`pi0_fast.py` (pi0.5's vision tower is
`siglip.py`, which is self-contained), the entire `openpi.models_pytorch` package, `serving/`,
`training/droid_rlds_dataset.py`, and upstream's Aloha/DROID/Libero policy modules.

Note: because `tokenizer.py` is now the full upstream file, importing this package pulls in
`transformers` (for `FASTTokenizer`'s `AutoProcessor`) and `chex` (for `utils/fsq_tokenizer.py`)
at import time, exactly as it does upstream. pi0.5 itself only ever uses `PaligemmaTokenizer`.

## Training

Training configs are Python, not YAML -- upstream's `TrainConfig` + `tyro`, registered in
`training/config.py`'s `_CONFIGS`:

```bash
# 1. Norm stats (openpi's data pipeline hard-fails without them).
python tools/compute_pi05_norm_stats.py --config-name=pi05_pick_tube

# 2. Train. Any TrainConfig field is overridable on the command line.
python tools/train_pi05_jax.py pi05_pick_tube --exp_name=lora_r1
python tools/train_pi05_jax.py pi05_pick_tube --exp_name=lora_r1 --resume
python tools/train_pi05_jax.py pi05_pick_tube --exp_name=lora_r1 --batch_size=16 --fsdp_devices=2
```

`scripts/start_pi05_train.sh` wraps both steps and backgrounds the run in tmux.

Registered configs: `pi05_pick_tube` (LoRA, EMA off), `pi05_pick_tube_full` (full fine-tune,
EMA 0.999), `debug` (random data, no dataset or checkpoint needed -- use this to smoke-test the
whole stack).

Checkpoints follow openpi's layout (orbax `CheckpointManager`, one directory per step containing
`params/`, `train_state/` and `assets/`), so a checkpoint written here loads with
`policy_config.load_pi0` and with an upstream openpi checkout.

## Status

Verified on a Linux training server with two NVIDIA H100 80GB GPUs (before this rewrite):
- `uv sync --frozen` resolves and installs the pinned dependency set with NumPy 2.2.6.
- JAX 0.5.3 and PyTorch both detect both GPUs.
- `load_pi0("gs://openpi-assets/checkpoints/pi05_base", config=Pi0Config(pi05=True,
  action_dim=32, action_horizon=50))` restores the official checkpoint successfully.

Not yet run since the rewrite (this dev machine has no jax/torch installed; verification happens
on the training server):
1. `python tools/train_pi05_jax.py debug --exp_name=smoke` -- exercises the whole training stack
   on fake data, no dataset or checkpoint required. Run this first.
2. `python tools/compute_pi05_norm_stats.py --config-name=pi05_pick_tube`, then a short
   `pi05_pick_tube` run.
3. Cross-check `frs.build_prefix_cache` + `frs.denoise_step` against `Pi0.sample_actions` on the
   same input: running `sample_actions` normally vs. driving `denoise_step` manually one step at
   a time (both t:1->0) must produce identical actions. This is the one piece of genuinely new
   model code here, so it is the one most worth verifying independently before relying on it for
   FRS's reverse integration.
4. A complete `sample_and_reverse` on the configured real dataset via
   `modalities_eval/pi05_utils.py` + `utils/pi05_source_model.py`.

## What's built on top of this

- `modalities_eval/pi05_utils.py` (`Pi05SampleProcessor` / `Pi05EvalModel`) -- dataset sample ->
  `Observation`. Composes the same vendored transforms the trainer uses, in the same order, so
  action-cache generation and training preprocess identically.
- `utils/pi05_source_model.py` -- `frs.build_prefix_cache`/`frs.denoise_step` wired into
  `utils/integration.py`'s euler/fireflow solvers.
- `prepare_pi05.py` + `tools/prepare_frs_pi05_cache.py` -- the FRS action_cache tool.
