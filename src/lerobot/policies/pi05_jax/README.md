# pi05_jax

Pure-JAX pi0.5, vendored from [openpi](https://github.com/Physical-Intelligence/openpi)
(Physical Intelligence, Apache-2.0), commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`
(`main`, 2026-06-16). This is FRS's base model on this branch, replacing `../smolvla_jax/`
(see [pi05_frs_plan.md](../../../../pi05_frs_plan.md) at the repo root for the full decision
history and remaining TODOs).

## Why vendored instead of a dependency

`pip install openpi` pulls in the official `lerobot` package (pinned to a specific commit),
which collides by package name with this repo's own `lerobot` package (`src/lerobot/`, this very
tree). Rather than renaming this repo's package to work around that, or running pi0.5 in a
separate environment and shuttling data across a process boundary, we vendor just the pi0.5
*model* code (no training/data/policy infrastructure, no `openpi` package as a whole) directly
into this repo, the same way `../smolvla_jax/` is a from-scratch JAX port rather than a
dependency on PyTorch LeRobot.

This is why the main `pyproject.toml`'s jax/flax/transformers/orbax-checkpoint pins were changed
to match openpi's exactly on this branch (see the comment there) -- `../smolvla_jax/` will very
likely break as a result. That's expected: this branch's FRS uses pi0.5, not SmolVLA.

## What's here vs. what was trimmed

Vendored close to verbatim (only import paths rewritten to local modules):
`array_typing.py`, `nnx_utils.py`, `image_tools.py`, `download.py`, `lora.py`, `gemma.py`,
`siglip.py`, `pi0_config.py`.

Vendored and trimmed:
- `model.py`: dropped `BaseModelConfig.load_pytorch` and its `openpi.models_pytorch.pi0_pytorch`
  import. That path loads a *PyTorch* checkpoint (`model.safetensors`) via openpi's PyTorch
  mirror model (which itself vendors a large patch of transformers internals) -- not needed since
  we specifically want the native JAX/orbax checkpoint.
- `tokenizer.py`: kept only `PaligemmaTokenizer` (what pi0.5 actually uses). Dropped
  `FASTTokenizer` / `BinningTokenizer` / `FSQTokenizer`, which are for pi0-FAST and RoboArena
  baselines, not pi0/pi0.5; `FSQTokenizer` additionally needs
  `openpi.models.utils.fsq_tokenizer`, which was not vendored.

Not vendored, not needed: `vit.py`/`resnet.py` (pi0.py's vision tower is `siglip.py`, which is
self-contained and never imports `vit.py`); the entire `openpi.models_pytorch` package; `openpi`'s
training/data/transforms/policy layers (see "What's still missing" below).

New, not from openpi:
- `sharding.py`: a no-op stand-in for `openpi.training.sharding.activation_sharding_constraint`.
  Upstream's real implementation *is* a no-op unless a multi-host device mesh has been set up via
  `sharding.set_mesh(...)` (see its docstring for the exact source reasoning) -- FRS only runs
  pi0.5 for single-device action-cache inference and never calls `set_mesh`, so this is an exact
  behavioral match, not an approximation.
- `checkpoint.py`: `load_pi0()` restores a native JAX/orbax checkpoint, following the same
  `restore_params` + `Config.load(params)` pattern openpi's own
  `policies.policy_config.create_trained_policy` uses.
- `pi0.py`'s `Pi0PrefixCache` / `Pi0.build_prefix_cache` / `Pi0.denoise_step`: FRS needs to run
  the flow-matching velocity field at arbitrary `(x, t)` to integrate it *backwards* (t:0->1, see
  `../../../../utils/integration.py`'s euler/fireflow solvers -- SmolVLA's side of this is
  `../../../../utils/source_model.py`'s `sample_and_reverse`/`denoise_step`). Upstream
  `sample_actions` only exposes a fixed t:1->0 forward loop with the per-step logic inlined in a
  closure, so these three additions extract that closure into a reusable, independently callable
  form. `sample_actions` itself is untouched (byte-for-byte from upstream), so ordinary forward
  sampling still goes through the exact original code path -- if the added methods have a bug,
  they don't affect `sample_actions`.

## Status: UNTESTED

Nothing in this directory has been run. This dev machine is macOS with no `jax`/`flax`/GPU
installed (`jax[cuda12-local]` is Linux+NVIDIA only), so none of this could be imported, let
alone executed against a real checkpoint, while writing it. Only verified so far:
`python -m py_compile` (syntax only) on every file here.

Before trusting this on the training server:
1. `uv sync` at the repo root with the new pins -- confirm the dependency set actually resolves
   (see the big comment block in `pyproject.toml`; `numpy` was deliberately **not** downgraded to
   openpi's `<2.0.0` pin because `opencv-python-headless`/`pyarrow`/`datasets` in this repo
   already require `numpy>=2.0.0` -- if `jax==0.5.3`/`flax==0.10.2` turn out to need numpy<2 for
   real (not just as a conservative pin), that's a genuine unresolved conflict, not just an
   unverified one).
2. `load_pi0("gs://openpi-assets/checkpoints/pi05_base")` and confirm the restored params
   actually match `Pi0Config(pi05=True)`'s shapes (`BaseModelConfig.load`'s
   `check_pytree_equality` will raise immediately if not).
3. Cross-check `build_prefix_cache` + `denoise_step` against upstream `sample_actions` on the
   same input: running `sample_actions` normally vs. driving `denoise_step` manually one step at
   a time (both t:1->0) should produce identical actions. This is the one piece of genuinely new
   code here (as opposed to a vendored copy), so it's the one most worth independently verifying
   before relying on it for FRS's reverse integration.

## What's built on top of this (and still needs verifying)

The observation-building, normalization, reverse-integration glue, and action_cache tool are all
written now -- see [pi05_frs_plan.md](../../../../pi05_frs_plan.md) at the repo root for the
full, current list of what's done vs. what's still an open question (in particular: which norm
stats to use, since `pi05_base`'s shipped assets don't include one for a brand-new dataset like
pick_tube). Pointers, so this file doesn't drift out of sync with that one:

- `modalities_eval/pi05_utils.py` (`Pi05EvalModel`) -- dataset sample -> `Observation`.
- `utils/pi05_source_model.py` -- `build_prefix_cache`/`denoise_step` wired into
  `utils/integration.py`'s euler/fireflow solvers.
- `prepare_pi05.py` + `tools/prepare_frs_pi05_cache.py` -- the actual action_cache tool.

None of it has been run (this dev machine has no jax/flax/GPU) -- see "Status: UNTESTED" above
for the verification checklist before trusting any of this on real data.
