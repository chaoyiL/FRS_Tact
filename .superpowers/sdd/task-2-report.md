# Task 2 report: visual-only SmolVLA JAX core

## Status

DONE_WITH_CONCERNS

The standalone `train_smolvla` package now contains the JAX SmolVLA core with no tactile encoder, cache, fusion, validation-module, LoRA, preprocessing, policy, or trainer branches. The legacy `src/lerobot/policies/smolvla_jax` package was left unchanged.

## TDD evidence

### RED

After adding the two isolation tests from the task brief:

```text
.venv/bin/python -m pytest -q tests/train_smolvla/test_package_boundary.py
..F
ImportError: cannot import name 'JaxSmolVLAConfig' from 'train_smolvla'
1 failed, 2 passed in 0.01s
```

The failure was expected: the empty namespace did not yet expose a visual config.

### GREEN

Final focused verification:

```text
.venv/bin/python -m pytest -q tests/train_smolvla/test_package_boundary.py tests/jax/test_functional.py tests/jax/test_checkpoint.py tests/jax/test_lora.py tests/jax/test_modality_dropout.py tests/jax/test_training.py
...........................................
43 passed in 5.77s
```

An additional boundary audit imported all 15 visual package modules, verified that `JaxSmolVLAConfig` has no field containing `tactile`, and verified that neither `tactile_encoder` nor `train_vtsmolvla` was loaded. `rg` found no tactile or legacy-package references in `train_smolvla`; `train_smolvla/tactile_cache.py` and `train_smolvla/validation.py` do not exist.

## Implementation notes

- Copied the visual JAX core into the standalone top-level package.
- Removed tactile fields and override validation from `JaxSmolVLAConfig`.
- Removed tactile parameter initialization and effective-config persistence.
- Removed tactile cache/data-loading, preprocessing, model fusion, inference, LoRA, and trainer paths.
- Added a neutral model hook returning `(loss, metrics)` and changed training/evaluation to pass the complete batch mapping through `compute_training_loss(params, batch=batch, rng=...)`.
- Updated pure-visual focused tests to import `train_smolvla`; existing tactile assertions continue to target the legacy package pending Task 3.

## Concern: concurrent external commit/reset

The assigned baseline was `61554ce`. During execution, an external process advanced HEAD to `6c624b9` (`first round change`) and included unrelated deploy/docs changes together with the first half of this task (new modules, tests, and the already-purified configuration/checkpoint/LoRA files). It then reset the remaining working-tree changes to that HEAD. I did not rewrite, reset, restore, or stage the unrelated user changes. I reapplied only the remaining six visual-core patches and committed them separately with the required message. Consequently, the complete Task 2 change is split across external commit `6c624b9` and the Task 2 commit recorded below, rather than being wholly contained in a single clean commit based directly on `61554ce`.

## Commit

This report is included in the visual-core completion commit with subject
`refactor: move visual SmolVLA JAX core`; the resulting hash is reported to the parent task.
