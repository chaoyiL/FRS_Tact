# Naive Tactile Token Ratio Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a zero-parameter repeat-and-concat baseline that gives the four tactile streams approximately 16% or 32% of VT-SmolVLA conditioning-prefix slots.

**Architecture:** Keep the existing four `[512]` ResNet/cache embeddings and shared `tactile_proj(512 -> 960)`. Inside prefix construction, repeat every projected sensor token and its mask K times in key-major order, with K configured by `tactile_token_repeat_factor`; all downstream attention, RoPE, training, inference and evaluation paths consume the expanded prefix automatically.

**Tech Stack:** Python 3.12, JAX 0.8.3, Flax, PyYAML, pytest, LeRobot v3 datasets, safetensors.

## Global Constraints

- Conditioning-prefix accounting is `128 RGB + 4K tactile + 48 allocated language slots + 1 state`; action/time suffix is excluded.
- K=1 is the unchanged 2.21% baseline; K=8 is 15.31% (paper label: about 16%); K=21 is 32.18% (paper label: about 32%).
- Keep `tactile_num_tokens=4` as the number of source sensor/cache streams; do not change tactile cache v1 shape `[F,4,512]`.
- Add no trainable parameters, modality/type embeddings, copy-slot embeddings, spatial tokens or new position-encoding logic.
- Expansion occurs only after the shared tactile projection. Repeat masks with the same key-major/token-minor layout.
- Old configs/checkpoints missing the new field behave exactly as K=1.
- K=1/8/21 paper runs must differ only in K, output identity and experiment tags; all training/data hyperparameters remain fixed.
- Preserve the existing dirty worktree changes. Do not stage or commit unrelated files; do not create a commit unless the user explicitly authorizes it at execution time.
- Use `/home/yunjing/FRS/FRS_Tact/.venv/bin/python` with `PYTHONPATH=src:.` from `/home/yunjing/FRS/.worktrees/FRS_Tact-modalities-eval`.

---

## File Map

- `src/lerobot/policies/smolvla_jax/configuration.py`: owns the new static config field, parsing, validation and effective-token property.
- `src/lerobot/policies/smolvla_jax/checkpoint.py`: persists the field in checkpoint `config.json`.
- `src/lerobot/policies/smolvla_jax/modeling.py`: expands projected tactile tokens and masks immediately before prefix concatenation.
- `src/lerobot/policies/smolvla_jax/training.py`: canonicalizes old resume signatures that predate the field.
- `tools/train_vtsmolvla_jax.py`: rejects malformed factors before loading checkpoints/data.
- `configs/train_vtsmolvla_jax.yaml`: explicit K=1 baseline.
- `configs/train_vtsmolvla_jax_tactile16.yaml`: K=8 paper variant.
- `configs/train_vtsmolvla_jax_tactile32.yaml`: K=21 paper variant.
- `tests/jax/test_checkpoint.py`: config parsing and persistence tests.
- `tests/jax/test_tactile_integration.py`: zero-parameter token/mask expansion tests.
- `tests/jax/test_training.py`: old resume migration and K mismatch tests.
- `tests/jax/test_train_vtsmolvla_config.py`: launcher validation and three-config parity tests.
- `CODEBASE_MEMORY.md`: durable implementation and verification record.

---

### Task 1: Add and persist the configuration contract

**Files:**
- Modify: `src/lerobot/policies/smolvla_jax/configuration.py:78-86,181-188,210-260`
- Modify: `src/lerobot/policies/smolvla_jax/checkpoint.py:364-414`
- Modify: `tests/jax/test_checkpoint.py`

**Interfaces:**
- Produces: `JaxSmolVLAConfig.tactile_token_repeat_factor: int` with default `1`.
- Produces: `JaxSmolVLAConfig.effective_tactile_num_tokens -> int`.
- Persists: JSON key `tactile_token_repeat_factor`.

- [ ] **Step 1: Write failing config tests**

Add these tests to `tests/jax/test_checkpoint.py`:

```python
def test_tactile_repeat_factor_defaults_validates_and_derives_effective_tokens() -> None:
    legacy = JaxSmolVLAConfig(
        use_tactile_encoder=True,
        tactile_encoder_path="encoder",
        tactile_keys=("t0", "t1", "t2", "t3"),
        tactile_num_tokens=4,
    )
    assert legacy.tactile_token_repeat_factor == 1
    assert legacy.effective_tactile_num_tokens == 4

    expanded = legacy.with_overrides({"tactile_token_repeat_factor": 8})
    assert expanded.tactile_token_repeat_factor == 8
    assert expanded.effective_tactile_num_tokens == 32

    for invalid in (0, -1, 1.5, True, "8"):
        with pytest.raises(ValueError, match="tactile_token_repeat_factor"):
            legacy.with_overrides({"tactile_token_repeat_factor": invalid})


def test_effective_config_persists_tactile_repeat_factor(tmp_path: Path) -> None:
    config = replace(
        JaxSmolVLAConfig(),
        use_tactile_encoder=True,
        tactile_encoder_path="encoder",
        tactile_keys=("t0", "t1", "t2", "t3"),
        tactile_num_tokens=4,
        tactile_token_repeat_factor=21,
    )
    write_effective_config(tmp_path, config)
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["tactile_token_repeat_factor"] == 21
    assert JaxSmolVLAConfig.from_pretrained(tmp_path).tactile_token_repeat_factor == 21


def test_from_pretrained_rejects_invalid_tactile_repeat_factor(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"tactile_token_repeat_factor": 0})
    )
    with pytest.raises(ValueError, match="tactile_token_repeat_factor"):
        JaxSmolVLAConfig.from_pretrained(tmp_path)
```

- [ ] **Step 2: Run the tests and verify the expected RED state**

Run:

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_checkpoint.py -q
```

Expected: failures report missing `tactile_token_repeat_factor` and `effective_tactile_num_tokens`.

- [ ] **Step 3: Implement the minimal field, parser, property and validation**

Add one validator near `_coerce_override_value`:

```python
def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value
```

In `JaxSmolVLAConfig`, add next to `tactile_num_tokens`:

```python
tactile_num_tokens: int = 4
tactile_token_repeat_factor: int = 1
```

Add the property:

```python
@property
def effective_tactile_num_tokens(self) -> int:
    return int(self.tactile_num_tokens) * int(self.tactile_token_repeat_factor)
```

In `from_pretrained`, preserve legacy behavior while rejecting malformed saved configs:

```python
tactile_token_repeat_factor=_require_positive_int(
    raw.get("tactile_token_repeat_factor", 1),
    "tactile_token_repeat_factor",
),
```

Before coercing overrides in `with_overrides`, validate the special field without silently truncating floats or accepting booleans/strings:

```python
if key == "tactile_token_repeat_factor":
    cleaned[key] = _require_positive_int(value, key)
    continue
```

After `updated = replace(...)`, retain a defensive invariant:

```python
if updated.tactile_token_repeat_factor < 1:
    raise ValueError("tactile_token_repeat_factor must be a positive integer")
```

In `write_effective_config`, add:

```python
"tactile_token_repeat_factor": config.tactile_token_repeat_factor,
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all `tests/jax/test_checkpoint.py` tests pass.

- [ ] **Step 5: Review the diff boundary**

Run:

```bash
git diff --check
git diff -- src/lerobot/policies/smolvla_jax/configuration.py \
  src/lerobot/policies/smolvla_jax/checkpoint.py tests/jax/test_checkpoint.py
```

Expected: no parameter initialization, cache or model code changed in this task.

---

### Task 2: Validate the factor at the VT launcher boundary

**Files:**
- Modify: `tools/train_vtsmolvla_jax.py:38-82`
- Create: `tests/jax/test_train_vtsmolvla_config.py`

**Interfaces:**
- Consumes: YAML `model.tactile_token_repeat_factor` from Task 1.
- Produces: `_validate_vt_config(path)` rejects non-positive/non-integer values before checkpoint or dataset access.

- [ ] **Step 1: Write failing launcher tests**

Create `tests/jax/test_train_vtsmolvla_config.py`:

```python
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tools.train_vtsmolvla_jax import _validate_vt_config


def _valid_config() -> dict:
    return {
        "model": {
            "use_tactile_encoder": True,
            "tactile_encoder_path": "encoder",
            "tactile_keys": ["t0", "t1", "t2", "t3"],
            "tactile_embedding_dim": 512,
            "tactile_num_tokens": 4,
            "tactile_token_repeat_factor": 8,
            "image_keys": ["camera1", "camera2"],
        },
        "tactile_embedding_cache": {"enabled": False},
    }


def _write(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_vt_launcher_accepts_legacy_default_and_paper_factors(tmp_path: Path) -> None:
    config = _valid_config()
    for factor in (1, 8, 21):
        config["model"]["tactile_token_repeat_factor"] = factor
        _validate_vt_config(_write(tmp_path / f"k{factor}.yaml", config))

    del config["model"]["tactile_token_repeat_factor"]
    _validate_vt_config(_write(tmp_path / "legacy.yaml", config))


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "8"])
def test_vt_launcher_rejects_invalid_repeat_factor(tmp_path: Path, invalid: object) -> None:
    config = deepcopy(_valid_config())
    config["model"]["tactile_token_repeat_factor"] = invalid
    with pytest.raises(ValueError, match="tactile_token_repeat_factor"):
        _validate_vt_config(_write(tmp_path / "invalid.yaml", config))
```

- [ ] **Step 2: Run the test and verify RED**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_train_vtsmolvla_config.py -q
```

Expected: invalid factors do not yet raise.

- [ ] **Step 3: Add fail-closed launcher validation**

After tactile key/count validation in `_validate_vt_config`, add:

```python
repeat_factor = model.get("tactile_token_repeat_factor", 1)
if (
    isinstance(repeat_factor, bool)
    or not isinstance(repeat_factor, int)
    or repeat_factor < 1
):
    raise ValueError(
        "model.tactile_token_repeat_factor 必须是正整数，"
        f"当前值：{repeat_factor!r}"
    )
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Review the launcher-only diff**

```bash
git diff --check
git diff -- tools/train_vtsmolvla_jax.py tests/jax/test_train_vtsmolvla_config.py
```

Expected: launcher validation only; no data loading or precompute behavior changed.

---

### Task 3: Repeat projected tactile tokens and masks in prefix construction

**Files:**
- Modify: `src/lerobot/policies/smolvla_jax/modeling.py:172-227,292-308`
- Modify: `tests/jax/test_tactile_integration.py`

**Interfaces:**
- Consumes: `config.tactile_token_repeat_factor` from Task 1.
- Produces: `_repeat_tactile_tokens_and_masks(tokens, masks, factor) -> tuple[Array, Array]`.
- Keeps: `embed_tactile(...) -> [B,S,H]` as the per-sensor projection API.

- [ ] **Step 1: Write failing pure tensor-contract tests**

Import the new helper in `tests/jax/test_tactile_integration.py` and add:

```python
from lerobot.policies.smolvla_jax.modeling import (
    JaxSmolVLA,
    _repeat_tactile_tokens_and_masks,
    normalize_tactile_embeddings,
)


@pytest.mark.parametrize("factor,expected_tokens", [(1, 4), (8, 32), (21, 84)])
def test_repeat_tactile_tokens_and_masks_is_key_major(
    factor: int, expected_tokens: int
) -> None:
    tokens = jnp.arange(4, dtype=jnp.float32).reshape(1, 4, 1)
    masks = jnp.asarray([[True, False, True, False]])

    expanded, expanded_masks = _repeat_tactile_tokens_and_masks(tokens, masks, factor)

    assert expanded.shape == (1, expected_tokens, 1)
    assert expanded_masks.shape == (1, expected_tokens)
    np.testing.assert_array_equal(
        expanded[0, :, 0], np.repeat(np.arange(4), factor)
    )
    np.testing.assert_array_equal(
        expanded_masks[0], np.repeat([True, False, True, False], factor)
    )


def test_repeat_factor_one_is_value_preserving() -> None:
    tokens = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
    masks = jnp.asarray([[True] * 4, [True, False, True, False]])
    expanded, expanded_masks = _repeat_tactile_tokens_and_masks(tokens, masks, 1)
    np.testing.assert_array_equal(expanded, tokens)
    np.testing.assert_array_equal(expanded_masks, masks)
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_tactile_integration.py -q
```

Expected: import error for `_repeat_tactile_tokens_and_masks`.

- [ ] **Step 3: Add the pure expansion helper**

Near `normalize_tactile_embeddings`, add:

```python
def _repeat_tactile_tokens_and_masks(
    tokens: Array,
    masks: Array,
    factor: int,
) -> tuple[Array, Array]:
    tokens = jnp.asarray(tokens)
    masks = jnp.asarray(masks, dtype=jnp.bool_)
    if tokens.ndim != 3:
        raise ValueError(f"tactile tokens must be [B,S,H], got {tokens.shape}")
    if masks.shape != tokens.shape[:2]:
        raise ValueError(
            f"tactile masks must have shape {tokens.shape[:2]}, got {masks.shape}"
        )
    if factor < 1:
        raise ValueError(f"tactile repeat factor must be positive, got {factor}")
    return (
        jnp.repeat(tokens, factor, axis=1),
        jnp.repeat(masks, factor, axis=1),
    )
```

- [ ] **Step 4: Use the helper after projection and base-mask validation**

Keep `embed_tactile` unchanged through the shared projection. In `embed_prefix`, after validating the base mask `[B,S]`, add:

```python
tactile_embedding, tactile_masks = _repeat_tactile_tokens_and_masks(
    tactile_embedding,
    tactile_masks,
    self.config.tactile_token_repeat_factor,
)
embeddings.append(tactile_embedding)
pad_masks.append(tactile_masks)
attention_segments.append(jnp.zeros(tactile_embedding.shape[1], dtype=jnp.bool_))
```

Replace the existing three append calls rather than appending both base and expanded tokens.

- [ ] **Step 5: Add an embed-prefix contract test without the full VLM weights**

Add this test to `tests/jax/test_tactile_integration.py`:

```python
def test_embed_prefix_repeats_tactile_tokens_and_ablation_mask() -> None:
    hidden_size = 3

    class StubPrefixModel(JaxSmolVLA):
        def embed_image(self, params, image):
            del params
            return jnp.ones((image.shape[0], 2, hidden_size), dtype=jnp.float32)

        def embed_language(self, params, tokens):
            del params
            return jnp.full(
                (tokens.shape[0], tokens.shape[1], hidden_size),
                7.0,
                dtype=jnp.float32,
            )

        def embed_tactile(self, params, tactile_images=None, *, tactile_embeddings=None):
            del params, tactile_images
            return jnp.asarray(tactile_embeddings, dtype=jnp.float32)

        def _linear(self, params, name, value, *, bias=False, **kwargs):
            del params, bias, kwargs
            assert name == "model.state_proj"
            return jnp.zeros((value.shape[0], hidden_size), dtype=jnp.float32)

    config = JaxSmolVLAConfig(
        use_tactile_encoder=True,
        tactile_encoder_path="unused",
        tactile_keys=("t0", "t1", "t2", "t3"),
        tactile_num_tokens=4,
        tactile_token_repeat_factor=8,
        text_hidden_size=hidden_size,
        max_state_dim=2,
    )
    model = StubPrefixModel(config)
    tactile = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, hidden_size)
    common = dict(
        params={},
        images=jnp.zeros((1, 2, 3, 2, 2), dtype=jnp.float32),
        image_masks=jnp.ones((1, 2), dtype=jnp.bool_),
        language_tokens=jnp.ones((1, 3), dtype=jnp.int32),
        language_masks=jnp.ones((1, 3), dtype=jnp.bool_),
        state=jnp.zeros((1, 2), dtype=jnp.float32),
        tactile_embeddings=tactile,
    )

    prefix, pad_mask, _ = model.embed_prefix(
        **common,
        tactile_masks=jnp.ones((1, 4), dtype=jnp.bool_),
    )
    _, ablated_pad_mask, _ = model.embed_prefix(
        **common,
        tactile_masks=jnp.zeros((1, 4), dtype=jnp.bool_),
    )

    tactile_start = 4  # two image slots x two stub tokens
    tactile_stop = tactile_start + 32
    assert prefix.shape == (1, 4 + 32 + 3 + 1, hidden_size)
    np.testing.assert_array_equal(
        prefix[:, tactile_start:tactile_stop],
        jnp.repeat(tactile, 8, axis=1),
    )
    assert bool(jnp.all(pad_mask[:, tactile_start:tactile_stop]))
    assert bool(jnp.all(~ablated_pad_mask[:, tactile_start:tactile_stop]))
```

This test uses real `embed_prefix` concatenation and mask logic but no checkpoint weights.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all tactile integration tests pass for K=1/8/21.

- [ ] **Step 7: Review parameter invariance**

Run:

```bash
git diff --check
rg -n "tactile_token_repeat_factor|repeat_tactile" \
  src/lerobot/policies/smolvla_jax/modeling.py tests/jax/test_tactile_integration.py
```

Expected: no new parameter key, initializer, projection shape or cache shape.

---

### Task 4: Migrate legacy resume metadata and reject cross-K strict resume

**Files:**
- Modify: `src/lerobot/policies/smolvla_jax/training.py:401-455`
- Modify: `tests/jax/test_training.py`

**Interfaces:**
- Produces: `_canonicalize_resume_signature(signature)` inserts K=1 only when the saved model signature predates the field.
- Preserves: exact strict-resume rejection when saved and current K differ.

- [ ] **Step 1: Write failing migration and mismatch tests**

Add to `tests/jax/test_training.py`:

```python
import json


def _remove_repeat_factor_from_resume_metadata(checkpoint: Path) -> None:
    path = checkpoint / "resume_metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["resume_signature"]["model"]["tactile_token_repeat_factor"]
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def test_resume_treats_legacy_missing_tactile_repeat_factor_as_one(tmp_path: Path) -> None:
    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    trainer.step(batch)
    checkpoint = trainer.save(tmp_path / "checkpoint")
    _remove_repeat_factor_from_resume_metadata(checkpoint)

    resumed = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    resumed.restore(checkpoint)
    assert resumed.step_count == 1


def test_resume_rejects_changed_tactile_repeat_factor(tmp_path: Path) -> None:
    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    trainer.step(batch)
    checkpoint = trainer.save(tmp_path / "checkpoint")

    changed = dataclasses.replace(config, tactile_token_repeat_factor=8)
    resumed = JaxSmolVLATrainer(TinyModel(changed), params, seed=4, total_steps=10)
    with np.testing.assert_raises_regex(ValueError, "tactile_token_repeat_factor"):
        resumed.restore(checkpoint)
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_training.py -q
```

Expected: the legacy missing-field test fails with a resume configuration difference; the changed-K test already fails closed or lacks the named difference until Task 1 is present.

- [ ] **Step 3: Canonicalize only the legacy saved signature**

Add near `_mapping_differences` helpers in `training.py`:

```python
def _canonicalize_resume_signature(signature: Any) -> Any:
    if not isinstance(signature, Mapping):
        return signature
    normalized = dict(signature)
    model = normalized.get("model")
    if isinstance(model, Mapping):
        normalized["model"] = {
            **model,
            "tactile_token_repeat_factor": model.get(
                "tactile_token_repeat_factor", 1
            ),
        }
    return normalized
```

In `_validate_resume_compatibility`, immediately after loading `saved_signature`, call:

```python
saved_signature = _canonicalize_resume_signature(saved_signature)
```

Do not canonicalize a present K=8/21 value to 1.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: exact K=1 legacy resume passes; K mismatch raises mentioning `tactile_token_repeat_factor`.

- [ ] **Step 5: Review the compatibility boundary**

```bash
git diff --check
git diff -- src/lerobot/policies/smolvla_jax/training.py tests/jax/test_training.py
```

Expected: only missing saved fields are migrated; current configs and present saved factors are not rewritten.

---

### Task 5: Add the K=1/8/21 paper configs and parity test

**Files:**
- Modify: `configs/train_vtsmolvla_jax.yaml`
- Create: `configs/train_vtsmolvla_jax_tactile16.yaml`
- Create: `configs/train_vtsmolvla_jax_tactile32.yaml`
- Modify: `tests/jax/test_train_vtsmolvla_config.py`

**Interfaces:**
- Produces three explicit training configurations whose only scientific difference is K.

- [ ] **Step 1: Write the failing config-parity test**

Add to `tests/jax/test_train_vtsmolvla_config.py`:

```python
ROOT = Path(__file__).resolve().parents[2]


def _load_repo_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def _scientific_config(config: dict) -> dict:
    normalized = deepcopy(config)
    normalized.pop("output", None)
    wandb = normalized.get("wandb") or {}
    wandb.pop("name", None)
    wandb.pop("tags", None)
    normalized["wandb"] = wandb
    normalized["model"] = dict(normalized["model"])
    normalized["model"].pop("tactile_token_repeat_factor", None)
    return normalized


def test_paper_ratio_configs_only_change_factor_and_output_identity() -> None:
    base = _load_repo_yaml("train_vtsmolvla_jax.yaml")
    tactile16 = _load_repo_yaml("train_vtsmolvla_jax_tactile16.yaml")
    tactile32 = _load_repo_yaml("train_vtsmolvla_jax_tactile32.yaml")

    assert base["model"]["tactile_token_repeat_factor"] == 1
    assert tactile16["model"]["tactile_token_repeat_factor"] == 8
    assert tactile32["model"]["tactile_token_repeat_factor"] == 21
    assert _scientific_config(base) == _scientific_config(tactile16)
    assert _scientific_config(base) == _scientific_config(tactile32)
    assert len({base["output"], tactile16["output"], tactile32["output"]}) == 3
```

- [ ] **Step 2: Run the parity test and verify RED**

Run:

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_train_vtsmolvla_config.py -q
```

Expected: missing tactile16/tactile32 files and missing K=1 field fail.

- [ ] **Step 3: Make K=1 explicit in the base YAML**

Under `tactile_num_tokens: 4`, add:

```yaml
  # 每路 projected tactile token 的无参数复制数；1 为原始四-token baseline。
  tactile_token_repeat_factor: 1
```

- [ ] **Step 4: Create the two variants from the complete base YAML**

Create each file as a full copy of the now-updated base config, then change exactly these fields:

```yaml
# train_vtsmolvla_jax_tactile16.yaml
output: /workspace/vtsmolvla_tactile_repeat16
wandb:
  name: vtsmolvla_tactile_repeat16
  tags: [smolvla, tactile, tactile-encoder, vtsmolvla, naive-repeat, tactile16]
model:
  tactile_token_repeat_factor: 8
```

```yaml
# train_vtsmolvla_jax_tactile32.yaml
output: /workspace/vtsmolvla_tactile_repeat32
wandb:
  name: vtsmolvla_tactile_repeat32
  tags: [smolvla, tactile, tactile-encoder, vtsmolvla, naive-repeat, tactile32]
model:
  tactile_token_repeat_factor: 21
```

All omitted keys in these excerpts must remain byte-for-byte/semantically equal to the base YAML. Use `apply_patch` for the final files; do not introduce inheritance or a new config framework.

- [ ] **Step 5: Run launcher validation and parity tests**

Run the Step 2 command. Expected: all tests pass and `_validate_vt_config` accepts all three files.

- [ ] **Step 6: Print and verify the exact ratios**

```bash
/home/yunjing/FRS/FRS_Tact/.venv/bin/python -c \
  "for k in (1,8,21): print(k, 4*k, (4*k)/(177+4*k))"
```

Expected:

```text
1 4 0.022099...
8 32 0.153110...
21 84 0.321839...
```

---

### Task 6: Full regression, checkpoint-backed smoke, review and memory update

**Files:**
- Modify: `CODEBASE_MEMORY.md`
- No production file should be added in this task.

**Interfaces:**
- Verifies all previous tasks together.
- Produces reproducible test and smoke evidence; does not start a paper training run.

- [ ] **Step 1: Run the complete CPU regression**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp \
  HF_DATASETS_CACHE=/tmp/frs_tactile_ratio_hf_cache \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  modalities_eval/test tests/jax -q
```

Expected: all tests pass; the existing real-checkpoint reconstruction test may remain explicitly skipped unless its opt-in environment flag is set.

- [ ] **Step 2: Run syntax and whitespace checks**

```bash
env PYTHONPATH=src:. /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m py_compile \
  src/lerobot/policies/smolvla_jax/configuration.py \
  src/lerobot/policies/smolvla_jax/checkpoint.py \
  src/lerobot/policies/smolvla_jax/modeling.py \
  src/lerobot/policies/smolvla_jax/training.py \
  tools/train_vtsmolvla_jax.py \
  tests/jax/test_checkpoint.py \
  tests/jax/test_tactile_integration.py \
  tests/jax/test_training.py \
  tests/jax/test_train_vtsmolvla_config.py
git diff --check
```

Expected: exit 0 with no output from `git diff --check`.

- [ ] **Step 3: Verify no parameter or cache schema changes**

Run this read-only check against the existing VT checkpoint:

```bash
env PYTHONPATH=src:. /home/yunjing/FRS/FRS_Tact/.venv/bin/python - <<'PY'
from dataclasses import replace

from lerobot.policies.smolvla_jax.checkpoint import load_params
from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig

checkpoint = "/home/yunjing/FRS/KaiyueChen/vtsmolvla_01_3w"
base = JaxSmolVLAConfig.from_pretrained(checkpoint)
params = load_params(checkpoint)
assert params["model.tactile_proj.weight"].shape == (960, 512)
for factor, effective in ((1, 4), (8, 32), (21, 84)):
    config = replace(base, tactile_token_repeat_factor=factor)
    assert config.tactile_num_tokens == 4
    assert config.effective_tactile_num_tokens == effective
print("parameter_schema_unchanged", params["model.tactile_proj.weight"].shape)
PY
```

Expected: `parameter_schema_unchanged (960, 512)`.

- [ ] **Step 4: Run one optional GPU prefix/sample smoke per K**

Outside the filesystem/network sandbox, run each K in a fresh process so memory peaks are not cumulative:

```bash
for K in 1 8 21; do
  env K="$K" PYTHONPATH=src:. \
    HF_HOME=/home/yunjing/FRS/eval_cache/huggingface \
    HF_HUB_CACHE=/home/yunjing/FRS/eval_cache/huggingface/hub \
    HF_DATASETS_CACHE=/home/yunjing/FRS/eval_cache/huggingface/datasets_arrow \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /home/yunjing/FRS/FRS_Tact/.venv/bin/python - <<'PY'
import os
import time
from dataclasses import replace

import jax
import numpy as np

from modalities_eval.utils import _batch_observation, create_velocity_context, load_episode, load_model
from lerobot.policies.smolvla_jax.modeling import JaxSmolVLA

assert any(device.platform == "gpu" for device in jax.devices()), jax.devices()
k = int(os.environ["K"])
model = load_model(
    "/home/yunjing/FRS/KaiyueChen/vtsmolvla_01_3w",
    dataset_repo_id="KaiyueChen/pick_tube_02",
    dataset_root="/home/yunjing/FRS/eval_data/KaiyueChen/pick_tube_02",
    action_key="actions",
    rename_map={
        "observation.images.camera0": "observation.images.camera1",
        "observation.images.camera1": "observation.images.camera2",
    },
    normalization_source="checkpoint",
)
model.config = replace(model.config, tactile_token_repeat_factor=k)
model.model = JaxSmolVLA(model.config)
model._sample_cache = {}
episode = load_episode(model, 48, frame_indices=(249,))
observation = _batch_observation(episode.observations[0])

start = time.perf_counter()
context = create_velocity_context(model, observation)
actions = model.sample_actions(
    jax.random.key(0), observation, num_steps=1
)
jax.block_until_ready(actions)
elapsed = time.perf_counter() - start
expected_prefix = 177 + 4 * k
assert context.pad_mask.shape == (1, expected_prefix)
assert np.isfinite(np.asarray(actions)).all()
stats = jax.devices()[0].memory_stats() or {}
print(
    "K", k,
    "prefix", expected_prefix,
    "elapsed_s", elapsed,
    "peak_bytes", stats.get("peak_bytes_in_use"),
)
PY
done
```

Expected: prefix lengths `181`, `209`, `261`; all sampled actions finite. Record `elapsed_s` and `peak_bytes` when the backend exposes it. This is a compatibility/performance smoke, not a paper result.

- [ ] **Step 5: Request a read-only code review**

Reviewer checklist:

- K=1 numerical and checkpoint compatibility.
- key-major repeat order and mask parity.
- no cache shape, parameter key or projection shape changes.
- saved config and resume migration correctness.
- K=1/8/21 YAML parity.
- no accidental changes to the existing `modalities_eval` work.

Fix every Critical/Important issue and rerun Steps 1-4 as affected.

- [ ] **Step 6: Update directory memory with exact evidence**

Append to `CODEBASE_MEMORY.md`:

- changed files and final config contract;
- red/green test commands and counts;
- K=1/8/21 effective token counts and ratios;
- GPU wall time/peak memory if smoke ran;
- checkpoint/cache compatibility result;
- explicit statement that no training run was started;
- remaining limitations and output/config paths.

- [ ] **Step 7: Present the final integration boundary**

Report that implementation lives in `codex/modalities-eval-tactile` worktree and remains uncommitted unless the user authorizes commit/merge. Provide explicit options to apply only the token-baseline hunks, combine with the evaluator work, or keep the worktree isolated.
