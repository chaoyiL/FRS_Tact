# Offline Vision Cache and Four-GPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete offline VT-SmolVLA training cache and run K8 on GPUs 0/1 concurrently with K21 on GPUs 2/3 on a four-card RTX PRO 6000 Blackwell server.

**Architecture:** Store frozen vision+connector outputs as bit-exact BF16 memmaps and store the remaining deterministic sample fields as map-style arrays. Extend the model and loader with an optional cached-token path while retaining the live image path. Prepare all four datasets before launching two independent two-GPU JAX processes.

**Tech Stack:** Python 3.12, JAX 0.8.3, NumPy memmap, ml_dtypes BF16, PyTorch DataLoader, LeRobot v3, Bash, tmux, pytest.

## Global Constraints

- Repository path is `/home/ljl/FRS_Tact`; virtual environment path is `/home/ljl/.venvs/frs_tact`.
- Persistent storage root is `/DATA/ljl/substage`; no active config or script may fall back to `/workspace`.
- Target hardware is exactly four `NVIDIA RTX PRO 6000 Blackwell Server Edition` GPUs; K8 uses physical GPUs `0,1` and K21 uses `2,3`.
- Each experiment is one JAX process with two-device data parallelism; do not use `torchrun`, MPI, or one process per GPU.
- K8 and K21 run concurrently and fail independently.
- RGB augmentation is disabled in offline-cache mode.
- Vision and connector modules remain frozen; cached mode rejects any other module mode.
- Vision tokens are logically BF16 and stored as their exact `uint16` bit pattern; FP16 substitution is forbidden.
- K8 and K21 share source data, tactile cache, offline cache, split, and normalization protocol.
- State and action are cached raw as FP32 and normalized online with the existing train-only protocol.
- The existing live `images` inference path and portable checkpoint parameter schema remain unchanged.
- Do not add remote SHA pinning, publishing provenance, deployment changes, LMDB, or WebDataset.

## File Structure

New files:

- `src/lerobot/policies/smolvla_jax/offline_training_cache.py`: cache schema, BF16 bit conversion, validation, memmap reader, and cache directory naming.
- `src/lerobot/policies/smolvla_jax/offline_cache_precompute.py`: resumable per-dataset cache writer with injected dataset and vision encoder interfaces.
- `tools/precompute_smolvla_training_cache.py`: YAML/checkpoint/dataset CLI entry point.
- `tests/jax/test_offline_training_cache.py`: cache format, completion, compatibility, and BF16 tests.
- `tests/jax/test_offline_cache_precompute.py`: precompute, interruption, resume, and publication tests.
- `tests/jax/test_offline_vision_embeddings.py`: online/cached model parity and invalid-input tests.

Modified files:

- `src/lerobot/policies/smolvla_jax/modeling.py`: optional precomputed vision embedding path.
- `src/lerobot/policies/smolvla_jax/data.py`: cached map-style dataset and host prefetch.
- `src/lerobot/policies/smolvla_jax/training.py`: cached vision inputs in loss/evaluation and data-wait metrics boundary.
- `tools/train_smolvla_jax.py`: parse cache config, construct cached loaders, and log data-wait time.
- `tools/train_vtsmolvla_jax.py`: fail-closed cache/module/augmentation validation.
- `configs/train_vtsmolvla_jax_tactile16.yaml`: K8 storage paths and offline-cache config.
- `configs/train_vtsmolvla_jax_tactile32.yaml`: K21 storage paths and offline-cache config.
- `scripts/setup_env.sh`: `/home/ljl`/`/DATA` layout and four-RTX validation.
- `scripts/download_ckpt.sh`, `deploy_smolvla/src/download_ckpt.py`: encoder destination under `/DATA/ljl/substage`.
- `scripts/start_vtsmolvla_train.sh`: four-dataset cache preparation and concurrent two-pair training.
- Existing focused tests for data, training, config, setup, checkpoint download, and launcher behavior.

---

### Task 1: Bit-exact offline cache contract

**Files:**

- Create: `src/lerobot/policies/smolvla_jax/offline_training_cache.py`
- Create: `tests/jax/test_offline_training_cache.py`

**Interfaces:**

- Produces `offline_cache_dir(root: Path, repo_id: str) -> Path`.
- Produces `bfloat16_to_uint16(values: ArrayLike) -> np.ndarray` and `uint16_to_bfloat16(values: ArrayLike) -> np.ndarray`.
- Produces `OfflineCacheSpec` and `OfflineTrainingCache` used by Tasks 3 and 4.
- Cache fields are named by exported constants so writer and reader cannot drift.

- [ ] **Step 1: Write failing BF16 and metadata tests**

Create tests that exercise actual bits, not approximate values:

```python
def test_bfloat16_uint16_roundtrip_is_bit_exact():
    source = np.asarray(jnp.array([0.0, -1.0, 1.2345, np.inf], dtype=jnp.bfloat16))
    stored = bfloat16_to_uint16(source)
    restored = uint16_to_bfloat16(stored)
    assert stored.dtype == np.uint16
    assert restored.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(source.view(np.uint16), restored.view(np.uint16))


def test_cache_rejects_incomplete_and_incompatible_metadata(tmp_path):
    cache_dir = make_complete_fixture(tmp_path, status="incomplete")
    with pytest.raises(ValueError, match="incomplete"):
        OfflineTrainingCache(cache_dir, expected_spec())
    cache_dir = make_complete_fixture(tmp_path, camera_keys=("right", "left"))
    with pytest.raises(ValueError, match="camera_keys"):
        OfflineTrainingCache(cache_dir, expected_spec())
```

Also cover total frames, vision shape, chunk size, tokenizer length, checkpoint source, module modes, field shape/dtype, missing files, and `status != complete`. Numeric finiteness belongs to the publication tests in Task 3 so training startup does not rescan an 80 GiB cache.

- [ ] **Step 2: Run RED**

Run:

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_offline_training_cache.py -q
```

Expected: collection fails because `offline_training_cache` does not exist.

- [ ] **Step 3: Implement the cache contract**

Define the fixed schema:

```python
OFFLINE_CACHE_SCHEMA_VERSION = 1
METADATA_NAME = "metadata.json"
PROGRESS_NAME = "progress.json"
VISION_TOKENS_NAME = "vision_tokens.uint16.npy"
STATE_NAME = "state.npy"
ACTIONS_NAME = "actions.npy"
ACTION_IS_PAD_NAME = "action_is_pad.npy"
LANGUAGE_TOKENS_NAME = "language_tokens.npy"
LANGUAGE_MASKS_NAME = "language_masks.npy"
EPISODE_INDEX_NAME = "episode_index.npy"
FRAME_INDEX_NAME = "frame_index.npy"


@dataclass(frozen=True)
class OfflineCacheSpec:
    repo_id: str
    total_frames: int
    camera_keys: tuple[str, ...]
    vision_tokens_per_camera: int
    vision_hidden_size: int
    state_dim: int
    action_dim: int
    chunk_size: int
    tokenizer_max_length: int
    checkpoint_source: str
    vision_mode: str
    connector_mode: str
```

Store BF16 as raw bits:

```python
def bfloat16_to_uint16(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=ml_dtypes.bfloat16))
    return array.view(np.uint16)


def uint16_to_bfloat16(values):
    array = np.asarray(values)
    if array.dtype != np.uint16:
        raise TypeError(f"expected uint16 BF16 storage, got {array.dtype}")
    return array.view(ml_dtypes.bfloat16)
```

`OfflineTrainingCache` exposes the validated `OfflineCacheSpec` as `.spec`. `OfflineTrainingCache.__getitem__(index)` returns a fresh mapping with logical BF16 vision tokens and raw small arrays. Validate all static metadata and array contracts in `__init__`; validate numeric finiteness during final publication rather than scanning 80 GiB on every training start.

- [ ] **Step 4: Run GREEN**

Run the Task 1 command again. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/lerobot/policies/smolvla_jax/offline_training_cache.py \
  tests/jax/test_offline_training_cache.py
git commit -m "feat: define offline SmolVLA cache contract"
```

---

### Task 2: Cached vision-token model path

**Files:**

- Modify: `src/lerobot/policies/smolvla_jax/modeling.py`
- Modify: `src/lerobot/policies/smolvla_jax/training.py`
- Create: `tests/jax/test_offline_vision_embeddings.py`

**Interfaces:**

- Consumes logical BF16 `vision_embeddings` shaped `[B,Ncam,64,960]` from Task 1.
- Extends `embed_prefix`, `flow_velocity`, `loss`, `build_prefix_context`, `sample_actions`, and denoising call paths with optional cached tokens.
- Preserves existing positional argument behavior for live `images` callers.

- [ ] **Step 1: Write failing parity and exclusivity tests**

Use a tiny model configuration and monkeypatch `embed_image` to return deterministic `[B,64,H]` tokens. Test exact prefix parity:

```python
online_prefix = model.embed_prefix(
    params, images, image_masks, language_tokens, language_masks, state
)
cached = jnp.stack([model.embed_image(params, images[:, i]) for i in range(2)], axis=1)
cached_prefix = model.embed_prefix(
    params,
    None,
    image_masks,
    language_tokens,
    language_masks,
    state,
    vision_embeddings=cached,
)
for online, offline in zip(online_prefix, cached_prefix, strict=True):
    np.testing.assert_array_equal(np.asarray(online), np.asarray(offline))
```

Also assert both supplied and neither supplied raise clear `ValueError`, wrong camera/token/hidden shapes fail, cached loss is finite, and cached validation rollout never invokes `embed_image`.

- [ ] **Step 2: Run RED**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_offline_vision_embeddings.py -q
```

Expected: failures report unknown `vision_embeddings` arguments and mandatory `images` access.

- [ ] **Step 3: Implement a single image-source selector**

Add one internal method and reuse it everywhere:

```python
def _resolve_image_embeddings(self, params, images, vision_embeddings):
    if (images is None) == (vision_embeddings is None):
        raise ValueError("exactly one of images or vision_embeddings is required")
    if vision_embeddings is not None:
        values = jnp.asarray(vision_embeddings)
        patch_rows = self.config.resize_height // self.config.vision_patch_size
        patch_cols = self.config.resize_width // self.config.vision_patch_size
        expected_tokens = (patch_rows * patch_cols) // (
            self.config.connector_scale_factor**2
        )
        expected = (
            expected_tokens,
            self.config.text_hidden_size,
        )
        if values.ndim != 4 or values.shape[2:] != expected:
            raise ValueError(
                "vision_embeddings must be [B,Ncam,"
                f"{expected[0]},{expected[1]}], got {values.shape}"
            )
        return [values[:, index] for index in range(values.shape[1])]
    return [self.embed_image(params, images[:, index]) for index in range(images.shape[1])]
```

Use the resolved list in `embed_prefix`; retain the existing hidden-size square-root scaling and camera masks. Change loss/evaluation reads from `batch["images"]` to `batch.get("images")` and pass `batch.get("vision_embeddings")`.

- [ ] **Step 4: Run GREEN and live-path regression**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_offline_vision_embeddings.py \
  tests/jax/test_tactile_integration.py -q
```

Expected: all selected tests pass and existing image callers remain unchanged.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/lerobot/policies/smolvla_jax/modeling.py \
  src/lerobot/policies/smolvla_jax/training.py \
  tests/jax/test_offline_vision_embeddings.py
git commit -m "feat: accept cached SmolVLA vision tokens"
```

---

### Task 3: Resumable per-dataset precompute

**Files:**

- Create: `src/lerobot/policies/smolvla_jax/offline_cache_precompute.py`
- Create: `tools/precompute_smolvla_training_cache.py`
- Create: `tests/jax/test_offline_cache_precompute.py`
- Modify: `tests/jax/test_train_script.py`

**Interfaces:**

- Consumes Task 1 cache constants/spec and Task 2 `JaxSmolVLA.embed_image`.
- Produces CLI `--config PATH --dataset-index INDEX` where INDEX is one of `0,1,2,3`.
- Produces a final cache directory only after validation; interruption state lives at `<cache_dir>.incomplete`.

- [ ] **Step 1: Write failing writer tests with injected fakes**

Define a five-frame fake dataset, deterministic tokenizer, and encoder callback. Cover first creation, interruption after two frames, resume at frame two without rewriting rows zero/one, completed-cache skip, incorrect progress rejection, finite checks, and atomic publication.

```python
writer = OfflineCachePrecomputer(
    spec=spec,
    output_dir=cache_dir,
    dataset=fake_dataset,
    encode_vision=lambda images: deterministic_tokens(images),
    tokenize=lambda tasks: deterministic_language(tasks),
    batch_size=2,
)
with pytest.raises(InjectedStop):
    writer.run(stop_after=2)
assert read_progress(cache_dir.with_name(cache_dir.name + ".incomplete"))["next_index"] == 2
writer.run()
cache = OfflineTrainingCache(cache_dir, spec)
assert len(cache) == 5
```

CLI tests assert dataset index range validation and that index two selects only the third YAML source.

- [ ] **Step 2: Run RED**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_offline_cache_precompute.py -q
```

Expected: collection fails because the precompute module and CLI are absent.

- [ ] **Step 3: Implement the injected writer**

`OfflineCachePrecomputer.run()` creates every NPY array with `np.lib.format.open_memmap`, iterates monotonically from `next_index`, writes batch results, flushes all arrays, and atomically rewrites progress after each completed batch. Use the fixed sample contract:

```python
sample = {
    "vision_tokens": vision_bfloat16,
    "state": raw_state_float32,
    "actions": raw_action_chunk_float32,
    "action_is_pad": padding_bool,
    "language_tokens": tokens_int32,
    "language_masks": masks_bool,
    "episode_index": episode_int,
    "frame_index": frame_int,
}
```

At completion, require `next_index == total_frames`, scan each numeric field in bounded chunks for finiteness, write complete metadata, and publish with the repository's no-replace atomic rename primitive. A pre-existing complete compatible cache is validated and skipped; incompatible content is never overwritten.

- [ ] **Step 4: Implement the real CLI adapter**

The tool loads YAML with the existing `tools/train_smolvla_jax.py` helpers, resolves the same checkpoint/config/params, validates frozen vision+connector and disabled augmentation, selects one dataset source, constructs the unaugmented LeRobot dataset with action delta timestamps, and injects:

```python
@jax.jit
def encode(images_bchw):
    return model.embed_image(params, images_bchw)
```

Flatten `[B,2,C,H,W]` to `[B*2,C,H,W]`, encode, and reshape to `[B,2,64,960]`. Store raw connector output; do not apply the prefix square-root scale during precompute.

- [ ] **Step 5: Run GREEN and compile checks**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_offline_cache_precompute.py tests/jax/test_train_script.py -q
env PYTHONPATH=src:. /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m py_compile \
  src/lerobot/policies/smolvla_jax/offline_cache_precompute.py \
  tools/precompute_smolvla_training_cache.py
```

Expected: tests and compilation pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/lerobot/policies/smolvla_jax/offline_cache_precompute.py \
  tools/precompute_smolvla_training_cache.py \
  tests/jax/test_offline_cache_precompute.py tests/jax/test_train_script.py
git commit -m "feat: precompute complete SmolVLA training samples"
```

---

### Task 4: Cache-backed loader and host prefetch

**Files:**

- Modify: `src/lerobot/policies/smolvla_jax/data.py`
- Modify: `tools/train_smolvla_jax.py`
- Modify: `tests/jax/test_data.py`
- Modify: `tests/jax/test_training.py`

**Interfaces:**

- Consumes `OfflineTrainingCache` from Task 1.
- Adds `offline_training_cache_root` and `host_prefetch_batches` to `LeRobotJaxDataLoader`.
- `batches()` yields `vision_embeddings`, never `images`, in cache mode.

- [ ] **Step 1: Write failing loader parity tests**

Build equivalent three-episode online and cache fixtures. Assert equal relative ordering, action chunks, padding, state, language tokens, tactile rows, train/validation episode filtering, deterministic batch sampler sequence, fixed subset indices, and `start_batch` continuation.

```python
cached = LeRobotJaxDataLoader(
    checkpoint,
    config,
    sources=sources,
    offline_training_cache_root=cache_root,
    tactile_embedding_cache_root=tactile_root,
    image_transforms=None,
    host_prefetch_batches=2,
)
batch = next(cached.batches(start_batch=3))
assert "images" not in batch
assert batch["vision_embeddings"].shape == (batch_size, 2, 64, 960)
assert batch["vision_embeddings"].dtype == jnp.bfloat16
```

Add a test whose backing iterator raises from the prefetch thread and assert the same exception reaches the consumer without hanging.

- [ ] **Step 2: Run RED**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_data.py tests/jax/test_training.py -q
```

Expected: failures report unknown cache/prefetch arguments and missing cached dataset behavior. Record any pre-existing unrelated collection failure separately; do not mask it.

- [ ] **Step 3: Implement map-style cache selection**

Add a private dataset that maps episode-filtered relative indices to absolute cache rows:

```python
class _OfflineCachedDataset(Dataset):
    def __init__(self, cache, episode_rows, tactile_cache):
        self.cache = cache
        self.rows = np.asarray(episode_rows, dtype=np.int64)
        self.tactile_cache = tactile_cache

    def __getitem__(self, index):
        absolute = int(self.rows[index])
        sample = dict(self.cache[absolute])
        sample["tactile_embeddings"] = self.tactile_cache[absolute]
        sample["image_masks"] = np.ones((len(self.cache.spec.camera_keys),), dtype=np.bool_)
        token_count = int(self.tactile_cache.metadata["num_tactile_tokens"])
        sample["tactile_masks"] = np.ones((token_count,), dtype=np.bool_)
        return sample
```

Build `episode_rows` from persisted episode metadata ranges in the same source/episode order as the current LeRobot reader. Continue to use `DeterministicEpochBatchSampler` unchanged.

After collate, normalize only state and actions with the existing preprocessor. Do not call tokenizer or RGB preparation in cache mode.

- [ ] **Step 4: Implement bounded host prefetch**

Use one daemon thread and a bounded `queue.Queue(maxsize=depth)`. Queue tagged values so `StopIteration` and worker exceptions propagate deterministically. `depth=0` returns the original iterator. Start prefetch around fully prepared host batches, not around GPU arrays.

- [ ] **Step 5: Add training-loop data wait metrics**

Split the loop timing boundary:

```python
data_started = time.perf_counter()
batch = next(batches)
data_wait_seconds += time.perf_counter() - data_started
metrics = trainer.step(batch)
```

At each existing synchronized log boundary, emit `data_wait_ms` as window wait time divided by window steps, log it to W&B, then reset the window accumulator.

- [ ] **Step 6: Run GREEN and data regressions**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_data.py tests/jax/test_training.py tests/jax/test_train_script.py -q
```

Expected: all runnable selected tests pass; cache mode performs no RGB decode or tokenizer call.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/lerobot/policies/smolvla_jax/data.py \
  tools/train_smolvla_jax.py tests/jax/test_data.py \
  tests/jax/test_training.py tests/jax/test_train_script.py
git commit -m "feat: train from offline SmolVLA sample cache"
```

---

### Task 5: Real server paths and cache-enabled experiment configs

**Files:**

- Modify: `configs/train_vtsmolvla_jax_tactile16.yaml`
- Modify: `configs/train_vtsmolvla_jax_tactile32.yaml`
- Modify: `tools/train_vtsmolvla_jax.py`
- Modify: `scripts/setup_env.sh`
- Modify: `scripts/download_ckpt.sh`
- Modify: `deploy_smolvla/src/download_ckpt.py`
- Modify: `tests/jax/test_train_vtsmolvla_config.py`
- Modify: `tests/test_setup_env.py`
- Modify: `tests/test_download_ckpt.py`

**Interfaces:**

- Produces `.env.frs` with code-independent storage paths.
- Produces two parsed YAMLs identical except K/output/W&B experiment identity.
- Makes offline-cache compatibility fail before checkpoint/data training work.

- [ ] **Step 1: Write failing path and validator tests**

Assert exact paths:

```python
assert env["FRS_STORAGE_ROOT"] == "/DATA/ljl/substage"
assert env["FRS_VENV_DIR"] == "/home/ljl/.venvs/frs_tact"
assert k8["offline_training_cache"]["root"] == "/DATA/ljl/substage/smolvla_training_cache"
assert k8["model"]["tactile_encoder_path"] == "/DATA/ljl/substage/checkpoints/encoder_ckpt_05"
assert k8["image_transforms"]["enable"] is False
assert k8["output"] == "/DATA/ljl/substage/outputs/vtsmolvla_tactile_repeat16"
assert k21["output"] == "/DATA/ljl/substage/outputs/vtsmolvla_tactile_repeat32"
```

Assert no active value in either YAML begins with `/workspace`. Validator tests reject enabled offline cache with augmentation, trainable vision, trainable connector, wrong BF16 dtype, missing root, or non-positive worker/prefetch values.

- [ ] **Step 2: Run RED**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_train_vtsmolvla_config.py tests/test_setup_env.py tests/test_download_ckpt.py -q
```

Expected: path, hardware-profile, augmentation, and offline-cache assertions fail against the current H100 `/workspace` configuration.

- [ ] **Step 3: Apply the approved filesystem mapping**

Use these exact active roots in both YAML files:

```text
/DATA/ljl/substage/lerobot_v30/KaiyueChen/pick_tube_01..04
/DATA/ljl/substage/checkpoints/encoder_ckpt_05
/DATA/ljl/substage/tactile_embeddings
/DATA/ljl/substage/smolvla_training_cache
/DATA/ljl/substage/normalization_protocols/pick_tube_vt_k8_k21
/DATA/ljl/substage/outputs/vtsmolvla_tactile_repeat16
/DATA/ljl/substage/outputs/vtsmolvla_tactile_repeat32
/DATA/ljl/substage/logs
```

Set the shared offline block exactly as approved and set `image_transforms.enable: false`. Set cache-mode training `num_workers: 2`; keep the batch, optimizer, scheduler, seed, split, and all model settings other than K unchanged.

`setup_env.sh` defaults `FRS_STORAGE_ROOT=/DATA/ljl/substage` and `FRS_VENV_DIR=/home/ljl/.venvs/frs_tact`. All HF, Arrow, temporary, log, and uv cache paths derive from the storage root. Downstream scripts must source `.env.frs` and must not reconstruct a storage root from `PROJECT_ROOT`.

- [ ] **Step 4: Implement fail-closed VT validation**

In `_validate_vt_config`, parse the offline block and enforce:

```python
if offline_enabled:
    if model_modes["vision"] != "frozen" or model_modes["connector"] != "frozen":
        raise ValueError("offline training cache requires frozen vision and connector")
    if bool((cfg.get("image_transforms") or {}).get("enable", False)):
        raise ValueError("offline training cache requires image_transforms.enable=false")
    if offline.get("dtype") != "bfloat16":
        raise ValueError("offline training cache dtype must be bfloat16")
```

Validate positive `precompute_batch_size`, `precompute_num_workers`, `loader_num_workers`, and `host_prefetch_batches` and a nonempty root.

- [ ] **Step 5: Run GREEN and YAML parity**

Run the Task 5 test command again, then parse both YAMLs and compare after removing only repeat factor, output, and W&B name/tags. Expected: all tests pass and the remaining mappings are equal.

- [ ] **Step 6: Commit Task 5**

```bash
git add configs/train_vtsmolvla_jax_tactile16.yaml \
  configs/train_vtsmolvla_jax_tactile32.yaml \
  tools/train_vtsmolvla_jax.py scripts/setup_env.sh scripts/download_ckpt.sh \
  deploy_smolvla/src/download_ckpt.py \
  tests/jax/test_train_vtsmolvla_config.py tests/test_setup_env.py \
  tests/test_download_ckpt.py
git commit -m "feat: configure offline cache on four RTX GPUs"
```

---

### Task 6: Four-GPU preparation and concurrent launcher

**Files:**

- Modify: `scripts/start_vtsmolvla_train.sh`
- Modify: `tests/test_start_vtsmolvla_train.py`
- Modify: `tests/test_setup_env.py`

**Interfaces:**

- Consumes Task 3 CLI, Task 5 `.env.frs`, and K8/K21 YAMLs.
- Produces one preparation coordinator and independent `vtsmolvla_k8` / `vtsmolvla_k21` sessions.

- [ ] **Step 1: Write failing hardware and orchestration tests**

Fake `nvidia-smi`, JAX preflight, precompute commands, tmux, and training commands. Assert exact order:

```text
tactile cache once
offline dataset 0 on GPU 0
offline dataset 1 on GPU 1
offline dataset 2 on GPU 2
offline dataset 3 on GPU 3
wait for all four
K8 on 0,1
K21 on 2,3
```

Tests prove the four offline jobs overlap, any precompute failure prevents both trainings, each training sees exactly two approved GPUs, K8/K21 launch independently, one training failure does not terminate the other, existing sessions fail without overwrite, and all original arguments survive coordinator tmux forwarding.

- [ ] **Step 2: Run RED**

```bash
/home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_start_vtsmolvla_train.py tests/test_setup_env.py -q
```

Expected: current sequential two-H100 assertions fail and no four-dataset offline CLI calls exist.

- [ ] **Step 3: Update the four-GPU hardware gate**

Require four rows from `nvidia-smi` and exact name substring `NVIDIA RTX PRO 6000 Blackwell Server Edition`. Require PyTorch and JAX to see four devices during setup. Run the existing NCCL/sharding collective across a four-device mesh.

Each child training preflight runs after its `CUDA_VISIBLE_DEVICES` assignment and requires exactly two JAX devices with the approved model name.

- [ ] **Step 4: Implement one-time parallel preparation**

Run tactile precompute once on GPU 0. Launch four offline commands in the coordinator:

```bash
for dataset_index in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${dataset_index}" \
      "${UV_BIN}" run --no-sync python tools/precompute_smolvla_training_cache.py \
      --config "${K8_CONFIG}" --dataset-index "${dataset_index}" \
      > >(tee -a "${LOG_ROOT}/offline_dataset_${dataset_index}.log") 2>&1 &
    cache_pids+=("$!")
done
```

Wait every PID, collect every status, and refuse to start training if any status is nonzero. Do not use a bare `wait` that loses individual failure information.

- [ ] **Step 5: Launch independent training jobs**

Use `CUDA_VISIBLE_DEVICES=0,1` for K8 and `CUDA_VISIBLE_DEVICES=2,3` for K21. In the normal interactive path, create independent tmux sessions named `vtsmolvla_k8` and `vtsmolvla_k21`; in `--foreground` mode, start both children, wait for both, report both statuses, and return nonzero if either failed without sending a signal to the healthy child.

- [ ] **Step 6: Run GREEN and shell gates**

```bash
/home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_start_vtsmolvla_train.py tests/test_setup_env.py -q
bash -n scripts/setup_env.sh scripts/start_vtsmolvla_train.sh
git diff --check
```

Expected: all tests and syntax checks pass.

- [ ] **Step 7: Commit Task 6**

```bash
git add scripts/setup_env.sh scripts/start_vtsmolvla_train.sh \
  tests/test_setup_env.py tests/test_start_vtsmolvla_train.py
git commit -m "feat: train K8 and K21 on separate GPU pairs"
```

---

### Task 7: Integration verification and server handoff

**Files:**

- Modify: `CODEBASE_MEMORY.md`
- Create: `docs/reports/2026-08-09-offline-cache-four-gpu-training.md`
- Modify only when a test exposes a defect: files owned by Tasks 1-6 and their focused tests.

**Interfaces:**

- Consumes every prior task.
- Produces tested server commands and measured acceptance criteria; it does not claim an unrun RTX result.

- [ ] **Step 1: Run the complete focused CPU suite**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax/test_offline_training_cache.py \
  tests/jax/test_offline_cache_precompute.py \
  tests/jax/test_offline_vision_embeddings.py \
  tests/jax/test_data.py tests/jax/test_training.py tests/jax/test_train_script.py \
  tests/jax/test_train_vtsmolvla_config.py \
  tests/test_setup_env.py tests/test_download_ckpt.py \
  tests/test_start_vtsmolvla_train.py -q
```

Expected: zero failures. If an existing out-of-scope collection failure remains, record the exact module and also run the largest explicit VT-only subset; do not describe the unfiltered suite as passing.

- [ ] **Step 2: Run broad runnable regressions and static gates**

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/jax modalities_eval/test -q
bash -n scripts/setup_env.sh scripts/download_data.sh scripts/download_ckpt.sh \
  scripts/start_vtsmolvla_train.sh
env PYTHONPATH=src:. /home/yunjing/FRS/FRS_Tact/.venv/bin/python -m py_compile \
  src/lerobot/policies/smolvla_jax/offline_training_cache.py \
  src/lerobot/policies/smolvla_jax/offline_cache_precompute.py \
  tools/precompute_smolvla_training_cache.py
git diff --check
```

Record exact pass/fail/skip counts and never hide known baseline exclusions.

- [ ] **Step 3: Verify parameter and cache compatibility**

Use the existing checkpoint loader to prove no parameter keys/shapes changed. Build a tiny cache, compare online and cached prefix/loss under fixed inputs, and verify K8/K21 parse to the same cache compatibility spec.

- [ ] **Step 4: Write the server acceptance commands**

The report contains these concrete stages from `/home/ljl/FRS_Tact`:

```bash
bash scripts/setup_env.sh
bash scripts/download_data.sh
bash scripts/start_vtsmolvla_train.sh
tmux attach -t vtsmolvla_k8
tmux attach -t vtsmolvla_k21
```

It also contains one-dataset interruption/resume commands and concurrent one-step K8/K21 smoke commands. Pass criteria are four complete caches, two devices per child, finite loss/gradient, no loader/cache error, and recorded `data_wait_ms`, `samples_per_s`, and `nvidia-smi dmon` utilization.

- [ ] **Step 5: Update repository memory and commit**

Document implemented behavior, exact test evidence, and the boundary that real four-GPU throughput remains unverified until the server smoke runs.

```bash
git add CODEBASE_MEMORY.md docs/reports/2026-08-09-offline-cache-four-gpu-training.md
git commit -m "docs: hand off offline four GPU training"
```

- [ ] **Step 6: Independent final review**

Review the complete branch diff for cache correctness, online-path compatibility, deterministic sampling/resume, path safety, four-process isolation, and test evidence. Fix every Critical or Important finding with a focused failing test and rerun the affected gates before declaring completion.
