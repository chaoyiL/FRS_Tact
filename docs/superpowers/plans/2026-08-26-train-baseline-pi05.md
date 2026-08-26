# Train Baseline Pi0.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `train_baseline_pi05` project that caches frozen Pi0.5 coarse actions and frozen tactile embeddings, then trains and evaluates the approved two-layer direct tactile action decoder.

**Architecture:** Run JAX cache producers in separate processes, storing normalized `[N,50,20]` coarse/target actions and frame-indexed `[F,4,512]` tactile embeddings. A PyTorch-only training process reads those immutable caches, optimizes only the decoder with masked Smooth L1, and emits a strict `best.pt` contract for deployment.

**Tech Stack:** Python 3.12, JAX/Flax 0.5.3/0.10.2, Orbax, PyTorch 2.7+, NumPy memmaps, PyYAML, safetensors metadata, pytest, uv.

## Global Constraints

- Pi0.5 source checkpoint defaults to `checkpoints/model/pi05_bi_two_tubes_0102_step16000/checkpoint`; source policy receives visual RGB/state/task only.
- Tactile encoder defaults to `checkpoints/encoder/encoder_ckpt_0824`; it is frozen and emits four ordered current-frame 512D ResNet tokens.
- Default dataset is `KaiyueChen/two_tubes_03`; all server paths remain editable in YAML.
- Decoder input/output is `[B,50,20]`; tactile input is `[B,4,512]` in `left_0,right_0,left_1,right_1` order.
- Decoder is exactly two `nn.TransformerDecoderLayer` blocks with `d_model=128`, `nhead=4`, FFN 256, dropout 0.1, ReLU, `norm_first=True`.
- Decoder predicts the complete 20D action, including grippers 9 and 19; it is not residual.
- Cache generation must never perform FRS reverse integration or create `x_base`.
- Pi0.5 and tactile encoder must not be imported by the PyTorch training module.
- Preserve user-owned changes and the untracked `Tactile_Action_Decoder_Network_Structure.md`.
- Keep validation limited to functional shape/finite/checkpoint/cache contracts; do not build a general safety or asset-audit framework.

---

## File Map

```text
train_baseline_pi05/
  __init__.py                    package marker
  README.md                      server setup, stages, smoke and formal commands
  pyproject.toml                 isolated Python 3.12 dependencies
  uv.lock                        generated frozen dependency graph
  config.py                      dependency-light YAML parsing and dataclasses
  model.py                       decoder, masked loss, metrics
  checkpoint.py                  atomic best/last save and strict load
  action_cache.py                cache manifest, writer, reader and record split
  prepare_action_cache.py        frozen Pi0.5 forward-only cache producer
  tactile_cache.py               frozen 0824 current-frame embedding producer/reader
  data.py                        aligned Torch Dataset/DataLoader
  train.py                       PyTorch training and resume
  evaluate.py                    validation/test and shuffled-tactile reports
  pipeline.py                    ordered subprocess orchestration
  configs/train_baseline_pi05.yaml
  scripts/setup_env.sh
  scripts/start_train.sh
  src/lerobot/**                 vendored Pi0.5 and LeRobot dataset runtime
  tests/test_config.py
  tests/test_model_checkpoint.py
  tests/test_action_cache.py
  tests/test_cache_producers.py
  tests/test_training.py
  tests/test_pipeline.py
```

### Task 1: Standalone project and strict configuration

**Files:**
- Create: `train_baseline_pi05/__init__.py`
- Create: `train_baseline_pi05/config.py`
- Create: `train_baseline_pi05/configs/train_baseline_pi05.yaml`
- Create: `train_baseline_pi05/pyproject.toml`
- Create: `train_baseline_pi05/tests/__init__.py`
- Create: `train_baseline_pi05/tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: Path) -> BaselineTrainConfig`
- Produces: frozen dataclasses `DatasetConfig`, `SourcePolicyConfig`, `TactileConfig`, `CacheConfig`, `DecoderTrainConfig`, `BaselineTrainConfig`
- Produces: `validate_paths(config: BaselineTrainConfig) -> None`

- [ ] **Step 1: Write dependency-light failing configuration tests**

```python
def test_default_yaml_locks_direct_decoder_contract():
    config = load_config(ROOT / "train_baseline_pi05/configs/train_baseline_pi05.yaml")
    assert config.source.action_horizon == 50
    assert config.source.action_dim == 20
    assert config.decoder.num_layers == 2
    assert config.decoder.d_model == 128
    assert config.decoder.tactile_keys == (
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    )


def test_config_import_does_not_import_heavy_runtimes():
    completed = subprocess.run(
        [sys.executable, "-c", "import train_baseline_pi05.config; "
         "import sys; assert 'jax' not in sys.modules; assert 'torch' not in sys.modules"],
        cwd=ROOT, check=False,
    )
    assert completed.returncode == 0
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=. pytest -q train_baseline_pi05/tests/test_config.py`

Expected: FAIL because `train_baseline_pi05.config` and the YAML do not exist.

- [ ] **Step 3: Implement dataclasses and exact YAML parsing**

Implement `load_config` with this public shape:

```python
@dataclass(frozen=True)
class DecoderTrainConfig:
    output: Path
    action_horizon: int = 50
    action_dim: int = 20
    tactile_dim: int = 512
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    tactile_keys: tuple[str, ...] = (
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    )
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0


def load_config(path: Path) -> BaselineTrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = BaselineTrainConfig.from_mapping(raw, config_path=path.resolve())
    config.validate_contract()
    return config
```

Reject quoted booleans, non-positive dimensions, any decoder layer count other than 2, horizon other than 50, action dimension other than 20, tactile dimension other than 512, duplicate/reordered tactile keys, split fractions not summing to 1, and writable outputs overlapping input asset roots.

Create the YAML with `KaiyueChen/two_tubes_03`, `/workspace/lerobot_v30/KaiyueChen/two_tubes_03`, the two approved local reference asset paths, fixed Pi0.5 seed 0, sample steps 10, episode split 0.8/0.1/0.1 seed 42, and distinct `/workspace/baseline_pi05/...` cache/output roots.

- [ ] **Step 4: Add the isolated dependency manifest**

Copy the pinned dependency ranges and indexes from `train_pi05_frs/pyproject.toml`, change the project name to `pi05-direct-baseline-training`, and list only `train_baseline_pi05` plus its vendored `lerobot` packages. Keep `torch`, JAX/Flax/Orbax, datasets/video dependencies, PyYAML, matplotlib, safetensors, and pytest as direct dependencies.

- [ ] **Step 5: Run configuration tests and verify GREEN**

Run: `PYTHONPATH=. pytest -q train_baseline_pi05/tests/test_config.py`

Expected: all tests pass and the subprocess proves config import is dependency-light.

- [ ] **Step 6: Commit Task 1**

```bash
git add train_baseline_pi05/__init__.py train_baseline_pi05/config.py \
  train_baseline_pi05/configs/train_baseline_pi05.yaml train_baseline_pi05/pyproject.toml \
  train_baseline_pi05/tests/__init__.py train_baseline_pi05/tests/test_config.py
git commit -m "feat: scaffold pi05 direct baseline training"
```

### Task 2: Two-layer decoder and checkpoint contract

**Files:**
- Create: `train_baseline_pi05/model.py`
- Create: `train_baseline_pi05/checkpoint.py`
- Create: `train_baseline_pi05/tests/test_model_checkpoint.py`

**Interfaces:**
- Consumes: `DecoderTrainConfig` from Task 1
- Produces: `DirectDecoderConfig`
- Produces: `DirectTactileActionDecoder.forward(coarse, tactile) -> Tensor`
- Produces: `masked_smooth_l1(predicted, target, valid_mask) -> Tensor`
- Produces: `save_best_checkpoint`, `save_last_checkpoint`, `load_decoder_checkpoint`

- [ ] **Step 1: Write failing architecture, loss and round-trip tests**

```python
def test_decoder_is_exact_two_layer_50_step_model():
    config = DirectDecoderConfig()
    model = DirectTactileActionDecoder(config)
    assert len(model.decoder.layers) == 2
    output = model(torch.zeros(3, 50, 20), torch.zeros(3, 4, 512))
    assert output.shape == (3, 50, 20)
    assert torch.isfinite(output).all()


def test_masked_smooth_l1_ignores_invalid_tail():
    prediction = torch.tensor([[[0.0], [100.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])
    assert masked_smooth_l1(prediction, target, mask).item() == 0.0


def test_best_checkpoint_weights_only_strict_round_trip(tmp_path):
    model = DirectTactileActionDecoder(DirectDecoderConfig())
    path = save_best_checkpoint(tmp_path / "best.pt", model=model, epoch=3,
                                global_step=9, metrics={"val_loss": 0.2},
                                source_contract=SOURCE_CONTRACT)
    restored, metadata = load_decoder_checkpoint(path, map_location="cpu")
    assert metadata["mode"] == "action_tactile"
    for left, right in zip(model.state_dict().values(), restored.state_dict().values(), strict=True):
        torch.testing.assert_close(left, right)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=. pytest -q train_baseline_pi05/tests/test_model_checkpoint.py`

Expected: FAIL because model/checkpoint symbols do not exist.

- [ ] **Step 3: Implement the approved model and loss**

Implement the decoder without importing JAX:

```python
class DirectTactileActionDecoder(nn.Module):
    def __init__(self, config: DirectDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.action_position = nn.Parameter(torch.randn(50, 128) * 0.02)
        self.sensor_identity = nn.Parameter(torch.randn(4, 128) * 0.02)
        self.action_in = nn.Linear(20, 128)
        self.tactile_in = nn.Linear(512, 128)
        layer = nn.TransformerDecoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1,
            activation="relu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.action_out = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 20))

    def forward(self, coarse: Tensor, tactile: Tensor) -> Tensor:
        if coarse.ndim != 3 or coarse.shape[1:] != (50, 20):
            raise ValueError(f"coarse must be [B,50,20], got {tuple(coarse.shape)}")
        if tactile.ndim != 3 or tactile.shape[1:] != (4, 512):
            raise ValueError(f"tactile must be [B,4,512], got {tuple(tactile.shape)}")
        action_tokens = self.action_in(coarse) + self.action_position
        tactile = tactile.float()
        tactile = tactile / tactile.square().mean(-1, keepdim=True).sqrt().clamp_min(
            torch.finfo(torch.float32).eps
        )
        memory = self.tactile_in(tactile) + self.sensor_identity
        return self.action_out(self.decoder(action_tokens, memory))
```

Implement masked Smooth L1 by computing `reduction="none"`, multiplying `valid_mask[..., None]`, and dividing by `action_dim * valid_mask.sum().clamp_min(1)`.

- [ ] **Step 4: Implement atomic primitive-only checkpoints**

`best.pt` must contain schema/version/mode, epoch/global step, `asdict(config)`, CPU state dict, scalar metrics and a primitive `source_contract`. Write to `.<name>.<uuid>.tmp`, flush and `os.fsync`, then `os.replace`. Loader must call `torch.load(path, weights_only=True)`, validate every fixed config value, instantiate the model, and call `load_state_dict(strict=True)`.

- [ ] **Step 5: Run model/checkpoint tests and verify GREEN**

Run: `PYTHONPATH=. pytest -q train_baseline_pi05/tests/test_model_checkpoint.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add train_baseline_pi05/model.py train_baseline_pi05/checkpoint.py \
  train_baseline_pi05/tests/test_model_checkpoint.py
git commit -m "feat: add pi05 direct tactile decoder"
```

### Task 3: Minimal action-cache schema and episode records

**Files:**
- Create: `train_baseline_pi05/action_cache.py`
- Create: `train_baseline_pi05/tests/test_action_cache.py`
- Copy then trim: `train_pi05_frs/src/lerobot/datasets/**` to `train_baseline_pi05/src/lerobot/datasets/**`
- Copy: `train_pi05_frs/src/lerobot/configs/**` to `train_baseline_pi05/src/lerobot/configs/**`
- Copy: `train_pi05_frs/src/lerobot/utils/**` to `train_baseline_pi05/src/lerobot/utils/**`
- Copy: `train_pi05_frs/src/lerobot/__init__.py` and `__version__.py` to `train_baseline_pi05/src/lerobot/`

**Interfaces:**
- Consumes: `DatasetConfig`, `CacheConfig`
- Produces: `SampleRecord(dataset_index, episode_index, frame_index, split_id)`
- Produces: `build_records(...) -> tuple[SampleRecord, ...]`
- Produces: `ActionCacheWriter.create/resume/write_batch/finalize`
- Produces: `ActionCache.open(path)`, array properties and `indices(split)`

- [ ] **Step 1: Write failing record/cache tests**

```python
def test_episode_split_is_disjoint_and_deterministic(fake_metadata):
    first = build_records(fake_metadata, split_seed=42, fractions=(0.8, 0.1, 0.1), frame_stride=5)
    second = build_records(fake_metadata, split_seed=42, fractions=(0.8, 0.1, 0.1), frame_stride=5)
    assert first == second
    by_split = {split_id: {r.episode_index for r in first if r.split_id == split_id}
                for split_id in (0, 1, 2)}
    assert by_split[0].isdisjoint(by_split[1] | by_split[2])
    assert by_split[1].isdisjoint(by_split[2])


def test_action_cache_schema_has_no_frs_latent(tmp_path):
    writer = ActionCacheWriter.create(tmp_path, sample_count=2, horizon=50, action_dim=20,
                                      manifest=MANIFEST)
    writer.write_batch(0, coarse=np.zeros((2, 50, 20), np.float32),
                       expert=np.ones((2, 50, 20), np.float32),
                       valid=np.ones((2, 50), bool), records=RECORDS)
    writer.finalize()
    assert not (tmp_path / "x_base.npy").exists()
    cache = ActionCache.open(tmp_path)
    assert cache.coarse_actions.shape == (2, 50, 20)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=train_baseline_pi05/src:. pytest -q train_baseline_pi05/tests/test_action_cache.py`

Expected: FAIL because action-cache classes do not exist.

- [ ] **Step 3: Copy the dataset runtime mechanically**

Use `cp -a` for the listed source files, exclude every `__pycache__`, and keep source notices. Then remove imports/files proven unused by the import tests only; do not rewrite LeRobot dataset behavior.

- [ ] **Step 4: Implement deterministic episode records and memmaps**

Adapt the episode-disjoint logic from `train_pi05_frs/pi05_cache/cache.py:126`. Use split IDs `0=train, 1=validation, 2=test`; validate fractions sum to one. Cache writer creates exactly the seven arrays in the design, persists `completed_samples` after each batch, resumes only when immutable manifest fields match, and marks `status="complete"` only after flushing every memmap.

- [ ] **Step 5: Run cache tests and verify GREEN**

Run: `PYTHONPATH=train_baseline_pi05/src:. pytest -q train_baseline_pi05/tests/test_action_cache.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add train_baseline_pi05/action_cache.py train_baseline_pi05/src/lerobot \
  train_baseline_pi05/tests/test_action_cache.py
git commit -m "feat: add direct baseline action cache"
```

### Task 4: Frozen tactile and forward-only Pi0.5 cache producers

**Files:**
- Create: `train_baseline_pi05/tactile_cache.py`
- Create: `train_baseline_pi05/prepare_action_cache.py`
- Create: `train_baseline_pi05/tests/test_cache_producers.py`
- Copy then remove FRS exports: `train_pi05_frs/src/lerobot/policies/pi05_jax/**` to `train_baseline_pi05/src/lerobot/policies/pi05_jax/**`
- Copy/adapt: `train_pi05_frs/pi05_cache/policy_inputs.py` to `train_baseline_pi05/policy_inputs.py`
- Copy/adapt: `train_pi05_frs/pi05_cache/source_model.py` to `train_baseline_pi05/source_model.py`
- Copy/adapt minimal encoder modules from `deploy_pi05/frs_inference/` to `train_baseline_pi05/tactile_encoder/`

**Interfaces:**
- Consumes: config, `ActionCacheWriter`, vendored Pi0.5 and dataset runtime
- Produces: `fixed_noise(batch_size, *, seed, horizon, action_dim) -> jax.Array`
- Produces: `sample_coarse_actions(model, params, observation, noise, num_steps) -> np.ndarray`
- Produces: `prepare_action_cache(config) -> Path`
- Produces: `prepare_tactile_cache(config) -> Path`

- [ ] **Step 1: Write failing fixed-noise and no-reverse tests**

```python
def test_fixed_noise_repeats_deployment_batch_one_noise():
    one = np.asarray(fixed_noise(1, seed=0, horizon=50, action_dim=20))
    many = np.asarray(fixed_noise(4, seed=0, horizon=50, action_dim=20))
    np.testing.assert_array_equal(many, np.repeat(one, 4, axis=0))


def test_prepare_action_cache_only_calls_forward_sampler(monkeypatch, tiny_config):
    calls = []
    monkeypatch.setattr(producer, "sample_coarse_actions",
                        lambda *a, **k: calls.append("sample") or FAKE_COARSE)
    prepare_action_cache(tiny_config, dependencies=FAKE_DEPENDENCIES)
    assert calls == ["sample"]
    assert "reverse" not in inspect.getsource(producer.prepare_action_cache)


def test_tactile_cache_is_current_frame_four_token_512d(tiny_config):
    path = prepare_tactile_cache(tiny_config, dependencies=FAKE_ENCODER_DEPENDENCIES)
    embeddings = np.load(path / "embeddings.npy", mmap_mode="r")
    assert embeddings.shape == (FAKE_FRAME_COUNT, 4, 512)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=train_baseline_pi05/src:. pytest -q train_baseline_pi05/tests/test_cache_producers.py`

Expected: FAIL because producer functions do not exist.

- [ ] **Step 3: Vendor the Pi0.5 runtime and remove FRS entry points**

Copy the existing verified Pi0.5 source tree, exclude bytecode, and change `lerobot.policies.pi05_jax.__init__` so it exports only observation/model/policy symbols. No new baseline module may import `pi05_jax.frs`, `train_pi05_frs.utils.integration`, `reverse_integrate_actions`, or `sample_and_reverse`.

- [ ] **Step 4: Implement forward-only source sampling**

Adapt `Pi05SampleProcessor` exactly for repack/normalize/resize/tokenize/pad. Build prefix cache and sample `t=1→0` with the source model's standard sampler only. Construct one noise tensor from `jax.random.key(seed)` with shape `[1,50,source_action_dim]` and repeat it over the batch. Slice both coarse and normalized expert actions to the first 20 dimensions before writing.

- [ ] **Step 5: Implement current-frame tactile cache**

Load `encoder_ckpt_0824` with the minimal copied checkpoint/config/ResNet modules. Preprocess four current-frame RGB images to 224×224 using the same function later copied into deployment, run shared ResNet18 in inference mode, RMS-normalize per sensor, and write `[frames,4,512]` plus a manifest.

- [ ] **Step 6: Run producer tests and verify GREEN**

Run: `PYTHONPATH=train_baseline_pi05/src:. pytest -q train_baseline_pi05/tests/test_cache_producers.py`

Expected: all tests pass and the import scan finds no reverse integration references.

- [ ] **Step 7: Commit Task 4**

```bash
git add train_baseline_pi05/tactile_cache.py train_baseline_pi05/prepare_action_cache.py \
  train_baseline_pi05/policy_inputs.py train_baseline_pi05/source_model.py \
  train_baseline_pi05/tactile_encoder train_baseline_pi05/src/lerobot/policies \
  train_baseline_pi05/tests/test_cache_producers.py
git commit -m "feat: cache frozen pi05 and tactile features"
```

### Task 5: Aligned dataset, training, evaluation and resume

**Files:**
- Create: `train_baseline_pi05/data.py`
- Create: `train_baseline_pi05/train.py`
- Create: `train_baseline_pi05/evaluate.py`
- Create: `train_baseline_pi05/tests/test_training.py`

**Interfaces:**
- Consumes: action/tactile caches, model/checkpoint/config
- Produces: `BaselineCacheDataset(cache, tactile_cache, split) -> Dataset`
- Produces: `train_decoder(config) -> Path`
- Produces: `evaluate_decoder(model, loader, norm_stats, *, shuffle_tactile=False) -> dict[str, float]`

- [ ] **Step 1: Write failing alignment, optimizer and evaluation tests**

```python
def test_dataset_aligns_tactile_by_absolute_dataset_index(fake_caches):
    dataset = BaselineCacheDataset(*fake_caches, split="train")
    sample = dataset[0]
    np.testing.assert_array_equal(sample["tactile"], fake_caches.tactile[37])
    assert sample["coarse"].shape == (50, 20)
    assert sample["target"].shape == (50, 20)


def test_one_step_updates_only_decoder(tmp_path, synthetic_config):
    before = synthetic_config.asset_hashes()
    checkpoint = train_decoder(synthetic_config, max_steps=1)
    assert checkpoint.is_file()
    assert synthetic_config.asset_hashes() == before
    assert "jax" not in train_decoder.__globals__.get("sys", sys).modules


def test_evaluation_reports_coarse_decoder_and_shuffled_metrics(fake_loader):
    metrics = evaluate_decoder(IDENTITY_MODEL, fake_loader, FAKE_NORM_STATS)
    assert set(("decoder_smooth_l1", "coarse_smooth_l1", "decoder_mse",
                "coarse_mse", "relative_reduction", "physical_mae", "physical_rmse")) <= metrics.keys()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=train_baseline_pi05/src:. pytest -q train_baseline_pi05/tests/test_training.py`

Expected: FAIL because dataset/training/evaluation symbols do not exist.

- [ ] **Step 3: Implement aligned Torch datasets**

Open all arrays read-only with `mmap_mode="r"`. Validate action-cache `dataset_indices` are in tactile-cache range. Return CPU tensors for coarse, tactile, target, valid mask, dataset index and episode index. Training loader shuffles with a seeded generator; validation/test loaders never shuffle.

- [ ] **Step 4: Implement training and resume**

Set Python/NumPy/Torch seeds. Instantiate only the decoder and AdamW. For each batch compute `masked_smooth_l1`, update, and accumulate mask-weighted metrics. At epoch end evaluate validation, atomically write `last.pt`, and replace `best.pt` only for strictly lower validation loss. Resume restores model, optimizer, CPU/CUDA RNG, epoch/global step and best metric from `last.pt`.

- [ ] **Step 5: Implement evaluation**

Evaluate coarse, synchronized tactile decoder, and deterministically episode-mismatched shuffled tactile. Report normalized Smooth L1/MSE, physical MAE/RMSE after inverse quantile normalization, relative reduction, delta RMS and gripper-9/19 errors. Write JSON/CSV without mutating checkpoints or caches.

- [ ] **Step 6: Run training tests and verify GREEN**

Run: `PYTHONPATH=train_baseline_pi05/src:. pytest -q train_baseline_pi05/tests/test_training.py`

Expected: all tests pass; subprocess import test proves training does not import JAX/Flax/Pi0.5.

- [ ] **Step 7: Commit Task 5**

```bash
git add train_baseline_pi05/data.py train_baseline_pi05/train.py \
  train_baseline_pi05/evaluate.py train_baseline_pi05/tests/test_training.py
git commit -m "feat: train and evaluate pi05 tactile baseline"
```

### Task 6: Pipeline, environment, documentation and full training verification

**Files:**
- Create: `train_baseline_pi05/pipeline.py`
- Create: `train_baseline_pi05/scripts/setup_env.sh`
- Create: `train_baseline_pi05/scripts/start_train.sh`
- Create: `train_baseline_pi05/README.md`
- Create: `train_baseline_pi05/tests/test_pipeline.py`
- Generate: `train_baseline_pi05/uv.lock`

**Interfaces:**
- Consumes: all training modules
- Produces: `python -m train_baseline_pi05.pipeline --config PATH [--check]`
- Produces: one-command server workflow and bounded smoke options

- [ ] **Step 1: Write failing pipeline-order and check-mode tests**

```python
def test_pipeline_runs_three_separate_processes_in_order(monkeypatch, config_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(argv) or OK)
    pipeline.run(config_path)
    modules = [argv[argv.index("-m") + 1] for argv in calls]
    assert modules == [
        "train_baseline_pi05.tactile_cache",
        "train_baseline_pi05.prepare_action_cache",
        "train_baseline_pi05.train",
    ]


def test_check_mode_does_not_import_jax_create_outputs_or_connect(tmp_path):
    completed = run_cli("--check", config=write_valid_config(tmp_path))
    assert completed.returncode == 0
    assert not output_paths(tmp_path).exist_any()
```

- [ ] **Step 2: Run pipeline tests and verify RED**

Run: `PYTHONPATH=train_baseline_pi05/src:. pytest -q train_baseline_pi05/tests/test_pipeline.py`

Expected: FAIL because pipeline and scripts do not exist.

- [ ] **Step 3: Implement ordered subprocess orchestration**

`--check` loads config and checks readable inputs only. Normal mode launches three `sys.executable -m ... --config ...` commands sequentially with `check=True`, so JAX is released before PyTorch training. Support `--max-samples` and `--max-steps` only as explicit smoke overrides and record them in output metadata.

- [ ] **Step 4: Add isolated environment scripts and lock**

`setup_env.sh` must resolve `train_baseline_pi05/.venv`, reject root/deploy/train-FRS environments, and run `uv sync --frozen --python 3.12 --project train_baseline_pi05`. `start_train.sh` selects only that interpreter, sets `PYTHONSAFEPATH=1`, package-local `PYTHONPATH`, `PYTHONUNBUFFERED=1`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false` before Python import. Generate the lock with `uv lock --project train_baseline_pi05`.

- [ ] **Step 5: Write exact server handoff documentation**

Document path edits, environment setup, `--check`, small-cache smoke, one-step/one-epoch smoke, formal run, resume, evaluation, output files and the statement that local CPU tests do not prove real GPU training or robot success.

- [ ] **Step 6: Run focused and full project verification**

Run:

```bash
bash -n train_baseline_pi05/scripts/setup_env.sh train_baseline_pi05/scripts/start_train.sh
PYTHONPATH=train_baseline_pi05/src:. pytest -q train_baseline_pi05/tests
uv lock --check --project train_baseline_pi05
```

Expected: shell syntax exits 0, all training tests pass, and the lock is current.

- [ ] **Step 7: Commit Task 6**

```bash
git add train_baseline_pi05/pipeline.py train_baseline_pi05/scripts \
  train_baseline_pi05/README.md train_baseline_pi05/tests/test_pipeline.py \
  train_baseline_pi05/uv.lock
git commit -m "feat: finish standalone pi05 baseline trainer"
```
