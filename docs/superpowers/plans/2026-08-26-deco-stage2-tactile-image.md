# DECO Stage 2 Tactile-Image Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete DECO Stage2 training path that initializes from the existing vision-only Stage1 checkpoint, converts and freezes the supplied JAX tactile encoder, and trains four tactile-image tokens through gated cross-attention and rank-32 PI adapters.

**Architecture:** Four tactile RGB frames are independently preprocessed, passed through one shared frozen ResNet18, RMS-normalized, and augmented with learned sensor IDs to form `[B, 4, 512]` tokens. Each DECO block uses those tokens as tactile K/V for the existing visual/action queries. New cross-attention branches are zero-gated with `tanh(gate)`, while rank-32 PI adapters provide the trainable residual path. Stage1 weights and the tactile encoder remain frozen; only tactile projections, gates, sensor embeddings, and PI adapters are optimized.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, safetensors, JAX/Flax/Orbax-compatible NPZ checkpoints, LeRobot v2.1 datasets, pytest, TorchScript, distributed PyTorch.

## Global Constraints

- Preserve the Stage1 state-dict key contract and existing Stage1 training behavior.
- Use exactly these tactile fields in this stable token order:
  1. `observation.images.tactile_left_0`
  2. `observation.images.tactile_right_0`
  3. `observation.images.tactile_left_1`
  4. `observation.images.tactile_right_1`
- Treat tactile inputs as RGB images, not the upstream DECO 1062-dimensional tactile-vector format.
- Tactile preprocessing uses aspect-preserving letterbox resize with black padding and float values in `[0, 1]`; visual ImageNet normalization and visual augmentation must not be applied to tactile frames.
- The encoder source may be a converted `.safetensors` file or a JAX checkpoint directory. Directory input must be converted automatically, validated, and cached by content hash.
- JAX/Flax conversion must run on CPU and only on distributed rank 0. Other ranks wait at a barrier before loading the cached artifact.
- A Stage2 fresh start (`--stage1-checkpoint`) is distinct from exact Stage2 resume (`--resume`), and the two arguments are mutually exclusive.
- Do not modify, stage, reset, or commit unrelated dirty worktree files.

---

## Task 1: Frozen tactile encoder and automatic JAX-to-PyTorch conversion

**Files:**

- Create: `train_deco/models/tactile_resnet.py`
- Create: `train_deco/tactile_encoder_conversion.py`
- Create: `train_deco/tests/test_tactile_encoder_conversion.py`
- Modify: `train_deco/pyproject.toml`
- Modify: `train_deco/setup_environment.sh`

- [ ] **Step 1: Add failing encoder-shape and state-dict tests**

  Add tests proving that `TactileResNet18()` accepts `[N, 3, 224, 224]`, returns `[N, 512]`, has no projection head, and exposes stable PyTorch parameter names. Add a small synthetic Flax-like NPZ fixture and assert that conversion rejects missing keys and shape mismatches with actionable errors.

- [ ] **Step 2: Define the conversion artifact contract in tests**

  Test this public interface:

  ```python
  @dataclass(frozen=True)
  class ResolvedTactileEncoder:
      weights_path: Path
      metadata_path: Path
      source_sha256: str
      architecture: str
      embedding_dim: int

  def resolve_tactile_encoder(
      source: str | Path,
      cache_root: str | Path,
  ) -> ResolvedTactileEncoder: ...

  def load_tactile_encoder_weights(
      module: torch.nn.Module,
      artifact: ResolvedTactileEncoder,
  ) -> None: ...
  ```

  Assert that repeated resolution of the same source returns the same cache path without rewriting files, while changed source bytes produce a different cache path. Assert metadata records source path, source digest, conversion version, architecture `resnet18`, embedding dimension `512`, framework names, and tensor-shape inventory.

- [ ] **Step 3: Implement the shared ResNet18 encoder**

  Reuse the architecture contract from `deploy_smolvla/direct_decoder.py` and `train_encoder/utils/model.py`. Keep the final representation at 512 dimensions, omit classification/projection heads, and make input validation errors explicit.

- [ ] **Step 4: Implement real Flax/JAX parameter mapping and validation**

  Read the NPZ leaves without initializing an accelerator, map Flax convolution/dense/batch-normalization layouts into the PyTorch encoder, transpose kernels where required, and validate the full expected key set and every tensor shape before saving. Save tensors with `safetensors.torch.save_file`; write JSON metadata atomically only after the weights file is valid. Use a versioned, content-addressed directory under the requested cache root.

- [ ] **Step 5: Support already-converted artifacts and directory discovery**

  Accept either a direct `.safetensors` file or a checkpoint directory containing `checkpoint.json` and the referenced/unique `params-*.npz`. Reject ambiguous directories. Never silently substitute a different pre-existing encoder artifact.

- [ ] **Step 6: Add dependency declarations and CPU-only lazy imports**

  Add `safetensors` as a runtime dependency and a tactile-conversion extra containing CPU JAX/Flax dependencies. Update the environment setup path used by `train_deco` so Stage2 installs the conversion dependencies. Set JAX platform/environment configuration before lazy-importing JAX/Flax.

- [ ] **Step 7: Verify against the actual local encoder when present**

  Add an opt-in/local integration test for `/home/typhon/FRS_Tact/checkpoints/encoder/encoder_ckpt_0824`. It must resolve the artifact, load all weights strictly, run a deterministic image through the JAX and PyTorch encoders, and compare the 512-dimensional embeddings within a documented numerical tolerance. Skip only when the source directory or conversion dependencies are unavailable.

- [ ] **Step 8: Run focused tests and commit**

  Run:

  ```bash
  .venv/bin/python -m pytest -q train_deco/tests/test_tactile_encoder_conversion.py
  ```

  Commit only Task 1 files with message `feat(train-deco): add tactile encoder conversion`.

---

## Task 2: Load and preprocess four tactile image streams

**Files:**

- Modify: `train_deco/lerobot_vision_dataset.py`
- Modify: `train_deco/input_adapter.py`
- Create: `train_deco/tests/test_tactile_dataset.py`
- Modify: `train_deco/tests/test_input_adapter.py`

- [ ] **Step 1: Add failing dataset-schema tests**

  Build a minimal LeRobot v2.1 fixture containing two visual and four tactile image fields. Assert Stage2 schema validation requires all four tactile keys with `[H, W, 3]` image features and reports the exact missing/wrong key. Assert Stage1 mode still accepts the prior two-camera schema.

- [ ] **Step 2: Add stable tactile field constants and Stage2 dataset mode**

  Define `TACTILE_NAMES` in the required order. Add a Stage2/tactile flag to dataset construction, include tactile columns in episode reads only in that mode, and return `tactile_images` as `[4, 3, H, W]` per sample while retaining `images` for visual cameras.

- [ ] **Step 3: Add failing tactile preprocessing tests**

  Assert batched inputs become `[B, 4, 3, target_h, target_w]`, black letterbox padding remains exactly zero, non-padding pixels remain in `[0, 1]`, channel order is RGB, and the routine never applies ImageNet normalization or low-light augmentation.

- [ ] **Step 4: Implement tactile preprocessing separately from visual preprocessing**

  Add a dedicated function such as:

  ```python
  def letterbox_tactile_images(
      images: torch.Tensor,
      target_size: tuple[int, int],
  ) -> torch.Tensor: ...
  ```

  Validate `[B, 4, 3, H, W]`, resize all sensors consistently, preserve aspect ratio, and use black padding.

- [ ] **Step 5: Test actual metadata compatibility when present**

  Add a local integration assertion that the configured dataset metadata recognizes all four known 224x224 RGB tactile fields. Keep it skip-safe when the dataset is absent.

- [ ] **Step 6: Run focused tests and commit**

  Run:

  ```bash
  .venv/bin/python -m pytest -q train_deco/tests/test_input_adapter.py train_deco/tests/test_tactile_dataset.py
  ```

  Commit only Task 2 files with message `feat(train-deco): load tactile image streams`.

---

## Task 3: Add Stage2 tactile-token fusion and strict Stage1 initialization

**Files:**

- Modify: `train_deco/models/deco/deco.py`
- Modify: `train_deco/model_factory.py`
- Create: `train_deco/stage2_initialization.py`
- Create: `train_deco/tests/test_stage2_model.py`
- Create: `train_deco/tests/test_stage2_initialization.py`

- [ ] **Step 1: Add failing four-token fusion tests**

  Construct a small DECO test model and assert that four tactile RGB inputs are encoded through one shared tactile encoder into `[B, 4, 512]`. Assert learned sensor embeddings have shape `[4, 512]`, RMS normalization precedes fusion, and every Stage2 transformer block receives four tactile K/V tokens.

- [ ] **Step 2: Add failing zero-gate parity test**

  Load identical Stage1 weights into a Stage1 model and a Stage2 model. With all tactile gates initialized to zero and deterministic evaluation mode, assert their action outputs match for identical visual/state/action inputs regardless of tactile pixels. This is the core guarantee that Stage2 starts from the trained Stage1 policy.

- [ ] **Step 3: Generalize tactile attention to image-token input**

  Keep Stage1 paths and state-dict keys unchanged. Add an image-token mode to DECO blocks where visual/action queries attend to the `[B, 4, 512]` tactile K/V sequence. Preserve any upstream vector-tactile scaffold only if it does not complicate or alter Stage1 behavior. Implement the residual as:

  ```python
  hidden = hidden + torch.tanh(tactile_gate) * tactile_cross_attention
  ```

  Initialize one scalar gate per block to exactly zero.

- [ ] **Step 4: Add shared encoder, RMSNorm, and sensor IDs**

  Attach one shared `TactileResNet18` to the Stage2 model. Flatten `[B, 4, 3, H, W]` to encode all sensors in one call, reshape to `[B, 4, 512]`, apply RMS normalization, and add the learned sensor-ID table. Reject wrong sensor counts and missing tactile inputs in Stage2 mode.

- [ ] **Step 5: Activate rank-32 PI adapters**

  Wire the existing PI adapter mechanism into Stage2 blocks with configurable rank defaulting to 32. Verify adapter parameter shapes, placement, and gradient flow with unit tests.

- [ ] **Step 6: Implement robust Stage1 checkpoint initialization**

  Add an initializer that reads the training checkpoint's `model` state dict, strips only known distributed prefixes, verifies all Stage1 keys and shapes exactly, loads them into the Stage2 model, and reports the expected new Stage2-only keys. Reject missing, extra legacy, or shape-mismatched Stage1 tensors rather than falling back to non-strict loading.

- [ ] **Step 7: Freeze exactly the intended parameter set**

  Freeze the tactile encoder and every parameter loaded from Stage1. Leave trainable only sensor embeddings, tactile K/V/projection parameters, scalar tactile gates, and PI adapters. Add a categorized parameter report and tests that fail if any Stage1 or encoder parameter is trainable or if an intended Stage2 parameter is frozen.

- [ ] **Step 8: Verify zero-gate gradients**

  Backpropagate one synthetic loss. Assert gates receive gradients at initialization, tactile attention/projection parameters begin receiving gradients once gates are nonzero, adapters receive gradients, and frozen parameters do not.

- [ ] **Step 9: Run focused tests and commit**

  Run:

  ```bash
  .venv/bin/python -m pytest -q train_deco/tests/test_stage2_model.py train_deco/tests/test_stage2_initialization.py
  ```

  Commit only Task 3 files with message `feat(train-deco): fuse tactile tokens in stage2`.

---

## Task 4: Implement Stage2 training lifecycle, distributed conversion, and checkpoint semantics

**Files:**

- Modify: `train_deco/train.py`
- Modify: `train_deco/training_utils.py`
- Modify: `train_deco/configs/train_deco.yaml`
- Modify: `train_deco/scripts/train_deco.sh`
- Create: `train_deco/tests/test_stage2_training.py`
- Modify: `train_deco/tests/test_train_config.py`

- [ ] **Step 1: Add failing configuration tests**

  Extend the typed configuration/CLI contract with:

  ```yaml
  stage: 2
  stage1_checkpoint: checkpoints/deco/image_aug/deco_stage1_latest.pt
  tactile_encoder_checkpoint: checkpoints/encoder/encoder_ckpt_0824
  tactile_encoder_cache: checkpoints/deco/tactile_encoder_cache
  tactile_adapter_rank: 32
  ```

  Assert Stage2 requires both initialization paths on a fresh run, `--stage1-checkpoint` and `--resume` are mutually exclusive, and Stage1 defaults remain unchanged.

- [ ] **Step 2: Add a Stage2 shell entrypoint mode**

  Extend `train_deco/scripts/train_deco.sh` with an explicit Stage2 mode that passes both user paths and keeps the current Stage1 modes intact. Do not hard-code a converted `.safetensors` filename.

- [ ] **Step 3: Resolve the tactile encoder once under DDP**

  Before constructing/loading the Stage2 model, let rank 0 call `resolve_tactile_encoder`, broadcast its success/error and resolved artifact path, then synchronize all ranks. All ranks strictly load the same cached artifact. Ensure JAX is never imported by nonzero ranks.

- [ ] **Step 4: Wire tactile batches into train and validation steps**

  Enable tactile dataset mode only for Stage2, apply tactile preprocessing, move the tensor to the policy device, and pass it to the Stage2 model in both training and evaluation. Preserve the current Stage1 call signature.

- [ ] **Step 5: Build the optimizer from trainable Stage2 parameters only**

  Ensure frozen parameters are absent from optimizer groups. Log total/trainable parameter counts plus categorized trainable names. Fail startup if the trainable set is empty or contains a Stage1/tactile-encoder parameter.

- [ ] **Step 6: Separate fresh Stage2 initialization from exact resume**

  Fresh Stage2 startup must load Stage1 model weights plus tactile encoder weights, initialize new Stage2 parameters, then create a new optimizer/scheduler/scaler state. Exact Stage2 resume must restore model, optimizer, scheduler, scaler, epoch/step, statistics, configuration, stage marker, Stage1 provenance, and tactile-encoder provenance from a Stage2 checkpoint.

- [ ] **Step 7: Add Stage2 checkpoint metadata and names**

  Save `deco_stage2_latest.pt` and `deco_stage2_best.pt`. Record a checkpoint schema/version, `model_type=upstream-deco-stage2-tactile-image`, tactile field order, encoder source digest, cached artifact digest/path, adapter rank, gate values, frozen/trainable categories, and Stage1 checkpoint provenance. Reject Stage1 checkpoints passed to `--resume`.

- [ ] **Step 8: Add tactile-aware metrics without changing the objective**

  Keep the existing DECO action loss as the Stage2 optimization objective. Add inexpensive diagnostics for per-block gate values and trainable/frozen gradient norms. Avoid auxiliary tactile losses unless separately configured in a future change.

- [ ] **Step 9: Add a CPU synthetic training/resume smoke test**

  Use a tiny/mocked visual and tactile backbone where needed to run at least one optimizer step. Assert only the allowed Stage2 parameters change, save a checkpoint, resume it, and verify the next step/epoch and optimizer state continue exactly.

- [ ] **Step 10: Run focused tests and commit**

  Run:

  ```bash
  .venv/bin/python -m pytest -q train_deco/tests/test_train_config.py train_deco/tests/test_stage2_training.py
  ```

  Commit only Task 4 files with message `feat(train-deco): add stage2 training lifecycle`.

---

## Task 5: Export and validate a portable Stage2 TorchScript policy

**Files:**

- Modify: `train_deco/export_torchscript.py`
- Modify: `train_deco/tests/test_torchscript_export.py`

- [ ] **Step 1: Add failing Stage2 export tests**

  Assert a Stage2 checkpoint exports a module with inputs for two visual streams, four tactile streams, and robot state, and returns `[B, chunk_size, action_dim]`. Assert Stage1 export retains its existing interface.

- [ ] **Step 2: Add Stage2 wrapper and metadata**

  Construct the correct model from checkpoint metadata, load strictly, include both visual and tactile preprocessing contracts, and store tactile field order, encoder digest, adapter rank, gate values, shapes, statistics, and checkpoint schema in the exported artifact metadata.

- [ ] **Step 3: Fix device portability during export**

  Cover the known baseline failure where CUDA tracing embeds CUDA constants and a CPU caller fails. Export a self-contained CPU-portable module regardless of training device, then test loading and inference on CPU. Do not mutate the live training model during export.

- [ ] **Step 4: Assert output parity**

  For deterministic synthetic inputs, compare eager and exported Stage2 outputs within a documented tolerance. Include a nonzero tactile gate so the test proves tactile input is present in the graph.

- [ ] **Step 5: Run focused tests and commit**

  Run:

  ```bash
  .venv/bin/python -m pytest -q train_deco/tests/test_torchscript_export.py
  ```

  Commit only Task 5 files with message `feat(train-deco): export stage2 tactile policy`.

---

## Task 6: End-to-end integration, operator documentation, and regression verification

**Files:**

- Modify: `train_deco/README.md`
- Create: `train_deco/tests/test_stage2_integration.py`

- [ ] **Step 1: Add an end-to-end Stage2 contract test**

  Exercise dataset sample -> visual/tactile preprocessing -> Stage2 forward/backward -> checkpoint save/resume -> TorchScript export with small deterministic fixtures. Verify four-token order and frozen/trainable boundaries at each handoff.

- [ ] **Step 2: Document the exact local command**

  Add a concise Stage2 section showing how to start from:

  ```text
  /home/typhon/FRS_Tact/checkpoints/deco/image_aug/deco_stage1_latest.pt
  /home/typhon/FRS_Tact/checkpoints/encoder/encoder_ckpt_0824
  ```

  Explain that the first run creates a content-addressed cached safetensors artifact automatically, how DDP rank-0 conversion works, where Stage2 checkpoints are written, and how `--resume` differs from `--stage1-checkpoint`.

- [ ] **Step 3: Document dataset and dependency prerequisites**

  List all four tactile field names and required shapes, state that Stage2 initially supports the `lerobot-v21` backend, explain the actionable rejection of unsupported preprocessed backends, and document how to install conversion dependencies.

- [ ] **Step 4: Run focused integration tests**

  Run:

  ```bash
  .venv/bin/python -m pytest -q train_deco/tests/test_stage2_integration.py
  ```

- [ ] **Step 5: Run the full train_deco regression suite**

  Run:

  ```bash
  .venv/bin/python -m pytest -q train_deco/tests
  ```

  Compare failures with the recorded baseline (27 passed, one pre-existing TorchScript device-portability failure). The Task 5 change must eliminate that failure and introduce no new failures.

- [ ] **Step 6: Perform final checkpoint compatibility checks**

  Load the real Stage1 checkpoint on CPU, initialize the real Stage2 model, resolve/load the real tactile encoder directory, assert the trainable-parameter allowlist, and run one no-grad forward pass using correctly shaped synthetic visual/tactile/state inputs. This check must not start a training job or alter checkpoints.

- [ ] **Step 7: Run final diff review and commit**

  Review only files changed by this implementation, verify no unrelated dirty files are staged, and commit Task 6 files with message `docs(train-deco): document stage2 tactile training`.
