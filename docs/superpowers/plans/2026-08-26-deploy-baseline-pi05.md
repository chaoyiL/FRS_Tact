# Deploy Baseline Pi0.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `deploy_baseline_pi05` project that samples one visual Pi0.5 coarse chunk and applies the trained two-layer tactile decoder with the latest tactile observation before every indexed robot action.

**Architecture:** Reuse the existing server-directed `frs_steering_v1` wire sequence only as scheduling. Keep one normalized `[1,50,20]` Pi0.5 chunk per chunk ID, run the frozen Flax tactile ResNet plus PyTorch decoder for every unique action request, select the requested index, inverse-normalize with the Pi0.5 stats, and send the complete 20D action.

**Tech Stack:** Python 3.12, JAX/Flax/Orbax Pi0.5 and tactile encoder, PyTorch direct decoder, NumPy, msgpack/websockets, PyYAML, pytest, uv.

## Global Constraints

- Consume the checkpoint schema emitted by `train_baseline_pi05`; do not accept legacy SmolVLA `[20,20]` decoder checkpoints.
- Pi0.5 source input is visual RGB/state/task only; tactile images never enter Pi0.5.
- Source horizon/action dimensions are 50/20; decoder output is the complete `[1,50,20]` action including grippers 9/19.
- One chunk invokes Pi0.5 exactly once. Every unique action request uses the newest four tactile images and invokes encoder/decoder exactly once.
- Reuse `frs_steering_v1` message ordering without importing FRS decoder, loss or integration code.
- Never perform reverse integration, forward FRS integration, residual addition, or silent coarse fallback.
- Keep checks to checkpoint/config, shape/finite, request ordering and configured magnitude limits.
- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing JAX so JAX and the small Torch decoder can share the GPU.
- Preserve user-owned changes and unrelated files.

---

## File Map

```text
deploy_baseline_pi05/
  __init__.py
  README.md
  pyproject.toml
  uv.lock
  deployment.py                 dependency-light config and observation utilities
  policy.py                     frozen visual Pi0.5 sampler and inverse normalization
  tactile_encoder.py            frozen 0824 checkpoint runtime
  direct_decoder.py             identical two-layer Torch model
  checkpoint.py                 strict train-checkpoint loader
  runtime.py                    chunk/request state machine
  bridge_client.py              websocket/msgpack transport
  protocol.py                   frs_steering_v1 typed parsing
  remote_client.py              startup, warmup and server-directed loop
  configs/deploy_baseline_pi05.yaml
  scripts/start_baseline_pi05.sh
  src/lerobot/**                vendored inference-only Pi0.5 runtime
  tests/test_config_checkpoint.py
  tests/test_runtime.py
  tests/test_protocol_client.py
  tests/test_project.py
```

### Task 1: Deployment config, model and strict checkpoint loading

**Files:**
- Create: `deploy_baseline_pi05/__init__.py`
- Create: `deploy_baseline_pi05/deployment.py`
- Create: `deploy_baseline_pi05/direct_decoder.py`
- Create: `deploy_baseline_pi05/checkpoint.py`
- Create: `deploy_baseline_pi05/configs/deploy_baseline_pi05.yaml`
- Create: `deploy_baseline_pi05/tests/__init__.py`
- Create: `deploy_baseline_pi05/tests/test_config_checkpoint.py`

**Interfaces:**
- Consumes: `train_baseline_pi05` plan Task 2 `best.pt` schema
- Produces: `load_deployment_config(path) -> DeploymentConfig`
- Produces: deployment `DirectDecoderConfig` and `DirectTactileActionDecoder`
- Produces: `load_decoder(path, *, device, expected_source) -> DirectTactileActionDecoder`

- [ ] **Step 1: Write failing config/checkpoint contract tests**

```python
def test_deploy_yaml_locks_50_step_direct_contract():
    config = load_deployment_config(CONFIG)
    assert config.model.action_horizon == 50
    assert config.model.action_dim == 20
    assert config.direct_decoder.num_layers == 2
    assert config.observation.data_type == "vitac"


def test_training_checkpoint_loads_strictly(tmp_path):
    checkpoint = make_training_checkpoint(tmp_path, chunk_size=50, action_dim=20)
    model = load_decoder(checkpoint, device="cpu", expected_source=EXPECTED_SOURCE)
    output = model(torch.zeros(1, 50, 20), torch.zeros(1, 4, 512))
    assert output.shape == (1, 50, 20)


@pytest.mark.parametrize("field,value", [("chunk_size", 20), ("num_layers", 6), ("action_dim", 32)])
def test_loader_rejects_wrong_contract(tmp_path, field, value):
    checkpoint = make_training_checkpoint(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=field):
        load_decoder(checkpoint, device="cpu", expected_source=EXPECTED_SOURCE)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=. pytest -q deploy_baseline_pi05/tests/test_config_checkpoint.py`

Expected: FAIL because deployment modules do not exist.

- [ ] **Step 3: Implement dependency-light configuration**

Adapt path/token/config utilities from `deploy_pi05/deployment.py`, but define explicit `source`, `norm_stats`, `direct_decoder`, `connection`, `observation`, `control`, `runtime`, and `logging` dataclasses. Validate model/control horizon 50, 20D state/action, two visual camera mappings, `data_type=vitac`, four tactile keys and positive connection timeouts. Do not import JAX or Torch during config parsing.

- [ ] **Step 4: Implement byte-for-byte-compatible model structure and loader**

Copy the approved model implementation from `train_baseline_pi05/model.py` without training/loss functions. Loader uses `weights_only=True`, checks schema 1/formal/action_tactile, every fixed architecture field, tactile order, and the configured Pi0.5/norm/encoder identities, then calls strict state loading and freezes/evals the decoder.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `PYTHONPATH=. pytest -q deploy_baseline_pi05/tests/test_config_checkpoint.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add deploy_baseline_pi05/__init__.py deploy_baseline_pi05/deployment.py \
  deploy_baseline_pi05/direct_decoder.py deploy_baseline_pi05/checkpoint.py \
  deploy_baseline_pi05/configs/deploy_baseline_pi05.yaml \
  deploy_baseline_pi05/tests/__init__.py deploy_baseline_pi05/tests/test_config_checkpoint.py
git commit -m "feat: scaffold direct pi05 deployment"
```

### Task 2: Frozen source/encoder runtimes and direct steering state machine

**Files:**
- Create: `deploy_baseline_pi05/policy.py`
- Create: `deploy_baseline_pi05/tactile_encoder.py`
- Create: `deploy_baseline_pi05/runtime.py`
- Create: `deploy_baseline_pi05/tests/test_runtime.py`
- Copy then trim: `deploy_pi05/src/lerobot/policies/pi05_jax/**` to `deploy_baseline_pi05/src/lerobot/policies/pi05_jax/**`
- Copy minimal encoder modules: `deploy_pi05/frs_inference/{encoder_checkpoint.py,encoder_config.py,preprocess.py,resnet.py}` to `deploy_baseline_pi05/tactile_runtime/`

**Interfaces:**
- Consumes: deployment config/model/checkpoint
- Produces: `Pi05VisualPolicy.predict_action_chunk(...) -> np.ndarray [1,50,20]`
- Produces: `FrozenTactileEncoder.encode(observation) -> np.ndarray [1,4,512]`
- Produces: `DirectDecoderRuntime.begin_chunk(...) -> DirectChunkReady`
- Produces: `DirectDecoderRuntime.steer_action(...) -> DirectSteerResult`
- Produces: `DirectDecoderRuntime.end_chunk(chunk_id) -> None`

- [ ] **Step 1: Write failing state-machine tests with fakes**

```python
def test_one_pi_sample_and_latest_tactile_per_action(runtime, fakes):
    ready = runtime.begin_chunk(7, OBS0, TASK, seed=0, num_steps=10)
    first = runtime.steer_action(7, 10, OBS1, 0)
    second = runtime.steer_action(7, 11, OBS2, 1)
    assert fakes.policy.calls == 1
    assert fakes.encoder.observations == [OBS1, OBS2]
    assert fakes.decoder.calls == 2
    np.testing.assert_array_equal(first.selected_normalized, fakes.outputs[0][0, 0])
    np.testing.assert_array_equal(second.selected_normalized, fakes.outputs[1][0, 1])


def test_duplicate_request_is_idempotent(runtime, fakes):
    runtime.begin_chunk(7, OBS0, TASK, seed=0, num_steps=10)
    first = runtime.steer_action(7, 10, OBS1, 0)
    second = runtime.steer_action(7, 10, OBS1, 0)
    assert first is second
    assert fakes.decoder.calls == 1


def test_nonfinite_decoder_output_stops_without_fallback(runtime):
    runtime.begin_chunk(7, OBS0, TASK, seed=0, num_steps=10)
    runtime.decoder.output = np.full((1, 50, 20), np.nan, np.float32)
    with pytest.raises(ValueError, match="finite"):
        runtime.steer_action(7, 10, OBS1, 0)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=deploy_baseline_pi05/src:. pytest -q deploy_baseline_pi05/tests/test_runtime.py`

Expected: FAIL because runtime classes do not exist.

- [ ] **Step 3: Vendor Pi0.5 inference code without FRS exports**

Copy the existing inference-only tree and remove `frs` exports/imports from package initializers. Adapt `deploy_pi05/policy.py` into `Pi05VisualPolicy`: require two 224×224 visual images and 20D state, restore Orbax params strictly, run only source `sample_actions`, return normalized `[1,50,20]`, and expose `unnormalize_actions` using the configured quantile stats.

- [ ] **Step 4: Implement frozen current-frame tactile encoding**

Load `encoder_ckpt_0824` ResNet variables, permanently use inference BatchNorm, preprocess the four HWC RGB images in fixed order, encode independently, per-token RMS-normalize, and return contiguous finite float32 `[1,4,512]`. This function must match the training cache preprocessing helper.

- [ ] **Step 5: Implement the direct state machine**

Store immutable copies of coarse chunks. Hash request tactile payloads by key/dtype/shape/bytes. Enforce one active chunk, matching chunk ID, action indices 0–49 and strictly increasing unique indices. On every unique request call encoder then decoder, validate `[1,50,20]` finite output plus configurable absolute/delta limits, select the requested index, inverse-normalize all 20 dimensions, cache the result by request ID, and expose timing/delta diagnostics. Never return the coarse action when refinement fails.

- [ ] **Step 6: Run runtime tests and verify GREEN**

Run: `PYTHONPATH=deploy_baseline_pi05/src:. pytest -q deploy_baseline_pi05/tests/test_runtime.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add deploy_baseline_pi05/policy.py deploy_baseline_pi05/tactile_encoder.py \
  deploy_baseline_pi05/tactile_runtime deploy_baseline_pi05/runtime.py \
  deploy_baseline_pi05/src/lerobot deploy_baseline_pi05/tests/test_runtime.py
git commit -m "feat: add pi05 direct steering runtime"
```

### Task 3: Server-directed protocol, bridge and remote client

**Files:**
- Create: `deploy_baseline_pi05/protocol.py`
- Create: `deploy_baseline_pi05/bridge_client.py`
- Create: `deploy_baseline_pi05/remote_client.py`
- Create: `deploy_baseline_pi05/tests/test_protocol_client.py`

**Interfaces:**
- Consumes: `DirectDecoderRuntime`
- Produces: strict parsed start/steer/end protocol messages
- Produces: `run_direct_protocol(bridge, runtime, ...) -> None`
- Produces: `python -m deploy_baseline_pi05.remote_client --config PATH`

- [ ] **Step 1: Write failing protocol-order integration tests**

```python
def test_protocol_chunk_lifecycle(fake_bridge, fake_runtime):
    fake_bridge.messages = [START, STEER_0, STEER_1, END]
    run_direct_protocol(fake_bridge, fake_runtime, max_iterations=1)
    assert fake_runtime.calls == ["begin:3", "steer:3:0", "steer:3:1", "end:3"]
    assert [message["type"] for message in fake_bridge.sent] == [
        "frs_chunk_ready", "frs_steer_action", "frs_steer_action"
    ]


def test_runtime_error_sends_no_action_and_propagates(fake_bridge, failing_runtime):
    fake_bridge.messages = [START, STEER_0]
    with pytest.raises(ValueError, match="decoder"):
        run_direct_protocol(fake_bridge, failing_runtime, max_iterations=1)
    assert not any(message["type"] == "frs_steer_action" for message in fake_bridge.sent)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=deploy_baseline_pi05/src:. pytest -q deploy_baseline_pi05/tests/test_protocol_client.py`

Expected: FAIL because protocol/client modules do not exist.

- [ ] **Step 3: Adapt strict protocol and transport**

Copy the msgpack ndarray encoding, hello/authentication, timeout and close behavior from `deploy_pi05/bridge_client.py`. Copy the `frs_steering_v1` parser shapes from `deploy_pi05/frs_protocol.py` but keep the wire field names unchanged. Do not import `deploy_pi05` or any FRS runtime/model module.

- [ ] **Step 4: Implement remote lifecycle**

Load config bytes once, print config digest, set up logging, lazily initialize JAX/Torch runtimes, perform one fake-shape warmup for policy/encoder/decoder, connect/authenticate, send start, then run chunk lifecycle until `max_iterations`. Save observation/action traces asynchronously using bounded queues copied from deployment utilities. On exception, send stop/close when possible, propagate a nonzero exit, and never send a replacement coarse action.

- [ ] **Step 5: Run protocol tests and verify GREEN**

Run: `PYTHONPATH=deploy_baseline_pi05/src:. pytest -q deploy_baseline_pi05/tests/test_protocol_client.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add deploy_baseline_pi05/protocol.py deploy_baseline_pi05/bridge_client.py \
  deploy_baseline_pi05/remote_client.py deploy_baseline_pi05/tests/test_protocol_client.py
git commit -m "feat: add direct decoder robot protocol"
```

### Task 4: Isolated environment, launchers, docs and parity verification

**Files:**
- Create: `deploy_baseline_pi05/pyproject.toml`
- Create: `deploy_baseline_pi05/uv.lock`
- Create: `deploy_baseline_pi05/scripts/start_baseline_pi05.sh`
- Create: `deploy_baseline_pi05/README.md`
- Create: `deploy_baseline_pi05/tests/test_project.py`

**Interfaces:**
- Consumes: complete deploy project and training checkpoint fixture
- Produces: `bash deploy_baseline_pi05/scripts/start_baseline_pi05.sh [--check] [--max-iterations N]`

- [ ] **Step 1: Write failing project-boundary and cross-project parity tests**

```python
def test_deploy_package_has_no_frs_runtime_imports():
    sources = "\n".join(path.read_text() for path in DEPLOY.rglob("*.py"))
    assert "frs_runtime" not in sources
    assert "reverse_integrate" not in sources
    assert "sample_and_reverse" not in sources


def test_train_and_deploy_decoder_forward_parity():
    torch.manual_seed(7)
    train_model = TrainDecoder(TrainConfig()).eval()
    deploy_model = DeployDecoder(DeployConfig()).eval()
    deploy_model.load_state_dict(train_model.state_dict(), strict=True)
    coarse = torch.randn(2, 50, 20)
    tactile = torch.randn(2, 4, 512)
    torch.testing.assert_close(train_model(coarse, tactile), deploy_model(coarse, tactile), rtol=0, atol=0)


def test_launcher_sets_jax_memory_flag_before_python():
    text = (DEPLOY / "scripts/start_baseline_pi05.sh").read_text()
    assert text.index("XLA_PYTHON_CLIENT_PREALLOCATE") < text.index("exec")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=deploy_baseline_pi05/src:. pytest -q deploy_baseline_pi05/tests/test_project.py`

Expected: FAIL because project files/launcher do not exist.

- [ ] **Step 3: Add isolated dependency manifest and launcher**

Base dependencies on `deploy_pi05/pyproject.toml`, add direct `safetensors`, keep Torch/JAX/Flax/Orbax/websockets/msgpack/PyYAML/OpenCV, remove training-only datasets/matplotlib/optax. The launcher selects only `deploy_baseline_pi05/.venv`, loads token from environment or configured token file, sets package-local `PYTHONPATH`, `PYTHONSAFEPATH=1`, `PYTHONUNBUFFERED=1`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`, then invokes the remote client. `--check` parses config and paths without importing JAX/Torch or connecting.

- [ ] **Step 4: Generate lock and write deployment documentation**

Run `uv lock --project deploy_baseline_pi05`. Document asset paths, environment setup, server requirement for `frs_steering_v1` with vitac observations, `--check`, bounded iteration command, trace locations, fail-stop behavior, and the explicit statement that no FRS integration occurs despite reuse of wire message names.

- [ ] **Step 5: Run focused and full verification**

Run:

```bash
bash -n deploy_baseline_pi05/scripts/start_baseline_pi05.sh
PYTHONPATH=train_baseline_pi05/src:deploy_baseline_pi05/src:. \
  pytest -q deploy_baseline_pi05/tests train_baseline_pi05/tests/test_model_checkpoint.py
uv lock --check --project deploy_baseline_pi05
```

Expected: shell syntax exits 0, all selected tests pass, parity is exact, and the lock is current.

- [ ] **Step 6: Commit Task 4**

```bash
git add deploy_baseline_pi05/pyproject.toml deploy_baseline_pi05/uv.lock \
  deploy_baseline_pi05/scripts deploy_baseline_pi05/README.md \
  deploy_baseline_pi05/tests/test_project.py
git commit -m "feat: finish standalone pi05 baseline deployment"
```

### Task 5: Whole-feature verification and handoff

**Files:**
- Modify only if verification exposes a concrete defect: files covered by the failing test
- Test: `train_baseline_pi05/tests/**`
- Test: `deploy_baseline_pi05/tests/**`

**Interfaces:**
- Consumes: both completed standalone projects
- Produces: verified local CPU handoff and exact server GPU commands

- [ ] **Step 1: Run all baseline tests from a clean process**

```bash
PYTHONPATH=train_baseline_pi05/src:deploy_baseline_pi05/src:. \
  pytest -q train_baseline_pi05/tests deploy_baseline_pi05/tests
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run dependency-light preflights**

```bash
bash train_baseline_pi05/scripts/start_train.sh --check \
  train_baseline_pi05/configs/train_baseline_pi05.yaml
bash deploy_baseline_pi05/scripts/start_baseline_pi05.sh --check
```

Expected: both commands exit 0, print resolved assets/contracts, do not initialize JAX/GPU, create outputs, or connect to the robot.

- [ ] **Step 3: Inspect the final diff and contract coverage**

Run: `git diff --check 1be369f..HEAD && git status --short`

Expected: no whitespace errors; only intentional baseline files plus pre-existing user changes are present.

- [ ] **Step 4: Request whole-feature code review**

Dispatch a reviewer with the approved design, both plan files, merge-base, head SHA, full test output and review package. Fix every Critical/Important issue using a failing regression test first, rerun its covering tests, then rerun the full suite.

- [ ] **Step 5: Record server-only verification limits**

Handoff must state that local verification covers CPU synthetic/cache/checkpoint/protocol behavior only. Provide exact commands for server environment sync, small real cache, one-step/one-epoch GPU training, strict checkpoint reload/evaluation, deployment `--check`, and a bounded one-chunk robot run; do not claim those external runs were performed locally.
