# DECO Balanced-Light V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable `balanced-light-v2` training preset that keeps 25% of Bread images unchanged and augments the other 75% at 90%-120% brightness without breaking `low-light-v1` checkpoint compatibility.

**Architecture:** Reuse the existing `LowLightAugmentationConfig` and `augment_training_images` engine, adding a centralized preset resolver rather than duplicating transform code. The Python entry point remains legacy-v1 when no preset is given, while `train_deco/scripts/train.sh` explicitly selects v2 for new runs; checkpoints store the fully resolved canonical config and exact resume compares that dictionary.

**Tech Stack:** Python 3.12, PyTorch/torchvision, argparse, Bash, pytest.

## Global Constraints

- `balanced-light-v2`: identity `0.25`, low-light `0.00`, mild `0.75`, brightness `0.90-1.20`.
- Both presets retain contrast `0.85-1.10`, saturation `0.90-1.10`, blur probability `0.20`, kernels `(3, 5)`, and sigma `0.1-1.0`.
- Both cameras in one sample share all random augmentation parameters.
- Identity samples are bitwise unchanged and are never blurred.
- `low-light-v1` values and random behavior remain unchanged.
- Validation, export, offline evaluation, and deployment never apply random augmentation.
- The new Bread run starts from the normal ResNet34 initialization with a new `RUN_ID` and no `RESUME_FROM`.
- Do not invent a Bread manifest: the actual long training launch requires the valid target-machine manifest.

---

### Task 1: Add canonical augmentation presets

**Files:**
- Modify: `train_deco/input_adapter.py:15-139`
- Test: `train_deco/tests/test_image_augmentation.py`

**Interfaces:**
- Produces: `augmentation_preset(name: str, *, enabled: bool = True) -> LowLightAugmentationConfig`
- Produces: `AUGMENTATION_PRESET_NAMES: tuple[str, ...]`
- Preserves: `augment_training_images(images, config) -> torch.Tensor`

- [ ] **Step 1: Write failing preset-contract tests**

Add imports for `asdict`, `augmentation_preset`, `AUGMENTATION_PRESET_NAMES`, and
`torchvision.transforms.functional as TVF`, then add:

```python
def test_low_light_v1_preset_preserves_legacy_contract():
    assert asdict(augmentation_preset("low-light-v1")) == asdict(
        LowLightAugmentationConfig()
    )


def test_balanced_light_v2_preset_is_canonical():
    config = augmentation_preset("balanced-light-v2")

    assert config.version == "balanced-light-v2"
    assert config.identity_probability == 0.25
    assert config.low_light_probability == 0.0
    assert config.mild_probability == 0.75
    assert config.mild_brightness_range == (0.90, 1.20)
    assert config.contrast_range == (0.85, 1.10)
    assert config.saturation_range == (0.90, 1.10)
    assert config.blur_probability == 0.20
    assert config.shared_across_cameras is True


def test_unknown_augmentation_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown augmentation preset"):
        augmentation_preset("bright-ish")
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  train_deco/tests/test_image_augmentation.py::test_low_light_v1_preset_preserves_legacy_contract \
  train_deco/tests/test_image_augmentation.py::test_balanced_light_v2_preset_is_canonical \
  train_deco/tests/test_image_augmentation.py::test_unknown_augmentation_preset_is_rejected
```

Expected: collection/import failure because `augmentation_preset` and `AUGMENTATION_PRESET_NAMES` do not exist.

- [ ] **Step 3: Implement the preset resolver and version-neutral validation**

Add after `LowLightAugmentationConfig`:

```python
AUGMENTATION_PRESET_NAMES = ("low-light-v1", "balanced-light-v2")


def augmentation_preset(
    name: str,
    *,
    enabled: bool = True,
) -> LowLightAugmentationConfig:
    if name == "low-light-v1":
        return LowLightAugmentationConfig(enabled=enabled)
    if name == "balanced-light-v2":
        return LowLightAugmentationConfig(
            version="balanced-light-v2",
            enabled=enabled,
            identity_probability=0.25,
            low_light_probability=0.0,
            mild_probability=0.75,
            mild_brightness_range=(0.90, 1.20),
        )
    raise ValueError(
        f"Unknown augmentation preset {name!r}; "
        f"expected one of {AUGMENTATION_PRESET_NAMES}"
    )
```

In `validate_augmentation_config`, reject unknown versions and replace the hard-coded low-light camera error:

```python
if config.version not in AUGMENTATION_PRESET_NAMES:
    raise ValueError(f"Unknown augmentation version: {config.version!r}")
...
if not config.shared_across_cameras:
    raise ValueError("DECO image augmentation requires shared_across_cameras=True")
```

- [ ] **Step 4: Add a deterministic v2 transform test**

```python
def test_balanced_v2_brightness_transform_preserves_camera_pair():
    view = torch.full((3, 8, 8), 0.5)
    images = view[None, None].repeat(1, 2, 1, 1, 1)
    config = replace(
        augmentation_preset("balanced-light-v2"),
        identity_probability=0.0,
        mild_probability=1.0,
        mild_brightness_range=(1.2, 1.2),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
        blur_probability=0.0,
    )

    result = augment_training_images(images, config)

    torch.testing.assert_close(result, torch.full_like(images, 0.6))
    assert torch.equal(result[:, 0], result[:, 1])


def test_balanced_v2_retains_contrast_saturation_and_blur():
    torch.manual_seed(17)
    images = torch.rand(1, 2, 3, 8, 8)
    config = replace(
        augmentation_preset("balanced-light-v2"),
        identity_probability=0.0,
        mild_probability=1.0,
        mild_brightness_range=(1.1, 1.1),
        contrast_range=(0.9, 0.9),
        saturation_range=(0.8, 0.8),
        blur_probability=1.0,
        blur_kernel_sizes=(3,),
        blur_sigma_range=(0.5, 0.5),
    )
    expected = TVF.adjust_brightness(images, 1.1)
    expected = TVF.adjust_contrast(expected, 0.9)
    expected = TVF.adjust_saturation(expected, 0.8)
    expected = TVF.gaussian_blur(expected, [3, 3], [0.5, 0.5])

    torch.manual_seed(17)
    result = augment_training_images(images, config)

    torch.testing.assert_close(result, expected)
```

- [ ] **Step 5: Run augmentation tests and verify GREEN**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q train_deco/tests/test_image_augmentation.py
```

Expected: all tests pass, including the unchanged v1 exposure/gamma tests.

- [ ] **Step 6: Commit Task 1**

```bash
git add train_deco/input_adapter.py train_deco/tests/test_image_augmentation.py
git commit -m "feat: add DECO balanced light preset"
```

---

### Task 2: Resolve presets through training and checkpoint contracts

**Files:**
- Modify: `train_deco/train.py:121-138`
- Modify: `train_deco/train.py:1133-1186`
- Modify: `train_deco/train.py:1420-1436`
- Test: `train_deco/tests/test_training_contract.py`
- Test: `train_deco/tests/test_stage2_training.py`

**Interfaces:**
- Consumes: `augmentation_preset(name, enabled=...)` from Task 1
- Produces: CLI option `--augmentation-preset {low-light-v1,balanced-light-v2}`
- Produces: canonical checkpoint fields `augmentation_preset` and `augmentation`

- [ ] **Step 1: Write failing training-resolution tests**

Factor the existing argument fixture in `test_training_contract.py` into this helper,
then add the tests below:

```python
def _augmentation_args(**overrides):
    values = {
        "augmentation_preset": None,
        "augmentation_enabled": True,
        "augmentation_identity_probability": 0.25,
        "augmentation_low_light_probability": 0.55,
        "augmentation_mild_probability": 0.20,
        "augmentation_exposure_probability": 0.5,
        "augmentation_exposure_range": (0.58, 0.90),
        "augmentation_gamma_range": (1.10, 1.50),
        "augmentation_mild_brightness_range": (0.90, 1.10),
        "augmentation_contrast_range": (0.85, 1.10),
        "augmentation_saturation_range": (0.90, 1.10),
        "augmentation_blur_probability": 0.20,
        "augmentation_blur_kernel_sizes": (3, 5),
        "augmentation_blur_sigma_range": (0.1, 1.0),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_training_arguments_resolve_balanced_light_v2_atomically():
    args = _augmentation_args(augmentation_preset="balanced-light-v2")

    config = asdict(augmentation_config_from_args(args))

    assert config["version"] == "balanced-light-v2"
    assert config["identity_probability"] == 0.25
    assert config["low_light_probability"] == 0.0
    assert config["mild_probability"] == 0.75
    assert config["mild_brightness_range"] == (0.90, 1.20)


def test_named_preset_rejects_fine_grained_conflicts():
    args = _augmentation_args(
        augmentation_preset="balanced-light-v2",
        augmentation_mild_probability=0.50,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        augmentation_config_from_args(args)
```

Also add a parser test:

```python
def test_python_entrypoint_defaults_to_legacy_augmentation_path():
    args = build_argument_parser().parse_args([])
    assert args.augmentation_preset is None
    assert augmentation_config_from_args(args).version == "low-light-v1"
```

- [ ] **Step 2: Run the resolution tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  train_deco/tests/test_training_contract.py::test_training_arguments_resolve_balanced_light_v2_atomically \
  train_deco/tests/test_training_contract.py::test_named_preset_rejects_fine_grained_conflicts \
  train_deco/tests/test_training_contract.py::test_python_entrypoint_defaults_to_legacy_augmentation_path
```

Expected: failures because the parser and resolver do not support presets.

- [ ] **Step 3: Add CLI preset resolution while preserving the legacy path**

Import `AUGMENTATION_PRESET_NAMES` and `augmentation_preset` in `train.py`. Add to the parser before `--augmentation-enabled`:

```python
parser.add_argument(
    "--augmentation-preset",
    choices=AUGMENTATION_PRESET_NAMES,
    default=None,
)
```

Refactor `augmentation_config_from_args` so it first builds the existing legacy
config. If no named preset is supplied, return it unchanged. If a named preset is
supplied, accept only the untouched legacy defaults and return the canonical preset:

```python
legacy_config = LowLightAugmentationConfig(
    enabled=args.augmentation_enabled,
    identity_probability=args.augmentation_identity_probability,
    low_light_probability=args.augmentation_low_light_probability,
    mild_probability=args.augmentation_mild_probability,
    exposure_probability=args.augmentation_exposure_probability,
    exposure_range=tuple(args.augmentation_exposure_range),
    gamma_range=tuple(args.augmentation_gamma_range),
    mild_brightness_range=tuple(args.augmentation_mild_brightness_range),
    contrast_range=tuple(args.augmentation_contrast_range),
    saturation_range=tuple(args.augmentation_saturation_range),
    blur_probability=args.augmentation_blur_probability,
    blur_kernel_sizes=tuple(args.augmentation_blur_kernel_sizes),
    blur_sigma_range=tuple(args.augmentation_blur_sigma_range),
)
validate_augmentation_config(legacy_config)
name = getattr(args, "augmentation_preset", None)
if name is None:
    return legacy_config
expected_legacy = LowLightAugmentationConfig(enabled=args.augmentation_enabled)
if legacy_config != expected_legacy:
    raise ValueError(
        "--augmentation-preset cannot be combined with fine-grained "
        "augmentation overrides"
    )
return augmentation_preset(name, enabled=args.augmentation_enabled)
```

When building the saved config, make the resolved identity explicit:

```python
"augmentation_preset": augmentation_config.version,
"augmentation": asdict(augmentation_config),
```

- [ ] **Step 4: Add exact-resume and Stage2 restoration tests**

Extend the resume test with canonical dictionaries:

```python
def test_exact_resume_rejects_cross_preset_configuration():
    checkpoint = {
        "training_state_version": 2,
        "augmentation": asdict(augmentation_preset("low-light-v1")),
    }
    current = {
        **checkpoint,
        "augmentation": asdict(augmentation_preset("balanced-light-v2")),
    }

    with pytest.raises(ValueError, match="augmentation"):
        validate_resume_config(checkpoint, current, resume_mode="exact")
```

In `test_stage2_training.py`, add these tests using its existing
`_valid_resume_checkpoint` fixture:

```python
def test_stage2_resume_restores_saved_augmentation_preset() -> None:
    checkpoint = _valid_resume_checkpoint()
    checkpoint["config"]["augmentation_preset"] = "balanced-light-v2"
    args = build_argument_parser().parse_args([
        "--stage", "2", "--resume", "/runtime/stage2.pt",
    ])

    restore_stage2_resume_arguments(
        args, checkpoint_loader=lambda path, device: checkpoint
    )

    assert args.augmentation_preset == "balanced-light-v2"


def test_legacy_stage2_resume_keeps_legacy_augmentation_path() -> None:
    checkpoint = _valid_resume_checkpoint()
    checkpoint["config"].pop("augmentation_preset", None)
    args = build_argument_parser().parse_args([
        "--stage", "2", "--resume", "/runtime/stage2.pt",
    ])

    restore_stage2_resume_arguments(
        args, checkpoint_loader=lambda path, device: checkpoint
    )

    assert args.augmentation_preset is None
```

- [ ] **Step 5: Run contract and Stage2 tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  train_deco/tests/test_training_contract.py \
  train_deco/tests/test_stage2_training.py
```

Expected: all tests pass; same-preset resume is accepted and cross-preset resume is rejected.

- [ ] **Step 6: Commit Task 2**

```bash
git add train_deco/train.py train_deco/tests/test_training_contract.py \
  train_deco/tests/test_stage2_training.py
git commit -m "feat: version DECO augmentation training contract"
```

---

### Task 3: Select v2 in the launcher and document the fresh Bread run

**Files:**
- Modify: `train_deco/scripts/train.sh:45-55`
- Modify: `train_deco/scripts/train.sh:200-270`
- Modify: `train_deco/tests/test_training_contract.py`
- Modify: `train_deco/configs/train_pick_tube.yaml:19-33`
- Modify: `train_deco/README.md:140-176`

**Interfaces:**
- Produces: environment variable `AUGMENTATION_PRESET`
- Produces: launcher argument `--augmentation-preset <name>`
- Produces: auditable fresh-run command for Bread v2

- [ ] **Step 1: Write failing launcher tests**

Replace the old launcher-source assertion with behavioral dry-run tests:

```python
def _launcher_dry_run(**environment):
    launcher = Path(__file__).parents[1] / "scripts" / "train.sh"
    return subprocess.run(
        ["bash", str(launcher), "--mode", "local-smoke", "--dry-run"],
        cwd=Path(__file__).parents[2],
        env={**os.environ, **environment},
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def test_train_launcher_defaults_new_runs_to_balanced_light_v2():
    output = _launcher_dry_run(RUN_ID="balanced-light-test")
    assert "--augmentation-preset balanced-light-v2" in output
    assert "--run-id balanced-light-test" in output


def test_train_launcher_can_explicitly_select_low_light_v1():
    output = _launcher_dry_run(AUGMENTATION_PRESET="low-light-v1")
    assert "--augmentation-preset low-light-v1" in output
```

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  train_deco/tests/test_training_contract.py::test_train_launcher_defaults_new_runs_to_balanced_light_v2 \
  train_deco/tests/test_training_contract.py::test_train_launcher_can_explicitly_select_low_light_v1
```

Expected: failures because the launcher does not emit `--augmentation-preset`.

- [ ] **Step 3: Make the shell launcher select an explicit preset**

Add `AUGMENTATION_PRESET` to the documented environment variables and resolve it:

```bash
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-balanced-light-v2}"
```

Add this atomic argument immediately after the augmentation enable flag:

```bash
--augmentation-preset "${AUGMENTATION_PRESET}"
```

Keep the existing fine-grained arguments at their unchanged legacy defaults. The
Python resolver from Task 2 accepts those defaults but rejects altered values when a
named preset is selected. To reproduce or resume the original Bread run, the command
must explicitly use `AUGMENTATION_PRESET=low-light-v1`.

- [ ] **Step 4: Update the reference config and README**

Replace the augmentation block in `train_pick_tube.yaml` with the canonical new-run
contract:

```yaml
augmentation:
  version: balanced-light-v2
  enabled: true
  identity_probability: 0.25
  low_light_probability: 0.00
  mild_probability: 0.75
  mild_brightness_range: [0.90, 1.20]
  contrast_range: [0.85, 1.10]
  saturation_range: [0.90, 1.10]
  blur_probability: 0.20
  blur_kernel_sizes: [3, 5]
  blur_sigma_range: [0.1, 1.0]
  shared_across_cameras: true
```

Update the README to describe both presets and include the fresh Bread command:

```bash
CUDA_VISIBLE_DEVICES=2 \
OUTPUT_DIR=/DATA/ljl/substage/deco_runs \
BATCH_SIZE=512 \
WORKERS=16 \
RUN_ID=bread-deco-stage1-balanced-light-v2 \
AUGMENTATION_PRESET=balanced-light-v2 \
RESUME_FROM= \
bash train_deco/scripts/train.sh \
  --mode local-train \
  --manifest /home/ljl/FRS_Tact/train_deco/data_manifests/bread_01_03.json
```

State explicitly that this is the manifest path recorded by the original Bread
checkpoint, that the file must exist on the target machine, and that old exact
resume requires `AUGMENTATION_PRESET=low-light-v1`. With the same manifest,
this single-process topology reproduces world size 1, global batch 512, workers
16, and the expected 1259 steps per epoch.

- [ ] **Step 5: Run launcher and documentation checks**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q train_deco/tests/test_training_contract.py
bash train_deco/scripts/train.sh --mode local-smoke --dry-run
AUGMENTATION_PRESET=low-light-v1 \
  bash train_deco/scripts/train.sh --mode local-smoke --dry-run
```

Expected: tests pass; first command line contains `balanced-light-v2`, second contains `low-light-v1`.

- [ ] **Step 6: Commit Task 3**

```bash
git add train_deco/scripts/train.sh train_deco/tests/test_training_contract.py \
  train_deco/configs/train_pick_tube.yaml train_deco/README.md
git commit -m "docs: launch DECO with balanced light augmentation"
```

---

### Task 4: Full regression and Bread training handoff

**Files:**
- Verify only: `train_deco/`
- Verify only: `docs/superpowers/specs/2026-08-28-deco-balanced-light-v2-design.md`

**Interfaces:**
- Consumes: all prior tasks
- Produces: verified code and an exact target-machine training command

- [ ] **Step 1: Run the complete DECO training test suite**

Run:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/frs_tact_mpl \
  .venv/bin/python -m pytest -q train_deco/tests
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Verify formatting and preset references**

Run:

```bash
git diff --check
rg -n "balanced-light-v2|low-light-v1|augmentation-preset" \
  train_deco/input_adapter.py train_deco/train.py train_deco/scripts/train.sh \
  train_deco/configs/train_pick_tube.yaml train_deco/README.md
```

Expected: no whitespace errors; both preset names appear in implementation, launcher, and documentation.

- [ ] **Step 3: Verify the original checkpoint remains unchanged**

Run:

```bash
sha256sum checkpoints/model/deco_0828/bread/deco_stage1_best.pt
```

Expected: `cfb2244fbde6a8c45b38291a7c123efe3c2ac45cce9120b2e944f2a4465a1c29`.

- [ ] **Step 4: Produce the target-machine dry run**

On the original target training machine, after verifying the recorded manifest
exists, run:

```bash
CUDA_VISIBLE_DEVICES=2 \
OUTPUT_DIR=/DATA/ljl/substage/deco_runs \
BATCH_SIZE=512 \
WORKERS=16 \
RUN_ID=bread-deco-stage1-balanced-light-v2 \
AUGMENTATION_PRESET=balanced-light-v2 \
RESUME_FROM= \
bash train_deco/scripts/train.sh \
  --mode local-train \
  --manifest /home/ljl/FRS_Tact/train_deco/data_manifests/bread_01_03.json \
  --dry-run
```

Expected: one Python process (not `torchrun`) with `CUDA_VISIBLE_DEVICES=2`, batch
512, workers 16, a new run ID, no `--resume`, and
`--augmentation-preset balanced-light-v2`.

- [ ] **Step 5: Record the external launch blocker accurately**

If the real manifest is not present in the current workspace, report that code and
dry-run verification are complete but do not claim the multi-hour training job has
started. Request or use the valid target-machine manifest before launching training.
