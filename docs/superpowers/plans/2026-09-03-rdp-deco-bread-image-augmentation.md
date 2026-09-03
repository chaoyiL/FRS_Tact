# RDP DECO Bread Image Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DECO Bread photometric augmentation to single-right RDP LDP training while retaining the existing 0.9 random crop.

**Architecture:** Add a parameter-free train-only augmentation module to `MultiImageObsEncoder`. The encoder stacks its RGB observations along a view dimension so each flattened observation shares one set of photometric parameters across `camera1` and `camera2`, then resumes the existing per-camera resize, crop, normalization, and ResNet path.

**Tech Stack:** Python, PyTorch, torchvision, Hydra/OmegaConf, pytest

## Global Constraints

- Apply 25% identity and 75% brightness `[0.8, 1.2]`, contrast `[0.85, 1.30]`, and saturation `[0.80, 1.15]`.
- On a non-identity sample, apply Gaussian blur with probability 0.20, kernel 3 or 5, and sigma `[0.1, 1.0]`.
- Share photometric branch and factors across both RGB cameras for one flattened observation.
- Run photometric math in `[0, 1]`, restore RDP's `[-1, 1]` encoder domain, and bypass augmentation in eval mode.
- Retain `RandomCrop(ratio=0.9)` and ImageNet normalization.
- Do not change AT, tactile PCA embeddings, datasets, validation, offline evaluation, or deployment behavior.

---

### Task 1: Train-only shared Bread photometric augmentation

**Files:**
- Modify: `train_RDP/reactive_diffusion_policy/model/vision/multi_image_obs_encoder.py`
- Test: `train_RDP/tests/test_pick_tube_training_data.py`

**Interfaces:**
- Produces: `BreadPhotometricAugmentation(nn.Module)` accepting `[B, V, C, H, W]` float tensors and returning the same shape.
- Produces: `build_photometric_augmentation(spec)` returning `None` for disabled old configs or a configured Bread module.
- Extends: `MultiImageObsEncoder(..., photometric_augmentation: Optional[dict] = None)`.

- [ ] **Step 1: Write failing module tests**

Add imports for `BreadPhotometricAugmentation` and tests equivalent to:

```python
def test_bread_photometric_augmentation_is_bypassed_in_eval_mode():
    transform = BreadPhotometricAugmentation()
    images = torch.rand(2, 2, 3, 8, 8)
    transform.eval()
    assert transform(images) is images


def test_bread_photometric_augmentation_shares_factors_across_cameras():
    transform = BreadPhotometricAugmentation(
        identity_probability=0.0,
        brightness_range=(1.2, 1.2),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
        blur_probability=0.0,
    )
    images = torch.stack((
        torch.full((3, 8, 8), 0.2),
        torch.full((3, 8, 8), 0.4),
    )).unsqueeze(0)
    output = transform(images)
    torch.testing.assert_close(output[:, 0], images[:, 0] * 1.2)
    torch.testing.assert_close(output[:, 1], images[:, 1] * 1.2)


def test_bread_photometric_augmentation_is_seed_reproducible_and_bounded():
    transform = BreadPhotometricAugmentation(identity_probability=0.0)
    images = torch.rand(3, 2, 3, 8, 8)
    torch.manual_seed(7)
    first = transform(images)
    torch.manual_seed(7)
    second = transform(images)
    torch.testing.assert_close(first, second)
    assert first.min() >= 0.0
    assert first.max() <= 1.0
```

Also add parameter-validation tests for probabilities outside `[0,1]`, unordered/non-positive ranges, and non-positive/even blur kernels.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
cd train_RDP
.venv/bin/python -m pytest -q tests/test_pick_tube_training_data.py -k bread_photometric
```

Expected: collection fails because `BreadPhotometricAugmentation` does not exist.

- [ ] **Step 3: Implement the augmentation module and encoder integration**

In `multi_image_obs_encoder.py`:

```python
class BreadPhotometricAugmentation(nn.Module):
    def __init__(
        self,
        identity_probability=0.25,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.85, 1.30),
        saturation_range=(0.80, 1.15),
        blur_probability=0.20,
        blur_kernel_sizes=(3, 5),
        blur_sigma_range=(0.1, 1.0),
    ):
        # Validate and store the exact scalar and range configuration listed above.

    def forward(self, images):
        if not self.training:
            return images
        # Clone, sample once per B item, apply the same values to all V views,
        # optionally blur, clamp, and return without parameters or buffers.
```

Match torchvision's brightness, contrast, and saturation math with batched factors; use `torchvision.transforms.functional.gaussian_blur` only for selected blur samples. Validate every probability, ordered positive range, and odd positive kernel in `__init__`.

Add `build_photometric_augmentation(spec)` supporting only `type: Bread`, and raise `ValueError` for unsupported types.

In `MultiImageObsEncoder.__init__`, build and store the module. At the beginning of `forward`, stack all `self.rgb_keys` as `[B,V,C,H,W]`, invoke the module once, and use the resulting per-key tensors in both shared- and independent-backbone branches. Keep existing `key_transform_map` behavior unchanged so resize, random crop, and normalization remain downstream.

- [ ] **Step 4: Run focused module tests**

Run the command from Step 2.

Expected: all selected Bread photometric tests pass.

- [ ] **Step 5: Commit the module and tests**

```bash
git add train_RDP/reactive_diffusion_policy/model/vision/multi_image_obs_encoder.py train_RDP/tests/test_pick_tube_training_data.py
git commit -m "feat(rdp): add shared Bread image augmentation"
```

### Task 2: Enable Bread augmentation for single-right LDP

**Files:**
- Modify: `train_RDP/reactive_diffusion_policy/config/train_pick_tube_single_right_ldp_workspace.yaml`
- Test: `train_RDP/tests/test_pick_tube_training_data.py`

**Interfaces:**
- Consumes: `MultiImageObsEncoder.photometric_augmentation` and `type: Bread` from Task 1.
- Produces: a resolved single-right LDP config with Bread photometric augmentation plus existing `RandomCrop(ratio=0.9)`.

- [ ] **Step 1: Write a failing resolved-config test**

Compose `train_pick_tube_single_right_ldp_workspace` and assert:

```python
augmentation = single_right_ldp_cfg.policy.obs_encoder.photometric_augmentation
assert augmentation.type == "Bread"
assert augmentation.identity_probability == 0.25
assert list(augmentation.brightness_range) == [0.8, 1.2]
assert list(augmentation.contrast_range) == [0.85, 1.30]
assert list(augmentation.saturation_range) == [0.80, 1.15]
assert augmentation.blur_probability == 0.20
assert list(augmentation.blur_kernel_sizes) == [3, 5]
assert list(augmentation.blur_sigma_range) == [0.1, 1.0]
assert single_right_ldp_cfg.policy.obs_encoder.random_transforms[0].type == "RandomCrop"
assert single_right_ldp_cfg.policy.obs_encoder.random_transforms[0].ratio == 0.9
```

- [ ] **Step 2: Run the config test and verify failure**

Run:

```bash
cd train_RDP
.venv/bin/python -m pytest -q tests/test_pick_tube_training_data.py -k single_right_ldp_bread
```

Expected: FAIL because `photometric_augmentation` is absent.

- [ ] **Step 3: Add the single-right Hydra configuration**

Under `policy.obs_encoder` in `train_pick_tube_single_right_ldp_workspace.yaml`, retain `resize_shape` and add:

```yaml
photometric_augmentation:
  type: Bread
  identity_probability: 0.25
  brightness_range: [0.8, 1.2]
  contrast_range: [0.85, 1.30]
  saturation_range: [0.80, 1.15]
  blur_probability: 0.20
  blur_kernel_sizes: [3, 5]
  blur_sigma_range: [0.1, 1.0]
```

- [ ] **Step 4: Run focused and neighboring tests**

Run:

```bash
cd train_RDP
.venv/bin/python -m pytest -q tests/test_pick_tube_training_data.py -k 'bread_photometric or single_right_ldp_bread or color_jitter or pick_tube_configs'
.venv/bin/python train.py --config-name=train_pick_tube_single_right_ldp_workspace --cfg job
```

Expected: selected tests pass; printed config contains both `photometric_augmentation.type: Bread` and `random_transforms: [{type: RandomCrop, ratio: 0.9}]`.

- [ ] **Step 5: Commit configuration and test**

```bash
git add train_RDP/reactive_diffusion_policy/config/train_pick_tube_single_right_ldp_workspace.yaml train_RDP/tests/test_pick_tube_training_data.py
git commit -m "feat(rdp): enable Bread augmentation for single-right training"
```

### Task 3: Final regression verification

**Files:**
- Verify: `train_RDP/reactive_diffusion_policy/model/vision/multi_image_obs_encoder.py`
- Verify: `train_RDP/reactive_diffusion_policy/config/train_pick_tube_single_right_ldp_workspace.yaml`
- Verify: `train_RDP/tests/test_pick_tube_training_data.py`
- Modify: `deploy_RDP/reactive_diffusion_policy/model/vision/multi_image_obs_encoder.py`
- Test: `deploy_RDP/tests/test_bread_augmentation_checkpoint_compat.py`

**Interfaces:**
- Consumes: completed Tasks 1 and 2.
- Produces: evidence that the focused training data/config suite passes without changing unrelated files.

- [ ] **Step 1: Run the complete relevant test module**

```bash
cd train_RDP
.venv/bin/python -m pytest -q tests/test_pick_tube_training_data.py
```

Expected: PASS.

- [ ] **Step 2: Check the scoped diff**

```bash
git diff HEAD~2 --check -- train_RDP/reactive_diffusion_policy/model/vision/multi_image_obs_encoder.py train_RDP/reactive_diffusion_policy/config/train_pick_tube_single_right_ldp_workspace.yaml train_RDP/tests/test_pick_tube_training_data.py
git status --short
```

Expected: no whitespace errors; unrelated pre-existing dirty files remain untouched.
