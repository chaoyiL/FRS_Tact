# DECO Image Resize Scheme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the DECO robot-server profile an explicit A/B image resize selection, defaulting DECO to scheme `B`.

**Architecture:** Keep image resize ownership entirely in `vb3_robot_server`. `DecoServerConfig` overrides the inherited server field, while the existing shared runtime continues validating, printing, and passing the selected value to `BimanualUmiEnv`.

**Tech Stack:** Python 3.12, frozen dataclasses, pytest

## Global Constraints

- Default DECO to scheme `B`, which performs full-frame resize.
- Scheme `A` remains selectable and performs center-crop preprocessing.
- Do not change `deploy_deco.yaml`, the WebSocket protocol, shared server defaults, or other policy profiles.
- Preserve the existing unrelated modification in `/home/typhon/vb3_robot_server/configs/server_config.py`.

---

### Task 1: Add the DECO-specific resize scheme

**Files:**
- Modify: `/home/typhon/vb3_robot_server/tests/test_deco_server_config.py:1-42`
- Modify: `/home/typhon/vb3_robot_server/configs/deco_server_config.py:5-29`

**Interfaces:**
- Consumes: `ImageResizeScheme = Literal["A", "B"]` from `configs.camera_config` and the inherited `SmolVLAServerConfig.image_resize_scheme` field.
- Produces: `DECO_SERVER_CONFIG.image_resize_scheme == "B"`; callers may create an A-config with `dataclasses.replace(DECO_SERVER_CONFIG, image_resize_scheme="A")`.

- [ ] **Step 1: Write the failing DECO configuration test**

Add the following assertions to `/home/typhon/vb3_robot_server/tests/test_deco_server_config.py`:

```python
def test_deco_server_defaults_match_training_and_deployment_contract():
    assert DECO_SERVER_CONFIG.image_resize_scheme == "B"
    assert DECO_SERVER_CONFIG.policy_family == "deco"
    # Keep the existing contract assertions below these lines.


@pytest.mark.parametrize("scheme", ["A", "B"])
def test_deco_server_allows_supported_image_resize_schemes(scheme):
    config = replace(DECO_SERVER_CONFIG, image_resize_scheme=scheme)

    assert config.image_resize_scheme == scheme


def test_deco_server_rejects_unknown_image_resize_scheme():
    with pytest.raises(ValueError, match="image_resize_scheme"):
        replace(DECO_SERVER_CONFIG, image_resize_scheme="invalid")
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd /home/typhon/vb3_robot_server
uv run --no-sync pytest -q tests/test_deco_server_config.py
```

Expected: one failure because `DECO_SERVER_CONFIG.image_resize_scheme` is currently `"A"`, while the new assertion requires `"B"`. The supported/invalid value tests should already pass through inherited validation.

- [ ] **Step 3: Add the minimal DECO configuration override**

Modify `/home/typhon/vb3_robot_server/configs/deco_server_config.py`:

```python
from configs.camera_config import ImageResizeScheme
from configs.server_config import SmolVLAServerConfig


@dataclass(frozen=True)
class DecoServerConfig(SmolVLAServerConfig):
    """DECO-specific extensions to the shared bimanual hardware defaults."""

    image_resize_scheme: ImageResizeScheme = "B"
    observation_resolution: tuple[int, int] = (224, 224)
```

Keep all existing DECO expected-contract fields unchanged.

- [ ] **Step 4: Run focused DECO and preprocessing tests and verify GREEN**

Run:

```bash
cd /home/typhon/vb3_robot_server
uv run --no-sync pytest -q \
  tests/test_deco_server_config.py \
  tests/test_smolvla_runtime_contract.py \
  tests/test_camera_frame_preprocessing.py
```

Expected: all selected tests pass. Existing generic SmolVLA default remains scheme `A`; DECO default is scheme `B`; both resize implementations remain valid.

- [ ] **Step 5: Verify the effective DECO value without starting hardware**

Run:

```bash
cd /home/typhon/vb3_robot_server
uv run --no-sync python -c 'from configs.deco_server_config import DECO_SERVER_CONFIG; print(DECO_SERVER_CONFIG.image_resize_scheme)'
```

Expected output:

```text
B
```

- [ ] **Step 6: Review the exact repository diff**

Run:

```bash
git -C /home/typhon/vb3_robot_server diff --check -- configs/deco_server_config.py tests/test_deco_server_config.py
git -C /home/typhon/vb3_robot_server diff -- configs/deco_server_config.py tests/test_deco_server_config.py
```

Expected: no whitespace errors and no changes outside the DECO profile and its test.

- [ ] **Step 7: Commit only the DECO server changes if requested**

```bash
git -C /home/typhon/vb3_robot_server add configs/deco_server_config.py tests/test_deco_server_config.py
git -C /home/typhon/vb3_robot_server commit -m "feat: configure DECO image resize scheme"
```

Do not stage the existing unrelated `configs/server_config.py` modification.
