# Tactile Orientation Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all four deployed tactile streams use the raw orientation currently shown by `observation.images.tactile_right_0`.

**Architecture:** Keep orientation normalization at the robot-server acquisition boundary. Extract the three-panel split into a small pure helper that preserves every panel's orientation, test it with asymmetric coordinate markers, and use it from the existing camera transform without changing observation keys or RGB processing.

**Tech Stack:** Python 3.11, NumPy, OpenCV, pytest, uv

## Global Constraints

- `tactile_right_0` is the canonical orientation, and the user confirmed all training data uses it.
- Do not swap tactile keys or the `0`/`1` wrist assignment.
- Do not modify SmolVLA YAML, `rename_map`, robot control, state, or action code.
- Do not enable robot motion during verification.

---

### Task 1: Preserve Raw Tactile Panel Orientation

**Files:**
- Create: `/home/typhon/vb3_robot_server/tests/test_tactile_orientation.py`
- Modify: `/home/typhon/vb3_robot_server/real_world/bimanual_umi_env.py:27-70,184-192`

**Interfaces:**
- Consumes: one HWC NumPy image containing three equal-width panels and an integer `panel_width`
- Produces: `_split_vitac_panels(image: np.ndarray, panel_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]`, ordered as left tactile, RGB, right tactile, with no orientation transform

- [ ] **Step 1: Write the failing regression test**

```python
import numpy as np

from real_world.bimanual_umi_env import _split_vitac_panels


def _marked_panel(marker: tuple[int, int, int]) -> np.ndarray:
    panel = np.zeros((3, 4, 3), dtype=np.uint8)
    panel[0, 0] = marker
    panel[2, 3] = np.asarray(marker, dtype=np.uint8) // 2
    return panel


def test_split_vitac_panels_preserves_right_zero_orientation_for_all_panels() -> None:
    left = _marked_panel((240, 20, 10))
    visual = _marked_panel((10, 240, 20))
    right = _marked_panel((20, 10, 240))
    frame = np.concatenate((left, visual, right), axis=1)

    actual_left, actual_visual, actual_right = _split_vitac_panels(frame, panel_width=4)

    assert np.array_equal(actual_left, left)
    assert np.array_equal(actual_visual, visual)
    assert np.array_equal(actual_right, right)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd /home/typhon/vb3_robot_server
uv run --no-sync pytest -q tests/test_tactile_orientation.py
```

Expected: collection fails because `_split_vitac_panels` does not exist. This proves the new behavior is not yet represented by testable production code.

- [ ] **Step 3: Add the minimal pure helper and use it**

Add near the existing camera alignment helper:

```python
def _split_vitac_panels(
    image: np.ndarray,
    panel_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a ViTac frame without changing any panel orientation."""
    return (
        image[:, 0:panel_width],
        image[:, panel_width : 2 * panel_width],
        image[:, 2 * panel_width : 3 * panel_width],
    )
```

Replace the three manual slices and the left-only rotation with:

```python
left_tactile, visual, right_tactile = _split_vitac_panels(img, panel_width)
```

Do not change resize, BGR-to-RGB conversion, output dictionary names, or camera-index mapping.

- [ ] **Step 4: Run focused and related tests and verify GREEN**

Run:

```bash
cd /home/typhon/vb3_robot_server
uv run --no-sync pytest -q \
  tests/test_tactile_orientation.py \
  tests/test_camera_timestamp_alignment.py \
  tests/test_smolvla_runtime_contract.py
```

Expected: all selected tests pass with no warnings or errors.

- [ ] **Step 5: Run static and diff checks**

Run:

```bash
cd /home/typhon/vb3_robot_server
uv run --no-sync ruff check real_world/bimanual_umi_env.py tests/test_tactile_orientation.py
git diff --check
git diff -- real_world/bimanual_umi_env.py tests/test_tactile_orientation.py
```

Expected: ruff and whitespace checks exit 0; the diff contains only the pure split helper, its use, removal of the left-only rotation, and the regression test.

- [ ] **Step 6: Perform the no-motion observation gate**

Start the server/client without issuing robot `START`, save one observation, and compare all four tactile JPEGs to the known asymmetric calibration orientation. `left_0`, `right_0`, `left_1`, and `right_1` must use the same top/bottom convention as the reference `right_0`. Then run VT-only and FRS-enabled inference smoke checks and require finite, correctly shaped actions before enabling motion.

- [ ] **Step 7: Commit the implementation**

```bash
cd /home/typhon/vb3_robot_server
git add real_world/bimanual_umi_env.py tests/test_tactile_orientation.py
git commit -m "fix: align tactile image orientation"
```
