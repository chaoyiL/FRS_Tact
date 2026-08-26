# Real-time Camera Brightness Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a VB3 standalone dashboard that compares the exact A/B 224x224 RGB camera inputs with brightness distributions decoded from LeRobot episode parquet files.

**Architecture:** Extract the production A/B resize function into a lightweight shared utility, keep parquet RGB decoding and brightness math in a hardware-free metrics module, and put camera acquisition plus OpenCV rendering in a thin CLI. Scheme B uses exact parquet RGB frames; scheme A uses a clearly labelled center-crop approximation derived from those B frames.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Pillow, PyArrow, pytest, VB3 `V4L2Camera`.

## Global Constraints

- Runtime code is implemented in `/home/typhon/vb3_robot_server`; the approved design is `/home/typhon/FRS_Tact/docs/superpowers/specs/2026-08-26-realtime-camera-brightness-design.md`.
- Parquet JPEG values are decoded once with Pillow as RGB and are never channel-swapped.
- Live V4L2/OpenCV frames are BGR; statistics are computed only after production resize and BGR-to-RGB conversion.
- Primary brightness is the untouched 224x224 RGB channel mean on the 0-255 scale.
- Scheme B reference is exact; scheme A reference is always named `A APPROX`.
- `camera0 = left_hand` and `camera1 = right_hand`; print both device paths at startup.
- Do not save frames/logs, control lighting, or control the robot.
- Preserve unrelated dirty changes in `/home/typhon/vb3_robot_server`.

## File Structure

- Create `utils/camera_frame_preprocessing.py`: pure middle-panel extraction and A/B resize.
- Modify `real_world/bimanual_umi_env.py`: import the shared resize function under its existing private name.
- Create `deploy_scripts/camera_brightness_metrics.py`: RGB metrics, rolling mean, and parquet reference loader.
- Create `deploy_scripts/preview_camera_brightness.py`: CLI, live processing, camera loop, and dashboard.
- Modify `pyproject.toml`: declare PyArrow.
- Modify `README.md`: operator instructions.
- Modify/create focused tests under `tests/`.

---

### Task 1: Extract shared A/B preprocessing

**Files:**
- Create: `/home/typhon/vb3_robot_server/utils/camera_frame_preprocessing.py`
- Modify: `/home/typhon/vb3_robot_server/real_world/bimanual_umi_env.py:1-70`
- Modify: `/home/typhon/vb3_robot_server/tests/test_camera_frame_preprocessing.py`

**Interfaces:**
- Produces: `extract_visual_panel(frame_bgr: np.ndarray) -> np.ndarray`.
- Produces: `resize_panel_for_model(panel, output_resolution, obs_float32, image_resize_scheme, image_color_order="RGB") -> np.ndarray`.
- Preserves: `real_world.bimanual_umi_env._resize_panel_for_model` as a compatibility import.

- [ ] **Step 1: Write failing extraction and compatibility tests**

```python
from utils.camera_frame_preprocessing import extract_visual_panel
from utils.camera_frame_preprocessing import resize_panel_for_model


def test_extract_visual_panel_selects_middle_third(triptych_bgr_frame):
    _, expected, _ = np.split(triptych_bgr_frame, 3, axis=1)
    np.testing.assert_array_equal(extract_visual_panel(triptych_bgr_frame), expected)


@pytest.mark.parametrize("scheme", ["A", "B"])
def test_environment_resize_alias_matches_shared_utility(triptych_bgr_frame, scheme):
    panel = extract_visual_panel(triptych_bgr_frame)
    shared = resize_panel_for_model(panel, (224, 224), False, scheme, "RGB")
    environment = env_module._resize_panel_for_model(panel, (224, 224), False, scheme, "RGB")
    np.testing.assert_array_equal(environment, shared)


def test_extract_visual_panel_rejects_non_triptych_width():
    with pytest.raises(ValueError, match="three equal panels"):
        extract_visual_panel(np.zeros((800, 3839, 3), dtype=np.uint8))
```

- [ ] **Step 2: Verify the tests fail before implementation**

Run: `uv run --no-sync pytest -q tests/test_camera_frame_preprocessing.py`

Expected: collection fails because `utils.camera_frame_preprocessing` does not exist.

- [ ] **Step 3: Implement the lightweight utility**

```python
from __future__ import annotations

import math
import cv2
import numpy as np

from configs.camera_config import ImageColorOrder
from configs.camera_config import ImageResizeScheme


def extract_visual_panel(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("camera frame must be an HWC three-channel image")
    width = frame_bgr.shape[1]
    if width % 3 != 0:
        raise ValueError("camera frame width must contain three equal panels")
    panel_width = width // 3
    return frame_bgr[:, panel_width:2 * panel_width]


def resize_panel_for_model(
    panel: np.ndarray,
    output_resolution: tuple[int, int],
    obs_float32: bool,
    image_resize_scheme: ImageResizeScheme,
    image_color_order: ImageColorOrder = "RGB",
) -> np.ndarray:
    output_width, output_height = output_resolution
    if output_width != output_height:
        raise ValueError("output_resolution must be square (width must equal height)")
    if image_resize_scheme == "A":
        input_height, input_width = panel.shape[:2]
        scale = output_height / input_height
        resized_width = math.ceil(input_width * scale)
        if resized_width < output_width:
            raise ValueError("resized panel width is smaller than output width after scaling")
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(panel, (resized_width, output_height), interpolation=interpolation)
        left = (resized_width - output_width) // 2
        resized = resized[:, left:left + output_width]
    elif image_resize_scheme == "B":
        resized = cv2.resize(panel, output_resolution, interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError("image_resize_scheme must be 'A' or 'B'")
    if image_color_order == "RGB":
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    elif image_color_order != "BGR":
        raise ValueError("image_color_order must be 'RGB' or 'BGR'")
    if obs_float32:
        resized = resized.astype(np.float32) / 255
    return resized
```

In `bimanual_umi_env.py`, delete the local implementation and import:

```python
from utils.camera_frame_preprocessing import resize_panel_for_model as _resize_panel_for_model
```

- [ ] **Step 4: Run preprocessing tests**

Run: `uv run --no-sync pytest -q tests/test_camera_frame_preprocessing.py`

Expected: all existing and new preprocessing tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add utils/camera_frame_preprocessing.py real_world/bimanual_umi_env.py tests/test_camera_frame_preprocessing.py
git commit -m "refactor: share camera frame preprocessing"
```

---

### Task 2: Implement RGB metrics and parquet references

**Files:**
- Create: `/home/typhon/vb3_robot_server/deploy_scripts/camera_brightness_metrics.py`
- Create: `/home/typhon/vb3_robot_server/tests/test_camera_brightness_metrics.py`
- Modify: `/home/typhon/vb3_robot_server/pyproject.toml`

**Interfaces:**
- Produces immutable `ImageMetrics`, `DistributionSummary`, and `ReferenceStatistics`.
- Produces `compute_rgb_metrics`, `approximate_scheme_a_rgb`, `load_reference_statistics`, and `RollingMean`.
- `load_reference_statistics(paths)` returns `dict[camera0|camera1][A|B]`.

- [ ] **Step 1: Declare PyArrow and write failing pure-metric tests**

Add `"pyarrow>=18,<30"` to dependencies, then add:

```python
def test_compute_rgb_metrics_uses_rgb_channel_order():
    image = np.array([[[30, 20, 10], [90, 60, 30]]], dtype=np.uint8)
    metrics = compute_rgb_metrics(image)
    assert metrics.channel_means == pytest.approx((60.0, 40.0, 20.0))
    assert metrics.rgb_mean == pytest.approx(40.0)


def test_approximate_scheme_a_rgb_does_not_swap_channels():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[:, 42:182] = (200, 100, 25)
    actual = approximate_scheme_a_rgb(image)
    assert actual.shape == (224, 224, 3)
    assert actual[112, 112].tolist() == [200, 100, 25]
```

- [ ] **Step 2: Verify the metric tests fail**

Run: `uv run pytest -q tests/test_camera_brightness_metrics.py`

Expected: import failure because the metrics module does not exist. This first run may install the newly declared dependency.

- [ ] **Step 3: Implement metric data contracts and RGB math**

```python
@dataclass(frozen=True)
class ImageMetrics:
    rgb_mean: float
    channel_means: tuple[float, float, float]
    luma_mean: float
    dark_percent: float
    saturated_percent: float


@dataclass(frozen=True)
class DistributionSummary:
    mean: float
    median: float
    q05: float
    q95: float


@dataclass(frozen=True)
class ReferenceStatistics:
    frame_count: int
    primary: DistributionSummary
    channel_means: tuple[float, float, float]
    luma_mean: float
    dark_percent: float
    saturated_percent: float
    approximate: bool
```

Validate `uint8 HWC RGB`. Compute luma with `0.2126 R + 0.7152 G + 0.0722 B`, dark percentage with `luma <= 16`, and saturation percentage with `luma >= 245`. Compute A approximation with:

```python
crop_width = round(image_rgb.shape[1] * 800 / 1280)
left = (image_rgb.shape[1] - crop_width) // 2
cropped = image_rgb[:, left:left + crop_width]
return cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
```

- [ ] **Step 4: Write a failing embedded-JPEG parquet test**

```python
def _jpeg_bytes(rgb):
    stream = io.BytesIO()
    Image.new("RGB", (224, 224), rgb).save(stream, format="JPEG", quality=100)
    return stream.getvalue()


def _write_reference_parquet(path):
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table({
        "observation.images.camera0": pa.array([
            {"bytes": _jpeg_bytes((30, 60, 90)), "path": None},
            {"bytes": _jpeg_bytes((60, 90, 120)), "path": None},
        ], type=image_type),
        "observation.images.camera1": pa.array([
            {"bytes": _jpeg_bytes((120, 90, 60)), "path": None},
            {"bytes": _jpeg_bytes((90, 60, 30)), "path": None},
        ], type=image_type),
    })
    pq.write_table(table, path)


def test_load_reference_statistics_preserves_rgb_and_labels_a_approx(tmp_path):
    parquet = tmp_path / "episode.parquet"
    _write_reference_parquet(parquet)
    reference = load_reference_statistics([parquet])
    assert reference["camera0"]["B"].approximate is False
    assert reference["camera0"]["A"].approximate is True
    assert reference["camera0"]["B"].frame_count == 2
    assert reference["camera0"]["B"].channel_means[0] < reference["camera0"]["B"].channel_means[2]
```

- [ ] **Step 5: Implement streaming parquet aggregation**

Read only the two camera columns. Iterate Arrow chunks/scalars, decode `scalar.as_py()["bytes"]` with Pillow `convert("RGB")`, aggregate secondary sums, and retain only each frame's primary scalar for q05/median/q95. Errors name the parquet and camera column for missing columns, bytes, decode failures, invalid shape, and an empty reference.

- [ ] **Step 6: Test and implement the rolling mean**

```python
def test_rolling_mean_evicts_samples_older_than_window():
    rolling = RollingMean(window_s=1.0)
    assert rolling.add(0.0, 10.0) == 10.0
    assert rolling.add(0.5, 20.0) == 15.0
    assert rolling.add(1.1, 40.0) == 30.0
```

Implement `RollingMean.add(timestamp, value)` with a deque retaining timestamps `>= newest - window_s`.

- [ ] **Step 7: Run tests and commit Task 2**

Run: `uv run --no-sync pytest -q tests/test_camera_brightness_metrics.py`

Expected: all tests pass.

```bash
git add pyproject.toml deploy_scripts/camera_brightness_metrics.py tests/test_camera_brightness_metrics.py
git commit -m "feat: load camera brightness references"
```

Add `uv.lock` only if it already exists and the dependency command updates it.

---

### Task 3: Build live processing and the dashboard renderer

**Files:**
- Create: `/home/typhon/vb3_robot_server/deploy_scripts/preview_camera_brightness.py`
- Create: `/home/typhon/vb3_robot_server/tests/test_preview_camera_brightness.py`

**Interfaces:**
- Consumes preprocessing and metric interfaces from Tasks 1 and 2.
- Produces `process_live_frame`, `classify_brightness`, `render_tile`, and `compose_dashboard`.
- `process_live_frame(frame_bgr, output_resolution=(224, 224))` returns RGB uint8 arrays under keys A/B.

- [ ] **Step 1: Write a failing BGR-to-RGB live test**

```python
def test_process_live_frame_returns_exact_rgb_for_both_schemes():
    left = np.zeros((4, 6, 3), dtype=np.uint8)
    visual = np.full((4, 6, 3), (10, 20, 30), dtype=np.uint8)
    right = np.zeros((4, 6, 3), dtype=np.uint8)
    frame = np.concatenate((left, visual, right), axis=1)
    result = process_live_frame(frame, output_resolution=(4, 4))
    assert set(result) == {"A", "B"}
    assert result["A"][0, 0].tolist() == [30, 20, 10]
    assert result["B"][0, 0].tolist() == [30, 20, 10]
```

- [ ] **Step 2: Verify the live test fails**

Run: `uv run --no-sync pytest -q tests/test_preview_camera_brightness.py::test_process_live_frame_returns_exact_rgb_for_both_schemes`

Expected: import failure because the preview module does not exist.

- [ ] **Step 3: Implement exact live processing**

```python
def process_live_frame(frame_bgr, output_resolution=(224, 224)):
    panel_bgr = extract_visual_panel(frame_bgr)
    return {
        scheme: resize_panel_for_model(
            panel_bgr, output_resolution, False, scheme, "RGB"
        )
        for scheme in ("A", "B")
    }
```

Do not annotate or mutate returned RGB arrays.

- [ ] **Step 4: Test and implement reference-band classification**

```python
@pytest.mark.parametrize(
    ("value", "expected"),
    [(79.9, "DARK"), (80.0, "MATCH"), (100.0, "MATCH"), (100.1, "BRIGHT")],
)
def test_classify_brightness_uses_reference_quantiles(value, expected):
    reference = DistributionSummary(mean=90.0, median=90.0, q05=80.0, q95=100.0)
    assert classify_brightness(value, reference) == expected
```

Below q05 is `DARK`, above q95 is `BRIGHT`, and the closed interval is `MATCH`.

- [ ] **Step 5: Prove rendering does not mutate the statistical input**

```python
@pytest.fixture
def reference_statistics():
    return ReferenceStatistics(
        frame_count=2,
        primary=DistributionSummary(mean=60.0, median=60.0, q05=50.0, q95=70.0),
        channel_means=(30.0, 60.0, 90.0),
        luma_mean=55.0,
        dark_percent=0.0,
        saturated_percent=0.0,
        approximate=False,
    )


@pytest.fixture
def references(reference_statistics):
    approx = dataclasses.replace(reference_statistics, approximate=True)
    return {
        "camera0": {"A": approx, "B": reference_statistics},
        "camera1": {"A": approx, "B": reference_statistics},
    }


def test_render_tile_does_not_mutate_rgb_input(reference_statistics):
    image_rgb = np.full((224, 224, 3), (30, 60, 90), dtype=np.uint8)
    before = image_rgb.copy()
    metrics = compute_rgb_metrics(image_rgb)
    tile = render_tile(
        image_rgb=image_rgb,
        metrics=metrics,
        rolling_mean=metrics.rgb_mean,
        reference=reference_statistics,
        camera_label="camera0 / left",
        scheme="B",
        selected=True,
    )
    np.testing.assert_array_equal(image_rgb, before)
    assert tile.ndim == 3 and tile.shape[2] == 3
```

`render_tile` copies the RGB image, converts only that copy to BGR for OpenCV, adds header/footer space, and renders current/rolling/reference/delta/ratio/status plus channel/luma/dark/saturation values. Label approximate references `A APPROX` and highlight the selected scheme with a border.

- [ ] **Step 6: Test and implement fixed 2x2 composition**

```python
def test_compose_dashboard_keeps_camera_rows_and_scheme_columns(rendered_tiles):
    dashboard = compose_dashboard(rendered_tiles)
    assert dashboard.shape[0] == rendered_tiles[("camera0", "A")].shape[0] * 2
    assert dashboard.shape[1] == rendered_tiles[("camera0", "A")].shape[1] * 2
```

Rows are camera0/left then camera1/right; columns are A then B. Render an unavailable tile for a missing camera so meanings never move.

- [ ] **Step 7: Run renderer tests and commit Task 3**

Run: `uv run --no-sync pytest -q tests/test_preview_camera_brightness.py`

Expected: all hardware-free renderer tests pass.

```bash
git add deploy_scripts/preview_camera_brightness.py tests/test_preview_camera_brightness.py
git commit -m "feat: render camera brightness dashboard"
```

---

### Task 4: Wire the CLI to VB3 camera readers

**Files:**
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/preview_camera_brightness.py`
- Modify: `/home/typhon/vb3_robot_server/tests/test_preview_camera_brightness.py`
- Reuse: `/home/typhon/vb3_robot_server/deploy_scripts/preview_cameras.py`
- Read: `/home/typhon/vb3_robot_server/configs/deco_server_config.py`

**Interfaces:**
- Consumes existing `CameraReader`, `open_camera`, `release_failed_initialization`, `DEFAULT_CAMERA_CONFIG`, and `DECO_SERVER_CONFIG`.
- Produces `build_parser`, `build_config`, `run_dashboard`, and `main(argv=None) -> int`.

- [ ] **Step 1: Write failing parser and mapping tests**

```python
def test_parser_accepts_repeated_reference_parquets():
    args = build_parser().parse_args([
        "--reference-parquet", "episode_1.parquet",
        "--reference-parquet", "episode_2.parquet",
        "--left-device", "/dev/video8",
        "--right-device", "/dev/video10",
    ])
    assert args.reference_parquet == ["episode_1.parquet", "episode_2.parquet"]


def test_build_config_preserves_camera_semantics():
    args = build_parser().parse_args([
        "--reference-parquet", "episode.parquet",
        "--left-device", "/dev/video8",
        "--right-device", "/dev/video10",
    ])
    config = build_config(args)
    assert (config.devices[0].name, config.devices[0].path) == ("left_hand", "/dev/video8")
    assert (config.devices[1].name, config.devices[1].path) == ("right_hand", "/dev/video10")
```

- [ ] **Step 2: Implement parser and startup contract**

Required CLI:

```text
--reference-parquet PATH   repeatable and required
--side {left,right,both}   default both
--left-device PATH         optional override
--right-device PATH        optional override
```

Use `DEFAULT_CAMERA_CONFIG` for capture settings. Use `DECO_SERVER_CONFIG` for the highlighted scheme, output resolution, and color order. Reject a non-RGB DECO color contract. Print current scheme/size/color plus `camera0 = left_hand = path` and `camera1 = right_hand = path` before camera initialization.

- [ ] **Step 3: Write lifecycle tests with fake readers**

Monkeypatch reference loading, camera opening/readers, `cv2.imshow`, and `cv2.waitKey`. Use a concrete fake reader:

```python
class FakeReader:
    instances = []

    def __init__(self, name, camera):
        self.name = name
        self.camera = camera
        self.started = False
        self.stopped = False
        FakeReader.instances.append(self)

    def start(self):
        self.started = True

    def snapshot(self):
        panel = np.full((800, 1280, 3), (10, 20, 30), dtype=np.uint8)
        frame = np.concatenate((panel, panel, panel), axis=1)
        return FrameSnapshot(frame=frame, fps=20.0, error=None)

    def stop(self):
        self.stopped = True


def test_run_dashboard_continues_when_left_camera_fails(monkeypatch, references):
    FakeReader.instances.clear()

    def fake_open(device, config):
        if device.name == "left_hand":
            raise OSError("left unavailable")
        return object()

    monkeypatch.setattr(preview_module, "open_camera", fake_open)
    monkeypatch.setattr(preview_module, "CameraReader", FakeReader)
    monkeypatch.setattr(preview_module.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(preview_module.cv2, "waitKey", lambda _delay: ord("q"))
    result = run_dashboard(
        DEFAULT_CAMERA_CONFIG, "both", references, "B", (224, 224)
    )
    assert result == 0
    assert [reader.name for reader in FakeReader.instances] == ["right_hand"]
    assert FakeReader.instances[0].stopped is True


def test_run_dashboard_returns_one_when_both_cameras_fail(monkeypatch, references):
    def fail_open(_device, _config):
        raise OSError("unavailable")

    monkeypatch.setattr(preview_module, "open_camera", fail_open)
    result = run_dashboard(
        DEFAULT_CAMERA_CONFIG, "both", references, "B", (224, 224)
    )
    assert result == 1


def test_main_reports_reference_error_and_returns_two(monkeypatch, capsys):
    def fail_reference(_paths):
        raise ValueError("episode.parquet: missing observation.images.camera0")

    monkeypatch.setattr(preview_module, "load_reference_statistics", fail_reference)
    result = main(["--reference-parquet", "episode.parquet"])
    assert result == 2
    assert "episode.parquet" in capsys.readouterr().out
```

Fake snapshots use a synthetic 3840x800 BGR frame. Assert renderer metrics equal metrics from `process_live_frame`, proving annotations are downstream of statistics.

- [ ] **Step 4: Implement runtime orchestration**

Load references before hardware. Open selected devices with existing preview helpers. Maintain one `RollingMean` per active camera/scheme. Each loop processes the raw snapshot, computes untouched RGB metrics, updates rolling means with `time.monotonic()`, renders the dashboard, and calls:

```python
cv2.imshow("VB3 A/B Camera Brightness", dashboard_bgr)
```

On a reader error, stop only that reader. Return 1 when no reader remains, 0 on Q/Ctrl+C, and 2 for CLI/reference/config errors. Always release readers and destroy OpenCV windows in `finally`.

- [ ] **Step 5: Run CLI tests and commit Task 4**

Run: `uv run --no-sync pytest -q tests/test_preview_camera_brightness.py`

Expected: all tests pass without camera hardware.

```bash
git add deploy_scripts/preview_camera_brightness.py tests/test_preview_camera_brightness.py
git commit -m "feat: add realtime camera brightness CLI"
```

---

### Task 5: Document and verify the complete tool

**Files:**
- Modify: `/home/typhon/vb3_robot_server/README.md`
- Verify: all Task 1-4 files.

**Interfaces:**
- Consumes the completed CLI.
- Produces operator instructions and verification evidence.

- [ ] **Step 1: Add operator documentation**

Document one/multiple parquet commands, Q/Ctrl+C, device overrides, camera semantics, and these warnings:

```text
B reference is computed directly from the parquet RGB images.
A APPROX is reconstructed from the center of an already-resized B image and is not an exact raw-frame A reference.
Live brightness is computed from the untouched 224x224 RGB image before dashboard annotations.
```

- [ ] **Step 2: Run formatting and focused tests**

```bash
uv run --no-sync ruff check utils/camera_frame_preprocessing.py real_world/bimanual_umi_env.py deploy_scripts/camera_brightness_metrics.py deploy_scripts/preview_camera_brightness.py tests/test_camera_frame_preprocessing.py tests/test_camera_brightness_metrics.py tests/test_preview_camera_brightness.py
uv run --no-sync pytest -q tests/test_camera_frame_preprocessing.py tests/test_camera_brightness_metrics.py tests/test_preview_camera_brightness.py tests/test_deco_server_config.py
```

Expected: Ruff exits 0 and all focused tests pass.

- [ ] **Step 3: Run the broader VB3 tests**

Run: `uv run --no-sync pytest -q`

Expected: all non-manual tests pass. Record exact traceback for any unrelated pre-existing failure without editing unrelated user files.

- [ ] **Step 4: Inspect changes and commit documentation**

```bash
git diff --check
git status --short
git diff -- README.md
git add README.md
git commit -m "docs: explain camera brightness comparison"
```

Confirm all dirty files that existed before implementation remain untouched unless explicitly named in this plan.

- [ ] **Step 5: Perform the hardware smoke test when cameras are free**

```bash
uv run --no-sync python deploy_scripts/preview_camera_brightness.py \
  --reference-parquet /home/typhon/FRS_Tact/train_deco/episode_000004.parquet
```

Expected: startup identifies both semantic/device mappings, current DECO scheme/RGB contract, and reference counts. The fixed 2x2 window shows B exact and `A APPROX` targets. Q releases all opened cameras.
