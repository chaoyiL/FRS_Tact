# Real-time Camera Brightness Comparison Design

## Goal

Add a standalone VB3 camera tool that previews the two wrist-camera visual
panels, produces the exact 224x224 RGB images that image resize schemes A and B
would send over the DECO wire contract, and compares their brightness with RGB
images embedded in one or more LeRobot episode parquet files.

The tool is intended to help an operator adjust the physical light source before
robot deployment. It does not control the robot, change camera controls at
runtime, save frames by default, or modify DECO inference.

## Location and invocation

The runtime implementation belongs in `/home/typhon/vb3_robot_server` because
that repository owns the camera configuration and online A/B preprocessing
contract.

The main entry point will be:

```text
deploy_scripts/preview_camera_brightness.py
```

Example invocation:

```bash
python deploy_scripts/preview_camera_brightness.py \
  --reference-parquet /home/typhon/FRS_Tact/train_deco/episode_000004.parquet
```

`--reference-parquet` may be repeated. All supplied episodes are combined into
one reference distribution per camera and scheme.

## Color and image contracts

The reference and live paths remain separate until they produce the same model
input contract.

### Reference parquet path

1. Read only `observation.images.camera0` and
   `observation.images.camera1` with PyArrow.
2. Read the embedded JPEG bytes from each parquet struct value.
3. Decode with Pillow and call `convert("RGB")`.
4. The decoded 224x224 RGB image is the exact scheme-B training reference.
5. Estimate the scheme-A reference by taking the centered horizontal crop that
   corresponds to the center 800 pixels of the original 1280x800 visual panel,
   then resizing it back to 224x224. This distribution must be labelled
   `A APPROX` because the original full-resolution frame is unavailable.

Reference images never pass through BGR-to-RGB conversion.

### Live camera path

1. Reuse the VB3 V4L2 camera configuration and threaded reader behavior.
2. Treat decoded OpenCV/V4L2 frames as BGR uint8, regardless of misleading RGB
   variable names in older camera code.
3. Split the 3840x800 composite into three equal panels and select the middle
   1280x800 visual panel.
4. Apply the shared scheme-A or scheme-B geometry.
5. Convert the processed output from BGR to RGB.
6. Compute statistics on the untouched 224x224 uint8 RGB result before drawing
   any annotations.

This RGB result is byte-for-byte equivalent to what the server would send as
the selected DECO image input. DECO's later division by 255, letterboxing, and
ImageNet normalization are outside the human-readable brightness metric.

## Shared preprocessing

Move the pure A/B panel-resize implementation from
`real_world/bimanual_umi_env.py` into a lightweight camera preprocessing module,
then have both the robot environment and brightness tool call that module.
Keep a compatibility import or wrapper in `bimanual_umi_env.py` if existing
callers or tests import the current private function.

The shared function preserves the current behavior:

- Scheme A scales to the target height while preserving aspect ratio and then
  horizontally center-crops to the square target.
- Scheme B directly resizes the full visual panel to the square target.
- Color order remains an explicit parameter independent of A/B.
- The default DECO wire image remains RGB uint8 at 224x224.

No robot controller dependency is imported by the brightness tool merely to
reuse image preprocessing.

## Reference statistics

Decode and process parquet rows incrementally. Do not retain all decoded image
arrays in memory.

For camera0 and camera1, and separately for B and `A APPROX`, compute:

- primary per-frame RGB channel mean on the 0-255 scale;
- aggregate mean, median, q05, and q95 of the per-frame primary metric;
- mean R, G, and B channel values;
- sRGB luma mean using `0.2126 R + 0.7152 G + 0.0722 B`;
- percentage of pixels with luma at or below 16;
- percentage of pixels with luma at or above 245.

The primary metric remains the full-image RGB channel mean so it is directly
comparable with the existing DECO brightness measurements. Dark fisheye borders
are part of the model input and must not be removed from the primary metric.

## Live display

Use one OpenCV dashboard arranged as two rows and two columns:

| | Scheme A | Scheme B |
|---|---|---|
| camera0 / left hand | A image and metrics | B image and metrics |
| camera1 / right hand | A image and metrics | B image and metrics |

Each tile shows:

- current primary brightness;
- an approximately one-second rolling mean;
- the matching reference mean and q05-q95 interval;
- absolute delta and live/reference ratio;
- a `DARK`, `MATCH`, or `BRIGHT` indicator;
- compact secondary metrics for R/G/B, luma, dark pixels, and saturated pixels.

The scheme selected by the current DECO server configuration is visually
highlighted. Scheme B compares against the exact B reference; scheme A compares
against the clearly labelled `A APPROX` reference.

At startup, print the semantic and device mapping explicitly:

```text
camera0 = left_hand = <device path>
camera1 = right_hand = <device path>
```

This avoids silently inheriting the opposite `/dev/video0` and `/dev/video2`
mapping used by the older VB-VLA collection configuration.

Pressing `Q` or `Ctrl+C` exits cleanly. The first version does not save images or
metric logs.

## Dependencies

The VB3 project needs PyArrow to read embedded parquet image columns. Add it to
the project's declared runtime dependencies rather than relying on an unrelated
environment to provide it. Pillow, NumPy, and OpenCV are already project
dependencies.

## Error handling

- Reject an empty reference list.
- Report the exact parquet and column when a required camera column is missing,
  an embedded byte value is absent, or a JPEG cannot be decoded.
- Reject a reference set with no valid images.
- If one live camera fails to initialize, continue with the available camera and
  report the missing side. Exit nonzero if both fail.
- Skip a malformed live frame whose composite geometry cannot provide the
  expected three equal panels, with rate-limited diagnostics.
- Print the selected DECO scheme, RGB color contract, image size, and device
  mapping before entering the preview loop.

## Testing

Tests use synthetic images and temporary parquet files; normal unit tests do not
require camera hardware.

Required coverage:

1. Shared scheme-A and scheme-B results remain pixel-equivalent to the existing
   deployment contract.
2. A live BGR panel becomes the expected RGB DECO wire image, including a test
   with intentionally different R and B values.
3. Pillow-decoded parquet RGB is never channel-swapped.
4. The middle visual panel is selected correctly from a synthetic three-panel
   camera frame.
5. Exact B reference statistics match known synthetic inputs.
6. The `A APPROX` center-crop estimate matches the documented geometry and is
   labelled approximate.
7. Rolling metrics and reference deltas are deterministic.
8. Dashboard annotations do not affect the image used for statistics.
9. Camera0/left and camera1/right mappings are preserved when device paths are
   overridden.

After unit tests pass, run a hardware smoke test with one known episode parquet.
Confirm that the scheme highlighted in the dashboard matches the DECO server
configuration and that the displayed image matches a saved DECO observation for
the same camera scene.

## Acceptance criteria

- A single command opens a live two-camera A/B dashboard using a reference
  episode parquet.
- Every reported live primary value is computed from the exact 224x224 RGB image
  that the corresponding A/B server path would provide to DECO.
- Scheme B is compared with exact parquet statistics and scheme A is compared
  only with a clearly marked approximate reference.
- Camera side/device mappings are unambiguous.
- The production deployment and tool share one A/B preprocessing implementation.
- Existing camera preprocessing tests and new brightness tests pass.
