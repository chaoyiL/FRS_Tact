# DECO Color Contract and Observation Saving Design

## Goal

Restore the color distribution expected by the existing DECO TorchScript
artifact while ensuring saved evaluation JPEG files display with correct RGB
colors. The change must not alter PI0.5 or other policy image inputs.

## Evidence

The camera decoder produces BGR arrays. The current environment converts those
arrays to semantic RGB before publishing observations. `ObsSaver` then passes
the RGB array directly to `cv2.imwrite`, which interprets it as BGR and encodes
a JPEG with red and blue exchanged.

The DECO artifact has a historical color-contract mismatch. On the two new
scheme-B observations, using the current semantic-RGB input produced mean
left/right eight-step paths of approximately `0.8/1.0 mm`. Exchanging only the
red and blue input channels produced `5.2-5.9/1.7-1.8 mm`, and the left path was
larger for all 32 tested random seeds. This behavior is consistent with a
checkpoint trained from historically channel-swapped images.

## Architecture

Add an explicit server-owned image color order alongside the existing resize
scheme:

- Shared/default policy color order: `RGB`.
- DECO color order: `BGR`, matching the current artifact's learned input.
- Supported values: exactly `RGB` and `BGR`.

`BimanualUmiEnv` will receive the selected order and produce observation image
arrays in that order. For `RGB`, it retains the current BGR-to-RGB conversion.
For `BGR`, it retains the decoder's BGR channel order after resize. Geometry,
camera ordering, dtype, and scaling remain unchanged.

The shared SmolVLA `ObsSaver` will also receive the effective color order. It
will convert RGB arrays to BGR before calling `cv2.imwrite`; BGR arrays will be
written directly. Both paths therefore produce standard JPEG files whose
colors display correctly in PIL, browsers, IDEs, and image viewers.

The runtime startup line will report both the resize scheme and image color
order so the effective DECO contract is visible before robot execution.

## Components

- `configs/camera_config.py`: define the `ImageColorOrder` literal type.
- `configs/server_config.py`: add the shared default `image_color_order="RGB"`
  and reject unsupported values without disturbing existing camera-path edits.
- `configs/deco_server_config.py`: override the DECO value to `"BGR"` while
  retaining resize scheme `"B"`.
- `real_world/bimanual_umi_env.py`: make panel conversion conditional on the
  explicit color order and pass that setting through transform construction.
- `deploy_scripts/bimanual_smolvla_online.py`: pass the effective setting into
  the environment and `ObsSaver`, print it at startup, and encode JPEGs using
  the correct OpenCV channel order.

No client YAML, WebSocket protocol, TorchScript file, action conversion, or
PI0.5-specific saver is changed.

## Validation and Error Handling

Configuration construction rejects values outside `RGB` and `BGR` before
camera or robot initialization. Image saving only accepts the existing HWC
three-channel image contract for channel conversion; the existing handling of
non-image arrays remains unchanged.

Tests will establish:

- the shared server default is RGB and DECO explicitly selects BGR;
- both supported values work and invalid values fail;
- an asymmetric BGR sentinel is preserved in DECO mode;
- an asymmetric BGR camera frame becomes RGB in default mode;
- `ObsSaver` produces correctly encoded JPEG colors for both in-memory orders;
- the runtime passes the configured order to the environment;
- existing resize A/B, tactile orientation, message serialization, and DECO
  configuration tests remain passing.

## Compatibility

This is an artifact compatibility setting, not a claim that BGR is the desired
format for future models. A future DECO artifact trained on canonical RGB can
switch its server profile to `image_color_order="RGB"` without changing the
runtime or saver. Existing saved observations are not rewritten.
