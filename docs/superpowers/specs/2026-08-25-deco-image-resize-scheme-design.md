# DECO Image Resize Scheme Configuration

## Goal

Allow the DECO robot server profile to select image resize scheme `A` or `B`
without changing the shared defaults or the DECO client protocol. DECO should
default to scheme `B`, matching the full-frame resize used by the retained RDP
data and the DECO training image distribution.

## Design

The selection remains server-owned. `DecoServerConfig` in
`vb3_robot_server/configs/deco_server_config.py` will explicitly override the
inherited `image_resize_scheme` field:

```python
image_resize_scheme: ImageResizeScheme = "B"
```

Operators can switch DECO between the two supported modes by changing this
single value to `"A"` or `"B"` in the DECO server profile. The shared
`SmolVLAServerConfig` validation already rejects every other value.

No field will be added to `deploy_deco.yaml`, and no WebSocket configuration or
negotiation behavior will change. Other policy profiles continue to use their
own existing resize setting.

## Runtime Data Flow

1. `bimanual_deco_online.py` selects `configs.deco_server_config`.
2. `DECO_SERVER_CONFIG` supplies the explicit resize scheme.
3. The shared SmolVLA runtime passes that value to `BimanualUmiEnv`.
4. `_resize_panel_for_model` applies scheme `A` (center crop) or scheme `B`
   (full-frame resize) to both wrist-camera visual panels.
5. Startup output reports the selected scheme for operator confirmation.

## Testing

Update the DECO server configuration tests to establish these requirements:

- the default DECO resize scheme is `B`;
- replacing the DECO configuration with scheme `A` succeeds;
- replacing it with scheme `B` succeeds;
- an unsupported scheme is rejected by the existing dataclass validation.

The implementation will use a failing DECO-specific test before changing the
server configuration, then run the focused DECO/configuration tests and the
existing camera preprocessing tests.

## Scope

Only the DECO server profile and its tests are changed. The shared server
default, image transformation implementation, client YAML, deployment
protocol, model artifact, and other policy profiles remain unchanged.
