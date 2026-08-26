"""Factory and contract checks for the DECO Stage 1 model."""

from pathlib import Path

from .models.deco.deco import DECO, modeling


MODEL_TYPE = "upstream-deco-stage1"
STAGE2_MODEL_TYPE = "upstream-deco-stage2-tactile-image"


def observation_indices(contract: dict) -> list[int]:
    """Map the source state vector to upstream DECO's action-sized observation."""
    state_columns = list(contract["state_columns"])
    action_columns = list(contract["action_columns"])
    if len(state_columns) != int(contract["obs_dim"]):
        raise ValueError("State-column count does not match the shard observation dimension")
    if len(action_columns) != int(contract["action_dim"]):
        raise ValueError("Action-column count does not match the shard action dimension")
    explicit = contract.get("observation_indices")
    if explicit is not None:
        indices = [int(index) for index in explicit]
        if len(indices) != int(contract["action_dim"]):
            raise ValueError(
                "observation_indices count must match the action dimension"
            )
        if len(set(indices)) != len(indices):
            raise ValueError("observation_indices must be unique")
        if any(index < 0 or index >= int(contract["obs_dim"]) for index in indices):
            raise ValueError("observation_indices contains an out-of-range index")
        return indices
    if len(set(state_columns)) != len(state_columns):
        raise ValueError("State columns must be unique for the DECO observation adapter")
    missing = [name for name in action_columns if name not in state_columns]
    if missing:
        raise ValueError(
            "Upstream DECO requires an action-sized observation; action columns are "
            f"missing from the state vector: {missing}"
        )
    return [state_columns.index(name) for name in action_columns]


def build_model(config: dict, load_backbone: bool = True) -> DECO:
    """Instantiate the two- or three-camera DECO variant from its saved config."""
    backbone = config.get("backbone_weights") if load_backbone else None
    if backbone and not Path(backbone).is_file():
        raise FileNotFoundError(f"Upstream DECO ResNet34 weights not found: {backbone}")
    return modeling(
        action_dim=int(config["action_dim"]),
        chunk_size=int(config["chunk_size"]),
        obs_state=True,
        use_tactile=False,
        plugin=False,
        use_task_condition=bool(config.get("use_task_condition", False)),
        num_tasks=int(config.get("num_tasks", 1)),
        inf_step=int(config["inference_steps"]),
        num_attn_blocks=int(config["layers"]),
        heads=int(config["heads"]),
        dim=int(config["hidden_dim"]),
        rope_axes_dim=(int(config["rope_height"]), int(config["rope_width"])),
        img_pretrain=backbone,
        freeze_backbone=False,
        pretrain_model_path=False,
        adapter_model_path=False,
        num_cameras=len(config.get("camera_names", ("camera_0", "camera_1"))),
    )


def build_stage2_model(
    config: dict,
    load_backbone: bool = False,
    tactile_encoder=None,
) -> DECO:
    """Build the four-image tactile Stage2 policy without loading Stage1."""
    backbone = config.get("backbone_weights") if load_backbone else None
    if backbone and not Path(backbone).is_file():
        raise FileNotFoundError(f"Upstream DECO ResNet34 weights not found: {backbone}")
    adapter_config = config.get("adapter", {})
    adapter_rank = int(
        config.get("tactile_adapter_rank", adapter_config.get("rank", 32))
    )
    if adapter_rank < 1:
        raise ValueError(f"Stage2 tactile adapter rank must be positive, got {adapter_rank}")
    return modeling(
        action_dim=int(config["action_dim"]),
        chunk_size=int(config["chunk_size"]),
        obs_state=True,
        use_tactile=True,
        tactile_image_mode=True,
        tactile_encoder=tactile_encoder,
        plugin=True,
        plugin_rank=adapter_rank,
        use_task_condition=bool(config.get("use_task_condition", False)),
        num_tasks=int(config.get("num_tasks", 1)),
        inf_step=int(config["inference_steps"]),
        num_attn_blocks=int(config["layers"]),
        heads=int(config["heads"]),
        dim=int(config["hidden_dim"]),
        rope_axes_dim=(int(config["rope_height"]), int(config["rope_width"])),
        img_pretrain=backbone,
        freeze_backbone=False,
        pretrain_model_path=False,
        adapter_model_path=False,
        num_cameras=len(config.get("camera_names", ("camera_0", "camera_1"))),
    )
