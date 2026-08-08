from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from .architecture import SMOLVLA_TEXT_HIDDEN_SIZE
from .rtc import JaxRTCConfig


def compute_expert_sizes(text_hidden_size: int, expert_width_multiplier: float) -> tuple[int, int]:
    """Match PyTorch SmolVLA get_intermediate_size() for the action expert."""

    expert_hidden_size = int(text_hidden_size * expert_width_multiplier)
    intermediate = int(4 * int(2 * expert_hidden_size / 3))
    expert_intermediate_size = 256 * ((intermediate + 255) // 256)
    return expert_hidden_size, expert_intermediate_size


def _coerce_override_value(value: Any, annotation: Any) -> Any:
    """Coerce YAML/CLI override values to the dataclass field type."""

    if value is None:
        return None
    if annotation is float or annotation == "float":
        return float(value)
    if annotation is int or annotation == "int":
        return int(value)
    if annotation is bool or annotation == "bool":
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"cannot coerce {value!r} to bool")
        return bool(value)
    if annotation is str or annotation == "str":
        return str(value)
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class JaxSmolVLAConfig:
    """Static model settings required by the JAX implementation."""

    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    action_dim: int = 32
    state_dim: int = 32
    image_keys: tuple[str, ...] = ()
    empty_cameras: int = 0
    resize_height: int = 512
    resize_width: int = 512
    tokenizer_max_length: int = 48
    tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    pad_language_to: str = "max_length"
    num_steps: int = 10
    num_vlm_layers: int = 16
    num_expert_layers: int = 16
    self_attn_every_n_layers: int = 2
    attention_mode: str = "cross_attn"
    expert_width_multiplier: float = 0.75
    min_period: float = 4e-3
    max_period: float = 4.0
    use_cache: bool = True
    add_image_special_tokens: bool = False
    prefix_length: int = 0
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    trainable_compute_dtype: str = "bfloat16"
    module_modes: dict[str, str] | None = None
    lora_rank: int = 8
    lora_alpha: float = 16.0
    vlm_lora_target_modules: tuple[str, ...] = ()
    use_tactile_encoder: bool = False
    tactile_encoder_path: str | None = None
    freeze_tactile_encoder: bool = True
    tactile_keys: tuple[str, ...] = ()
    tactile_embedding_dim: int = 512
    tactile_num_tokens: int = 4
    tactile_token_repeat_factor: int = 1
    tactile_image_size: int = 224
    adapt_to_pi_aloha: bool = False
    optimizer_lr: float = 1e-4
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.95
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    rtc_config: JaxRTCConfig | None = None

    # SmolVLM2-500M architecture.
    vision_hidden_size: int = 768
    vision_intermediate_size: int = 3072
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    vision_patch_size: int = 16
    vision_layer_norm_eps: float = 1e-6
    connector_scale_factor: int = 4
    text_hidden_size: int = SMOLVLA_TEXT_HIDDEN_SIZE
    text_intermediate_size: int = 2560
    text_num_heads: int = 15
    text_num_kv_heads: int = 5
    head_dim: int = 64
    text_rms_norm_eps: float = 1e-5
    vocab_size: int = 49280
    fake_image_token_id: int = 49189
    global_image_token_id: int = 49152
    expert_hidden_size: int = 720
    expert_intermediate_size: int = 2048

    def __post_init__(self) -> None:
        _require_positive_int(self.tactile_token_repeat_factor, "tactile_token_repeat_factor")
        if self.trainable_compute_dtype != "bfloat16":
            raise ValueError(
                "trainable_compute_dtype must be 'bfloat16', "
                f"got {self.trainable_compute_dtype!r}"
            )

    @property
    def effective_tactile_num_tokens(self) -> int:
        return int(self.tactile_num_tokens) * int(self.tactile_token_repeat_factor)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> JaxSmolVLAConfig:
        path = Path(path)
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open() as config_file:
            raw: dict[str, Any] = json.load(config_file)

        output_features = raw.get("output_features", {})
        input_features = raw.get("input_features", {})
        action_dim = output_features.get("action", {}).get("shape", [raw.get("max_action_dim", 32)])[0]
        state_dim = input_features.get("observation.state", {}).get("shape", [raw.get("max_state_dim", 32)])[
            0
        ]
        image_keys = tuple(key for key, feature in input_features.items() if feature.get("type") == "VISUAL")
        resize = raw.get("resize_imgs_with_padding") or (512, 512)
        num_vlm_layers = int(raw.get("num_vlm_layers", 16))
        requested_expert_layers = int(raw.get("num_expert_layers", -1))
        num_expert_layers = requested_expert_layers if requested_expert_layers > 0 else num_vlm_layers
        expert_width_multiplier = float(raw.get("expert_width_multiplier", 0.75))
        expert_hidden_size, expert_intermediate_size = compute_expert_sizes(
            cls.text_hidden_size, expert_width_multiplier
        )
        rtc_raw = raw.get("rtc_config")
        rtc_config = None
        if rtc_raw is not None:
            rtc_config = JaxRTCConfig(
                enabled=bool(rtc_raw.get("enabled", True)),
                prefix_attention_schedule=str(rtc_raw.get("prefix_attention_schedule", "LINEAR")),
                max_guidance_weight=float(rtc_raw.get("max_guidance_weight", 10.0)),
                execution_horizon=int(rtc_raw.get("execution_horizon", 10)),
            )

        return cls(
            chunk_size=int(raw.get("chunk_size", 50)),
            n_action_steps=int(raw.get("n_action_steps", 50)),
            max_state_dim=int(raw.get("max_state_dim", 32)),
            max_action_dim=int(raw.get("max_action_dim", 32)),
            action_dim=int(action_dim),
            state_dim=int(state_dim),
            image_keys=image_keys,
            empty_cameras=int(raw.get("empty_cameras", 0)),
            resize_height=int(resize[1]),
            resize_width=int(resize[0]),
            tokenizer_max_length=int(raw.get("tokenizer_max_length", 48)),
            tokenizer_name=str(raw.get("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")),
            pad_language_to=str(raw.get("pad_language_to", "max_length")),
            num_steps=int(raw.get("num_steps", 10)),
            num_vlm_layers=num_vlm_layers,
            num_expert_layers=num_expert_layers,
            self_attn_every_n_layers=int(raw.get("self_attn_every_n_layers", 2)),
            attention_mode=str(raw.get("attention_mode", "cross_attn")),
            expert_width_multiplier=expert_width_multiplier,
            min_period=float(raw.get("min_period", 4e-3)),
            max_period=float(raw.get("max_period", 4.0)),
            use_cache=bool(raw.get("use_cache", True)),
            add_image_special_tokens=bool(raw.get("add_image_special_tokens", False)),
            prefix_length=max(0, int(raw.get("prefix_length", 0))),
            freeze_vision_encoder=bool(raw.get("freeze_vision_encoder", True)),
            train_expert_only=bool(raw.get("train_expert_only", True)),
            train_state_proj=bool(raw.get("train_state_proj", True)),
            trainable_compute_dtype=raw.get("trainable_compute_dtype", "bfloat16"),
            module_modes=raw.get("module_modes"),
            lora_rank=int(raw.get("lora_rank", 8)),
            lora_alpha=float(raw.get("lora_alpha", 16.0)),
            vlm_lora_target_modules=tuple(raw.get("vlm_lora_target_modules") or ()),
            use_tactile_encoder=bool(raw.get("use_tactile_encoder", False)),
            tactile_encoder_path=raw.get("tactile_encoder_path"),
            freeze_tactile_encoder=bool(raw.get("freeze_tactile_encoder", True)),
            tactile_keys=tuple(raw.get("tactile_keys") or ()),
            tactile_embedding_dim=int(raw.get("tactile_embedding_dim", 512)),
            tactile_num_tokens=int(raw.get("tactile_num_tokens", 4)),
            tactile_token_repeat_factor=_require_positive_int(
                raw.get("tactile_token_repeat_factor", 1),
                "tactile_token_repeat_factor",
            ),
            tactile_image_size=int(raw.get("tactile_image_size", 224)),
            adapt_to_pi_aloha=bool(raw.get("adapt_to_pi_aloha", False)),
            optimizer_lr=float(raw.get("optimizer_lr", 1e-4)),
            optimizer_beta1=float(raw.get("optimizer_betas", [0.9, 0.95])[0]),
            optimizer_beta2=float(raw.get("optimizer_betas", [0.9, 0.95])[1]),
            optimizer_eps=float(raw.get("optimizer_eps", 1e-8)),
            optimizer_weight_decay=float(raw.get("optimizer_weight_decay", 1e-10)),
            optimizer_grad_clip_norm=float(raw.get("optimizer_grad_clip_norm", 10.0)),
            scheduler_warmup_steps=int(raw.get("scheduler_warmup_steps", 1_000)),
            scheduler_decay_steps=int(raw.get("scheduler_decay_steps", 30_000)),
            scheduler_decay_lr=float(raw.get("scheduler_decay_lr", 2.5e-6)),
            rtc_config=rtc_config,
            expert_hidden_size=expert_hidden_size,
            expert_intermediate_size=expert_intermediate_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, overrides: dict[str, Any] | None) -> JaxSmolVLAConfig:
        """Apply YAML/CLI model overrides and refresh derived expert sizes."""

        if not overrides:
            return self
        if not isinstance(overrides, dict):
            raise ValueError("model overrides must be a mapping")
        field_types = {field.name: field.type for field in fields(JaxSmolVLAConfig)}
        unknown = sorted(set(overrides) - set(field_types))
        if unknown:
            raise ValueError(f"unknown model override fields: {unknown}")

        cleaned: dict[str, Any] = {}
        for key, value in overrides.items():
            if key == "tactile_token_repeat_factor":
                cleaned[key] = _require_positive_int(value, key)
                continue
            if key in {"image_keys", "tactile_keys", "vlm_lora_target_modules"} and value is not None:
                cleaned[key] = tuple(value)
            else:
                # PyYAML 1.2 treats bare scientific notation like ``1e-4`` as a
                # string; coerce to the dataclass field type so AdamW etc. get
                # real floats instead of ``"1e-4" * array``.
                cleaned[key] = _coerce_override_value(value, field_types[key])

        updated = replace(self, **cleaned)
        if updated.tactile_token_repeat_factor < 1:
            raise ValueError("tactile_token_repeat_factor must be a positive integer")

        num_vlm_layers = int(updated.num_vlm_layers)
        num_expert_layers = int(updated.num_expert_layers)
        if num_expert_layers <= 0:
            # Use as many source expert layers as possible while preserving the
            # transformer's requirement that expert depth divides VLM depth.
            max_expert_layers = min(num_vlm_layers, int(self.num_expert_layers))
            num_expert_layers = next(
                candidate
                for candidate in range(max_expert_layers, 0, -1)
                if num_vlm_layers % candidate == 0
            )
        if num_vlm_layers <= 0 or num_expert_layers <= 0:
            raise ValueError(
                f"VLM/expert layer counts must be positive, got "
                f"{num_vlm_layers}/{num_expert_layers}"
            )
        if updated.use_tactile_encoder:
            if not updated.tactile_encoder_path:
                raise ValueError("tactile_encoder_path is required when use_tactile_encoder=True")
            if not updated.tactile_keys:
                raise ValueError("tactile_keys is required when use_tactile_encoder=True")
            if len(updated.tactile_keys) != int(updated.tactile_num_tokens):
                raise ValueError(
                    "tactile_keys length must match tactile_num_tokens "
                    f"({len(updated.tactile_keys)} != {updated.tactile_num_tokens})"
                )
            overlap = sorted(set(updated.image_keys) & set(updated.tactile_keys))
            if overlap:
                raise ValueError(f"tactile_keys must not also appear in image_keys: {overlap}")
        valid_vlm_lora_targets = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        unknown_vlm_lora_targets = sorted(
            set(updated.vlm_lora_target_modules) - valid_vlm_lora_targets
        )
        if unknown_vlm_lora_targets:
            raise ValueError(
                "unknown vlm_lora_target_modules: "
                f"{unknown_vlm_lora_targets}; expected a subset of "
                f"{sorted(valid_vlm_lora_targets)}"
            )
        if num_vlm_layers % num_expert_layers:
            raise ValueError(
                f"num_expert_layers ({num_expert_layers}) must divide "
                f"num_vlm_layers ({num_vlm_layers})"
            )

        derived: dict[str, Any] = {
            "num_vlm_layers": num_vlm_layers,
            "num_expert_layers": num_expert_layers,
        }
        # Recompute expert sizes from width knobs unless the user pinned both.
        if {"expert_hidden_size", "expert_intermediate_size"} - set(cleaned):
            expert_hidden_size, expert_intermediate_size = compute_expert_sizes(
                int(updated.text_hidden_size), float(updated.expert_width_multiplier)
            )
            if "expert_hidden_size" not in cleaned:
                derived["expert_hidden_size"] = expert_hidden_size
            if "expert_intermediate_size" not in cleaned:
                derived["expert_intermediate_size"] = expert_intermediate_size

        return replace(updated, **derived)
