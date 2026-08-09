from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import SafetensorError, safe_open

from .architecture import SMOLVLA_TEXT_HIDDEN_SIZE

_PREPROCESSOR_FILE = "policy_preprocessor.json"
_POSTPROCESSOR_FILE = "policy_postprocessor.json"
_PREPROCESSOR_STATS_FILE = "policy_preprocessor_step_5_normalizer_processor.safetensors"
_POSTPROCESSOR_STATS_FILE = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
_SIDECAR_FILENAMES = (
    "config.json",
    _PREPROCESSOR_FILE,
    _POSTPROCESSOR_FILE,
    _PREPROCESSOR_STATS_FILE,
    _POSTPROCESSOR_STATS_FILE,
)
_BASE_MODULE_NAMES = frozenset(("vision", "connector", "vlm_text", "expert", "action", "state_proj"))
_VALID_MODULE_MODES = frozenset(("frozen", "full", "lora"))
_ATTENTION_LORA_TARGETS = frozenset(("q_proj", "k_proj", "v_proj", "o_proj"))
_MLP_LORA_TARGETS = frozenset(("gate_proj", "up_proj", "down_proj"))
_VLM_LORA_TARGETS = _ATTENTION_LORA_TARGETS | _MLP_LORA_TARGETS
_VLM_LAYER_RE = re.compile(r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.(\d+)\.")
_VLM_TEXT_PREFIX = "model.vlm_with_expert.vlm.model.text_model"
_VLM_EMBED_TOKENS = f"{_VLM_TEXT_PREFIX}.embed_tokens.weight"
@dataclass(frozen=True)
class CheckpointContract:
    state_dim: int
    action_dim: int
    chunk_size: int
    image_keys: tuple[str, ...]
    lora_rank: int = 0
    vlm_lora_target_modules: tuple[str, ...] = ()


def contract_from_config(config: Any) -> CheckpointContract:
    """Build the validator contract from an effective training configuration."""

    return CheckpointContract(
        state_dim=int(config.state_dim),
        action_dim=int(config.action_dim),
        chunk_size=int(config.chunk_size),
        image_keys=tuple(config.image_keys),
        lora_rank=int(config.lora_rank),
        vlm_lora_target_modules=tuple(config.vlm_lora_target_modules),
    )


class CheckpointValidationExtension:
    """Neutral hook set for validating additional model input modalities."""

    module_names: frozenset[str] = frozenset()

    def parse_config(self, raw: Mapping[str, Any], issues: list[str]) -> Any:
        del raw, issues
        return None

    def build_contract(
        self,
        base: CheckpointContract,
        config: Any,
    ) -> CheckpointContract | None:
        del config
        return base

    def compare_contract(
        self,
        config: Any,
        expected: CheckpointContract,
        issues: list[str],
    ) -> None:
        del config, expected, issues

    def validate_config(
        self,
        raw: Mapping[str, Any],
        config: Any,
        image_keys: tuple[str, ...],
        issues: list[str],
    ) -> None:
        del raw, config, image_keys, issues

    def needs_model_structure(self, contract: CheckpointContract) -> bool:
        del contract
        return False

    def validate_model(
        self,
        tensors: Any,
        keys: set[str],
        *,
        path: Path,
        contract: CheckpointContract,
        hidden_size: int,
        issues: list[str],
    ) -> None:
        del tensors, keys, path, contract, hidden_size, issues


DEFAULT_VALIDATION_EXTENSION = CheckpointValidationExtension()


@dataclass(frozen=True)
class CheckpointValidationReport:
    path: Path
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def format_errors(self) -> str:
        if not self.issues:
            return f"checkpoint validation passed: {self.path}"
        details = "\n".join(f"- {issue}" for issue in self.issues)
        return f"checkpoint validation failed for {self.path}:\n{details}"

    def require_valid(self) -> None:
        if self.issues:
            raise ValueError(self.format_errors())


@dataclass(frozen=True)
class _ConfigView:
    state_dim: int | None
    action_dim: int | None
    chunk_size: int | None
    image_keys: tuple[str, ...]
    lora_rank: int | None
    vlm_lora_target_modules: tuple[str, ...]
    num_vlm_layers: int | None
    extension: Any = None

    def as_contract(
        self,
        extension: CheckpointValidationExtension,
    ) -> CheckpointContract | None:
        required = (
            self.state_dim,
            self.action_dim,
            self.chunk_size,
            self.lora_rank,
        )
        if any(value is None for value in required):
            return None
        base = CheckpointContract(
            state_dim=int(self.state_dim),
            action_dim=int(self.action_dim),
            chunk_size=int(self.chunk_size),
            image_keys=self.image_keys,
            lora_rank=int(self.lora_rank),
            vlm_lora_target_modules=self.vlm_lora_target_modules,
        )
        return extension.build_contract(base, self.extension)


def _load_json(path: Path, issues: list[str], *, description: str | None = None) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(f"could not read {description or path.name}: {exc}")
        return None
    if not isinstance(value, dict):
        issues.append(f"{description or path.name} must contain a JSON object")
        return None
    return value


def _integer(value: Any, field: str, issues: list[str], *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        issues.append(f"config {field} must be an integer, got {value!r}")
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        issues.append(f"config {field} must be an integer, got {value!r}")
        return None


def _feature_dim(
    features: Any,
    key: str,
    label: str,
    issues: list[str],
) -> int | None:
    if not isinstance(features, Mapping):
        issues.append(f"config {label} must be an object")
        return None
    feature = features.get(key)
    if not isinstance(feature, Mapping):
        issues.append(f"config {label}.{key} is missing or is not an object")
        return None
    shape = feature.get("shape")
    if not isinstance(shape, list | tuple) or len(shape) != 1:
        issues.append(f"config {label}.{key}.shape must be one-dimensional, got {shape!r}")
        return None
    return _integer(shape[0], f"{label}.{key}.shape[0]", issues)


def _string_tuple(value: Any, field: str, issues: list[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple) or any(not isinstance(item, str) for item in value):
        issues.append(f"config {field} must be a list of strings, got {value!r}")
        return ()
    return tuple(value)


def _parse_config(
    raw: Mapping[str, Any],
    issues: list[str],
    extension: CheckpointValidationExtension,
) -> _ConfigView:
    input_features = raw.get("input_features")
    output_features = raw.get("output_features")
    state_dim = _feature_dim(input_features, "observation.state", "input_features", issues)
    action_dim = _feature_dim(output_features, "action", "output_features", issues)

    image_keys: tuple[str, ...] = ()
    if isinstance(input_features, Mapping):
        image_keys = tuple(
            key
            for key, feature in input_features.items()
            if isinstance(key, str)
            and isinstance(feature, Mapping)
            and str(feature.get("type", "")).upper() == "VISUAL"
        )
    lora_targets = _string_tuple(raw.get("vlm_lora_target_modules"), "vlm_lora_target_modules", issues)
    if "chunk_size" not in raw or raw.get("chunk_size") is None:
        issues.append("config chunk_size is missing")
        chunk_size = None
    else:
        chunk_size = _integer(raw.get("chunk_size"), "chunk_size", issues)
    return _ConfigView(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        image_keys=image_keys,
        lora_rank=_integer(raw.get("lora_rank"), "lora_rank", issues, default=0),
        vlm_lora_target_modules=lora_targets,
        num_vlm_layers=_integer(raw.get("num_vlm_layers"), "num_vlm_layers", issues),
        extension=extension.parse_config(raw, issues),
    )


def _compare_contract(
    config: _ConfigView,
    expected: CheckpointContract,
    issues: list[str],
    extension: CheckpointValidationExtension,
) -> None:
    actual_fields: dict[str, Any] = {
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        "chunk_size": config.chunk_size,
        "image_keys": config.image_keys,
        "lora_rank": config.lora_rank,
        "vlm_lora_target_modules": config.vlm_lora_target_modules,
    }
    for field, actual in actual_fields.items():
        wanted = getattr(expected, field)
        if actual is not None and actual != wanted:
            issues.append(f"config {field} expected {wanted!r}, got {actual!r}")
    extension.compare_contract(config.extension, expected, issues)


def _validate_config_consistency(
    raw: Mapping[str, Any],
    config: _ConfigView,
    issues: list[str],
    extension: CheckpointValidationExtension,
) -> None:
    if len(config.image_keys) != len(set(config.image_keys)):
        issues.append(f"config image_keys contain duplicates: {config.image_keys!r}")
    extension.validate_config(raw, config.extension, config.image_keys, issues)

    modes = raw.get("module_modes")
    if modes is not None:
        if not isinstance(modes, Mapping):
            issues.append("config module_modes must be an object")
            modes = None
        else:
            unknown = sorted(set(modes) - (_BASE_MODULE_NAMES | extension.module_names))
            missing = sorted(_BASE_MODULE_NAMES - set(modes))
            if unknown:
                issues.append(f"config module_modes has unknown modules: {unknown}")
            if missing:
                issues.append(f"config module_modes is missing modules: {missing}")
            invalid = {
                str(module): str(mode)
                for module, mode in modes.items()
                if str(mode).lower() not in _VALID_MODULE_MODES
            }
            if invalid:
                issues.append(f"config module_modes has invalid modes: {invalid}")
    if config.vlm_lora_target_modules:
        if config.lora_rank is None or config.lora_rank <= 0:
            issues.append("config lora_rank must be positive when vlm_lora_target_modules is configured")
        if not isinstance(modes, Mapping) or str(modes.get("vlm_text", "")).lower() != "lora":
            issues.append(
                "config module_modes.vlm_text must be 'lora' when vlm_lora_target_modules is configured"
            )
        if config.num_vlm_layers is None or config.num_vlm_layers <= 0:
            issues.append("config num_vlm_layers must be positive when vlm_lora_target_modules is configured")
        unknown_targets = sorted(set(config.vlm_lora_target_modules) - _VLM_LORA_TARGETS)
        if unknown_targets:
            issues.append(f"config has unknown vlm_lora_target_modules: {unknown_targets}")


def _step(
    processor: Mapping[str, Any],
    registry_name: str,
    filename: str,
    issues: list[str],
) -> Mapping[str, Any] | None:
    steps = processor.get("steps")
    if not isinstance(steps, list):
        issues.append(f"{filename} steps must be a list")
        return None
    matches = [
        item for item in steps if isinstance(item, Mapping) and item.get("registry_name") == registry_name
    ]
    if not matches:
        issues.append(f"{filename} is missing the {registry_name} step")
        return None
    if len(matches) > 1:
        issues.append(f"{filename} contains multiple {registry_name} steps")
    return matches[0]


def _validate_state_file(
    step: Mapping[str, Any],
    registry_name: str,
    expected_filename: str,
    issues: list[str],
) -> None:
    actual = step.get("state_file")
    if actual != expected_filename:
        issues.append(f"{registry_name} state_file expected {expected_filename!r}, got {actual!r}")


def _processor_shape(
    features: Any,
    key: str,
    processor_label: str,
    expected_dim: int | None,
    issues: list[str],
) -> None:
    feature = features.get(key) if isinstance(features, Mapping) else None
    shape = feature.get("shape") if isinstance(feature, Mapping) else None
    if expected_dim is None:
        if feature is None:
            issues.append(f"{processor_label} is missing feature {key!r}")
        return
    wanted = [expected_dim]
    if shape != wanted:
        issues.append(f"{processor_label} {key} shape must agree with config {wanted}, got {shape!r}")


def _validate_processors(
    preprocessor: Mapping[str, Any] | None,
    postprocessor: Mapping[str, Any] | None,
    config: _ConfigView | None,
    issues: list[str],
) -> None:
    if preprocessor is not None:
        normalizer = _step(preprocessor, "normalizer_processor", _PREPROCESSOR_FILE, issues)
        if normalizer is not None:
            _validate_state_file(
                normalizer,
                "normalizer_processor",
                _PREPROCESSOR_STATS_FILE,
                issues,
            )
            normalizer_config = normalizer.get("config")
            features = normalizer_config.get("features") if isinstance(normalizer_config, Mapping) else None
            if not isinstance(features, Mapping):
                issues.append(f"{_PREPROCESSOR_FILE} normalizer features must be an object")
            else:
                _processor_shape(
                    features,
                    "observation.state",
                    "normalizer",
                    config.state_dim if config else None,
                    issues,
                )
                _processor_shape(
                    features,
                    "action",
                    "normalizer",
                    config.action_dim if config else None,
                    issues,
                )
                if config is not None:
                    visual_keys = tuple(
                        key
                        for key, feature in features.items()
                        if isinstance(feature, Mapping) and str(feature.get("type", "")).upper() == "VISUAL"
                    )
                    if visual_keys != config.image_keys:
                        issues.append(
                            "normalizer visual feature keys must agree with config image_keys "
                            f"{config.image_keys!r}, got {visual_keys!r}"
                        )
    if postprocessor is not None:
        unnormalizer = _step(postprocessor, "unnormalizer_processor", _POSTPROCESSOR_FILE, issues)
        if unnormalizer is not None:
            _validate_state_file(
                unnormalizer,
                "unnormalizer_processor",
                _POSTPROCESSOR_STATS_FILE,
                issues,
            )
            unnormalizer_config = unnormalizer.get("config")
            features = (
                unnormalizer_config.get("features") if isinstance(unnormalizer_config, Mapping) else None
            )
            if not isinstance(features, Mapping):
                issues.append(f"{_POSTPROCESSOR_FILE} unnormalizer features must be an object")
            else:
                _processor_shape(
                    features,
                    "action",
                    "unnormalizer",
                    config.action_dim if config else None,
                    issues,
                )


def _inspect_stats(
    path: Path,
    required: Mapping[str, int | None],
    issues: list[str],
) -> None:
    if not path.is_file():
        return
    try:
        with safe_open(path, framework="numpy") as tensors:
            keys = set(tensors.keys())
            for key, expected_dim in required.items():
                if key not in keys:
                    issues.append(f"{path.name}: missing tensor {key!r}")
                    continue
                if expected_dim is None:
                    continue
                shape = list(tensors.get_slice(key).get_shape())
                if shape != [expected_dim]:
                    issues.append(f"{path.name}: tensor {key!r} expected shape [{expected_dim}], got {shape}")
    except (OSError, SafetensorError, ValueError) as exc:
        issues.append(f"could not inspect {path.name}: {exc}")


def _vlm_target_prefix(layer_index: int, target: str) -> str:
    block = "self_attn" if target in _ATTENTION_LORA_TARGETS else "mlp"
    return f"{_VLM_TEXT_PREFIX}.layers.{layer_index}.{block}.{target}"


def _validate_lora_target(
    tensors: Any,
    keys: set[str],
    *,
    path: Path,
    layer_index: int,
    target: str,
    rank: int,
    hidden_size: int | None,
    issues: list[str],
) -> None:
    prefix = _vlm_target_prefix(layer_index, target)
    base_key = f"{prefix}.weight"
    adapter_keys = {component: f"{prefix}.{component}" for component in ("lora_a", "lora_b", "lora_scale")}
    if base_key not in keys:
        issues.append(f"{path.name}: missing {target} base weight for VLM layer {layer_index}")
        base_shape = None
    else:
        base_shape = list(tensors.get_slice(base_key).get_shape())
        if len(base_shape) != 2:
            issues.append(
                f"{path.name}: {target} base weight for VLM layer {layer_index} "
                f"must be rank 2, got {base_shape}"
            )
            base_shape = None

    missing = [component for component, key in adapter_keys.items() if key not in keys]
    if missing:
        issues.append(f"{path.name}: missing {target} LoRA tensors for VLM layer {layer_index}: {missing}")

    if base_shape is not None:
        out_features, in_features = base_shape
        if hidden_size is not None:
            if target in {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}:
                if in_features != hidden_size:
                    issues.append(
                        f"{path.name}: {target} base weight input dimension expected "
                        f"{hidden_size}, got {in_features} for VLM layer {layer_index}"
                    )
            elif out_features != hidden_size:
                issues.append(
                    f"{path.name}: {target} base weight output dimension expected "
                    f"{hidden_size}, got {out_features} for VLM layer {layer_index}"
                )

        expected_shapes = {
            "lora_a": [rank, in_features],
            "lora_b": [out_features, rank],
        }
        for component, expected_shape in expected_shapes.items():
            key = adapter_keys[component]
            if key not in keys:
                continue
            actual_shape = list(tensors.get_slice(key).get_shape())
            if actual_shape != expected_shape:
                issues.append(
                    f"{path.name}: {target} {component} expected shape {expected_shape}, "
                    f"got {actual_shape} for VLM layer {layer_index}"
                )

    scale_key = adapter_keys["lora_scale"]
    if scale_key in keys:
        scale_shape = list(tensors.get_slice(scale_key).get_shape())
        if scale_shape:
            issues.append(
                f"{path.name}: {target} lora_scale expected scalar shape [], got "
                f"{scale_shape} for VLM layer {layer_index}"
            )


def _validate_model(
    path: Path,
    contract: CheckpointContract | None,
    config: _ConfigView | None,
    issues: list[str],
    extension: CheckpointValidationExtension = DEFAULT_VALIDATION_EXTENSION,
) -> None:
    if not path.is_file() or contract is None:
        return
    try:
        with safe_open(path, framework="numpy") as tensors:
            keys = set(tensors.keys())
            needs_vlm_structure = bool(
                extension.needs_model_structure(contract)
                or contract.vlm_lora_target_modules
            )
            hidden_size = SMOLVLA_TEXT_HIDDEN_SIZE
            if needs_vlm_structure:
                if _VLM_EMBED_TOKENS not in keys:
                    issues.append(f"{path.name}: missing tensor {_VLM_EMBED_TOKENS!r}")
                else:
                    embed_shape = list(tensors.get_slice(_VLM_EMBED_TOKENS).get_shape())
                    if len(embed_shape) != 2:
                        issues.append(
                            f"{path.name}: tensor {_VLM_EMBED_TOKENS!r} must be rank 2, got {embed_shape}"
                        )
                    else:
                        model_hidden_size = embed_shape[1]
                        if model_hidden_size != hidden_size:
                            issues.append(
                                f"{path.name}: runtime text_hidden_size expected {hidden_size}, got "
                                f"{model_hidden_size} from {_VLM_EMBED_TOKENS!r}"
                            )

            extension.validate_model(
                tensors,
                keys,
                path=path,
                contract=contract,
                hidden_size=hidden_size,
                issues=issues,
            )

            if contract.vlm_lora_target_modules:
                actual_layers = {
                    int(match.group(1)) for key in keys if (match := _VLM_LAYER_RE.match(key)) is not None
                }
                num_layers = config.num_vlm_layers if config is not None else None
                if num_layers is None or num_layers <= 0:
                    issues.append(
                        f"{path.name}: cannot validate VLM layers without positive config num_vlm_layers"
                    )
                    layers_to_validate = sorted(actual_layers)
                else:
                    expected_layers = set(range(num_layers))
                    missing_layers = sorted(expected_layers - actual_layers)
                    unexpected_layers = sorted(actual_layers - expected_layers)
                    if missing_layers:
                        issues.append(f"{path.name}: missing VLM text layers: {missing_layers}")
                    if unexpected_layers:
                        issues.append(f"{path.name}: unexpected VLM text layers: {unexpected_layers}")
                    layers_to_validate = range(num_layers)

                for layer_index in layers_to_validate:
                    for target in contract.vlm_lora_target_modules:
                        if target not in _VLM_LORA_TARGETS:
                            continue
                        _validate_lora_target(
                            tensors,
                            keys,
                            path=path,
                            layer_index=layer_index,
                            target=target,
                            rank=contract.lora_rank,
                            hidden_size=hidden_size,
                            issues=issues,
                        )
    except (OSError, SafetensorError, ValueError) as exc:
        issues.append(f"could not inspect {path.name}: {exc}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_contract_differs(
    filename: str,
    effective: CheckpointContract,
    base: CheckpointContract,
) -> bool:
    if filename == "config.json":
        return effective != base
    if filename in {_PREPROCESSOR_FILE, _PREPROCESSOR_STATS_FILE}:
        return (
            effective.state_dim,
            effective.action_dim,
            effective.image_keys,
        ) != (
            base.state_dim,
            base.action_dim,
            base.image_keys,
        )
    if filename in {_POSTPROCESSOR_FILE, _POSTPROCESSOR_STATS_FILE}:
        return effective.action_dim != base.action_dim
    return False


def _check_base_sidecars(
    checkpoint: Path,
    base: Path,
    effective: CheckpointContract | None,
    issues: list[str],
    extension: CheckpointValidationExtension = DEFAULT_VALIDATION_EXTENSION,
) -> None:
    if not base.is_dir():
        issues.append(f"base sidecar directory does not exist: {base}")
        return
    base_raw = _load_json(base / "config.json", issues, description="base config.json")
    if base_raw is None or effective is None:
        return
    base_parse_issues: list[str] = []
    base_config = _parse_config(base_raw, base_parse_issues, extension)
    if base_parse_issues:
        issues.extend(f"base config: {issue}" for issue in base_parse_issues)
        return
    base_contract = base_config.as_contract(extension)
    if base_contract is None or base_contract == effective:
        return
    for filename in _SIDECAR_FILENAMES:
        if not _sidecar_contract_differs(filename, effective, base_contract):
            continue
        candidate = checkpoint / filename
        base_file = base / filename
        if not candidate.is_file() or not base_file.is_file():
            continue
        try:
            identical = candidate.stat().st_size == base_file.stat().st_size and _sha256(
                candidate
            ) == _sha256(base_file)
        except OSError as exc:
            issues.append(f"could not fingerprint sidecar {filename!r}: {exc}")
            continue
        if identical:
            issues.append(
                f"sidecar {filename!r} is byte-identical to base despite differing checkpoint contracts"
            )


def validate_checkpoint(
    path: str | Path,
    *,
    expected: CheckpointContract | None = None,
    base_sidecars: str | Path | None = None,
    require_weight: bool = True,
    extension: CheckpointValidationExtension = DEFAULT_VALIDATION_EXTENSION,
) -> CheckpointValidationReport:
    """Validate a SmolVLA inference checkpoint without materializing model tensors."""

    checkpoint = Path(path).expanduser().resolve()
    issues: list[str] = []
    if not checkpoint.is_dir():
        issues.append(f"checkpoint directory does not exist: {checkpoint}")
        return CheckpointValidationReport(checkpoint, tuple(issues))

    required_files = list(_SIDECAR_FILENAMES)
    if require_weight:
        required_files.append("model.safetensors")
    for filename in required_files:
        if not (checkpoint / filename).is_file():
            issues.append(f"missing required checkpoint file: {filename}")

    config_raw = _load_json(checkpoint / "config.json", issues)
    config = _parse_config(config_raw, issues, extension) if config_raw is not None else None
    if config is not None:
        _validate_config_consistency(config_raw, config, issues, extension)
        if expected is not None:
            _compare_contract(config, expected, issues, extension)

    preprocessor = _load_json(checkpoint / _PREPROCESSOR_FILE, issues)
    postprocessor = _load_json(checkpoint / _POSTPROCESSOR_FILE, issues)
    _validate_processors(preprocessor, postprocessor, config, issues)

    target = expected or (config.as_contract(extension) if config is not None else None)
    state_dim = target.state_dim if target is not None else config.state_dim if config else None
    action_dim = target.action_dim if target is not None else config.action_dim if config else None
    _inspect_stats(
        checkpoint / _PREPROCESSOR_STATS_FILE,
        {
            "observation.state.mean": state_dim,
            "observation.state.std": state_dim,
            "action.mean": action_dim,
            "action.std": action_dim,
        },
        issues,
    )
    _inspect_stats(
        checkpoint / _POSTPROCESSOR_STATS_FILE,
        {"action.mean": action_dim, "action.std": action_dim},
        issues,
    )
    model_path = checkpoint / "model.safetensors"
    if model_path.is_file():
        _validate_model(model_path, target, config, issues, extension)

    if base_sidecars is not None:
        _check_base_sidecars(
            checkpoint,
            Path(base_sidecars).expanduser().resolve(),
            expected or (config.as_contract(extension) if config is not None else None),
            issues,
            extension,
        )
    return CheckpointValidationReport(checkpoint, tuple(issues))
