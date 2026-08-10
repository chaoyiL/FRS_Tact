#!/usr/bin/env python
"""Merge a SmolVLA PEFT adapter into its base checkpoint for JAX inference.

The JAX SmolVLA loader consumes a complete ``model.safetensors`` file.  LeRobot
PEFT checkpoints on the Hub instead contain ``adapter_model.safetensors`` plus
the fully-trained expert/action modules listed in ``modules_to_save``.  This
tool combines both forms without importing the PyTorch LeRobot policy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ASSET_FILES = (
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
ADAPTER_INFERENCE_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    *ASSET_FILES,
)
ADAPTER_PREFIX = "base_model.model."


def _resolve_repo_or_path(
    value: str | Path,
    *,
    revision: str | None,
    allow_download: bool,
    patterns: list[str],
) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    snapshot = snapshot_download(
        repo_id=str(value),
        revision=revision,
        local_files_only=not allow_download,
        allow_patterns=patterns,
    )
    return Path(snapshot)


def _strip_adapter_prefix(name: str) -> str:
    if not name.startswith(ADAPTER_PREFIX):
        raise ValueError(f"unexpected PEFT tensor name without {ADAPTER_PREFIX!r}: {name}")
    return name[len(ADAPTER_PREFIX) :]


def merge_peft_state_dicts(
    base: Mapping[str, torch.Tensor],
    adapter: Mapping[str, torch.Tensor],
    *,
    lora_alpha: float,
    lora_rank: int,
) -> dict[str, torch.Tensor]:
    """Return full parameters with modules-to-save and LoRA deltas applied."""

    if lora_rank <= 0:
        raise ValueError(f"lora_rank must be positive, got {lora_rank}")
    scale = float(lora_alpha) / float(lora_rank)
    output = {name: tensor.detach().cpu() for name, tensor in base.items()}
    lora_parts: dict[str, dict[str, torch.Tensor]] = {}

    for adapter_name, value in adapter.items():
        name = _strip_adapter_prefix(adapter_name)
        if name.endswith(".lora_A.weight"):
            target = name[: -len(".lora_A.weight")] + ".weight"
            lora_parts.setdefault(target, {})["A"] = value.detach().cpu()
        elif name.endswith(".lora_B.weight"):
            target = name[: -len(".lora_B.weight")] + ".weight"
            lora_parts.setdefault(target, {})["B"] = value.detach().cpu()
        else:
            if name not in output:
                raise KeyError(f"adapter modules_to_save tensor is absent from base checkpoint: {name}")
            if tuple(output[name].shape) != tuple(value.shape):
                raise ValueError(
                    f"shape mismatch for adapter tensor {name}: "
                    f"base={tuple(output[name].shape)} adapter={tuple(value.shape)}"
                )
            output[name] = value.detach().cpu().to(dtype=output[name].dtype)

    for target, parts in lora_parts.items():
        if set(parts) != {"A", "B"}:
            raise ValueError(f"incomplete LoRA pair for {target}: found {sorted(parts)}")
        if target not in output:
            raise KeyError(f"LoRA target is absent from base checkpoint: {target}")
        base_value = output[target]
        delta = torch.matmul(parts["B"].float(), parts["A"].float()) * scale
        if tuple(delta.shape) != tuple(base_value.shape):
            raise ValueError(
                f"LoRA delta shape mismatch for {target}: "
                f"base={tuple(base_value.shape)} delta={tuple(delta.shape)}"
            )
        output[target] = (base_value.float() + delta).to(dtype=base_value.dtype)

    return output


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def validate_supported_adapter_config(config: Mapping[str, Any]) -> None:
    """Fail loudly for PEFT variants whose merge math differs from alpha/r."""

    unsupported: list[str] = []
    if bool(config.get("use_rslora", False)):
        unsupported.append("use_rslora")
    if bool(config.get("use_dora", False)):
        unsupported.append("use_dora")
    for key in ("rank_pattern", "alpha_pattern"):
        if config.get(key):
            unsupported.append(key)
    if unsupported:
        raise ValueError(
            "unsupported PEFT adapter options for the JAX merger: "
            f"{unsupported}. Merge this adapter with PEFT/PyTorch first."
        )


def _validate_existing_merge(
    output: Path,
    *,
    adapter: str | Path,
    base: str | Path,
    adapter_revision: str | None,
    base_revision: str | None,
) -> None:
    manifest_path = output / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(
            f"merged checkpoint exists without a provenance manifest: {output}; "
            "rerun with --overwrite"
        )
    manifest = _load_json(manifest_path)
    expected = {
        "source_adapter": str(adapter),
        "source_base": str(base),
        "adapter_revision": adapter_revision,
        "base_revision": base_revision,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"existing merged checkpoint has different sources: {mismatches}; "
            "choose a new output or rerun with --overwrite"
        )


def _effective_state_dim(adapter_dir: Path, config: Mapping[str, Any]) -> int:
    stats_path = adapter_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    if stats_path.is_file():
        stats = load_file(str(stats_path), device="cpu")
        mean = stats.get("observation.state.mean")
        if mean is not None and mean.ndim == 1:
            return int(mean.shape[0])
    features = config.get("input_features") or {}
    return int(features.get("observation.state", {}).get("shape", [32])[0])


def _write_effective_config(adapter_dir: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_json(adapter_dir / "config.json")
    state_dim = _effective_state_dim(adapter_dir, config)
    input_features = config.setdefault("input_features", {})
    state_feature = input_features.setdefault("observation.state", {"type": "STATE"})
    state_feature["shape"] = [state_dim]
    state_feature["type"] = "STATE"
    config["use_peft"] = False
    config.pop("pretrained_path", None)
    config.pop("pretrained_revision", None)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return config


def _repair_preprocessor_config(output_dir: Path, *, state_dim: int) -> None:
    """Keep the copied processor metadata consistent with the effective policy shape."""

    path = output_dir / "policy_preprocessor.json"
    if not path.is_file():
        return
    processor = _load_json(path)
    steps = processor.get("steps") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_config = step.get("config") if isinstance(step.get("config"), dict) else step
        features = step_config.get("features") or step_config.get("feature_specs") or {}
        state = features.get("observation.state") if isinstance(features, dict) else None
        if isinstance(state, dict) and "shape" in state:
            state["shape"] = [state_dim]
    with path.open("w", encoding="utf-8") as file:
        json.dump(processor, file, indent=2, ensure_ascii=False)
        file.write("\n")


def merge_checkpoint(
    *,
    adapter: str | Path,
    base: str | Path,
    output: Path,
    adapter_revision: str | None = None,
    base_revision: str | None = None,
    allow_download: bool = True,
    overwrite: bool = False,
) -> Path:
    adapter_dir = _resolve_repo_or_path(
        adapter,
        revision=adapter_revision,
        allow_download=allow_download,
        patterns=list(ADAPTER_INFERENCE_FILES),
    )
    base_dir = _resolve_repo_or_path(
        base,
        revision=base_revision,
        allow_download=allow_download,
        patterns=["config.json", "model.safetensors"],
    )
    adapter_model = adapter_dir / "adapter_model.safetensors"
    adapter_config_path = adapter_dir / "adapter_config.json"
    base_model = base_dir / "model.safetensors"
    for path in (adapter_model, adapter_config_path, base_model, adapter_dir / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    adapter_config = _load_json(adapter_config_path)
    validate_supported_adapter_config(adapter_config)

    output = output.expanduser().resolve()
    output_model = output / "model.safetensors"
    if output_model.exists() and not overwrite:
        _validate_existing_merge(
            output,
            adapter=adapter,
            base=base,
            adapter_revision=adapter_revision,
            base_revision=base_revision,
        )
        print(f"merged checkpoint already exists, skip: {output}")
        return output
    output.mkdir(parents=True, exist_ok=True)

    rank = int(adapter_config.get("r", 0))
    alpha = float(adapter_config.get("lora_alpha", rank))
    print(f"loading base checkpoint: {base_model}", flush=True)
    base_state = load_file(str(base_model), device="cpu")
    print(f"loading PEFT adapter: {adapter_model}", flush=True)
    adapter_state = load_file(str(adapter_model), device="cpu")
    merged = merge_peft_state_dicts(
        base_state,
        adapter_state,
        lora_alpha=alpha,
        lora_rank=rank,
    )
    print(
        f"saving merged checkpoint: tensors={len(merged)} rank={rank} alpha={alpha:g} -> {output_model}",
        flush=True,
    )
    save_file(merged, str(output_model))

    for filename in ASSET_FILES:
        source = adapter_dir / filename
        if source.is_file() and filename != "config.json":
            shutil.copy2(source, output / filename)
    config = _write_effective_config(adapter_dir, output)
    _repair_preprocessor_config(
        output,
        state_dim=int(config["input_features"]["observation.state"]["shape"][0]),
    )
    manifest = {
        "format_version": 1,
        "backend": "jax",
        "source_base": str(base),
        "source_adapter": str(adapter),
        "adapter_revision": adapter_revision,
        "base_revision": base_revision,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "tensor_count": len(merged),
        "state_dim": config["input_features"]["observation.state"]["shape"][0],
        "action_dim": config["output_features"]["action"]["shape"][0],
        "chunk_size": config["chunk_size"],
    }
    with (output / "conversion_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"merged JAX checkpoint ready: {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", default="lerobot/smolvla_base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-revision")
    parser.add_argument("--base-revision")
    parser.add_argument("--allow-download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merge_checkpoint(
        adapter=args.adapter,
        base=args.base,
        output=args.output,
        adapter_revision=args.adapter_revision,
        base_revision=args.base_revision,
        allow_download=args.allow_download,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
