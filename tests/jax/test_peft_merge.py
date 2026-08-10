from __future__ import annotations

import torch
import pytest

from tools.merge_smolvla_peft_to_jax import merge_peft_state_dicts
from tools.merge_smolvla_peft_to_jax import merge_checkpoint
from tools.merge_smolvla_peft_to_jax import validate_supported_adapter_config


def test_merge_peft_replaces_saved_modules_and_applies_lora() -> None:
    base = {
        "model.saved.weight": torch.zeros((2, 2), dtype=torch.float32),
        "model.block.q_proj.weight": torch.ones((3, 2), dtype=torch.float32),
    }
    adapter = {
        "base_model.model.model.saved.weight": torch.full((2, 2), 7.0),
        "base_model.model.model.block.q_proj.lora_A.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "base_model.model.model.block.q_proj.lora_B.weight": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
    }

    merged = merge_peft_state_dicts(base, adapter, lora_alpha=4.0, lora_rank=2)
    torch.testing.assert_close(merged["model.saved.weight"], torch.full((2, 2), 7.0))
    expected_delta = (
        adapter["base_model.model.model.block.q_proj.lora_B.weight"]
        @ adapter["base_model.model.model.block.q_proj.lora_A.weight"]
    )
    torch.testing.assert_close(
        merged["model.block.q_proj.weight"],
        base["model.block.q_proj.weight"] + 2.0 * expected_delta,
    )


@pytest.mark.parametrize(
    "config",
    [
        {"use_rslora": True},
        {"use_dora": True},
        {"rank_pattern": {"q_proj": 4}},
        {"alpha_pattern": {"q_proj": 8}},
    ],
)
def test_merge_rejects_unsupported_peft_math(config) -> None:
    with pytest.raises(ValueError, match="unsupported PEFT"):
        validate_supported_adapter_config(config)


def test_merge_downloads_adapter_from_exact_inference_allowlist(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def capture_resolver(value, *, revision, allow_download, patterns):
        calls.append(
            {
                "value": value,
                "revision": revision,
                "allow_download": allow_download,
                "patterns": patterns,
            }
        )
        raise RuntimeError("stop after adapter resolution")

    monkeypatch.setattr(
        "tools.merge_smolvla_peft_to_jax._resolve_repo_or_path", capture_resolver
    )

    with pytest.raises(RuntimeError, match="stop after adapter resolution"):
        merge_checkpoint(
            adapter="owner/adapter",
            adapter_revision="adapter-sha",
            base="owner/base",
            base_revision="base-sha",
            output=tmp_path / "merged",
        )

    assert calls == [
        {
            "value": "owner/adapter",
            "revision": "adapter-sha",
            "allow_download": True,
            "patterns": [
                "adapter_model.safetensors",
                "adapter_config.json",
                "config.json",
                "train_config.json",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "policy_preprocessor_step_5_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            ],
        }
    ]
