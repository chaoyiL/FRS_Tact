from __future__ import annotations

import torch

from tools.merge_smolvla_peft_to_jax import merge_peft_state_dicts


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
