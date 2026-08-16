from pathlib import Path

import numpy as np
import pytest
import torch

from deploy_smolvla.direct_decoder import (
    DIRECT_TACTILE_KEYS,
    DirectDecoderRuntime,
    DirectTactileActionDecoder,
)

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / "checkpoints" / "ablation"


def test_released_decoder_state_loads_strictly() -> None:
    checkpoint = torch.load(
        ABLATION / "decoder" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = DirectTactileActionDecoder.from_config(checkpoint["decoder_config"])
    model.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    assert sum(parameter.numel() for parameter in model.parameters()) == 471_828


def test_fixed_noise_matches_training_contract() -> None:
    noise = np.load(ABLATION / "fixed_noise.npy", allow_pickle=False)
    assert noise.dtype == np.float32
    assert noise.shape == (1, 20, 32)
    assert np.isfinite(noise).all()
    np.testing.assert_array_equal(noise[:, :, 20:], 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="deployment uses cuda:0")
def test_runtime_refine_returns_finite_normalized_chunk() -> None:
    runtime = DirectDecoderRuntime.from_bundle(ABLATION, device="cuda:0")
    observation = {
        key: np.zeros((240, 320, 3), dtype=np.uint8)
        for key in DIRECT_TACTILE_KEYS
    }
    result = runtime.refine(np.zeros((1, 20, 20), dtype=np.float32), observation)
    assert result.shape == (1, 20, 20)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
