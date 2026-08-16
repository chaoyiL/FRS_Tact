from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import yaml

from deploy_smolvla import remote_client
from deploy_smolvla import direct_decoder as direct_decoder_module
from deploy_smolvla.direct_decoder import (
    DIRECT_TACTILE_KEYS,
    DirectDecoderRuntime,
    DirectTactileActionDecoder,
)

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / "checkpoints" / "ablation"


def _direct_backend_config() -> dict[str, object]:
    config = yaml.safe_load(
        (ROOT / "deploy_smolvla/configs/deploy_smolvla_jax.yaml").read_text()
    )
    config["backend"] = "direct_tactile_decoder"
    config["direct_decoder"] = {"bundle": str(ABLATION), "device": "cpu"}
    config["observation"]["data_type"] = "vitac"
    return config


def test_direct_decoder_config_and_launcher() -> None:
    config_path = ROOT / "deploy_smolvla/configs/deploy_direct_decoder.yaml"
    launcher_path = ROOT / "deploy_smolvla/scripts/start_direct_decoder.sh"
    config = remote_client.load_config(config_path)
    assert config["backend"] == "direct_tactile_decoder"
    assert config["observation"]["data_type"] == "vitac"
    assert config["control"]["action_horizon"] == 20
    assert config["control"]["steps_per_inference"] == 10
    launcher = launcher_path.read_text()
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "start_remote_client.sh" in launcher


def _direct_policy(*, use_tactile_encoder: bool, rtc_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            use_tactile_encoder=use_tactile_encoder,
            rtc_config=(
                SimpleNamespace(enabled=True, execution_horizon=20)
                if rtc_enabled
                else None
            ),
            chunk_size=20,
            action_dim=20,
            image_keys=("observation.images.camera0",),
            state_dim=14,
            empty_cameras=0,
        ),
        reset=lambda: None,
    )


def test_run_rejects_direct_backend_with_checkpoint_rtc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _direct_backend_config()
    monkeypatch.setattr(remote_client, "load_config", lambda path: config)
    monkeypatch.setattr(
        remote_client,
        "_load_validated_policy",
        lambda *args, **kwargs: _direct_policy(
            use_tactile_encoder=False, rtc_enabled=True
        ),
    )

    with pytest.raises(ValueError, match="does not support checkpoint RTC"):
        remote_client.run(tmp_path / "deploy.yaml")


def test_run_rejects_direct_backend_with_tactile_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _direct_backend_config()
    monkeypatch.setattr(remote_client, "load_config", lambda path: config)
    monkeypatch.setattr(
        remote_client,
        "_load_validated_policy",
        lambda *args, **kwargs: _direct_policy(
            use_tactile_encoder=True, rtc_enabled=False
        ),
    )

    with pytest.raises(ValueError, match="requires a visual-only JaxSmolVLAPolicy"):
        remote_client.run(tmp_path / "deploy.yaml")


def test_direct_backend_requires_vitac_horizon_and_bundle(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (ROOT / "deploy_smolvla/configs/deploy_smolvla_jax.yaml").read_text()
    )
    config["backend"] = "direct_tactile_decoder"
    config["direct_decoder"] = {"bundle": str(ABLATION), "device": "cpu"}
    config["observation"]["data_type"] = "vision"
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="requires observation.data_type='vitac'"):
        remote_client.load_config(path)


def test_predict_chunk_refines_normalized_actions_before_unnormalizing() -> None:
    coarse = jnp.ones((1, 2, 1), dtype=jnp.float32)
    refined = np.full((1, 2, 1), 4.0, dtype=np.float32)
    unnormalized: list[np.ndarray] = []
    refined_inputs: list[tuple[np.ndarray, object]] = []
    predict_kwargs: dict[str, object] = {}

    class Preprocessor:
        @staticmethod
        def unnormalize_actions(actions: np.ndarray) -> np.ndarray:
            unnormalized.append(np.asarray(actions))
            return np.asarray(actions) * 10.0

    class Policy:
        config = SimpleNamespace(chunk_size=2, action_dim=1)
        preprocessor = Preprocessor()

        @staticmethod
        def predict_action_chunk(observation, task, **kwargs):
            del observation, task
            predict_kwargs.update(kwargs)
            return coarse

    observation = {"tactile": np.zeros((1,), dtype=np.float32)}
    runtime = SimpleNamespace(
        fixed_noise_jax=jnp.zeros((1, 2, 1), dtype=jnp.float32),
        refine=lambda normalized, received_observation: (
            refined_inputs.append((np.asarray(normalized), received_observation)) or refined
        ),
    )

    action, normalized = remote_client._predict_chunk(
        Policy(),
        observation,
        "task",
        seed=7,
        jit=False,
        num_steps=3,
        previous_chunk=np.zeros((1, 1), dtype=np.float32),
        inference_delay=1,
        execution_horizon=2,
        direct_decoder=runtime,
    )

    assert predict_kwargs == {
        "seed": 7,
        "noise": runtime.fixed_noise_jax,
        "jit": False,
        "normalized": True,
        "num_steps": 3,
        "previous_chunk": None,
        "inference_delay": None,
        "execution_horizon": None,
    }
    assert len(refined_inputs) == 1
    np.testing.assert_array_equal(refined_inputs[0][0], np.asarray(coarse))
    assert refined_inputs[0][1] is observation
    assert len(unnormalized) == 1
    np.testing.assert_array_equal(unnormalized[0], refined)
    np.testing.assert_array_equal(action, np.full((2, 1), 40.0, dtype=np.float32))
    np.testing.assert_array_equal(normalized, np.full((2, 1), 4.0, dtype=np.float32))


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


def test_runtime_refine_records_copied_snapshots_and_reset_clears_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(DirectDecoderRuntime)
    runtime.device = torch.device("cpu")
    runtime.tactile_keys = DIRECT_TACTILE_KEYS
    runtime.encoder = lambda images: torch.ones((4, 512), dtype=torch.float32)
    runtime.decoder = lambda coarse, tactile: coarse + 2.0
    runtime.last_vla_normalized = None
    runtime.last_direct_normalized = None
    monkeypatch.setattr(
        direct_decoder_module,
        "_preprocess_image",
        lambda image: np.zeros((3, 224, 224), dtype=np.float32),
    )

    coarse = np.ones((1, 20, 20), dtype=np.float32)
    observation = {
        key: np.zeros((1, 1, 3), dtype=np.uint8) for key in DIRECT_TACTILE_KEYS
    }

    direct = runtime.refine(coarse, observation)

    np.testing.assert_array_equal(runtime.last_vla_normalized, coarse)
    np.testing.assert_array_equal(runtime.last_direct_normalized, direct)
    coarse[0, 0, 0] = -1.0
    direct[0, 0, 0] = -2.0
    assert runtime.last_vla_normalized[0, 0, 0] == 1.0
    assert runtime.last_direct_normalized[0, 0, 0] == 3.0

    runtime.reset()

    assert runtime.last_vla_normalized is None
    assert runtime.last_direct_normalized is None


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
