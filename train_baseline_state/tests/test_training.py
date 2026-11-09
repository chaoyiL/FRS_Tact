from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest
import torch

from train_baseline_state.action_cache import ActionCache, ActionCacheWriter, SampleRecord
from train_baseline_state.config import DecoderTrainConfig, load_config
from train_baseline_state.data import BaselineCacheDataset
from train_baseline_state.model import StateDecoderConfig, StateConditionedFlowDecoder, flow_matching_loss
from train_baseline_state.train import train_decoder
from train_baseline_state.checkpoint import save_best_checkpoint
from deploy_baseline_state.checkpoint import load_decoder

ROOT = Path(__file__).resolve().parents[2]


def caches(tmp_path, width=10):
    manifest = dict(
        dataset_identity=dict(root=str(tmp_path / "dataset"), repo_id="synthetic", revision=None),
        split=dict(seed=42),
        source_checkpoint="frozen/pi05",
        source_variant={"name": "pi05"},
        norm_stats={"asset_id": "synthetic"},
        sample_steps=10,
        noise_seed=0,
        source_model_action_width=width,
        decoder_action_width=width,
        action_space="normalized_pi05",
        origin="structured_noise",
        reverse_steps=50,
        reverse_solver="slerpflow",
    )
    writer = ActionCacheWriter.create(
        tmp_path / "cache", sample_count=4, horizon=50, action_dim=width, manifest=manifest
    )
    coarse = np.full((4, 50, width), 0.1, np.float32)
    target = np.full_like(coarse, 0.2)
    noise = np.full_like(coarse, 0.5)
    states = np.arange(4 * (7 if width == 10 else 20), dtype=np.float32).reshape(4, -1) / 100
    records = [SampleRecord(i, i // 2, i % 2, 0 if i < 2 else 1) for i in range(4)]
    writer.write_batch(
        0,
        coarse=coarse,
        expert=target,
        structured_noise=noise,
        states=states,
        valid=np.ones((4, 50), bool),
        records=records,
    )
    writer.finalize()
    return ActionCache.open(tmp_path / "cache")


@pytest.mark.parametrize("width", [10, 20])
def test_network_uses_one_state_token_and_structured_noise_origin(width):
    model = StateConditionedFlowDecoder(StateDecoderConfig(action_dim=width)).eval()
    state = torch.randn(1, 7 if width == 10 else 20)
    assert model.encode_condition(state).shape == (1, 1, 256)
    assert len(model.blocks) == 6
    assert not any(any(part in name for part in ("tactile", "gru")) for name in model.state_dict())
    noise = torch.full((1, 50, width), 0.5)
    coarse = torch.full_like(noise, 0.1)
    with torch.no_grad():
        model.action_out.weight.zero_()
        model.action_out.bias.zero_()
        output = model.decode(noise, state=state, vla_action=coarse)
    mask = torch.ones(width, dtype=torch.bool)
    mask[9::10] = False
    torch.testing.assert_close(output[..., mask], noise[..., mask], rtol=0, atol=0)
    torch.testing.assert_close(output[..., 9::10], coarse[..., 9::10], rtol=0, atol=0)
    with pytest.raises(TypeError):
        model(noise, torch.zeros(1), state=state, tactile_seq=torch.zeros(1, 10, 2, 512))


@pytest.mark.parametrize("width", [10, 20])
def test_cache_dataset_and_portable_checkpoint_have_no_sensory_dependency(tmp_path, width):
    action = caches(tmp_path, width)
    dataset = BaselineCacheDataset(action, "train")
    sample = dataset[0]
    assert set(sample) == {"coarse", "structured_noise", "target", "state", "valid"}
    assert not torch.equal(sample["structured_noise"], sample["coarse"])
    source = {
        "pi": {"model_action_width": width},
        "flow": {"origin": "structured_noise", "conditioning": "state_only"},
    }
    model = StateConditionedFlowDecoder(StateDecoderConfig(action_dim=width)).eval()
    checkpoint = save_best_checkpoint(
        tmp_path / "decoder",
        model,
        model.config,
        epoch=1,
        global_step=2,
        metrics={"validation_loss": 0.1},
        source_contract=source,
    )
    deployed = load_decoder(checkpoint, expected_source=source)
    noise = sample["structured_noise"][None]
    state = sample["state"][None]
    coarse = sample["coarse"][None]
    with torch.no_grad():
        torch.testing.assert_close(
            deployed.decode(noise, state=state, vla_action=coarse),
            model.decode(noise, state=state, vla_action=coarse),
            rtol=0,
            atol=0,
        )
    assert all(not value.requires_grad for value in deployed.parameters())
    raw = torch.load(checkpoint, weights_only=True)
    raw["mode"] = "action_origin_flow"
    torch.save(raw, checkpoint)
    with pytest.raises(ValueError, match="contract"):
        load_decoder(checkpoint, expected_source=source)


@pytest.mark.parametrize("width", [10, 20])
def test_train_resume_is_exact_and_does_not_mutate_frozen_inputs(tmp_path, width):
    action = caches(tmp_path, width)
    source_file = tmp_path / "source"
    source_file.write_text("frozen")
    source = SimpleNamespace(
        checkpoint=source_file,
        norm_stats_dir=tmp_path,
        norm_stats_asset_id="synthetic",
        model_action_dim=width,
        sample_steps=10,
        paligemma_variant="p",
        action_expert_variant="a",
        reverse_steps=50,
        reverse_solver="slerpflow",
    )
    settings = DecoderTrainConfig(
        output=tmp_path / "decoder",
        action_dim=width,
        state_dim=7 if width == 10 else 20,
        device="cpu",
        batch_size=2,
        epochs=2,
        learning_rate=1e-4,
        aux_decode_weight=0.0,
        seed=3,
    )
    config = SimpleNamespace(
        source=source,
        decoder=settings,
        action_cache=action,
        norm_stats={"q01": np.zeros(width), "q99": np.ones(width)},
    )
    before = (action.cache_dir / "structured_noise.npy").read_bytes()
    last = train_decoder(config, max_steps=1)
    assert torch.load(last, weights_only=True)["global_step"] == 1
    assert (settings.output / "best.pt").is_file()
    config.decoder = replace(settings, resume=True)
    resumed = torch.load(train_decoder(config, max_steps=2), weights_only=True)
    config.decoder = replace(settings, output=tmp_path / "uninterrupted")
    complete = torch.load(train_decoder(config, max_steps=2), weights_only=True)
    assert all(
        torch.equal(resumed["decoder_state"][key], value) for key, value in complete["decoder_state"].items()
    )
    assert source_file.read_text() == "frozen"
    assert (action.cache_dir / "structured_noise.npy").read_bytes() == before


def test_gt_flow_supervision_masks_invalid_actions_and_depends_on_reverse_origin():
    class Capture(torch.nn.Module):
        def forward(self, x, t, *, state, state_keep_mask):
            self.x = x.clone()
            return torch.zeros_like(x)

    noise = torch.full((1, 50, 10), 0.5)
    target = torch.ones_like(noise)
    coarse = torch.zeros_like(noise)
    valid = torch.zeros((1, 50), dtype=torch.bool)
    valid[:, :2] = True
    model = Capture()
    torch.manual_seed(3)
    loss = flow_matching_loss(
        model, noise, target, torch.zeros(1, 7), valid, vla_action=coarse, aux_decode_weight=0.0
    )
    # Every valid dimension has a +/- .5 target velocity, including frozen VLA grip endpoint.
    assert loss.item() == pytest.approx(0.25)
    target[:, 2:] = 9999
    changed = flow_matching_loss(
        model, noise, target, torch.zeros(1, 7), valid, vla_action=coarse, aux_decode_weight=0.0
    )
    assert changed.item() == pytest.approx(0.25)


def test_producer_saves_actual_reverse_result_and_reuses_cache(tmp_path):
    from train_baseline_state.prepare_action_cache import prepare_action_cache

    config = load_config(ROOT / "train_baseline_state/configs/train_baseline_state_task3.yaml")
    config = replace(
        config,
        dataset=replace(config.dataset, frame_stride=1),
        cache=replace(config.cache, action_root=tmp_path / "cache"),
    )

    class Dataset:
        def __getitem__(self, index):
            return {
                "observation.state": np.full(7, index, np.float32),
                "actions": np.zeros(10, np.float32) if index == 2 else np.ones(10, np.float32),
            }

    class Processor:
        def prepare_sample(self, sample):
            return {"state": sample["observation.state"]}, sample["actions"], "task"

    calls = []

    def reverse(model, obs, noise, **kwargs):
        calls.append(kwargs)
        return np.full_like(noise, 0.2), np.full_like(noise, 0.7)

    deps = dict(
        metadata=SimpleNamespace(
            total_episodes=1, episodes=[{"dataset_from_index": 0, "dataset_to_index": 3}]
        ),
        dataset=Dataset(),
        processor=Processor(),
        model=SimpleNamespace(pi05=True, action_dim=10, action_horizon=50),
        sample_and_reverse=reverse,
    )
    cache = ActionCache.open(prepare_action_cache(config, deps))
    assert len(calls) == 1
    assert calls[0] == dict(sample_steps=10, reverse_steps=50, solver="slerpflow")
    np.testing.assert_allclose(cache.structured_noise, 0.7)
    np.testing.assert_allclose(cache.coarse_actions, 0.2)
    np.testing.assert_array_equal(cache.states[:, 0], [0, 1])
    prepare_action_cache(config, deps)
    assert len(calls) == 1
    changed = replace(config, source=replace(config.source, reverse_steps=20))
    with pytest.raises(ValueError, match="contract"):
        prepare_action_cache(changed, deps)


def test_configs_and_check_imports_are_lightweight_and_do_not_request_touch():
    for path in (ROOT / "train_baseline_state/configs").glob("*.yaml"):
        config = load_config(path)
        assert config.source.reverse_steps == 50 and config.source.reverse_solver == "slerpflow"
        assert not hasattr(config, "tactile")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import train_baseline_state.config; import deploy_baseline_state.deployment; import sys; assert 'torch' not in sys.modules and 'jax' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_offline_physical_metrics_follow_source_standard_normalization():
    from train_baseline_state.evaluate import evaluate_decoder

    class Zero(torch.nn.Module):
        def decode(self, structured_noise, *, state, vla_action):
            return torch.zeros_like(structured_noise)

    values = torch.zeros(1, 50, 10)
    batch = {
        "coarse": values,
        "structured_noise": values,
        "state": torch.zeros(1, 7),
        "target": torch.ones_like(values),
        "valid": torch.ones(1, 50, dtype=torch.bool),
    }
    metrics = evaluate_decoder(Zero(), [batch], {"std": np.full(10, 2.0)}, use_quantile_norm=False)
    assert metrics["physical_mae"] == pytest.approx(2.0, abs=2e-6)
    assert metrics["physical_rmse"] == pytest.approx(2.0, abs=2e-6)


def test_direct_training_rejects_cache_with_a_different_reverse_contract(tmp_path):
    from train_baseline_state.train import _open_cache

    action = caches(tmp_path)
    config = SimpleNamespace(
        action_cache=action,
        source=SimpleNamespace(
            reverse_steps=20, reverse_solver="slerpflow", model_action_dim=10, sample_steps=10
        ),
    )
    with pytest.raises(ValueError, match="reverse_steps"):
        _open_cache(config)
