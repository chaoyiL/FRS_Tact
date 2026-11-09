"""Numerical parity with TacFRS after removing all sensory condition tokens."""

import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
from train_pi05_frs.utils.model import DecoderConfig, TactileConditionedFlowDecoder
from train_pi05_frs.utils.integration import fireflow_integrate_velocity
from train_baseline_state.model import StateDecoderConfig, StateConditionedFlowDecoder


def _copy_linear(destination, source):
    destination.weight.copy_(
        torch.tensor(np.asarray(source.kernel.value).reshape(source.kernel.value.shape[0], -1).T.copy())
    )
    if destination.bias is not None:
        destination.bias.copy_(torch.tensor(np.asarray(source.bias.value).reshape(-1).copy()))


def _copy_norm(destination, source):
    destination.weight.copy_(torch.tensor(np.asarray(source.scale.value).copy()))
    destination.bias.copy_(torch.tensor(np.asarray(source.bias.value).copy()))


def _copy_attention(destination, source):
    weights, biases = [], []
    for name in ("query", "key", "value"):
        layer = getattr(source, name)
        weights.append(torch.tensor(np.asarray(layer.kernel.value).reshape(256, 256).T.copy()))
        biases.append(torch.tensor(np.asarray(layer.bias.value).reshape(-1).copy()))
    destination.in_proj_weight.copy_(torch.cat(weights))
    destination.in_proj_bias.copy_(torch.cat(biases))
    destination.out_proj.weight.copy_(
        torch.tensor(np.asarray(source.out.kernel.value).reshape(256, 256).T.copy())
    )
    destination.out_proj.bias.copy_(torch.tensor(np.asarray(source.out.bias.value).copy()))


@pytest.mark.parametrize("width,sensors,state_dim", [(10, 2, 7), (20, 4, 20)])
def test_state_only_backbone_matches_tacfrs_state_token_only(width, sensors, state_dim):
    reference = TactileConditionedFlowDecoder(
        DecoderConfig(
            action_dim=width,
            action_horizon=50,
            tactile_window=10,
            model_dim=256,
            depth=6,
            num_heads=4,
            mlp_ratio=4,
            num_tactile_tokens=sensors,
            state_conditioning=True,
            state_dim=state_dim,
        ),
        rngs=nnx.Rngs(3),
    )
    model = StateConditionedFlowDecoder(StateDecoderConfig(action_dim=width)).eval()
    with torch.no_grad():
        for name in ("action_in", "state_fc1", "state_fc2", "action_out"):
            _copy_linear(getattr(model, name), getattr(reference, name))
        for name in ("fc1", "fc2"):
            _copy_linear(getattr(model.time_mlp, name), getattr(reference.time_mlp, name))
        for name in ("state_norm", "out_norm"):
            _copy_norm(getattr(model, name), getattr(reference, name))
        for block, ref in zip(model.blocks, reference.blocks, strict=True):
            for name in ("norm_self", "norm_cross_q", "norm_cross_kv", "norm_mlp"):
                _copy_norm(getattr(block, name), getattr(ref, name))
            for name in ("fc1", "fc2"):
                _copy_linear(getattr(block, name), getattr(ref, name))
            for name in ("self_attention", "cross_attention"):
                _copy_attention(getattr(block, name), getattr(ref, name))
    rng = np.random.default_rng(8)
    noise = rng.normal(size=(1, 50, width)).astype(np.float32)
    state = rng.normal(size=(1, state_dim)).astype(np.float32)
    condition = reference.encode_condition(np.zeros((1, 10, sensors, 512), np.float32), state)[:, :1]
    with torch.no_grad():
        actual = model(torch.from_numpy(noise), torch.tensor([0.37]), state=torch.from_numpy(state)).numpy()
        decoded = model.decode(
            torch.from_numpy(noise), state=torch.from_numpy(state), preserve_grippers=False
        ).numpy()
    expected = np.asarray(reference.velocity_from_condition(noise, np.array([0.37], np.float32), condition))
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-5)
    result = fireflow_integrate_velocity(
        lambda x, t: reference.velocity_from_condition(x, t, condition), noise, num_steps=10
    )
    np.testing.assert_allclose(decoded, np.asarray(result), rtol=4e-4, atol=5e-5)


def test_native_vla_forward_and_reverse_share_prefix_and_refresh_weights():
    root = Path(__file__).resolve().parents[2]
    script = """
import numpy as np
import jax.numpy as jnp
from flax import nnx
from train_baseline_state import reverse
from utils.integration import slerpflow_integrate_velocity as reference_solver
calls=[]
class Toy(nnx.Module):
    def __init__(self):
        self.action_dim=10;self.action_horizon=50;self.max_token_len=1;self.pi05=True;self.image_keys=()
        self.weight=nnx.Param(jnp.asarray(.2))
model=Toy()
def prefix(model,observation):
    calls.append('prefix')
    return observation
reverse._frs.build_prefix_cache=prefix
reverse._frs.denoise_step=lambda model,cache,x,t:x*model.weight.value
noise=jnp.ones((1,50,10))
coarse,base=reverse.sample_and_reverse(model,jnp.asarray(1.),noise,sample_steps=10,reverse_steps=50,solver='slerpflow')
np.testing.assert_allclose(coarse,(1-.2/10)**10,atol=1e-5)
expected=reference_solver(lambda x,t:x*.2,coarse,num_steps=50)
np.testing.assert_allclose(base,expected,atol=1e-6)
assert not np.allclose(base,noise)
assert not np.allclose(coarse,base)
model.weight.value=jnp.asarray(.4)
coarse2,base2=reverse.sample_and_reverse(model,jnp.asarray(1.),noise,sample_steps=10,reverse_steps=50,solver='slerpflow')
np.testing.assert_allclose(coarse2,(1-.4/10)**10,atol=1e-5)
expected2=reference_solver(lambda x,t:x*.4,coarse2,num_steps=50)
np.testing.assert_allclose(base2,expected2,atol=1e-6)
assert calls==['prefix']
"""
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(root / "train_baseline_state/src"), str(root))),
        JAX_PLATFORMS="cpu",
        PYTHONSAFEPATH="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=environment, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
