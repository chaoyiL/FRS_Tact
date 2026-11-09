"""TacFRS Flow in PyTorch, integrating directly from frozen VLA actions."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train_baseline_act_inter.config import RIGHT_TACTILE_KEYS, TACTILE_KEYS


@dataclasses.dataclass(frozen=True)
class DirectDecoderConfig:
    action_horizon: int = 50
    action_dim: int = 20
    tactile_dim: int = 512
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.0
    tactile_keys: tuple[str, ...] = TACTILE_KEYS
    state_dim: int | None = None
    tactile_window: int = 10
    history_stride: int = 3
    gru_hidden_dim: int = 256
    decode_steps: int = 10
    decode_solver: str = "fireflow"

    def __post_init__(self):
        if self.state_dim is None:
            object.__setattr__(self, "state_dim", 7 if self.action_dim == 10 else 20)

    def validate(self):
        fixed = dict(
            action_horizon=50,
            tactile_dim=512,
            d_model=256,
            nhead=4,
            num_layers=6,
            dim_feedforward=1024,
            tactile_window=10,
            history_stride=3,
            gru_hidden_dim=256,
            decode_steps=10,
        )
        for name, required in fixed.items():
            if type(getattr(self, name)) is not int or getattr(self, name) != required:
                raise ValueError(f"{name} must be {required} for the TacFRS Flow contract.")
        if type(self.action_dim) is not int or self.action_dim not in (10, 20):
            raise ValueError("action_dim must be 10 or 20.")
        expected = 7 if self.action_dim == 10 else 20
        if type(self.state_dim) is not int or self.state_dim != expected:
            raise ValueError(f"state_dim must be {expected} for {self.action_dim}D actions.")
        if type(self.dropout) is not float or self.dropout != 0.0:
            raise ValueError("dropout must be 0.0 for TacFRS Flow.")
        if self.decode_solver != "fireflow":
            raise ValueError("decode_solver must be fireflow.")
        if self.tactile_keys not in (TACTILE_KEYS, RIGHT_TACTILE_KEYS):
            raise ValueError("tactile_keys must use canonical order.")

    @classmethod
    def from_train_config(cls, config: Any):
        result = cls(**{f.name: getattr(config, f.name, f.default) for f in dataclasses.fields(cls)})
        result.validate()
        return result

    def to_primitive(self):
        self.validate()
        result = dataclasses.asdict(self)
        result["tactile_keys"] = list(self.tactile_keys)
        return result

    @classmethod
    def from_primitive(cls, raw: object):
        if not isinstance(raw, dict) or set(raw) != {f.name for f in dataclasses.fields(cls)}:
            raise ValueError("decoder_config must describe action-origin TacFRS Flow; retrain old decoders.")
        raw = dict(raw)
        raw["tactile_keys"] = tuple(raw["tactile_keys"])
        result = cls(**raw)
        result.validate()
        return result


def sinusoidal_embedding(x: Tensor, dim: int, max_period: float = 10_000.0):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / max(half - 1, 1)
    )
    arguments = x.float().unsqueeze(-1) * frequencies
    return F.pad(torch.cat((arguments.sin(), arguments.cos()), dim=-1), (0, dim % 2))


class TimeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc1, self.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def forward(self, t):
        return self.fc2(F.silu(self.fc1(sinusoidal_embedding(t, self.dim))))


class SharedTactileGRU(nn.Module):
    """Same single-layer GRU equations and bias placement as flax.nnx.GRUCell."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dense_i = nn.Linear(input_dim, 3 * hidden_dim)
        self.dense_h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

    def forward(self, xs):
        carry = xs.new_zeros((xs.shape[0], self.hidden_dim))
        for x in xs.unbind(dim=1):
            ir, iz, candidate_in = self.dense_i(x).chunk(3, dim=-1)
            hr, hz, candidate_hidden = self.dense_h(carry).chunk(3, dim=-1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            candidate = torch.tanh(candidate_in + reset * candidate_hidden)
            carry = (1.0 - update) * candidate + update * carry
        return carry


class ConditionedTransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm_cross_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_cross_kv = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6)
        self.fc1, self.fc2 = nn.Linear(dim, hidden_dim), nn.Linear(hidden_dim, dim)

    def forward(self, x, condition):
        normalized = self.norm_self(x)
        x = x + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        q, kv = self.norm_cross_q(x), self.norm_cross_kv(condition)
        x = x + self.cross_attention(q, kv, kv, need_weights=False)[0]
        return x + self.fc2(F.gelu(self.fc1(self.norm_mlp(x)), approximate="tanh"))


class DirectTactileActionDecoder(nn.Module):
    """v_theta(x_t, t, tactile_history, state), matching TacFRS's Flow interface."""

    def __init__(self, config: DirectDecoderConfig):
        super().__init__()
        config.validate()
        self.config = config
        dim = config.d_model
        self.action_in = nn.Linear(config.action_dim, dim)
        self.time_mlp = TimeMLP(dim)
        self.tactile_gru = SharedTactileGRU(config.tactile_dim, config.gru_hidden_dim)
        self.tactile_proj = nn.Linear(config.gru_hidden_dim, dim)
        self.state_norm = nn.LayerNorm(config.state_dim, eps=1e-6)
        self.state_fc1, self.state_fc2 = nn.Linear(config.state_dim, dim), nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(
            [
                ConditionedTransformerBlock(dim, config.nhead, config.dim_feedforward)
                for _ in range(config.num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim, eps=1e-6)
        self.action_out = nn.Linear(dim, config.action_dim)
        # Match Flax's LeCun-normal/zero-bias defaults and orthogonal recurrent kernel.
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    std = 1.0 / math.sqrt(module.in_features) / 0.8796256610342398
                    nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.MultiheadAttention):
                    std = 1.0 / math.sqrt(dim) / 0.8796256610342398
                    nn.init.trunc_normal_(module.in_proj_weight, std=std, a=-2 * std, b=2 * std)
                    nn.init.zeros_(module.in_proj_bias)
            nn.init.orthogonal_(self.tactile_gru.dense_h.weight)

    def encode_condition(self, tactile_seq, state, state_keep_mask=None):
        cfg = self.config
        if tactile_seq.ndim != 4 or tuple(tactile_seq.shape[1:]) != (
            cfg.tactile_window,
            len(cfg.tactile_keys),
            cfg.tactile_dim,
        ):
            raise ValueError(
                f"tactile must have shape [B,{cfg.tactile_window},{len(cfg.tactile_keys)},{cfg.tactile_dim}]."
            )
        batch, length, sensors, features = tactile_seq.shape
        if state.shape != (batch, cfg.state_dim):
            raise ValueError(f"state must have shape [B,{cfg.state_dim}] and matching batch size.")
        dtype = self.action_in.weight.dtype
        sequences = tactile_seq.to(dtype=dtype).transpose(1, 2).reshape(batch * sensors, length, features)
        tactile = self.tactile_proj(self.tactile_gru(sequences).reshape(batch, sensors, cfg.gru_hidden_dim))
        state_token = self.state_fc2(F.silu(self.state_fc1(self.state_norm(state.to(dtype=dtype)))))
        if state_keep_mask is not None:
            if state_keep_mask.shape != (batch,):
                raise ValueError("state_keep_mask must have shape [B].")
            state_token = state_token * state_keep_mask[:, None]
        return torch.cat((state_token[:, None, :], tactile), dim=1)

    def velocity_from_condition(self, x_t, t, condition):
        cfg = self.config
        if x_t.ndim != 3 or tuple(x_t.shape[1:]) != (cfg.action_horizon, cfg.action_dim):
            raise ValueError(f"action must have shape [B,{cfg.action_horizon},{cfg.action_dim}].")
        if t.shape != (x_t.shape[0],):
            raise ValueError("time must have shape [B].")
        x = self.action_in(x_t.to(dtype=self.action_in.weight.dtype))
        x = x + sinusoidal_embedding(torch.arange(x.shape[1], device=x.device), cfg.d_model)[None, ...]
        x = x + self.time_mlp(t)[:, None, :]
        for block in self.blocks:
            x = block(x, condition)
        return self.action_out(self.out_norm(x))

    def forward(self, x_t, t, tactile_seq, *, state, state_keep_mask=None):
        return self.velocity_from_condition(
            x_t, t, self.encode_condition(tactile_seq, state, state_keep_mask)
        )

    def decode(self, vla_action, tactile_seq, *, state, state_keep_mask=None, preserve_grippers=True):
        condition = self.encode_condition(tactile_seq, state, state_keep_mask)
        result = fireflow_integrate_velocity(
            lambda x, t: self.velocity_from_condition(x, t, condition),
            vla_action,
            num_steps=self.config.decode_steps,
        )
        if preserve_grippers:
            result = result.clone()
            result[..., 9::10] = vla_action[..., 9::10]
        return result


def fireflow_integrate_velocity(velocity_fn, x, *, num_steps):
    """TacFRS modified-midpoint FireFlow; N steps require N+1 evaluations."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    x = x.float()
    times = torch.linspace(0.0, 1.0, num_steps + 1, device=x.device)
    dt = times[1] - times[0]
    previous = velocity_fn(
        x + 0.5 * dt * velocity_fn(x, times[0].expand(x.shape[0])), (times[0] + 0.5 * dt).expand(x.shape[0])
    )
    x = x + dt * previous
    for step in range(1, num_steps):
        dt = times[step + 1] - times[step]
        previous = velocity_fn(x + 0.5 * dt * previous, (times[step] + 0.5 * dt).expand(x.shape[0]))
        x = x + dt * previous
    return x


def masked_smooth_l1(predicted, target, valid_mask):
    if predicted.shape != target.shape or valid_mask.dtype != torch.bool:
        raise ValueError("predicted/target shapes and boolean valid_mask must agree.")
    mask = (
        valid_mask.unsqueeze(-1).expand_as(predicted)
        if valid_mask.shape == predicted.shape[:-1]
        else valid_mask
    )
    if mask.shape != predicted.shape or not mask.any():
        raise ValueError("valid_mask must match action tokens/elements and contain a valid element.")
    return F.smooth_l1_loss(predicted, target, reduction="none")[mask].mean()


def flow_matching_loss(
    model,
    coarse,
    target,
    tactile,
    state,
    valid,
    *,
    baseline_tactile,
    state_dropout_rate=0.1,
    gate_tau=None,
    gate_temperature=None,
    aux_decode_weight=4.0,
    low_gate_safety_weight=0.5,
    rank_weight=4.0,
    repair_weight=2.0,
):
    """TacFRS composite/bimanual objective, replacing only the Flow origin.

    Per-episode initial tactile features determine the same low/mid/high Gate;
    state-token dropout is shared by FM and endpoint supervision in this step.
    Valid masks additionally exclude terminal/padded dataset frames.
    """
    single = coarse.shape[-1] == 10
    tau = (0.4 if single else 0.65) if gate_tau is None else gate_tau
    temperature = (0.1 if single else 0.15) if gate_temperature is None else gate_temperature
    current = F.normalize(tactile[:, -1], dim=-1, eps=1e-8)
    baseline = F.normalize(baseline_tactile, dim=-1, eps=1e-8)
    change = 1.0 - (current * baseline).sum(-1)
    if single:
        change = change.mean(-1, keepdim=True)
    elif change.shape[-1] == 4:
        change = change.reshape(-1, 2, 2).mean(-1)
    else:
        change = change.mean(-1, keepdim=True).expand(-1, 2)
    gates = torch.sigmoid((change - tau) / temperature).detach()
    effective = ((gates - 0.3) / 0.4).clamp(0.0, 1.0)
    weights = effective.repeat_interleave(10, dim=-1)[:, None, :]
    endpoint = weights * target + (1.0 - weights) * coarse
    if single:
        endpoint = endpoint.clone()
        endpoint[..., 9] = coarse[..., 9]
    t = torch.rand(coarse.shape[0], device=coarse.device)
    keep = (torch.rand(coarse.shape[0], device=coarse.device) >= state_dropout_rate).to(coarse.dtype)
    x_t = (1.0 - t[:, None, None]) * coarse + t[:, None, None] * endpoint
    predicted = model(x_t, t, tactile, state=state, state_keep_mask=keep)
    if not valid.any():
        raise ValueError("flow matching requires valid action tokens.")
    loss = (predicted - (endpoint - coarse)).square()[valid].mean()
    if not any((aux_decode_weight, low_gate_safety_weight, rank_weight, repair_weight)):
        return loss
    decoded = model.decode(coarse, tactile, state=state, state_keep_mask=keep, preserve_grippers=False)
    slices = (slice(0, 9),) if single else (slice(0, 10), slice(10, 20))

    def wrist_mse(left, right):
        mask = valid.to(left.dtype)
        return torch.stack(
            [
                ((left[..., sl] - right[..., sl]).square().mean(-1) * mask).sum(-1)
                / mask.sum(-1).clamp_min(1)
                for sl in slices
            ],
            dim=-1,
        )

    mse_gt, mse_vla = wrist_mse(decoded, target), wrist_mse(decoded, coarse)
    loss = loss + aux_decode_weight * (effective * mse_gt + (1.0 - effective) * mse_vla).mean()

    def active_group_mean(penalty, strength):
        total = strength.sum(0)
        means = (penalty * strength).sum(0) / total.clamp_min(torch.finfo(penalty.dtype).tiny)
        active = total > 0
        return (means * active).sum() / active.sum().clamp_min(1)

    low = (1.0 - gates) * (gates <= 0.3)
    high = gates * (gates >= 0.7)
    loss = loss + low_gate_safety_weight * active_group_mean(
        F.relu(torch.minimum(mse_gt, mse_vla) - 0.03), low
    )
    loss = loss + rank_weight * active_group_mean(F.relu(mse_gt - mse_vla), high)
    loss = loss + repair_weight * active_group_mean(F.relu(mse_gt - wrist_mse(coarse, target)), high)
    return loss
