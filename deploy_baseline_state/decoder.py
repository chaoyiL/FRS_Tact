"""TacFRS Flow backbone conditioned only on state, starting from structured noise."""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclasses.dataclass(frozen=True)
class StateDecoderConfig:
    action_horizon: int = 50
    action_dim: int = 20
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.0
    state_dim: int | None = None
    decode_steps: int = 10
    decode_solver: str = "fireflow"

    def __post_init__(self):
        if self.state_dim is None:
            object.__setattr__(self, "state_dim", 7 if self.action_dim == 10 else 20)

    def validate(self):
        for name, required in dict(
            action_horizon=50, d_model=256, nhead=4, num_layers=6, dim_feedforward=1024, decode_steps=10
        ).items():
            if type(getattr(self, name)) is not int or getattr(self, name) != required:
                raise ValueError(f"{name} must be {required} for the state-only TacFRS backbone")
        if type(self.action_dim) is not int or self.action_dim not in (10, 20):
            raise ValueError("action_dim must be 10 or 20")
        if self.state_dim != (7 if self.action_dim == 10 else 20) or type(self.state_dim) is not int:
            raise ValueError("state_dim does not match the action profile")
        if type(self.dropout) is not float or self.dropout != 0.0 or self.decode_solver != "fireflow":
            raise ValueError("state Flow requires dropout 0.0 and fireflow decoding")

    @classmethod
    def from_train_config(cls, config):
        result = cls(
            **{field.name: getattr(config, field.name, field.default) for field in dataclasses.fields(cls)}
        )
        result.validate()
        return result

    def to_primitive(self):
        self.validate()
        return dataclasses.asdict(self)

    @classmethod
    def from_primitive(cls, raw):
        if not isinstance(raw, dict) or set(raw) != {field.name for field in dataclasses.fields(cls)}:
            raise ValueError("decoder_config must describe a state-only structured-noise Flow model")
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


class StateConditionedFlowDecoder(nn.Module):
    """v_theta(x_t,t,state), with one state token and no sensory-token branch."""

    def __init__(self, config: StateDecoderConfig):
        super().__init__()
        config.validate()
        self.config = config
        dim = config.d_model
        self.action_in = nn.Linear(config.action_dim, dim)
        self.time_mlp = TimeMLP(dim)
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
        # Match Flax's LeCun-normal and zero-bias defaults.
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

    def encode_condition(self, state, state_keep_mask=None):
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(f"state must have shape [B,{self.config.state_dim}]")
        token = self.state_fc2(
            F.silu(self.state_fc1(self.state_norm(state.to(dtype=self.action_in.weight.dtype))))
        )
        if state_keep_mask is not None:
            if state_keep_mask.shape != (state.shape[0],):
                raise ValueError("state_keep_mask must have shape [B]")
            token = token * state_keep_mask[:, None]
        return token[:, None, :]

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

    def forward(self, x_t, t, *, state, state_keep_mask=None):
        return self.velocity_from_condition(x_t, t, self.encode_condition(state, state_keep_mask))

    def decode(
        self, structured_noise, *, state, vla_action=None, state_keep_mask=None, preserve_grippers=True
    ):
        condition = self.encode_condition(state, state_keep_mask)
        result = fireflow_integrate_velocity(
            lambda x, t: self.velocity_from_condition(x, t, condition),
            structured_noise,
            num_steps=self.config.decode_steps,
        )
        if preserve_grippers:
            if vla_action is None or vla_action.shape != structured_noise.shape:
                raise ValueError("vla_action with matching shape is required to preserve VLA grippers")
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
