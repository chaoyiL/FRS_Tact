"""Pure tactile interventions used by FRS modality evaluations."""

from dataclasses import dataclass

import numpy as np


def _l2_normalize(vectors: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def tactile_change_from_tokens(
    current_tokens: np.ndarray,
    baseline_tokens: np.ndarray,
) -> np.ndarray:
    """Return per-sample mean ``1 - cosine`` tactile change."""

    if current_tokens.ndim != 3 or baseline_tokens.ndim != 3:
        raise ValueError(
            f"Expected tokens [B, 4, D], got current={current_tokens.shape}, "
            f"baseline={baseline_tokens.shape}."
        )
    if current_tokens.shape != baseline_tokens.shape:
        raise ValueError(
            f"current/baseline shape mismatch: {current_tokens.shape} vs {baseline_tokens.shape}."
        )
    current_n = _l2_normalize(current_tokens.astype(np.float32))
    baseline_n = _l2_normalize(baseline_tokens.astype(np.float32))
    cosine = np.sum(current_n * baseline_n, axis=-1)
    return np.mean(1.0 - cosine, axis=-1).astype(np.float32)


def gate_weights_from_change(
    change: np.ndarray,
    *,
    tau: float,
    temperature: float,
) -> np.ndarray:
    """Return ``sigmoid((change - tau) / temperature)`` gate weights."""

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    logits = (np.asarray(change, dtype=np.float32) - float(tau)) / float(temperature)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


@dataclass(frozen=True)
class Intervention:
    """A named tactile counterfactual."""

    name: str


DEFAULT_INTERVENTIONS = (
    Intervention("baseline_fixed"),
    Intervention("baseline_recomputed"),
    Intervention("current_only"),
    *(Intervention(f"drop_sensor_{sensor}") for sensor in range(4)),
)


@dataclass(frozen=True)
class InterventionResult:
    tactile: np.ndarray
    gate: np.ndarray
    recomputed_gate: bool


def apply_intervention(
    name,
    tactile,
    baseline,
    original_gate,
    *,
    tau,
    temperature,
    sensor_count=4,
):
    """Return a counterfactual tactile window and its reporting gate labels."""

    if isinstance(name, Intervention):
        name = name.name
    tactile = np.asarray(tactile, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    original_gate = np.asarray(original_gate, dtype=np.float32)
    if tactile.ndim != 4 or baseline.shape != (
        tactile.shape[0],
        tactile.shape[2],
        tactile.shape[3],
    ):
        raise ValueError("expected tactile [B,T,N,D] and matching baseline [B,N,D]")
    if tactile.shape[1] == 0:
        raise ValueError("expected tactile window with positive time dimension")
    if original_gate.shape != (tactile.shape[0],):
        raise ValueError("expected original_gate [B]")
    result = tactile.copy()
    gate = original_gate.copy()
    recomputed = False
    if name in {"baseline_fixed", "baseline_recomputed"}:
        result[:] = baseline[:, None, :, :]
        recomputed = name == "baseline_recomputed"
    elif name == "current_only":
        result[:] = tactile[:, -1:, :, :]
    elif name.startswith("drop_sensor_"):
        sensor = int(name.rsplit("_", 1)[1])
        if not 0 <= sensor < min(sensor_count, tactile.shape[2]):
            raise ValueError(f"sensor index out of range: {sensor}")
        result[:, :, sensor, :] = baseline[:, None, sensor, :]
    else:
        raise ValueError(f"unsupported intervention: {name}")
    if recomputed:
        change = tactile_change_from_tokens(result[:, -1], baseline)
        gate = gate_weights_from_change(change, tau=tau, temperature=temperature)
    return InterventionResult(result, gate, recomputed)
