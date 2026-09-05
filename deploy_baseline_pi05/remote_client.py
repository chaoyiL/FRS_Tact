"""Run direct Pi0.5 tactile decoding through the existing scheduling bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
from uuid import uuid4
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .deployment import (
    TACTILE_KEYS,
    DeploymentConfig,
    load_deployment_config,
    make_server_config,
    preflight_deployment_assets,
)
from .protocol import ScheduleChunkEnd, ScheduleChunkStart, ScheduleSteerAck, ScheduleSteerRequest


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "deploy_baseline_pi05.yaml"


def _array_copy(value: Any) -> np.ndarray:
    return np.array(np.asarray(value, dtype=np.float32), copy=True)


def _require_matching_id(result: Any, name: str, expected: int, source: str) -> None:
    try:
        actual = getattr(result, name)
    except AttributeError as error:
        raise RuntimeError(f"{source} result must expose {name}") from error
    if actual != expected:
        raise RuntimeError(f"{source} result {name} does not match request")


def _require_matching_ids(result: Any, *expected: tuple[str, int], source: str) -> None:
    for name, value in expected:
        _require_matching_id(result, name, value, source)


def _trace_identity(config: DeploymentConfig) -> dict[str, str]:
    return {
        "config_path": str(config.config_path),
        "config_sha256": hashlib.sha256(config.config_path.read_bytes()).hexdigest(),
        "source_checkpoint": str(config.source.checkpoint),
        "tactile_encoder_checkpoint": str(config.tactile_encoder.checkpoint),
        "direct_decoder_checkpoint": str(config.direct_decoder.checkpoint),
    }


def _chunk_trace(ready: Any, identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "protocol": "frs_steering_v1",
        "identity": dict(identity or {}),
        "chunk_id": ready.chunk_id,
        "coarse_normalized_action": _array_copy(ready.action_vla_normalized),
        "coarse_action": _array_copy(ready.action_vla),
        "prediction_started_at": float(ready.prediction_started_at),
        "prediction_finished_at": float(ready.prediction_finished_at),
        "prediction_duration_s": float(ready.prediction_finished_at - ready.prediction_started_at),
    }


def _steer_trace(result: Any, identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {
        "protocol": "frs_steering_v1",
        "identity": dict(identity or {}),
        "chunk_id": result.chunk_id,
        "request_id": result.request_id,
        "action_index": result.action_index,
        "coarse_normalized_action": _array_copy(result.action_vla_normalized),
        "corrected_normalized_action": _array_copy(result.decoded_normalized),
        "selected_normalized_action": _array_copy(result.selected_normalized),
        "selected_action": _array_copy(result.selected_action),
        "delta_rms": float(diagnostics.delta_rms),
        "max_normalized_action_abs": float(diagnostics.max_normalized_action_abs),
        "encode_started_at": float(result.encode_started_at),
        "encode_finished_at": float(result.encode_finished_at),
        "encode_duration_s": float(result.encode_finished_at - result.encode_started_at),
        "decode_started_at": float(result.decode_started_at),
        "decode_finished_at": float(result.decode_finished_at),
        "decode_duration_s": float(result.decode_finished_at - result.decode_started_at),
    }


def _copy_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _copy_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_payload(item) for item in value)
    if isinstance(value, list):
        return [_copy_payload(item) for item in value]
    return value


class BoundedTraceSaver:
    """Persist diagnostic observation/action records on a bounded, fail-visible queue."""

    _STOP = object()

    def __init__(self, output_dir: Path, *, queue_size: int, writer: Callable[[dict[str, Any]], None] | None = None) -> None:
        if queue_size < 1:
            raise ValueError("logging.queue_size must be positive")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.session_dir: Path | None = None
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(queue_size)
        self._writer = writer or self._write
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._sequence = 0

    def start(self) -> None:
        if self._started:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.output_dir / f"session_{uuid4().hex}"
        self.session_dir.mkdir()
        self._started = True
        self._thread = threading.Thread(target=self._worker, name="direct-trace-saver", daemon=True)
        self._thread.start()

    def submit(self, payload: Mapping[str, Any]) -> None:
        if not self._started or self._closed:
            raise RuntimeError("trace saver is not accepting records")
        self._raise_failure()
        self._sequence += 1
        copied = _copy_payload(payload)
        copied["_sequence"] = self._sequence
        try:
            self._queue.put_nowait(copied)
        except queue.Full as error:
            raise RuntimeError("trace saver queue is full; diagnostic control record was not dropped") from error

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, dict)
                self._writer(item)
            except BaseException as error:
                with self._failure_lock:
                    if self._failure is None:
                        self._failure = error
            finally:
                self._queue.task_done()

    def _raise_failure(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("trace saver failed") from failure

    def flush(self) -> None:
        self._queue.join()
        self._raise_failure()

    def close(self) -> None:
        if self._closed:
            self._raise_failure()
            return
        failure: RuntimeError | None = None
        try:
            self.flush()
        except RuntimeError as error:
            failure = error
        finally:
            if self._started:
                self._queue.put(self._STOP)
                assert self._thread is not None
                self._thread.join()
            self._closed = True
        if failure is not None:
            raise failure
        self._raise_failure()

    def _write(self, payload: dict[str, Any]) -> None:
        arrays: dict[str, np.ndarray] = {}

        def encode(value: Any, path: str) -> Any:
            if isinstance(value, np.ndarray):
                key = path.replace(".", "_")
                arrays[key] = np.array(value, copy=True)
                return {"ndarray": key, "shape": list(value.shape), "dtype": value.dtype.str}
            if isinstance(value, Mapping):
                return {str(key): encode(item, f"{path}.{key}") for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item, f"{path}_{index}") for index, item in enumerate(value)]
            return value

        metadata = encode(payload, "record")
        kind = str(payload.get("kind", "trace"))
        iteration = int(payload.get("iteration", 0))
        sequence = int(payload["_sequence"])
        assert self.session_dir is not None
        prefix = self.session_dir / f"{iteration:06d}_{kind}_{sequence:06d}"
        prefix.mkdir()
        (prefix / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        if arrays:
            np.savez_compressed(prefix / "arrays.npz", **arrays)


def _make_trace_saver(config: DeploymentConfig) -> BoundedTraceSaver | None:
    if not config.logging.save_observations:
        return None
    saver = BoundedTraceSaver(config.logging.output_dir, queue_size=config.logging.queue_size)
    saver.start()
    return saver


def check(config_path: Path) -> DeploymentConfig:
    """Parse config and preflight local assets without heavyweight imports or connecting."""
    path = Path(config_path).expanduser().resolve()
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    print(f"[check] deploy config path: {path}")
    print(f"[check] deploy config sha256: {digest}")
    config = load_deployment_config(path)
    preflight_deployment_assets(config)
    return config


def _make_runtime(config: DeploymentConfig) -> Any:
    """Import heavyweight JAX/Torch deployment code only for a real run."""
    from .checkpoint import load_decoder
    from .deployment import expected_source_contract
    from .policy import Pi05VisualPolicy
    from .runtime import DirectDecoderRuntime
    from .tactile_encoder import FrozenTactileEncoder

    policy = Pi05VisualPolicy(config)
    encoder = FrozenTactileEncoder(config.tactile_encoder.checkpoint, tactile_keys=config.tactile_encoder.tactile_keys, key_map=config.tactile_encoder.key_map)
    decoder = load_decoder(config.direct_decoder.checkpoint, device=config.direct_decoder.device, expected_source=expected_source_contract(config))
    return DirectDecoderRuntime(policy=policy, tactile_encoder=encoder, decoder=decoder, action_dim=config.source.action_dim, max_normalized_action_abs=config.control.max_normalized_action_abs, max_normalized_delta_rms=config.control.max_normalized_delta_rms, device=config.direct_decoder.device)


def _warmup_observation() -> dict[str, np.ndarray]:
    observation = {
        "observation.state": np.zeros(20, dtype=np.float32),
        "observation.images.camera0": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.camera1": np.zeros((224, 224, 3), dtype=np.uint8),
    }
    observation.update({key: np.zeros((224, 224, 3), dtype=np.uint8) for key in TACTILE_KEYS})
    return observation


def warmup(runtime: Any, config: DeploymentConfig, observation: Mapping[str, Any]) -> None:
    """Warm every direct runtime stage without constructing a robot bridge or action message."""
    for index in range(config.runtime.warmup_runs):
        warmup_id = -(index + 1)
        begun = False
        try:
            ready = runtime.begin_chunk(warmup_id, observation, config.observation.language_prompt, seed=config.source.seed, num_steps=config.source.sample_steps)
            begun = True
            _require_matching_id(ready, "chunk_id", warmup_id, "begin_chunk")
            result = runtime.steer_action(warmup_id, 0, observation, 0)
            _require_matching_ids(result, ("chunk_id", warmup_id), ("request_id", 0), ("action_index", 0), source="steer_action")
        finally:
            if begun:
                runtime.end_chunk(warmup_id)


def run_schedule(bridge: Any, runtime: Any, *, task: str, observation_timeout_s: float, action_ack_timeout_s: float, seed: int, sample_steps: int, max_iterations: int, trace_identity: Mapping[str, Any] | None = None, saver: BoundedTraceSaver | None = None, save_every: int = 1) -> None:
    """Process only start/steer/ack/end scheduling messages with fail-stop ordering."""
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if save_every < 1:
        raise ValueError("logging.save_every must be positive")
    completed = 0
    previous_chunk_id: int | None = None
    identity = dict(trace_identity or {})
    while max_iterations == 0 or completed < max_iterations:
        start = bridge.receive_schedule_message(observation_timeout_s)
        if not isinstance(start, ScheduleChunkStart):
            raise RuntimeError(f"expected ScheduleChunkStart, got {type(start).__name__}")
        if start.execution_mode != "block":
            raise RuntimeError("direct decoder scheduling requires block execution mode")
        if start.action_horizon != 50:
            raise RuntimeError("direct decoder scheduling requires a 50-step action horizon")
        if previous_chunk_id is not None and start.chunk_id <= previous_chunk_id:
            raise RuntimeError("scheduling chunk ids must be strictly increasing")
        active = False
        try:
            ready = runtime.begin_chunk(start.chunk_id, start.observation, task, seed=seed, num_steps=sample_steps)
            active = True
            _require_matching_id(ready, "chunk_id", start.chunk_id, "begin_chunk")
            prediction_trace = _chunk_trace(ready, identity)
            bridge.send_frs_chunk_ready(start.obs_seq, start.chunk_id, prediction_trace)
            if saver is not None and (completed + 1) % save_every == 0:
                saver.submit({"kind": "chunk", "iteration": completed + 1, "obs_seq": start.obs_seq, "chunk_id": start.chunk_id, "observation": start.observation, "trace": prediction_trace})
            while True:
                message = bridge.receive_schedule_message(observation_timeout_s)
                if isinstance(message, ScheduleChunkEnd):
                    if message.chunk_id != start.chunk_id:
                        raise RuntimeError("ScheduleChunkEnd does not match the active chunk")
                    runtime.end_chunk(start.chunk_id)
                    active = False
                    previous_chunk_id = start.chunk_id
                    completed += 1
                    break
                if not isinstance(message, ScheduleSteerRequest):
                    raise RuntimeError(f"expected ScheduleSteerRequest, got {type(message).__name__}")
                if message.chunk_id != start.chunk_id:
                    raise RuntimeError("ScheduleSteerRequest does not match the active chunk")
                result = runtime.steer_action(message.chunk_id, message.request_id, message.observation, message.action_index)
                _require_matching_ids(result, ("chunk_id", message.chunk_id), ("request_id", message.request_id), ("action_index", message.action_index), source="steer_action")
                action = np.asarray(result.selected_action)
                if action.shape != (20,) or action.dtype.kind != "f" or not np.isfinite(action).all():
                    raise RuntimeError("direct decoder must return one finite full physical 20D action")
                steer_trace = _steer_trace(result, identity)
                bridge.send_frs_steer_action(message.chunk_id, message.request_id, message.action_index, action, trace=steer_trace)
                if saver is not None and (completed + 1) % save_every == 0:
                    saver.submit({"kind": "steer", "iteration": completed + 1, "chunk_id": message.chunk_id, "request_id": message.request_id, "action_index": message.action_index, "observation": message.observation, "trace": steer_trace})
                acknowledgement = bridge.receive_schedule_message(action_ack_timeout_s)
                if not isinstance(acknowledgement, ScheduleSteerAck):
                    raise RuntimeError(f"expected ScheduleSteerAck, got {type(acknowledgement).__name__}")
                if (acknowledgement.chunk_id, acknowledgement.request_id, acknowledgement.action_index) != (message.chunk_id, message.request_id, message.action_index):
                    raise RuntimeError("ScheduleSteerAck does not match its request")
        finally:
            if active:
                try:
                    runtime.end_chunk(start.chunk_id)
                except Exception:
                    pass


def _token(config: DeploymentConfig) -> str | None:
    token = config.connection.token or os.environ.get(config.connection.token_env)
    if config.connection.require_token and not token:
        raise ValueError(f"{config.connection.token_env} must be set because connection.require_token is true")
    return token


def _close_saver(saver: BoundedTraceSaver | Any, active_error: BaseException | None) -> None:
    try:
        saver.close()
    except Exception as error:
        if active_error is None:
            raise
        active_error.add_note(f"trace saver close failed: {error!r}")


def run(config_path: Path, max_iterations_override: int | None = None, *, bridge_factory: Callable[..., Any] | None = None, runtime_factory: Callable[[DeploymentConfig], Any] | None = None) -> None:
    """Run a real bridge session and clean up/stop best-effort after every exit path."""
    config = check(config_path)
    max_iterations = config.runtime.max_iterations if max_iterations_override is None else int(max_iterations_override)
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if bridge_factory is None:
        from .bridge_client import RobotBridgeClient

        bridge_factory = RobotBridgeClient
    runtime = (runtime_factory or _make_runtime)(config)
    bridge: Any | None = None
    saver: BoundedTraceSaver | None = None
    try:
        saver = _make_trace_saver(config)
        bridge = bridge_factory(config.connection.address, config.connection.port, _token(config), retry_interval_s=config.connection.retry_interval_s, ping_interval_s=config.connection.ping_interval_s, ping_timeout_s=config.connection.ping_timeout_s)
        bridge.send_config(make_server_config(config))
        obs_seq, observation = bridge.receive_observation(config.connection.observation_timeout_s)
        warmup(runtime, config, observation)
        if not config.runtime.auto_start:
            input("[client] Ready. Press Enter to send START to the robot server... ")
        bridge.send_state("start", obs_seq=obs_seq)
        run_schedule(bridge, runtime, task=config.observation.language_prompt, observation_timeout_s=config.connection.observation_timeout_s, action_ack_timeout_s=config.connection.action_ack_timeout_s, seed=config.source.seed, sample_steps=config.source.sample_steps, max_iterations=max_iterations, trace_identity=_trace_identity(config), saver=saver, save_every=config.logging.save_every)
    finally:
        active_error = sys.exception()
        if bridge is not None:
            try:
                bridge.send_state("stop")
            except Exception:
                pass
            try:
                bridge.close()
            except Exception:
                pass
        if saver is not None:
            _close_saver(saver, active_error)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.check:
            check(args.config)
        else:
            run(args.config, args.max_iterations)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"[client] fail-stop: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
