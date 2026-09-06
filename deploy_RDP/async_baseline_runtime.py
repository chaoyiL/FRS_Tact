"""Latest-observation LDP worker with foreground tactile-conditioned AT decoding."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import threading
import time

import numpy as np
import torch

from baseline_runtime import BaselineRDPRuntime


@dataclass(frozen=True)
class _Plan:
    generation: int
    timestamp: float
    bases: dict
    observation: dict
    ready_event: object = None
    latent: object = None
    planning_ms: float = 0.0


class AsyncBaselineRDPRuntime(BaselineRDPRuntime):
    """Only LDP runs in the worker; each predict still decodes fresh tactile input.

    Plan age is measured from its captured observation, not from adoption. A
    single replaceable mailbox bounds pending work even when LDP is slow.
    """

    def __init__(self, *args, inference_fps=6.0, **kwargs):
        if not math.isfinite(inference_fps) or inference_fps <= 0:
            raise ValueError('inference_fps must be finite and positive')
        self.inference_fps = float(inference_fps)
        self._condition = threading.Condition()
        self._generation = 0
        self._closed = False
        self._pending_plan = None
        self._completed_plan = None
        self._worker_error = None
        self._terminal_worker_error = None
        self._planning = False
        super().__init__(*args, **kwargs)
        if self.inference_fps > self.control_frequency:
            raise ValueError('inference_fps cannot exceed control_frequency')
        raw = self.n_obs_steps * self.temporal_downsample_ratio
        self.action_horizon = min(int(self.policy.n_action_steps), int(self.policy.original_horizon) - raw + 1)
        if self.action_horizon < 1 or self.slow_update_interval > self.action_horizon:
            raise ValueError('slow_update_interval exceeds the usable action horizon')
        self._worker = threading.Thread(target=self._planning_loop, name='rdp-ldp-planner', daemon=True)
        try:
            self._worker.start()
        except Exception:
            self._closed = True
            raise

    def reset(self):
        with self._condition:
            super().reset()
            self._generation += 1
            self._pending_plan = None
            self._completed_plan = None
            self._worker_error = self._terminal_worker_error
            self._last_request_timestamp = None
            self._last_adoption_timestamp = None
            self.last_planning_ms = 0.0
            self.last_diagnostics = {}
            self._condition.notify_all()

    @property
    def planner_status(self):
        with self._condition:
            if self._worker_error is not None:
                return 'failed'
            return 'planning' if self._planning else 'ready' if self._completed_plan is not None else 'idle'

    def close(self):
        with self._condition:
            self._closed = True
            self._generation += 1
            self._pending_plan = None
            self._completed_plan = None
            self._condition.notify_all()
        worker = getattr(self, '_worker', None)
        if worker is not None and worker.ident is not None:
            worker.join()

    def _raise_worker_error(self):
        if self._closed:
            raise RuntimeError('Async RDP runtime is closed')
        if self._worker_error is not None:
            raise RuntimeError('RDP background planner failed') from self._worker_error

    @torch.inference_mode()
    def _planning_loop(self):
        cuda = torch.device(self.device).type == 'cuda'
        try:
            stream = torch.cuda.Stream(device=self.device) if cuda else None
        except Exception as error:
            with self._condition:
                self._terminal_worker_error = error
                self._worker_error = error
                self._condition.notify_all()
            return
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending_plan is not None)
                if self._closed:
                    return
                request = self._pending_plan
                self._pending_plan = None
                self._planning = True
            started = time.perf_counter()
            try:
                if cuda:
                    with torch.cuda.stream(stream):
                        stream.wait_event(request.ready_event)
                        result = self.policy.predict_action(request.observation,
                            dataset_obs_temporal_downsample_ratio=self.temporal_downsample_ratio,
                            return_latent_action=True)
                        latent = result['action'][:, 0].detach()
                    # Publish only completed tensors, without stalling the AT stream.
                    stream.synchronize()
                else:
                    result = self.policy.predict_action(request.observation,
                        dataset_obs_temporal_downsample_ratio=self.temporal_downsample_ratio,
                        return_latent_action=True)
                    latent = result['action'][:, 0].detach()
                plan = _Plan(request.generation, request.timestamp, request.bases,
                             {}, latent=latent, planning_ms=(time.perf_counter() - started) * 1000)
                with self._condition:
                    if not self._closed and request.generation == self._generation:
                        self._completed_plan = plan
            except Exception as error:
                with self._condition:
                    if not self._closed and request.generation == self._generation:
                        self._worker_error = error
                        self._pending_plan = None
            finally:
                with self._condition:
                    self._planning = False
                    self._condition.notify_all()

    def _request_plan(self, timestamp):
        active_bases = self.chunk_bases
        self.chunk_bases = {}
        try:
            snapshot = {key: value.detach().clone() for key, value in self._slow_policy_observation().items()}
            bases = {arm: base.copy() for arm, base in self.chunk_bases.items()}
        finally:
            self.chunk_bases = active_bases
        event = None
        if torch.device(self.device).type == 'cuda':
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        with self._condition:
            self._raise_worker_error()
            self._pending_plan = _Plan(self._generation, timestamp, bases, snapshot, event)
            self._last_request_timestamp = timestamp
            self._condition.notify_all()

    def _tick(self, timestamp, plan_timestamp):
        return int(math.floor((timestamp - plan_timestamp) * self.control_frequency + .5))

    def _decode_at(self, latent, plan_timestamp, target_timestamp, tactile, *, causal, capture_timestamp=None):
        """Decode the execution tick, assuming unknown future touch stays constant.

        Hold-last extrapolation is a deployment assumption, not a forecast of
        contact. Keep the original nearest-sample history and its final actual
        touch replacement, then repeat that actual sample at future grid steps.
        Never condition on observations captured after this action's capture.
        """
        tick = self._tick(target_timestamp, plan_timestamp)
        raw = self.n_obs_steps * self.temporal_downsample_ratio
        times = plan_timestamp + np.arange(1 - raw, tick + 1) / self.control_frequency
        if causal:
            capture_timestamp = (self.observation_timestamps[-1] if capture_timestamp is None
                                 else capture_timestamp)
            capture_tick = self._tick(capture_timestamp, plan_timestamp)
            prefix_times = plan_timestamp + np.arange(1 - raw, capture_tick + 1) / self.control_frequency
            captured = np.asarray(self.observation_timestamps)
            available_indices = np.flatnonzero(captured <= capture_timestamp)
            if not len(available_indices):
                raise RuntimeError('No tactile observation is available at the capture timestamp')
            available = captured[available_indices]
            nearest = np.abs(available[:, None] - prefix_times[None, :]).argmin(axis=0)
            history = [self.observation_history[available_indices[i]]['tactile_embedding']
                       for i in nearest]
            history[-1] = tactile
            history.extend([tactile] * (tick - capture_tick))
        else:
            # Preserve the historical offline / warmup capture-time path.
            history = [frame['tactile_embedding'] for frame in self._history_at_times(times)]
            history[-1] = tactile
        result = self.policy.predict_from_latent_action(latent,
            {'tactile_embedding': torch.cat(history, dim=1)},
            extended_obs_last_step=len(history),
            dataset_obs_temporal_downsample_ratio=self.temporal_downsample_ratio)
        action = result['action'][0, -1].detach().float().cpu().numpy()
        if action.shape != (self.profile.action_dim,) or not np.isfinite(action).all():
            raise RuntimeError(f'Expected finite {self.profile.action_dim}D action, got {action.shape}')
        return action, history

    def _plan_diagnostics(self, action, tick, latent, plan_timestamp, bases):
        active_bases = self.chunk_bases
        self.chunk_bases = bases
        try:
            absolute = self.absolute_action_target(action).tolist() if action is not None else None
        finally:
            self.chunk_bases = active_bases
        return {
            'capture_timestamp': float(plan_timestamp),
            'target_tick': int(tick),
            'bases': {arm: base.tolist() for arm, base in bases.items()},
            'latent': latent.detach().float().cpu().tolist(),
            # Absolute within each arm's episode frame, not world coordinates.
            'absolute_action': absolute,
        }

    @torch.inference_mode()
    def predict(self, observation):
        with self._condition:
            self._raise_worker_error()
        timestamp = observation.get('observation.timestamp')
        if timestamp is None:
            timestamp = self.step / self.control_frequency
            if self.observation_timestamps and timestamp <= self.observation_timestamps[-1]:
                raise ValueError('observation.timestamp disappeared during a timed rollout')
        if isinstance(timestamp, (bool, np.bool_)) or not isinstance(timestamp, Real) or not math.isfinite(timestamp):
            raise ValueError('observation.timestamp must be finite')
        timestamp = float(timestamp)
        if self.observation_timestamps and timestamp <= self.observation_timestamps[-1]:
            raise ValueError('observation.timestamp must strictly increase')
        deadline_key = 'observation.action_target_timestamp'
        has_deadline = deadline_key in observation
        target_timestamp = observation.get(deadline_key, timestamp)
        if has_deadline and (isinstance(target_timestamp, (bool, np.bool_))
                or not isinstance(target_timestamp, Real)
                or not math.isfinite(target_timestamp) or target_timestamp <= timestamp):
            raise ValueError('observation.action_target_timestamp must be finite and after observation.timestamp')
        target_timestamp = float(target_timestamp)
        current, tactile = self._prepare_observation(observation)
        self.observation_history.append(current)
        self.observation_timestamps.append(timestamp)
        raw = self.n_obs_steps * self.temporal_downsample_ratio
        retained = raw + self.action_horizon + int(self.policy.original_horizon) + 1
        self.observation_history = self.observation_history[-retained:]
        self.observation_timestamps = self.observation_timestamps[-retained:]
        # Epoch-second float precision is about 0.2 us; tolerate that rounding.
        if self._last_request_timestamp is None or timestamp - self._last_request_timestamp >= 1 / self.inference_fps - 1e-6:
            self._request_plan(timestamp)
        previous = None
        with self._condition:
            if self.latent_action is None:
                # Bootstrap is the only blocking LDP call in an episode.
                self._condition.wait_for(lambda: self._completed_plan is not None or self._worker_error is not None or self._closed)
            self._raise_worker_error()
            candidate = self._completed_plan
            # Prefer the configured cadence, but do not discard an available
            # fresh plan when a previously delayed plan reaches its last step.
            due = (self.latent_action is None
                   or self._tick(timestamp, self._last_adoption_timestamp) >= self.slow_update_interval
                   or self._tick(target_timestamp, self.plan_timestamp) >= self.action_horizon - 1)
            adopted = False
            if candidate is not None and due:
                candidate_tick = self._tick(target_timestamp, candidate.timestamp)
                self._completed_plan = None
                if 0 <= candidate_tick < self.action_horizon:
                    if self.latent_action is not None:
                        previous = (self.latent_action, self.plan_timestamp, self.chunk_bases)
                    self.latent_action = candidate.latent
                    self.plan_timestamp = candidate.timestamp
                    self.chunk_bases = candidate.bases
                    self._last_adoption_timestamp = timestamp
                    self.last_planning_ms = candidate.planning_ms
                    adopted = True
        if self.latent_action is None:
            raise RuntimeError('RDP action plan exhausted before its execution deadline; no valid LDP plan is ready')
        tick = self._tick(target_timestamp, self.plan_timestamp)
        if tick < 0 or tick >= self.action_horizon:
            raise RuntimeError(f'RDP action plan exhausted at tick {tick}; no fresh LDP plan is ready')
        switch = None
        if previous is not None:
            old_latent, old_timestamp, old_bases = previous
            old_tick = self._tick(target_timestamp, old_timestamp)
            old_action = None
            if 0 <= old_tick < self.action_horizon:
                old_action, _ = self._decode_at(old_latent, old_timestamp, target_timestamp,
                                                tactile, causal=has_deadline, capture_timestamp=timestamp)
            switch = {
                'execution_timestamp': target_timestamp,
                'frame': 'per_arm_episode',
                'old': self._plan_diagnostics(old_action, old_tick, old_latent, old_timestamp, old_bases),
            }
        action, self.tactile_history = self._decode_at(self.latent_action,
            self.plan_timestamp, target_timestamp, tactile, causal=has_deadline, capture_timestamp=timestamp)
        if adopted:
            new = self._plan_diagnostics(action, tick, self.latent_action, self.plan_timestamp, self.chunk_bases)
            if switch is None:
                switch = {'execution_timestamp': target_timestamp, 'frame': 'per_arm_episode', 'old': None}
            switch['new'] = new
        capture_tick = self._tick(timestamp, self.plan_timestamp)
        self.last_diagnostics = {
            'capture_timestamp': timestamp,
            'execution_timestamp': target_timestamp,
            'plan_capture_timestamp': float(self.plan_timestamp),
            'capture_tick': capture_tick,
            'target_tick': tick,
            'lookahead_ticks': tick - capture_tick,
            'lookahead_seconds': target_timestamp - timestamp,
            'tactile_extrapolation': 'hold_last' if has_deadline else 'capture_time_legacy',
            'adopted': adopted,
            'planner_status': self.planner_status,
            'plan_switch': switch,
        }
        self.last_decoder_tick = tick
        self.step += 1
        return self._action_for_execution(action, observation)[None].astype(np.float32, copy=False), adopted
