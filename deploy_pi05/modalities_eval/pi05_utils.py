"""pi0.5 analogue of modalities_eval/utils.py's SmolVLAEvalModel: checkpoint + dataset-sample ->
Observation glue for FRS action_cache generation.

A separate class rather than a SmolVLAEvalModel subclass/branch: pi0.5's observation format
differs enough (fixed base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb image keys, and -- critically
-- state baked into the tokenized prompt rather than passed as a continuous input, see
pi05_jax/tokenizer.py) that sharing SmolVLAEvalModel's code would mean more branching than
duplication.

The preprocessing itself is *not* hand-written: `prepare_sample` composes the same vendored openpi
transforms the trainer uses (`pi05_jax/transforms.py`, driven by
`pi05_jax/policies/pick_tube_policy.py`), in the same order openpi's
`training/config.py:ModelTransformFactory` and `training/data_loader.py:transform_dataset` use --

    repack -> PickTubeInputs -> Normalize -> ResizeImages -> TokenizePrompt -> PadStatesAndActions

-- so a sample fed to the action cache is preprocessed byte-identically to one fed to training.
The class keeps its own constructor (camera_map / explicit norm stats) rather than taking a
`TrainConfig`, because FRS's cache tools are configured from YAML per dataset; see
prepare_pi05.py.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.sample_utils import (
    action_delta_timestamps,
    lerobot_sample_to_observation,
    resolve_action_key,
)
from lerobot.policies.pi05_jax import Observation, PaligemmaTokenizer, Pi0, Pi0Config, load_pi0, transforms
from lerobot.policies.pi05_jax.model import IMAGE_KEYS, IMAGE_RESOLUTION
from lerobot.policies.pi05_jax.normalize import NormStats
from lerobot.policies.pi05_jax.policies.pick_tube_policy import PickTubeInputs

Array = jax.Array


def _match_norm_stats_dim(stats: NormStats, dim: int, *, label: str) -> NormStats:
    """Reconcile borrowed norm stats (e.g. an official asset_id for a different robot) with this
    dataset's actual feature width.

    See pi05_frs_plan.md: pi05_base's shipped assets (droid/franka/trossen/ur5e_dual/...) don't
    include one for a new dataset like pick_tube. If told to reuse one anyway and it's narrower
    than this dataset's state/action dim, the extra dims are padded to an identity transform
    (mean=0/std=1, or q01=-1/q99=1 under quantile norm -- both reduce `Normalize`'s formula to `x`,
    up to the `1e-6` epsilon it adds to the denominator, i.e. ~1e-6 relative; verified against the
    real trossen stats). This is a real approximation, not just unit padding: those extra dims pass
    through *unnormalized*, which for pi0.5 also means they won't be meaningfully discretized
    into the tokenized prompt (see tokenizer.py) since they aren't guaranteed to be in [-1, 1].
    Loud on purpose (prints, doesn't just silently do this) -- narrower-than-needed stats mean
    someone chose to reuse a mismatched asset_id rather than compute real ones with
    `tools/compute_pi05_norm_stats.py`.
    """
    current = np.asarray(stats.mean).shape[-1]
    if current == dim:
        return stats
    if current > dim:
        raise ValueError(f"{label} norm stats have {current} dims, wider than this dataset's {dim}")
    pad = dim - current
    print(
        f"WARNING: {label} norm stats only cover {current}/{dim} dims (borrowed asset_id is for "
        f"a different robot) -- padding the extra {pad} dims to an identity transform. "
        "See _match_norm_stats_dim's docstring / pi05_frs_plan.md.",
        flush=True,
    )

    def pad_with(value: np.ndarray | None, fill: float) -> np.ndarray | None:
        return None if value is None else np.pad(np.asarray(value), (0, pad), constant_values=fill)

    return NormStats(
        mean=np.pad(np.asarray(stats.mean), (0, pad), constant_values=0.0),
        std=np.pad(np.asarray(stats.std), (0, pad), constant_values=1.0),
        q01=pad_with(stats.q01, -1.0),
        q99=pad_with(stats.q99, 1.0),
    )


class Pi05SampleProcessor:
    """Normalize and convert one LeRobot sample into pi0.5 model inputs."""

    def __init__(
        self,
        *,
        dataset_repo_id: str,
        dataset_root: str | pathlib.Path | None = None,
        dataset_revision: str | None = None,
        action_key: str | None = None,
        rename_map: Mapping[str, str] | None = None,
        camera_map: Mapping[str, str],
        state_stats: NormStats,
        action_stats: NormStats,
        use_quantile_norm: bool = True,
        action_dim: int = 32,
        action_horizon: int = 50,
        max_token_len: int | None = None,
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
    ):
        """
        Args:
            camera_map: pi0.5 image key (subset of `base_0_rgb`/`left_wrist_0_rgb`/
                `right_wrist_0_rgb`) -> dataset observation key, *after* `rename_map` is applied
                (e.g. `{"base_0_rgb": "observation.images.camera1"}`). Keys from `IMAGE_KEYS` not
                present in `camera_map` are zero-filled and masked off by `PickTubeInputs`.
            state_stats / action_stats: required, not loaded automatically -- see this module's
                docstring and pi05_frs_plan.md for why (no norm stats exist for a brand-new
                dataset in the pretrained pi05_base checkpoint's assets).
            use_quantile_norm: pi0.5 (pi05=True) bakes the discretized *state* into the tokenized
                prompt assuming it is already in [-1, 1] (see pi05_jax/tokenizer.py) -- quantile
                normalization guarantees that range; z-score normalization does not. Leave True
                unless you have a specific reason and have re-checked that assumption.
            paligemma_variant / action_expert_variant: must match the `TrainConfig` the checkpoint
                was trained with. The defaults describe the official `pi05_base`; a LoRA
                fine-tune from `tools/train_pi05_jax.py pi05_pick_tube` needs
                `gemma_2b_lora`/`gemma_300m_lora` instead. Getting this wrong is caught by
                `policy_config.load_pi0` rather than silently discarding the LoRA weights.
        """
        unknown_cameras = set(camera_map) - set(IMAGE_KEYS)
        if unknown_cameras:
            raise ValueError(f"camera_map keys must be a subset of {IMAGE_KEYS}, got extra {sorted(unknown_cameras)}")

        self.config = Pi0Config(
            pi05=True,
            action_dim=action_dim,
            action_horizon=action_horizon,
            max_token_len=max_token_len,
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
        )

        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = pathlib.Path(dataset_root).expanduser() if dataset_root is not None else None
        self.dataset_revision = dataset_revision
        metadata = LeRobotDatasetMetadata(dataset_repo_id, root=self.dataset_root, revision=dataset_revision)
        self.dataset_root = metadata.root
        self.dataset_revision = metadata.revision
        self.action_key = resolve_action_key(metadata.features, action_key)
        self.fps = metadata.fps

        self.rename_map = dict(rename_map or {})
        self.camera_map = dict(camera_map)
        # Norm stats must match *this dataset's* raw feature width (e.g. pick_tube's 20-dim
        # state/actions), not the model's action_dim=32 -- normalization happens before the
        # separate PadStatesAndActions step below.
        state_dim = int(metadata.features["observation.state"]["shape"][0])
        action_feature_dim = int(metadata.features[self.action_key]["shape"][0])
        self.state_stats = _match_norm_stats_dim(state_stats, state_dim, label="state")
        self.action_stats = _match_norm_stats_dim(action_stats, action_feature_dim, label="action")
        self.use_quantile_norm = use_quantile_norm

        # The training-time pipeline, rebuilt here for a single sample. Order and contents match
        # `training/data_loader.py:transform_dataset` + `ModelTransformFactory` for PI05.
        norm_stats = {"state": self.state_stats, "actions": self.action_stats}
        self._inputs = transforms.compose(
            [
                transforms.RepackTransform(
                    {
                        "image": dict(self.camera_map),
                        "state": "observation.state",
                        "actions": self.action_key,
                        "prompt": "prompt",
                    }
                ),
                PickTubeInputs(model_type=self.config.model_type),
                transforms.Normalize(norm_stats, use_quantiles=use_quantile_norm),
                transforms.ResizeImages(*IMAGE_RESOLUTION),
                transforms.TokenizePrompt(
                    PaligemmaTokenizer(self.config.max_token_len),
                    discrete_state_input=self.config.discrete_state_input,
                ),
                transforms.PadStatesAndActions(self.config.action_dim),
            ]
        )
        self._unnormalize_actions = transforms.Unnormalize(
            {"actions": self.action_stats}, use_quantiles=use_quantile_norm
        )

    @property
    def action_horizon(self) -> int:
        return self.config.action_horizon

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    def _renamed_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Dataset sample -> the flat key space the repack transform reads.

        Mirrors `training/data_loader.py:RenameKeys` + `PromptFromTask`, except the action column
        keeps its own name here (the repack maps it explicitly, since a `Pi05SampleProcessor` is
        built per dataset and already knows `self.action_key`).
        """
        observation = lerobot_sample_to_observation(sample)
        renamed = {self.rename_map.get(key, key): value for key, value in observation.items()}
        renamed[self.action_key] = sample[self.action_key]
        renamed["prompt"] = np.asarray(str(sample.get("task", "")))
        return renamed

    def prepare_sample(self, sample: Mapping[str, Any]) -> tuple[Observation, Array, str]:
        renamed = self._renamed_sample(sample)
        if "observation.state" not in renamed:
            raise KeyError("observation.state is required")
        missing = [key for key in self.camera_map.values() if key not in renamed]
        if missing:
            raise KeyError(f"camera_map points at keys not in the sample: {missing}; have {sorted(renamed)}")

        prompt = str(np.asarray(renamed["prompt"]).item())
        # Normalize every leaf to a numpy array before building the Observation. This is the
        # single-sample analogue of what openpi's `data_loader._collate_fn` does for a batch, and
        # it is required, not cosmetic: `Observation` is `@at.typecheck`'d and its fields share
        # one `ArrayT` TypeVar, so the mixed types coming out of the transform chain --
        # `ResizeImages` returns *jax* arrays (it is jitted) while `state`/`actions`/tokens stay
        # numpy -- would fail the check. `image_mask` is worse: `PickTubeInputs` emits `np.True_`,
        # a numpy *scalar*, which is not an `np.ndarray` at all.
        data = jax.tree.map(np.asarray, self._inputs(renamed))
        return Observation.from_dict(data), jnp.asarray(data["actions"], dtype=jnp.float32), prompt

    def unnormalize_actions(self, actions: Array) -> Array:
        """Model-space actions -> real units, dropping the PadStatesAndActions zero padding."""
        real_dim = np.asarray(self.action_stats.mean).shape[-1]
        unnormalized = self._unnormalize_actions({"actions": np.asarray(actions)[..., :real_dim]})
        return jnp.asarray(unnormalized["actions"])


class Pi05EvalModel(Pi05SampleProcessor):
    """pi0.5 checkpoint plus the sample processor used by FRS cache generation."""

    def __init__(
        self,
        checkpoint: str | pathlib.Path,
        *,
        loaded_model: Pi0 | None = None,
        **processor_kwargs: Any,
    ):
        super().__init__(**processor_kwargs)
        if loaded_model is None:
            self.model = load_pi0(checkpoint, config=self.config)
            return

        actual = (
            loaded_model.action_dim,
            loaded_model.action_horizon,
            loaded_model.max_token_len,
            loaded_model.pi05,
        )
        expected = (
            self.config.action_dim,
            self.config.action_horizon,
            self.config.max_token_len,
            self.config.pi05,
        )
        if actual != expected:
            raise ValueError(
                "loaded pi0.5 model config does not match requested config: "
                f"actual={actual}, expected={expected}"
            )
        self.model = loaded_model


def stack_observations(observations: list[Observation]) -> Observation:
    if not observations:
        raise ValueError("at least one observation is required")
    return jax.tree.map(lambda *values: jnp.stack(values, axis=0), *observations)


def action_delta_timestamps_for(model: Pi05EvalModel) -> dict[str, list[float]]:
    return action_delta_timestamps(model.action_key, model.action_horizon, model.fps)
